"""SQLite-backed conversation history and per-turn feedback."""

import re
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


DATABASE_PATH = Path(__file__).resolve().parent.parent / "memory.db"
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_ -]?key|password|passwd|secret|token|access[_ -]?key)\b\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\b(?:gsk_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,})\b"),
)


def redact_secrets(text: str) -> str:
    """Remove common credential patterns before conversation text is persisted."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 2:
            redacted = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


@contextmanager
def _database() -> Iterator[sqlite3.Connection]:
    """Open the local database and ensure its schema exists."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session_time
                ON conversation_turns(session_id, id DESC);
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL UNIQUE,
                session_id TEXT NOT NULL,
                rating TEXT NOT NULL CHECK (rating IN ('helpful', 'not_helpful')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (turn_id) REFERENCES conversation_turns(id) ON DELETE CASCADE
            );
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(conversation_turns)")
        }
        if "sources_json" not in columns:
            connection.execute(
                "ALTER TABLE conversation_turns ADD COLUMN sources_json "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        yield connection
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        connection.close()


def save_turn(
    session_id: str,
    question: str,
    answer: str,
    user_id: str | None = None,
    sources: list[dict] | None = None,
) -> int:
    """Persist one user/assistant turn and return its database identifier."""
    if not session_id.strip():
        raise ValueError("session_id cannot be empty.")
    with _database() as connection:
        cursor = connection.execute(
            "INSERT INTO conversation_turns "
            "(session_id, user_id, question, answer, sources_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                user_id,
                redact_secrets(question),
                redact_secrets(answer),
                json.dumps(sources or [], ensure_ascii=True),
            ),
        )
        return int(cursor.lastrowid)


def get_recent_history(session_id: str, limit: int = 5) -> list[dict]:
    """Return the most recent conversation turns in chronological order."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    with _database() as connection:
        rows = connection.execute(
            "SELECT t.id, t.question, t.answer, t.sources_json, "
            "t.created_at, f.rating AS feedback "
            "FROM conversation_turns AS t "
            "LEFT JOIN feedback AS f ON f.turn_id = t.id "
            "WHERE t.session_id = ? ORDER BY t.id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    history = []
    for row in reversed(rows):
        turn = dict(row)
        turn["sources"] = json.loads(turn.pop("sources_json", "[]"))
        history.append(turn)
    return history


def clear_conversation(session_id: str) -> int:
    """Delete conversation turns for one session and return the count removed."""
    with _database() as connection:
        cursor = connection.execute(
            "DELETE FROM conversation_turns WHERE session_id = ?", (session_id,)
        )
        return cursor.rowcount


def save_feedback(session_id: str, turn_id: int, rating: str) -> None:
    """Save or update feedback for a turn owned by the specified session."""
    if rating not in {"helpful", "not_helpful"}:
        raise ValueError("rating must be 'helpful' or 'not_helpful'.")
    with _database() as connection:
        turn = connection.execute(
            "SELECT id FROM conversation_turns WHERE id = ? AND session_id = ?",
            (turn_id, session_id),
        ).fetchone()
        if turn is None:
            raise ValueError("Conversation turn was not found for this session.")
        connection.execute(
            "INSERT INTO feedback (turn_id, session_id, rating) VALUES (?, ?, ?) "
            "ON CONFLICT(turn_id) DO UPDATE SET rating = excluded.rating, "
            "created_at = CURRENT_TIMESTAMP",
            (turn_id, session_id, rating),
        )


def count_session_turns(session_id: str) -> int:
    """Return the number of persisted turns for one session."""
    with _database() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return int(row["count"])