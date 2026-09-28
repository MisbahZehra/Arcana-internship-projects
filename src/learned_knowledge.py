"""Explicitly approved, non-personal additions to the shared knowledge base."""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document


LEARNED_KNOWLEDGE_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "data" / "learned_knowledge"
)
_METADATA_START = "<!-- approved-knowledge-metadata\n"
_METADATA_END = "\n-->\n"
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)"
)
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_ -]?key|password|passwd|secret|token|access[_ -]?key)\b\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\b(?:gsk_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,})\b"),
)


def contains_sensitive_data(text: str) -> bool:
    """Detect common credentials and email addresses before global storage."""
    return bool(
        _EMAIL_PATTERN.search(text)
        or _PHONE_PATTERN.search(text)
        or _SSN_PATTERN.search(text)
        or any(pattern.search(text) for pattern in _SECRET_PATTERNS)
    )


def learned_knowledge_count() -> int:
    """Count approved text and Markdown files in the shared knowledge store."""
    if not LEARNED_KNOWLEDGE_DIRECTORY.exists():
        return 0
    return sum(
        1
        for path in LEARNED_KNOWLEDGE_DIRECTORY.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )


def save_approved_knowledge(
    content: str, session_id: str, description: str = ""
) -> Path:
    """Save content only when explicitly invoked by the user's approval action."""
    cleaned_content = content.strip()
    if not cleaned_content:
        raise ValueError("Approved knowledge cannot be empty.")
    if contains_sensitive_data(cleaned_content) or contains_sensitive_data(description):
        raise ValueError(
            "This content appears to contain a credential or personal email. "
            "Remove sensitive information before approving it."
        )

    LEARNED_KNOWLEDGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "source_type": "user-approved",
        "timestamp": timestamp,
        "session_id": session_id,
        "description": description.strip(),
    }
    filename = f"approved-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.md"
    file_path = LEARNED_KNOWLEDGE_DIRECTORY / filename
    payload = (
        _METADATA_START
        + json.dumps(metadata, ensure_ascii=True)
        + _METADATA_END
        + cleaned_content
        + "\n"
    )
    file_path.write_text(payload, encoding="utf-8")
    return file_path


def load_approved_knowledge_documents() -> list[Document]:
    """Load approved Markdown/text files and retain their source metadata."""
    if not LEARNED_KNOWLEDGE_DIRECTORY.exists():
        return []

    documents = []
    for file_path in sorted(LEARNED_KNOWLEDGE_DIRECTORY.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = file_path.read_text(encoding="utf-8")
        metadata = {"source_type": "user-approved"}
        if text.startswith(_METADATA_START):
            metadata_end = text.find(_METADATA_END, len(_METADATA_START))
            if metadata_end < 0:
                raise ValueError(f"Invalid approval metadata in {file_path.name}.")
            raw_metadata = text[len(_METADATA_START) : metadata_end]
            metadata.update(json.loads(raw_metadata))
            text = text[metadata_end + len(_METADATA_END) :]
        if contains_sensitive_data(text):
            raise ValueError(
                f"Sensitive information detected in learned knowledge file "
                f"{file_path.name}; remove it before indexing."
            )
        if text.strip():
            metadata["source"] = str(file_path)
            documents.append(Document(page_content=text.strip(), metadata=metadata))
    return documents