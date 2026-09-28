"""Streamlit interface for the AI & Data Science Knowledge Assistant."""

from html import escape
from pathlib import Path
from uuid import UUID, uuid4

import streamlit as st

st.set_page_config(
    page_title="AI & Data Science Knowledge Assistant",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="auto",
)

from src.llm import LLMConfigurationError
from src.learned_knowledge import learned_knowledge_count, save_approved_knowledge
from src.memory import (
    clear_conversation,
    count_session_turns,
    get_recent_history,
    save_feedback,
    save_turn,
)
from src.rag_pipeline import answer_question


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index" / "index.faiss"
SUGGESTED_QUESTIONS = (
    "What is RAG?",
    "What is supervised learning?",
    "What is a SQL JOIN?",
    "Explain NumPy and Pandas",
)


@st.cache_data(ttl=60, show_spinner=False)
def get_knowledge_base_stats() -> tuple[int, int | None]:
    """Read actual PDF and FAISS vector counts without loading the embedding model."""
    pdf_count = len(list(DATA_DIRECTORY.rglob("*.pdf"))) if DATA_DIRECTORY.exists() else 0
    chunk_count = None
    if FAISS_INDEX_PATH.exists():
        try:
            import faiss

            chunk_count = int(faiss.read_index(str(FAISS_INDEX_PATH)).ntotal)
        except (ImportError, OSError, RuntimeError):
            chunk_count = None
    return pdf_count, chunk_count


def render_sources(sources: list[dict]) -> None:
    """Render source references as compact, escaped cards."""
    if not sources:
        return

    st.markdown('<div class="sources-heading">Sources</div>', unsafe_allow_html=True)
    cards = []
    for source in sources:
        filename = escape(str(source.get("source", "Unknown source")))
        page = source.get("page")
        page_label = f"Page {escape(str(page))}" if page is not None else "Page unavailable"
        cards.append(
            '<div class="source-card">'
            '<span class="source-mark">PDF</span>'
            '<span class="source-copy">'
            f'<span class="source-name">{filename}</span>'
            f'<span class="source-page">{page_label}</span>'
            "</span></div>"
        )
    st.markdown(f'<div class="source-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --ink: #182b2a;
        --muted: #647674;
        --green: #164b43;
        --green-dark: #103c36;
        --mint: #dceee8;
        --paper: #f5f7f4;
        --line: #dce5e0;
        --white: #ffffff;
        --coral: #c2765e;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 1.2rem; }
    [data-testid="stSidebar"] { background: var(--green-dark); border-right: 1px solid rgba(255,255,255,.08); }
    [data-testid="stSidebar"] * { color: #eef5f1; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c8d9d3; }
    .sidebar-brand { font-family: 'Manrope', 'Segoe UI', sans-serif; font-size: 1.25rem; font-weight: 800; line-height: 1.15; color: white; margin: .45rem 0 1.45rem; }
    .sidebar-label { color: #9fbdb4; text-transform: uppercase; font-size: .68rem; font-weight: 700; letter-spacing: .12em; margin: 1.45rem 0 .7rem; }
    .stat-row { display: flex; align-items: center; gap: .65rem; padding: .66rem .72rem; margin: .38rem 0; border: 1px solid rgba(228,242,236,.13); border-radius: 10px; background: rgba(255,255,255,.045); color: #f3f7f5; font-size: .88rem; }
    .stat-dot { width: 7px; height: 7px; flex: 0 0 7px; border-radius: 50%; background: #9ad1bc; box-shadow: 0 0 0 4px rgba(154,209,188,.12); }
    .stack-list { display: flex; flex-wrap: wrap; gap: .42rem; }
    .stack-item { border: 1px solid rgba(220,238,232,.2); border-radius: 7px; padding: .36rem .55rem; color: #dcebe5; font-size: .78rem; }
    .about-copy { font-size: .83rem; line-height: 1.55; color: #c4d4ce; }
    .clear-button button { width: 100%; color: #f5fbf8; border: 1px solid rgba(255,255,255,.3); background: transparent; border-radius: 9px; font-weight: 600; }
    .clear-button button:hover { border-color: #a6d1bf; color: white; background: rgba(255,255,255,.08); }
    .hero { position: relative; overflow: hidden; padding: 2rem 2.1rem 1.85rem; border-radius: 16px; background: var(--green); color: white; box-shadow: 0 12px 30px rgba(21,68,60,.12); }
    .hero:after { content: ''; position: absolute; right: -50px; top: -120px; width: 300px; height: 300px; border: 1px solid rgba(222,242,234,.12); border-radius: 50%; box-shadow: 0 0 0 34px rgba(222,242,234,.035), 0 0 0 70px rgba(222,242,234,.025); pointer-events: none; }
    .hero-kicker { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .13em; color: #b8ddd0; margin-bottom: .72rem; }
    .hero h1 { font-family: 'Manrope', 'Segoe UI', sans-serif; max-width: 750px; margin: 0; color: white; font-size: 2.15rem; line-height: 1.16; font-weight: 800; letter-spacing: 0; }
    .hero-subtitle { color: #e1eee8; font-size: 1.02rem; margin: .7rem 0 1.05rem; }
    .hero-description { max-width: 730px; color: #c3d9d1; font-size: .88rem; line-height: 1.55; margin: .85rem 0 0; }
    .tech-badge { display: inline-block; padding: .39rem .68rem; border: 1px solid rgba(225,242,235,.25); border-radius: 7px; color: #edf6f2; font-size: .75rem; font-weight: 600; }
    .section-kicker { margin: 1.6rem 0 .35rem; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size: .68rem; font-weight: 700; }
    .welcome-panel { margin: .65rem 0 1rem; padding: 1.1rem 1.2rem; border: 1px solid var(--line); border-radius: 12px; background: var(--white); }
    .welcome-title { color: var(--ink); font-family: 'Manrope', 'Segoe UI', sans-serif; font-size: 1.18rem; font-weight: 800; margin-bottom: .25rem; }
    .welcome-subtitle { color: var(--muted); font-size: .9rem; }
    div[data-testid="stChatMessage"] { gap: .8rem; padding: 1rem 1.05rem; margin: .7rem 0; border: 1px solid var(--line); border-radius: 12px; background: white; }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { line-height: 1.65; }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: #e8f0ec; border-color: #dce8e1; }
    .sources-heading { color: #49615c; text-transform: uppercase; letter-spacing: .11em; font-size: .67rem; font-weight: 700; margin: .25rem 0 .5rem; }
    .source-grid { display: flex; flex-wrap: wrap; gap: .55rem; margin: 0 0 .65rem 3rem; }
    .source-card { min-width: 190px; display: flex; align-items: center; gap: .62rem; padding: .62rem .75rem; border: 1px solid var(--line); border-radius: 9px; background: #fbfcfa; }
    .source-mark { color: var(--green); background: var(--mint); border-radius: 6px; padding: .34rem .4rem; font-size: .59rem; font-weight: 800; }
    .source-copy { display: flex; min-width: 0; flex-direction: column; gap: .13rem; }
    .source-name { color: var(--ink); font-size: .78rem; font-weight: 700; overflow-wrap: anywhere; }
    .source-page { color: var(--muted); font-size: .7rem; }
    .stButton > button { min-height: 3.15rem; text-align: left; white-space: normal; border: 1px solid var(--line); border-radius: 10px; background: white; color: var(--ink); font-weight: 600; transition: border-color .15s ease, background .15s ease; }
    .stButton > button:hover { border-color: #8db5a7; background: #f0f7f3; color: var(--green-dark); }
    [data-testid="stChatInput"] { border-color: #cbd9d2; border-radius: 12px; background: white; box-shadow: 0 6px 18px rgba(30,68,57,.055); }
    [data-testid="stChatInput"] textarea { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
    .footer { margin: 1.5rem 0 .25rem; padding-top: .8rem; border-top: 1px solid var(--line); color: #758681; text-align: center; font-size: .74rem; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1.1rem 1rem .8rem; }
        .hero { padding: 1.45rem 1.25rem; }
        .hero h1 { font-size: 1.65rem; }
        .source-grid { margin-left: .25rem; }
        .source-card { min-width: min(100%, 220px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

session_id = st.session_state.get("session_id")
if not session_id:
    supplied_session_id = st.query_params.get("session")
    try:
        session_id = str(UUID(supplied_session_id)) if supplied_session_id else str(uuid4())
    except ValueError:
        session_id = str(uuid4())
    st.session_state.session_id = session_id
    if not supplied_session_id:
        st.query_params["session"] = session_id

if not st.session_state.get("history_loaded"):
    saved_turns = get_recent_history(session_id, limit=100)
    restored_messages = []
    for turn in saved_turns:
        restored_messages.extend(
            [
                {"role": "user", "content": turn["question"]},
                {
                    "role": "assistant",
                    "content": turn["answer"],
                    "sources": turn["sources"],
                    "turn_id": turn["id"],
                    "feedback": turn["feedback"],
                },
            ]
        )
    st.session_state.messages = restored_messages
    st.session_state.history_loaded = True


with st.sidebar:
    st.markdown('<div class="sidebar-brand">Knowledge<br>Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Knowledge Base</div>', unsafe_allow_html=True)
    pdf_count, chunk_count = get_knowledge_base_stats()
    chunk_display = str(chunk_count) if chunk_count is not None else "Index unavailable"
    sidebar_stats = (
        f"{pdf_count} Educational PDFs",
        f"{chunk_display} Indexed Chunks",
        "Semantic Retrieval",
        "Groq LLM",
    )
    for label in sidebar_stats:
        st.markdown(
            f'<div class="stat-row"><span class="stat-dot"></span>{escape(label)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-label">Memory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="stat-row"><span class="stat-dot"></span>🧠 Conversation Memory</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="about-copy">Recent conversations are remembered locally.</div>',
        unsafe_allow_html=True,
    )
    learned_count = learned_knowledge_count()
    st.markdown(
        f'<div class="about-copy" style="margin-top:.5rem">Knowledge Base: '
        f'{pdf_count} PDFs + {learned_count} approved items</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Technology Stack</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="stack-list">'
        '<span class="stack-item">Python</span><span class="stack-item">LangChain</span>'
        '<span class="stack-item">FAISS</span><span class="stack-item">Hugging Face</span>'
        '<span class="stack-item">Groq</span><span class="stack-item">Streamlit</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="about-copy">Answers are grounded in the indexed knowledge base. '
        'When information is unavailable, the assistant is instructed not to invent it.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="clear-button">', unsafe_allow_html=True)
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        clear_conversation(session_id)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    '<section class="hero">'
    '<div class="hero-kicker">A curated learning companion</div>'
    '<h1>AI &amp; Data Science Knowledge Assistant</h1>'
    '<div class="hero-subtitle">Ask questions from your custom AI &amp; Data Science knowledge base.</div>'
    '<span class="tech-badge">RAG &nbsp;•&nbsp; FAISS &nbsp;•&nbsp; Hugging Face &nbsp;•&nbsp; Groq</span>'
    '<p class="hero-description">This assistant retrieves relevant information from a curated educational knowledge base before generating an answer.</p>'
    '</section>',
    unsafe_allow_html=True,
)


if not st.session_state.messages:
    st.markdown('<div class="section-kicker">Ask the knowledge base</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="welcome-panel"><div class="welcome-title">Welcome</div>'
        '<div class="welcome-subtitle">Ask me anything about the knowledge base, or start with a suggested question.</div></div>',
        unsafe_allow_html=True,
    )
    question_columns = st.columns(2)
    for index, example in enumerate(SUGGESTED_QUESTIONS):
        with question_columns[index % 2]:
            if st.button(example, key=f"suggestion_{index}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            turn_id = message.get("turn_id")
            if turn_id:
                feedback_columns = st.columns([1, 1, 5])
                with feedback_columns[0]:
                    if st.button("👍 Helpful", key=f"helpful_{turn_id}"):
                        save_feedback(session_id, turn_id, "helpful")
                        message["feedback"] = "helpful"
                with feedback_columns[1]:
                    if st.button("👎 Not helpful", key=f"not_helpful_{turn_id}"):
                        save_feedback(session_id, turn_id, "not_helpful")
                        message["feedback"] = "not_helpful"
                if message.get("feedback"):
                    st.caption(f"Feedback saved: {message['feedback'].replace('_', ' ')}")

                with st.expander("Add useful information to knowledge base"):
                    st.caption(
                        "Explicit approval only: this becomes shared knowledge. "
                        "Do not include personal information, credentials, or secrets."
                    )
                    approval_text = st.text_area(
                        "Approved knowledge",
                        value=message["content"],
                        key=f"approved_text_{turn_id}",
                        help="Review and edit this content before approving it.",
                    )
                    approval_description = st.text_input(
                        "Optional description",
                        key=f"approved_description_{turn_id}",
                    )
                    confirmed = st.checkbox(
                        "I reviewed this and confirm it contains no private information or secrets.",
                        key=f"approved_confirm_{turn_id}",
                    )
                    if st.button("Approve and add", key=f"approve_{turn_id}"):
                        if not confirmed:
                            st.warning("Confirm the content is safe before adding it.")
                        else:
                            try:
                                saved_path = save_approved_knowledge(
                                    approval_text,
                                    session_id,
                                    approval_description,
                                )
                                st.success(
                                    "Approved knowledge saved. Run `python ingest.py` "
                                    "to add it to semantic search."
                                )
                                st.caption(saved_path.name)
                            except ValueError as error:
                                st.error(str(error))


submitted_question = st.chat_input("Ask a question about AI, data science, Python, or SQL...")
question = submitted_question or st.session_state.pending_question
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    assistant_message = {"role": "assistant", "content": "", "sources": []}
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching the knowledge base and generating an answer..."):
            try:
                recent_history = get_recent_history(session_id, limit=5)
                result = answer_question(question, history=recent_history)
                assistant_message["content"] = result["answer"]
                assistant_message["sources"] = result["sources"]
                st.markdown(result["answer"])
                render_sources(result["sources"])
            except LLMConfigurationError:
                assistant_message["content"] = (
                    "The assistant is not configured yet. Add `GROQ_API_KEY` and "
                    "`GROQ_MODEL` to the local `.env` file, then restart the app."
                )
                st.info(assistant_message["content"])
            except FileNotFoundError:
                assistant_message["content"] = (
                    "Knowledge base index not found. Please run `python ingest.py` first."
                )
                st.warning(assistant_message["content"])
            except Exception:
                assistant_message["content"] = (
                    "Something went wrong while answering. Please try again in a moment."
                )
                st.error(assistant_message["content"])
    turn_id = save_turn(
        session_id,
        question,
        assistant_message["content"],
        sources=assistant_message["sources"],
    )
    assistant_message["turn_id"] = turn_id
    assistant_message["feedback"] = None
    st.session_state.messages.append(assistant_message)
    st.rerun()


st.markdown(
    '<div class="footer">AI &amp; Data Science Knowledge Assistant &nbsp;•&nbsp; RAG-based Knowledge Retrieval</div>',
    unsafe_allow_html=True,
)
