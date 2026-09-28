"""Split loaded documents into chunks suitable for RAG retrieval."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents so RAG can retrieve focused, relevant text passages."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
    )
    return text_splitter.split_documents(documents)
