"""Load PDF documents from the project's knowledge-base directory."""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pypdf.errors import PdfReadError

from src.learned_knowledge import load_approved_knowledge_documents


DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"


def load_documents():
    """Load all PDF pages from the ``data/`` directory.

    ``PyPDFLoader`` returns one LangChain ``Document`` per page and includes
    source and page metadata for each document.
    """
    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Knowledge-base directory was not found: {DATA_DIRECTORY}"
        )

    if not DATA_DIRECTORY.is_dir():
        raise NotADirectoryError(
            f"Knowledge-base path is not a directory: {DATA_DIRECTORY}"
        )

    try:
        pdf_paths = list(DATA_DIRECTORY.rglob("*.pdf"))
        documents = []
        if pdf_paths:
            loader = DirectoryLoader(
                str(DATA_DIRECTORY),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                recursive=True,
                silent_errors=False,
            )
            documents = loader.load()

        learned_documents = load_approved_knowledge_documents()
        for document in learned_documents:
            document.metadata["source_type"] = "user-approved"
        return documents + learned_documents
    except (OSError, PdfReadError, ValueError) as error:
        raise RuntimeError(f"Failed to load PDF documents: {error}") from error
