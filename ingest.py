"""Build and safely replace the FAISS index from all approved sources."""

import shutil
import uuid
from pathlib import Path

from src.chunking import split_documents
from src.document_loader import load_documents
from src.vector_store import (
    FAISS_INDEX_DIRECTORY,
    create_vector_store,
    load_vector_store,
    save_vector_store,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
LEARNED_DIRECTORY = DATA_DIRECTORY / "learned_knowledge"


def replace_index_safely(vector_store, expected_vectors: int) -> None:
    """Validate a staged index before replacing the active index, with rollback."""
    staging_directory = PROJECT_ROOT / f".faiss_index_staging_{uuid.uuid4().hex}"
    backup_directory = PROJECT_ROOT / f".faiss_index_backup_{uuid.uuid4().hex}"
    moved_current_index = False

    try:
        save_vector_store(vector_store, staging_directory)
        staged_store = load_vector_store(staging_directory)
        if staged_store.index.ntotal != expected_vectors or expected_vectors == 0:
            raise RuntimeError("Staged FAISS index failed vector-count validation.")

        if FAISS_INDEX_DIRECTORY.exists():
            FAISS_INDEX_DIRECTORY.rename(backup_directory)
            moved_current_index = True

        try:
            staging_directory.rename(FAISS_INDEX_DIRECTORY)
        except OSError:
            if moved_current_index:
                backup_directory.rename(FAISS_INDEX_DIRECTORY)
                moved_current_index = False
            raise

        if moved_current_index:
            shutil.rmtree(backup_directory, ignore_errors=True)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory, ignore_errors=True)


def main():
    """Load source documents and safely rebuild the FAISS index."""
    try:
        documents = load_documents()
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as error:
        print(f"Unable to load PDF documents: {error}")
        return

    pdf_sources = list(DATA_DIRECTORY.rglob("*.pdf"))
    learned_sources = (
        [
            path
            for path in LEARNED_DIRECTORY.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        ]
        if LEARNED_DIRECTORY.exists()
        else []
    )
    print(f"Source documents: {len(pdf_sources) + len(learned_sources)}")
    print(f"Loaded documents/pages: {len(documents)}")

    if not documents:
        print("No PDF files found. Place PDF files inside the data/ folder.")
        return

    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        print("No chunks were created from the loaded documents.")
        return

    staging_vector_store = create_vector_store(chunks)
    vector_count = int(staging_vector_store.index.ntotal)
    print(f"Indexed vectors: {vector_count}")
    if vector_count != len(chunks) or vector_count == 0:
        print("Index validation failed; the existing FAISS index was preserved.")
        return

    try:
        replace_index_safely(staging_vector_store, len(chunks))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Index rebuild failed; the existing index was preserved: {error}")
        return
    print(f"FAISS index saved successfully: {FAISS_INDEX_DIRECTORY}")


if __name__ == "__main__":
    main()
