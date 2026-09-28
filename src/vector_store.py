"""Create, save, and load the local FAISS vector store."""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.embeddings import create_embeddings


FAISS_INDEX_DIRECTORY = Path(__file__).resolve().parent.parent / "faiss_index"


def create_vector_store(chunks):
	"""Create a FAISS vector store from document chunks and their metadata."""
	embeddings = create_embeddings()
	return FAISS.from_documents(chunks, embeddings)


def save_vector_store(vector_store, index_directory=FAISS_INDEX_DIRECTORY):
	"""Save the FAISS index and metadata files in the local index directory."""
	index_directory = Path(index_directory)
	index_directory.mkdir(parents=True, exist_ok=True)
	vector_store.save_local(str(index_directory))


def load_vector_store(index_directory=FAISS_INDEX_DIRECTORY):
	"""Load the saved FAISS index or raise a clear error when it is missing."""
	index_directory = Path(index_directory)
	index_file = index_directory / "index.faiss"
	metadata_file = index_directory / "index.pkl"
	if not index_file.exists() or not metadata_file.exists():
		raise FileNotFoundError(
			"FAISS index was not found. Run 'python ingest.py' first."
		)

	embeddings = create_embeddings()
	return FAISS.load_local(
		str(index_directory),
		embeddings,
		allow_dangerous_deserialization=True,
	)
