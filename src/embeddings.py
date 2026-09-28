"""Create local Hugging Face embeddings for document chunks and queries."""

from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings() -> HuggingFaceEmbeddings:
	"""Create the local embedding model used by the FAISS vector store."""
	return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
