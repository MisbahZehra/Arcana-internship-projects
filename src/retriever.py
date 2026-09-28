"""Retrieve relevant chunks from the saved FAISS vector store."""

from langchain_core.documents import Document

from src.vector_store import load_vector_store


def retrieve_documents_with_scores(
	query: str, k: int = 3
) -> list[tuple[Document, float]]:
	"""Return the top matching chunks and their FAISS similarity distances."""
	if not query or not query.strip():
		raise ValueError("Query must contain at least one non-whitespace character.")
	if k < 1:
		raise ValueError("k must be at least 1.")

	vector_store = load_vector_store()
	return vector_store.similarity_search_with_score(query.strip(), k=k)


def retrieve_documents(query: str, k: int = 3) -> list[Document]:
	"""Return the top relevant document chunks for a semantic query."""
	scored_documents = retrieve_documents_with_scores(query, k=k)
	return [document for document, _score in scored_documents]
