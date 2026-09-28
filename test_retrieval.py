"""Manual retrieval checks for the saved FAISS knowledge base."""

from src.retriever import retrieve_documents_with_scores


TEST_QUERIES = (
    "What is RAG?",
    "What is supervised learning?",
    "What is a SQL JOIN?",
)


def main():
    """Run representative semantic retrieval queries and print their results."""
    for query in TEST_QUERIES:
        print(f"Query:\n{query}\n")
        print("Retrieved chunks:")

        try:
            results = retrieve_documents_with_scores(query, k=3)
        except FileNotFoundError as error:
            print(f"Retrieval unavailable: {error}")
            return

        for index, (document, score) in enumerate(results, start=1):
            metadata = document.metadata
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page")
            page_text = str(page + 1) if page is not None else "Unknown"
            preview = " ".join(document.page_content.split())[:180]
            print(
                f"{index}. source={source} | page={page_text} | "
                f"score={score:.4f} | preview={preview}"
            )
        print()


if __name__ == "__main__":
    main()