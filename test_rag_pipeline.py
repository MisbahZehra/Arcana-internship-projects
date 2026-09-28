"""Manual end-to-end checks for knowledge-grounded answer generation."""

from src.llm import LLMConfigurationError
from src.rag_pipeline import answer_question


TEST_QUESTIONS = (
    "What is RAG?",
    "What is supervised learning?",
    "What is a SQL JOIN?",
    "What is the capital of Japan?",
)


def main():
    """Ask representative in-knowledge-base and out-of-scope questions."""
    for question in TEST_QUESTIONS:
        print(f"Question: {question}")
        try:
            result = answer_question(question)
        except LLMConfigurationError as error:
            print(f"RAG test cannot run: {error}")
            return
        except FileNotFoundError as error:
            print(f"RAG test cannot run: {error}")
            return

        print(f"Answer: {result['answer']}")
        print("Sources:")
        if result["sources"]:
            for source in result["sources"]:
                page = source["page"] if source["page"] is not None else "unknown"
                print(f"- {source['source']}, page {page}")
        else:
            print("- None")
        print()


if __name__ == "__main__":
    main()