## 🎓 Final Internship Project
This project, the **AI & Data Science RAG Knowledge Assistant**, is my **final project** for the Arcana internship. It builds on the concepts and skills I learned during the 9 weeks of internship tasks. The weekly tasks are available in the `weekly-tasks/` folder of this repository.

# AI & Data Science Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) chatbot designed for AI and data science learning. The application grounds each answer in a curated knowledge base of educational PDFs and supports follow-up questions, conversation memory, feedback, and controlled knowledge expansion.

## Overview

This project is a practical RAG assistant for exploring topics such as:

- Python fundamentals
- SQL essentials
- NumPy and Pandas
- Machine learning concepts
- Generative AI and RAG
- Data science workflow concepts

The assistant retrieves relevant chunks from the knowledge base, uses a Groq-backed language model to generate a grounded answer, and cites the source file and page when available.

## Key Features

- PDF-based knowledge ingestion and chunking
- FAISS vector indexing for semantic retrieval
- Groq-powered answer generation
- Streamlit interface for interactive chat
- Conversation memory with session-based history
- Feedback capture for helpful / not helpful responses
- Controlled knowledge approval workflow for learned content
- Safe FAISS rebuild process that preserves the active index on failure
- Sensitive-value redaction for stored conversation content

## Architecture

```text
PDF files in data/
       |
       v
document_loader.py -> chunking.py -> embeddings.py
       |
       v
vector_store.py -> FAISS index
       |
       v
retriever.py -> rag_pipeline.py -> Groq LLM
       |
       v
Streamlit app (app.py)
```

## Tech Stack

- Python
- LangChain
- PyPDF
- Hugging Face sentence-transformers
- FAISS
- Streamlit
- SQLite
- python-dotenv
- Groq API

## Project Structure

```text
Rag Chatbot/
├── app.py
├── ingest.py
├── evaluate.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── memory.db
├── faiss_index/
├── data/
│   ├── README.md
│   ├── python_fundamentals.pdf
│   ├── sql_essentials.pdf
│   ├── numpy_pandas.pdf
│   ├── machine_learning.pdf
│   ├── generative_ai_rag.pdf
│   └── learned_knowledge/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── chunking.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── learned_knowledge.py
│   ├── memory.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── vector_store.py
├── test_rag_pipeline.py
├── test_retrieval.py
└── .venv/
```

## Local Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your local environment file:

```powershell
Copy-Item .env.example .env
```

4. Add your Groq key in `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

> Keep the `.env` file local-only and never commit secrets or credentials to Git.

## Usage

Build or refresh the FAISS index:

```bash
python ingest.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Run the project verification checks:

```bash
python -m pytest
```

## Security and Repository Hygiene

- Never hardcode API keys in source files.
- Never commit `.env` to a public repository.
- Use `.env.example` as the safe template.
- Keep sensitive content out of the document library.
- Redact secrets before persisting chat or memory data.
- Review the working tree before staging or pushing.

## Internship / Portfolio Context

This project demonstrates practical implementation of:

- LLM application design
- Retrieval-augmented generation
- Vector search and semantic retrieval
- Local data workflows for ML and AI education
- Safe project structure for experimentation and portfolio submission

## Current Status

The application is implemented and working locally, with the core retrieval, indexing, chat interface, memory, feedback, and knowledge-approval flows in place.

## Future Improvements

- Add a richer source citation panel
- Expand the knowledge base with more course materials
- Add admin controls for approved knowledge review
- Support multi-file uploads and document metadata
- Improve evaluation and answer quality metrics

## License

This project is intended for educational and portfolio use.
