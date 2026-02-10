# Exam Assistant — Two Stage RAG Chatbot

A RAG-based chatbot that helps with exam preparation by retrieving answers from your uploaded study materials using a two-stage retrieval system.

## What it does

- Upload notes, sample papers, and reference books as PDFs
- Ask questions and get answers grounded in your materials
- Two-stage retrieval automatically picks the best source for each question

## How it works

**Stage 1 — Notes & Sample Papers**
- Smaller chunks (600 chars) for focused retrieval
- Priority scoring for exam-relevant content (questions, marks, solutions)
- Used first for fast, direct answers

**Stage 2 — Reference Books**
- Larger chunks (1000 chars) for detailed context
- Kicks in when Stage 1 confidence is low
- Gives comprehensive explanations

Both stages use a CrossEncoder reranker to improve retrieval quality.

## Tech Stack

- Sentence Transformers (all-MiniLM-L6-v2) — embeddings
- CrossEncoder (ms-marco-MiniLM-L-6-v2) — reranking
- Pinecone — vector database
- Groq (Llama 3.1 8B) — answer generation
- LangChain — text splitting
- Streamlit — UI

## Setup

```bash
pip install streamlit sentence-transformers pinecone-client groq pypdf langchain-text-splitters numpy httpx==0.27.2
```


Live: https://exam-assistant.streamlit.app/
