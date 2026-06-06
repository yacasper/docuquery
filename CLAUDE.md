# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

DocuQuery is a single-day RAG demo: upload a PDF or TXT, ask a question, get an answer with a page citation. The full spec is in `tz_1_docuquery.md`.

Stack: Python · FastAPI · ChromaDB · Claude API · HTML/JS (no framework).

## Running the app

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000
```

Requires `ANTHROPIC_API_KEY` in `.env`.

## Architecture

The entire project is intentionally two files:

- **`main.py`** — FastAPI backend. Three endpoints:
  - `GET /` — serves `index.html` from disk
  - `POST /upload` — extracts text from PDF (via PyMuPDF) or TXT, splits into 500-char chunks, upserts into ChromaDB with `{source, page}` metadata
  - `POST /ask` — queries ChromaDB for the 4 nearest chunks, builds a context string with `[File: …, p. N]` headers, sends to Claude, returns the answer
- **`index.html`** — single-page frontend. No build step. Two sections: file upload and question input.

ChromaDB runs in embedded mode (`PersistentClient`), storing data in `./chroma_data/`. It computes embeddings locally — no external embedding API. All documents share a single collection named `"documents"`.

The Claude model used is `claude-sonnet-4-20250514` with `max_tokens=1024`. The system prompt restricts answers to the provided context and requires citations.

## Key design decisions

- **chunk_size = 500 chars**: balances retrieval precision (smaller) vs. context quality (larger).
- **ChromaDB local embeddings**: free, no external API, persists to a folder.
- **upsert by `filename_N` IDs**: re-uploading the same file overwrites its chunks without duplicating them.
- No auth, no queues, no users — intentionally minimal.

## Potential improvements (from spec)

Streaming responses, DOCX support, per-document filtering, chat history.
