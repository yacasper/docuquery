# DocuQuery

Upload a PDF or TXT file, ask a question, get an answer with a page citation.

Built in one day as a focused RAG demo using **FastAPI · ChromaDB · Claude API**.

## How it works

```
Upload file → extract text → split into 500-char chunks → store in ChromaDB
Ask question → semantic search (top 4 chunks) → send context to Claude → return answer + citation
```

ChromaDB runs **embedded** (no separate server) and generates embeddings locally — no external embedding API needed.

## Quick start

### Docker (recommended)

```bash
cp .env.example .env        # add your ANTHROPIC_API_KEY
docker compose up --build
# Open http://localhost:8000
```

ChromaDB data is persisted in a named Docker volume (`chroma_data`).

### Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
uvicorn main:app --reload
```

## Tests

```bash
pytest
```

No API key or running services required — external dependencies are mocked.

## Design decisions

| Choice | Reason |
|---|---|
| ChromaDB embedded | Free local embeddings, persists to a folder, zero ops overhead |
| chunk_size = 500 chars | Balances retrieval precision (smaller) vs. context quality (larger) |
| Upsert by `filename_N` IDs | Re-uploading the same file overwrites chunks, no duplicates |
| Single collection | Keeps scope minimal; per-document filtering is an easy next step |

## Possible improvements

- Streaming responses via SSE
- DOCX support (python-docx)
- Per-document filtering in ChromaDB
- Conversation history
