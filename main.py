"""
DocuQuery — ask questions about your documents.

Endpoints:
  POST /upload  — receive a file, split into chunks, store in ChromaDB
  POST /ask     — find similar chunks, send to Claude, return answer
  GET  /        — serve index.html
  GET  /health  — liveness probe
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import anthropic
import chromadb
import fitz  # pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="DocuQuery", version="1.0.0")

# ---------------------------------------------------------------------------
# Clients (initialised once at startup)
# ---------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path="./chroma_data")
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Single collection — all documents live together.
# ChromaDB generates embeddings locally (no extra API key needed).
collection = chroma_client.get_or_create_collection(name="documents")

CHUNK_SIZE = 500  # characters; balances retrieval precision vs. context quality
CLAUDE_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024
N_RESULTS = 4


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _split_text(text: str, page: int) -> list[dict]:
    """Yield non-empty 500-char chunks with the given page label."""
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i : i + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({"text": chunk, "page": page})
    return chunks


def extract_text_from_pdf(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    chunks: list[dict] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            chunks.extend(_split_text(text, page_num))
    doc.close()
    return chunks


def extract_text_from_txt(file_path: str) -> list[dict]:
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    return [
        {"text": chunk["text"], "page": idx + 1}
        for idx, chunk in enumerate(_split_text(text, 0))  # page recalculated below
    ]


def extract_text(file_path: str, filename: str) -> list[dict]:
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    # .txt — page = chunk index
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    chunks = []
    for i, start in enumerate(range(0, len(text), CHUNK_SIZE), start=1):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({"text": chunk, "page": i})
    return chunks


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile) -> dict:
    """Accept PDF or TXT, chunk the text, upsert chunks into ChromaDB."""
    if not file.filename or not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunks = extract_text(tmp_path, file.filename)
    finally:
        os.unlink(tmp_path)

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from the file")

    collection.upsert(
        ids=[f"{file.filename}_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": file.filename, "page": c["page"]} for c in chunks],
    )

    return {
        "filename": file.filename,
        "chunks_saved": len(chunks),
        "message": f"Done! Saved {len(chunks)} chunks.",
    }


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(body: QuestionRequest) -> dict:
    """Find the most relevant chunks and ask Claude to answer based on them."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    results = collection.query(query_texts=[body.question], n_results=N_RESULTS)

    if not results["documents"][0]:
        return {"answer": "No documents found. Please upload a file first."}

    context = "\n\n".join(
        f"[File: {meta['source']}, p. {meta['page']}]\n{text}"
        for text, meta in zip(results["documents"][0], results["metadatas"][0])
    )

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=(
            "You are a document assistant. "
            "Answer questions based ONLY on the provided context. "
            "If the answer is not in the context, say so clearly. "
            "Always cite the source at the end of your answer: [File: ..., p. ...]"
        ),
        messages=[
            {
                "role": "user",
                "content": f"Context from documents:\n\n{context}\n\nQuestion: {body.question}",
            }
        ],
    )

    return {"answer": response.content[0].text}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> str:
    return Path("index.html").read_text(encoding="utf-8")
