# DocuQuery — Project Spec

> **Stack:** Python · FastAPI · ChromaDB · Claude API · HTML/JS
> **Time:** 1 day with Claude Code
> **Goal:** upload a file → ask a question → get an answer with a citation

---

## What it does

1. User uploads a PDF or TXT file
2. Backend splits text into chunks and stores them in ChromaDB
3. User types a question
4. System finds the most relevant chunks and sends them to Claude
5. Claude answers based on the document and cites the page number

No users, no auth, no queues. Just the core feature.

---

## Project structure

```
docuquery/
├── main.py          # entire backend — one file
├── index.html       # entire frontend — one file
├── requirements.txt
├── .env
└── README.md
```

---

## `requirements.txt`

```
fastapi==0.115.0
uvicorn==0.30.0
python-multipart==0.0.9
chromadb==0.5.0
anthropic==0.34.0
pymupdf==1.24.0
python-dotenv==1.0.0
```

---

## `main.py`

```python
# main.py
# DocuQuery — ask questions about your documents
#
# How it works:
# 1. POST /upload  — receives a file, splits it into chunks, stores in ChromaDB
# 2. POST /ask     — finds similar chunks, sends them to Claude, returns the answer
# 3. GET  /        — serves index.html

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
import anthropic
import fitz          # pymupdf — for reading PDFs
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- Clients ---
# ChromaDB stores data on disk in ./chroma_data folder
chroma_client = chromadb.PersistentClient(path="./chroma_data")
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Single collection for all documents
# ChromaDB generates embeddings locally for free — no external API needed
collection = chroma_client.get_or_create_collection(name="documents")


# --- Helper functions ---

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF and returns a list of text chunks.
    Each chunk is roughly one paragraph or ~300 words.
    We save the page number so we can cite it in answers.
    """
    doc = fitz.open(file_path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()

        # Skip blank pages
        if not text.strip():
            continue

        # Split each page into 500-character chunks.
        # Too small = lose context. Too large = noisy retrieval.
        chunk_size = 500
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_num
                })

    doc.close()
    return chunks


def extract_text_from_txt(file_path: str) -> list[dict]:
    """
    Reads a TXT file and splits it into 500-character chunks.
    For TXT there are no real page numbers — we use the chunk index instead.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = []
    chunk_size = 500

    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": (i // chunk_size) + 1
            })

    return chunks


# --- API endpoints ---

@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    Accepts a file, extracts text, saves chunks to ChromaDB.
    Returns how many chunks were saved.
    """
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if file.filename.endswith(".pdf"):
            chunks = extract_text_from_pdf(tmp_path)
        else:
            chunks = extract_text_from_txt(tmp_path)

        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from the file")

        # Save chunks to ChromaDB
        # Each chunk gets a unique ID: "filename_0", "filename_1", ...
        # upsert = insert or update if ID already exists
        collection.upsert(
            ids=[f"{file.filename}_{i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": file.filename, "page": c["page"]} for c in chunks]
        )

        return {
            "filename": file.filename,
            "chunks_saved": len(chunks),
            "message": f"Done! Saved {len(chunks)} chunks."
        }

    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(body: QuestionRequest):
    """
    Finds similar chunks in ChromaDB, sends them to Claude,
    and returns the answer with source citations.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Find the 4 most similar chunks to the question
    # ChromaDB computes similarity between the question and stored chunks
    results = collection.query(
        query_texts=[body.question],
        n_results=4
    )

    if not results["documents"][0]:
        return {"answer": "No documents found. Please upload a file first."}

    # Build the context string to pass to Claude
    # Format: [File: doc.pdf, p. 3]\nchunk text
    context_parts = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(
            f"[File: {meta['source']}, p. {meta['page']}]\n{text}"
        )
    context = "\n\n".join(context_parts)

    # Send to Claude
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=(
            "You are a document assistant. "
            "Answer questions based ONLY on the provided context. "
            "If the answer is not in the context, say so clearly. "
            "Always cite the source at the end of your answer: [File: ..., p. ...]"
        ),
        messages=[{
            "role": "user",
            "content": f"Context from documents:\n\n{context}\n\nQuestion: {body.question}"
        }]
    )

    return {"answer": response.content[0].text}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend — read index.html from disk"""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
```

---

## `index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DocuQuery</title>
    <style>
        body {
            font-family: system-ui, sans-serif;
            max-width: 700px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }
        h1 { color: #1a56a0; }
        .section {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        input[type="file"], input[type="text"] {
            width: 100%;
            padding: 8px;
            margin: 8px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background: #1a56a0;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:disabled { background: #999; cursor: not-allowed; }
        .status { margin-top: 10px; padding: 10px; border-radius: 4px; }
        .success { background: #e1f5ee; color: #0f6e56; }
        .error   { background: #faece7; color: #993c1d; }
        .answer  {
            background: #f4f5f7;
            padding: 15px;
            border-radius: 4px;
            white-space: pre-wrap;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <h1>DocuQuery</h1>
    <p>Upload a document and ask questions in plain English.</p>

    <!-- File upload section -->
    <div class="section">
        <h2>1. Upload document</h2>
        <input type="file" id="fileInput" accept=".pdf,.txt">
        <button onclick="uploadFile()" id="uploadBtn">Upload</button>
        <div id="uploadStatus"></div>
    </div>

    <!-- Question section -->
    <div class="section">
        <h2>2. Ask a question</h2>
        <input
            type="text"
            id="question"
            placeholder="e.g. What is the main topic of this document?"
            onkeydown="if(event.key==='Enter') askQuestion()"
        >
        <button onclick="askQuestion()" id="askBtn">Ask</button>
        <div id="answerBlock" style="display:none">
            <h3>Answer:</h3>
            <div id="answer" class="answer"></div>
        </div>
    </div>

    <script>
        async function uploadFile() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) {
                showStatus('uploadStatus', 'Please select a file', 'error');
                return;
            }

            const btn = document.getElementById('uploadBtn');
            btn.disabled = true;
            btn.textContent = 'Uploading...';
            showStatus('uploadStatus', '');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();

                if (res.ok) {
                    showStatus('uploadStatus', data.message, 'success');
                } else {
                    showStatus('uploadStatus', data.detail, 'error');
                }
            } catch (e) {
                showStatus('uploadStatus', 'Connection error', 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Upload';
            }
        }

        async function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) return;

            const btn = document.getElementById('askBtn');
            btn.disabled = true;
            btn.textContent = 'Thinking...';
            document.getElementById('answerBlock').style.display = 'none';

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await res.json();
                document.getElementById('answer').textContent = data.answer;
                document.getElementById('answerBlock').style.display = 'block';
            } catch (e) {
                document.getElementById('answer').textContent = 'Connection error';
                document.getElementById('answerBlock').style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Ask';
            }
        }

        function showStatus(elementId, message, type = '') {
            const el = document.getElementById(elementId);
            el.textContent = message;
            el.className = 'status ' + type;
        }
    </script>
</body>
</html>
```

---

## `.env`

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000
```

---

## What to say at the interview

- **Why ChromaDB:** built-in local embeddings (free), persists to a folder, no separate server to manage
- **Why chunk_size 500:** a balance between search precision (small chunks) and context quality (large chunks)
- **What could be improved:** streaming responses, DOCX support, per-document filtering, chat history
