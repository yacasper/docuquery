"""
Integration-style tests for DocuQuery API.

External dependencies (ChromaDB, Anthropic) are patched so the suite runs
without any real API keys or a running vector store.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """
    Return a TestClient with both ChromaDB and Anthropic clients mocked.
    We patch at the module level (main.*) so the running app uses fakes.
    """
    mock_collection = MagicMock()
    mock_claude = MagicMock()

    with (
        patch("main.collection", mock_collection),
        patch("main.claude_client", mock_claude),
    ):
        from main import app
        yield TestClient(app), mock_collection, mock_claude


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(client):
    tc, *_ = client
    res = tc.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------

def test_upload_txt_happy_path(client):
    tc, mock_collection, _ = client
    content = b"Hello world! " * 50  # enough text for at least one chunk

    res = tc.post(
        "/upload",
        files={"file": ("sample.txt", io.BytesIO(content), "text/plain")},
    )

    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "sample.txt"
    assert data["chunks_saved"] >= 1
    mock_collection.upsert.assert_called_once()


def test_upload_unsupported_format_returns_400(client):
    tc, mock_collection, _ = client

    res = tc.post(
        "/upload",
        files={"file": ("report.docx", io.BytesIO(b"data"), "application/octet-stream")},
    )

    assert res.status_code == 400
    assert "PDF" in res.json()["detail"] or "TXT" in res.json()["detail"]
    mock_collection.upsert.assert_not_called()


def test_upload_empty_txt_returns_400(client):
    tc, mock_collection, _ = client

    res = tc.post(
        "/upload",
        files={"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")},
    )

    assert res.status_code == 400
    mock_collection.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

def _make_query_result(texts: list[str], source: str = "doc.txt", page: int = 1):
    """Build the dict shape that ChromaDB collection.query() returns."""
    return {
        "documents": [texts],
        "metadatas": [[{"source": source, "page": page}] * len(texts)],
    }


def test_ask_returns_claude_answer(client):
    tc, mock_collection, mock_claude = client

    mock_collection.query.return_value = _make_query_result(["Relevant chunk."])
    mock_claude.messages.create.return_value = MagicMock(
        content=[MagicMock(text="The answer is 42. [File: doc.txt, p. 1]")]
    )

    res = tc.post("/ask", json={"question": "What is the answer?"})

    assert res.status_code == 200
    assert "42" in res.json()["answer"]
    mock_collection.query.assert_called_once()
    mock_claude.messages.create.assert_called_once()


def test_ask_no_documents_uploaded(client):
    tc, mock_collection, mock_claude = client

    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

    res = tc.post("/ask", json={"question": "Anything?"})

    assert res.status_code == 200
    assert "upload" in res.json()["answer"].lower()
    mock_claude.messages.create.assert_not_called()


def test_ask_empty_question_returns_400(client):
    tc, *_ = client

    res = tc.post("/ask", json={"question": "   "})

    assert res.status_code == 400


# ---------------------------------------------------------------------------
# / (frontend)
# ---------------------------------------------------------------------------

def test_root_serves_html(client):
    tc, *_ = client
    res = tc.get("/")
    assert res.status_code == 200
    assert "DocuQuery" in res.text
