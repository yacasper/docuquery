"""
Patch heavy C-extension modules (chromadb, fitz/pymupdf) at the sys.modules
level before main.py is imported.  This lets the test suite run without
needing these packages installed locally.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# --- chromadb stub ---
chromadb_mock = MagicMock()
chromadb_mock.PersistentClient.return_value.get_or_create_collection.return_value = MagicMock()
sys.modules.setdefault("chromadb", chromadb_mock)

# --- fitz (PyMuPDF) stub ---
sys.modules.setdefault("fitz", MagicMock())
