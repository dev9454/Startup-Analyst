# tools/loaders.py
from __future__ import annotations
from typing import List
from langchain_community.document_loaders import (
    PyMuPDFLoader,            # tolerant, fast
    PDFPlumberLoader,         # good for tables
    PyPDFLoader,              # current (pypdf)
    Docx2txtLoader,
    WebBaseLoader,
)
import logging
log = logging.getLogger(__name__)

def _load_pdf(path: str):
    # Try in order of robustness
    tried = []
    for Loader in (PyMuPDFLoader, PDFPlumberLoader, PyPDFLoader):
        try:
            docs = Loader(path).load()
            log.info(f"Loaded PDF with {Loader.__name__}: {path}")
            return docs
        except Exception as e:
            tried.append(f"{Loader.__name__}: {e}")
    raise RuntimeError(f"All PDF loaders failed for {path}. Tried -> " + " | ".join(tried))

def load_any(path_or_url: str):
    p = path_or_url.strip()
    if p.lower().endswith(".pdf"):
        return _load_pdf(p)
    if p.lower().endswith(".docx"):
        return Docx2txtLoader(p).load()
    # default: website
    return WebBaseLoader(p).load()

def load_many(items: List[str]):
    docs = []
    for x in items:
        docs += load_any(x)
    return docs
