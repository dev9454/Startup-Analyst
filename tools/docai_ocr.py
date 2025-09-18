# tools/docai_ocr.py
from __future__ import annotations

from typing import Tuple, Optional, List
import re
import fitz  # PyMuPDF
from google.cloud import documentai
from config import DOCAI_PROJECT, DOCAI_LOCATION, DOCAI_PROCESSOR

DOCAI_MAX_PAGES = 15
MAX_DOC_BYTES   = 35_000_000

_client = None
def _client_once():
    global _client
    if _client is None:
        _client = documentai.DocumentProcessorServiceClient()
    return _client

def _norm(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _field_text(doc: documentai.Document, layout) -> str:
    if not getattr(layout, "text_anchor", None) or not layout.text_anchor.text_segments:
        return ""
    parts = []
    for seg in layout.text_anchor.text_segments:
        start = seg.start_index or 0
        end = seg.end_index or 0
        parts.append(doc.text[start:end])
    return "".join(parts)

def _ocr_page_text(doc: documentai.Document, page) -> str:
    chunks: List[str] = []

    if getattr(page, "paragraphs", None):
        for par in page.paragraphs:
            chunks.append(_field_text(doc, par.layout))

    if getattr(page, "tables", None):
        for table in page.tables:
            for row in getattr(table, "body_rows", []) or []:
                cells = []
                for cell in getattr(row, "cells", []) or []:
                    cells.append(_field_text(doc, cell.layout).strip().replace("\n", " "))
                if any(cells):
                    chunks.append("\t".join(cells))

    if not any(s.strip() for s in chunks) and getattr(page, "lines", None):
        for line in page.lines:
            chunks.append(_field_text(doc, line.layout))

    if not any(s.strip() for s in chunks) and getattr(page, "tokens", None):
        chunks.append(" ".join(_field_text(doc, t.layout) for t in page.tokens))

    chunks = [c for c in chunks if c and c.strip()]
    return _norm("\n".join(chunks))

def _process_pdf_bytes(pdf_bytes: bytes) -> List[Tuple[str, Optional[float]]]:
    client = _client_once()
    name = client.processor_path(DOCAI_PROJECT, DOCAI_LOCATION, DOCAI_PROCESSOR)
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf"),
    )
    result = client.process_document(request=request)
    return result  # caller will handle extraction

def _pdf_to_chunks(pdf_bytes: bytes) -> list[bytes]:
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    blobs: list[bytes] = []
    cur = fitz.open()
    cur_pages = 0

    def cur_bytes() -> bytes:
        return cur.tobytes(deflate=True, garbage=3)

    for i in range(len(src)):
        cur.insert_pdf(src, from_page=i, to_page=i)
        cur_pages += 1

        too_many_pages = cur_pages > DOCAI_MAX_PAGES
        too_big = False
        if not too_many_pages:
            try:
                too_big = len(cur_bytes()) > MAX_DOC_BYTES
            except Exception:
                too_big = len(cur.tobytes()) > MAX_DOC_BYTES

        if too_many_pages or too_big:
            cur.delete_page(-1)
            cur_pages -= 1
            blobs.append(cur_bytes())
            cur = fitz.open()
            cur.insert_pdf(src, from_page=i, to_page=i)
            cur_pages = 1

    if cur_pages > 0:
        blobs.append(cur_bytes())

    cur.close()
    src.close()
    return blobs

def docai_ocr_pdf_bytes(pdf_bytes: bytes) -> list[tuple[str, Optional[float]]]:
    """
    Returns list of (page_text, None). For each page, we combine:
    - DocAI extracted text (paragraphs/tables/lines/tokens)
    - Native PyMuPDF text for the same page (as a fallback/merger)
    """
    pages_all: list[tuple[str, Optional[float]]] = []
    # Keep a copy opened with fitz to pull native text by page
    pdf_for_native = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_offset = 0

    for chunk in _pdf_to_chunks(pdf_bytes):
        # DocAI OCR on the chunk
        result = _process_pdf_bytes(chunk)
        doc = result.document

        # Also open chunk with fitz to align native texts
        chunk_pdf = fitz.open(stream=chunk, filetype="pdf")

        for idx, page in enumerate(doc.pages):
            ocr_text = _ocr_page_text(doc, page)
            # native text for the corresponding page in the chunk
            try:
                native_text = chunk_pdf[idx].get_text("text")
            except Exception:
                native_text = ""

            # Merge + dedupe
            merged = _norm((ocr_text + "\n" + (native_text or "")).strip())
            pages_all.append((merged, None))

        chunk_pdf.close()
        page_offset += len(doc.pages)

    pdf_for_native.close()
    return pages_all
