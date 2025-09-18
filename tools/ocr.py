# tools/ocr.py
from __future__ import annotations
import io, os
from typing import Tuple, Optional

def google_vision_ocr(image_bytes: bytes) -> Tuple[str, Optional[float]]:
    # Requires: pip install google-cloud-vision
    # ADC: gcloud auth application-default login
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        resp = client.document_text_detection(image=image)  # better for dense text
        if resp.error.message:
            raise RuntimeError(resp.error.message)
        text = resp.full_text_annotation.text or ""
        # Vision doesn't give a single confidence; return None
        return text, None
    except Exception:
        return "", None

def tesseract_ocr(image_bytes: bytes) -> Tuple[str, Optional[float]]:
    # Requires: pip install pytesseract pillow
    # Also install Tesseract binary (Windows: https://github.com/UB-Mannheim/tesseract/wiki)
    try:
        from PIL import Image
        import pytesseract, io as _io
        img = Image.open(_io.BytesIO(image_bytes))
        txt = pytesseract.image_to_string(img)
        return txt or "", None
    except Exception:
        return "", None
