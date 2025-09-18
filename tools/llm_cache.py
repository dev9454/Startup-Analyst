# tools/llm_cache.py
from __future__ import annotations
import hashlib, os, threading

_CACHE_DIR = os.path.join(".cache", "llm")
_LOCK = threading.Lock()
os.makedirs(_CACHE_DIR, exist_ok=True)

def _key(prompt: str, context: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8")); h.update(b"\x00")
    h.update(context.encode("utf-8")); h.update(b"\x00")
    h.update(model.encode("utf-8"))
    return h.hexdigest()

def get(prompt: str, context: str, model: str) -> str | None:
    p = os.path.join(_CACHE_DIR, f"{_key(prompt, context, model)}.json")
    try:
        if os.path.exists(p):
            return open(p, "r", encoding="utf-8").read()
    except Exception:
        pass
    return None

def set(prompt: str, context: str, model: str, text: str) -> None:
    p = os.path.join(_CACHE_DIR, f"{_key(prompt, context, model)}.json")
    with _LOCK:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text or "")
