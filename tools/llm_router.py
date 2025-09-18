# tools/llm_router.py
from typing import Optional
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL_ID, GEMINI_MODEL_NAME

_MODEL = GEMINI_MODEL_ID or GEMINI_MODEL_NAME

def _client_once():
    if not getattr(genai, "_cfg", None):
        genai.configure(api_key=GEMINI_API_KEY)
        genai._cfg = True
    return genai.GenerativeModel(_MODEL)

def call_llm_json(*, task_hint: str, schema: str, context: str = "", max_tokens: int = 2048) -> str:
    """
    Returns the *raw model text* (string). Use your parse_json_or_repair on it.
    """
    mdl = _client_once()
    sys = (
        "You are a strict JSON generator. "
        "ALWAYS return ONLY valid JSON that matches the schema exactly. "
        "No markdown, no prose, no extra keys."
    )
    prompt = (
        f"{sys}\n\n"
        f"SCHEMA:\n{schema}\n\n"
        f"TASK:\n{task_hint}\n\n"
        f"CONTEXT:\n{context}\n"
    )
    resp = mdl.generate_content(prompt)
    # robustly extract text (avoid .text when empty candidates)
    if not resp.candidates:
        return "{}"
    parts = []
    for part in resp.candidates[0].content.parts:
        if getattr(part, "text", None):
            parts.append(part.text)
    return "".join(parts).strip() if parts else "{}"

def call_llm_text(*, prompt: str, context: str = "", max_tokens: int = 512) -> str:
    mdl = _client_once()
    full = f"{prompt}\n\nCONTEXT:\n{context}"
    resp = mdl.generate_content(full)
    if not resp.candidates:
        return ""
    parts = []
    for part in resp.candidates[0].content.parts:
        if getattr(part, "text", None):
            parts.append(part.text)
    return "".join(parts).strip()
