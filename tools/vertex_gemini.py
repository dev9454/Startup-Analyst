# tools/vertex_gemini.py
from __future__ import annotations
from typing import Optional
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from config import GCP_PROJECT, GCP_LOCATION, GEMINI_MODEL_ID

_vertex_init = False

def _ensure_init():
    global _vertex_init
    if _vertex_init:
        return
    if not GCP_PROJECT or not GCP_LOCATION:
        raise RuntimeError("GCP_PROJECT / GCP_LOCATION not set in config.py")
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    _vertex_init = True

def call_vertex_gemini(user_prompt: str, context: str = "",
                       json_only: bool = True,
                       model: Optional[str] = None,
                       temperature: float = 0.0,
                       max_tokens: int = 2048) -> str:
    _ensure_init()
    model_id = model or GEMINI_MODEL_ID
    gen_cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    if json_only:
        gen_cfg.response_mime_type = "application/json"
    mdl = GenerativeModel(model_id)
    resp = mdl.generate_content([user_prompt, {"text": context}], generation_config=gen_cfg)
    return resp.text or ""
