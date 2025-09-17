from __future__ import annotations
import google.generativeai as genai
from typing import Optional
from config import GEMINI_API_KEY, GEMINI_MODEL_ID

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Export it or put it in config.py")

genai.configure(api_key=GEMINI_API_KEY)

def call_gemini_llm(user_prompt: str, context: str = "",
                    json_only: bool = True,
                    model: Optional[str] = None,
                    temperature: float = 0.0,
                    max_tokens: int = 2048) -> str:
    model_id = model or GEMINI_MODEL_ID
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if json_only:
        generation_config["response_mime_type"] = "application/json"

    mdl = genai.GenerativeModel(model_id)
    res = mdl.generate_content(
        contents=[user_prompt, {"text": context}],
        generation_config=generation_config
    )
    return res.text or ""
