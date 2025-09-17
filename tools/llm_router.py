from __future__ import annotations
from tools.json_prompt import build_json_prompt  # you already have this helper
from tools.gemini_llm import call_gemini_llm

def call_llm_json(task_hint: str, schema: str, context: str = "") -> str:
    """Builds a strict JSON-only prompt and calls Gemini."""
    prompt = build_json_prompt(schema, task_hint)
    return call_gemini_llm(user_prompt=prompt, context=context, json_only=True)
