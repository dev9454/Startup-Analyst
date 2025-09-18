# agents/brief.py
from __future__ import annotations
from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair

BRIEF_SCHEMA = """{
  "brief_1_2_sentences": ""
}"""

class BriefAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="brief")

    def summarize(self, vs, company: str):
        # Pull small, high-signal context
        q = f"{company} what it does problem solution product customers market geography"
        docs = vs.similarity_search(q, k=8)
        context = "\n\n---\n\n".join(d.page_content[:800] for d in docs) or f"COMPANY={company}"

        task_hint = (
            "Write a neutral, investor-style overview in 1–2 sentences: "
            "what the company does, for whom, and the core value proposition. "
            "No marketing fluff; avoid unverified claims. Return exactly this JSON."
        )
        raw = call_llm_json(task_hint=task_hint, schema=BRIEF_SCHEMA, context=context, max_tokens=512)
        out = parse_json_or_repair(raw)
        # safety fallback
        text = (out.get("brief_1_2_sentences") or "").strip() if isinstance(out, dict) else ""
        if not text:
            text = f"{company} is a startup; details were sparse in the provided materials."
            out = {"brief_1_2_sentences": text}
        self.log("brief_llm", {"chars": len(text)})
        return out
