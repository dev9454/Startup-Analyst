from typing import List, Dict
from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair
from tools.search_multi import multi_search

VERIFY_SCHEMA = '{"checks":[{"claim":"","status":"supported|mixed|refuted|unknown","rationale":"","evidence":[{"url":null,"quote":null}]}]}'

class DeepResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="deep_research")

    def verify(self, claims: List[str], local_evidence: List[Dict[str,str]] | None = None):
        # Search multiple backends per claim; keep it efficient
        web_ev: List[Dict[str, str]] = []
        for c in claims[:10]:
            for q in {c, f"{c} founder", f"{c} revenue", f"{c} market size", f"{c} site:news"}:
                web_ev += multi_search(q, k_total=8)

        # Build compact evidence set (merge web + local)
        merged: List[Dict[str, str]] = []
        seen = set()
        for src in (web_ev + (local_evidence or [])):
            url = (src.get("url") or src.get("source") or "local").strip()
            snippet = (src.get("snippet") or src.get("quote") or "").strip()
            key = (url, snippet[:100])
            if key in seen:
                continue
            seen.add(key)
            merged.append({"url": url, "quote": snippet[:260]})
            if len(merged) >= 25:
                break

        # Put claims up front so the model knows what to verify
        claims_block = "\n".join(f"- {c}" for c in claims[:10])
        context = f"CLAIMS TO VERIFY:\n{claims_block}\n\nEVIDENCE:\n" + "\n\n".join([f"{e['url']}\n{e['quote']}" for e in merged])

        task_hint = (
            "For each claim, set status in {supported|mixed|refuted|unknown}, add a 1–2 sentence rationale, "
            "and include up to 5 pieces of evidence (url, quote). Prefer high-credibility and recent sources."
        )
        raw = call_llm_json(task_hint=task_hint, schema=VERIFY_SCHEMA, context=context)
        res = parse_json_or_repair(raw)
        self.log("verified", {"n_claims": len(claims), "evidence_used": len(merged)})
        return res
