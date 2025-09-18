from typing import List, Dict
from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair
from tools.search_multi import multi_search

VERIFY_SCHEMA = '{"checks":[{"claim":"","status":"supported|mixed|refuted|unknown","rationale":"","evidence":[{"url":null,"quote":null}]}]}'

class DeepResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="deep_research")

    def _normalize_claims(self, claims_in) -> list[str]:
        """Coerce claims into a clean list[str]. Accepts list[str] or dict with 'claims', or list[dict]."""
        if isinstance(claims_in, dict):
            claims_in = claims_in.get("claims", [])
        out: list[str] = []
        for c in claims_in or []:
            if isinstance(c, str):
                s = c.strip()
                if s:
                    out.append(s)
            elif isinstance(c, dict):
                # common shapes: {"claim": "..."} or {"text": "..."}
                s = c.get("claim") or c.get("text") or ""
                s = str(s).strip()
                if s:
                    out.append(s)
            else:
                s = str(c).strip()
                if s:
                    out.append(s)
        # dedupe while preserving order
        seen = set(); dedup = []
        for s in out:
            if s not in seen:
                seen.add(s); dedup.append(s)
        return dedup

    def verify(self, claims, local_evidence: list[dict[str,str]] | None = None):
        # 1) normalize claims
        claims = self._normalize_claims(claims)
        if not claims:
            self.log("verified", {"n_claims": 0, "evidence_used": 0})
            return {"checks": []}

        # 2) fetch web evidence
        web_ev: list[dict[str, str]] = []
        for c in claims[:10]:
            queries = {c, f"{c} founder", f"{c} revenue", f"{c} market size", f"{c} site:news"}
            for q in queries:
                web_ev += multi_search(q, k_total=8)

        # 3) merge evidence (web + local)
        merged: list[dict[str, str]] = []
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

        claims_block = "\n".join(f"- {c}" for c in claims[:10])
        context = f"CLAIMS TO VERIFY:\n{claims_block}\n\nEVIDENCE:\n" + "\n\n".join(
            f"{e['url']}\n{e['quote']}" for e in merged
        )

        task_hint = (
            "For each claim, set status in {supported|mixed|refuted|unknown}, add a 1–2 sentence rationale, "
            "and include up to 5 pieces of evidence (url, quote). Prefer high-credibility and recent sources."
        )
        raw = call_llm_json(task_hint=task_hint, schema=VERIFY_SCHEMA, context=context)
        res = parse_json_or_repair(raw)
        self.log("verified", {"n_claims": len(claims), "evidence_used": len(merged)})
        return res
