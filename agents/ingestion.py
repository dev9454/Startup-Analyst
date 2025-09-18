# agents/ingestion.py
from typing import Dict, Any, List
from agents.base import BaseAgent
from tools.loaders import load_many
from tools.vectorstore import build_index
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair

EXTRACT_FACTS_SCHEMA = (
    '{"founders":[{"name":"","background":""}],'
    '"traction":[{"metric":"","value":""}],'
    '"unit_economics":[{"metric":"","value":""}],'
    '"market":[],"product":[],"legal":[]}'
)

CLAIMS_SCHEMA = '{"claims":["..."]}'

class IngestionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ingestion")

    def run(self, inputs: List[str]):
        docs = load_many(inputs)
        vs = build_index(docs)
        self.log("ingested", {"n_docs": len(docs)})
        return {"vs": vs, "docs": docs}

    def llm_extract_facts(self, vs, company: str) -> Dict[str, Any]:
        ctx = vs.similarity_search(f"{company} founders traction unit economics market product legal", k=15)
        context = "\n\n".join([d.page_content[:1000] for d in ctx]) or f"COMPANY={company}"
        task_hint = (
            "Extract structured facts about founders, traction, unit_economics, market, product, and legal. "
            "Include units like INR, USD, %, users if clearly present. Use empty arrays when unknown."
        )
        raw = call_llm_json(task_hint=task_hint, schema=EXTRACT_FACTS_SCHEMA, context=context)
        data = parse_json_or_repair(raw) or {}
        if not isinstance(data, dict):
            data = {}
        self.log("facts_llm", {"keys": list(data.keys())})
        return data

    def _normalize_claims_list(self, obj) -> List[str]:
        if isinstance(obj, dict):
            obj = obj.get("claims", [])
        out = []
        for c in obj or []:
            if isinstance(c, str):
                s = c.strip()
            elif isinstance(c, dict):
                s = (c.get("claim") or c.get("text") or "").strip()
            else:
                s = str(c).strip()
            if s:
                out.append(s)
        # dedupe
        seen = set(); dedup = []
        for s in out:
            if s not in seen:
                seen.add(s); dedup.append(s)
        return dedup

    def llm_mine_claims(self, vs, company: str) -> List[str]:
        ctx = vs.similarity_search(f"{company} TAM SAM SOM CAC LTV churn GM revenue runway founders", k=12)
        context = "\n\n".join([d.page_content[:1000] for d in ctx]) or f"COMPANY={company}"
        task_hint = (
            "Extract 10 concise, checkable claims covering TAM/SAM/SOM, growth, CAC, LTV, churn, gross margin, "
            "revenue, runway, and founder background. Keep each claim short."
        )
        raw = call_llm_json(task_hint=task_hint, schema=CLAIMS_SCHEMA, context=context)
        out = parse_json_or_repair(raw)
        claims = self._normalize_claims_list(out)
        self.log("claims_llm", {"n": len(claims)})
        return claims[:20]
