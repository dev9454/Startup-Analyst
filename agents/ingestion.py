from typing import Dict, Any, List
from agents.base import BaseAgent
from tools.loaders import load_many
from tools.vectorstore import build_index
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair

EXTRACT_FACTS_SCHEMA = '{"founders":[],"traction":[],"unit_economics":[],"market":[],"product":[],"legal":[]}'
CLAIMS_SCHEMA = '{"claims":["..."]}'

class IngestionAgent(BaseAgent):
    def __init__(self): super().__init__(name="ingestion")

    def run(self, inputs: List[str]):
        docs = load_many(inputs)
        vs = build_index(docs)
        self.log("ingested", {"n_docs": len(docs)})
        return {"vs": vs, "docs": docs}

    def llm_extract_facts(self, vs, company: str) -> Dict[str, Any]:
        ctx = vs.similarity_search(f"{company} founders traction unit economics market", k=12)
        context = "\n\n".join([d.page_content[:1000] for d in ctx])
        user_prompt = (
            "Extract structured facts from the context.\n"
            f"Return EXACTLY this JSON shape: {EXTRACT_FACTS_SCHEMA}\n"
            "No markdown. Use nulls if unknown. Return ONLY JSON."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=context)
        data = parse_json_or_repair(raw)
        self.log("facts_llm", {"keys": list(data.keys()) if isinstance(data, dict) else []})
        return data

    def llm_mine_claims(self, vs, company: str) -> List[str]:
        ctx = vs.similarity_search(f"{company} TAM SAM SOM CAC LTV churn GM revenue runway founders", k=12)
        context = "\n\n".join([d.page_content[:1000] for d in ctx])
        user_prompt = (
            "Extract 10 concise, checkable claims (TAM/SAM/SOM, growth, CAC, LTV, churn, GM, revenue, runway, founder background).\n"
            f"Return EXACT JSON: {CLAIMS_SCHEMA}\n"
            "Only JSON, no prose."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=context)
        out = parse_json_or_repair(raw)
        claims = out.get("claims", []) if isinstance(out, dict) else []
        self.log("claims_llm", {"n": len(claims)})
        return claims[:20]
