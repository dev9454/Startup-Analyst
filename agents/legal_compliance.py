from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair
from tools.search_multi import multi_search

LEGAL_SCHEMA = """{
  "requirements":["..."],
  "matched":["..."],
  "missing":["..."],
  "citations":[{"url":"","quote":""}]
}"""

class LegalComplianceAgent(BaseAgent):
    def __init__(self): super().__init__(name="legal_compliance")

    def check(self, sector: str, geo: str, company: str):
        results = []
        for q in [f"{sector} regulatory requirements {geo}", f"{sector} license required {geo}", f"{company} compliance {geo}"]:
            results += multi_search(q, k_total=5)
        ctx = "\n\n".join([f"{r.get('title','')}\n{r.get('url','')}\n{r.get('snippet','')}" for r in results[:20]])
        user_prompt = (
            f"List regulatory licenses and compliance requirements for {sector} in {geo}. "
            "Mark which the company appears to have vs missing. "
            f"Return EXACT JSON: {LEGAL_SCHEMA}\nOnly JSON."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=ctx)
        out = parse_json_or_repair(raw)
        self.log("legal_llm", {"reqs": out.get("requirements") if isinstance(out, dict) else None})
        return out
