from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair

RISKS_SCHEMA = '{"risks":[{"code":"","severity":"low|medium|high","message":"","evidence_excerpt":""}]}'

class RiskFlaggingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="risk_flagging")

    def run(self, facts: dict, verification: dict):
        context = f"FACTS:\n{facts}\n\nVERIFICATION:\n{verification}"
        task_hint = (
            "Scan for anomalies: TAM inflation, runway/ burn concerns, compliance/licensing gaps, metric inconsistencies, "
            "negative PR/litigation. Add a brief evidence_excerpt for each risk."
        )
        raw = call_llm_json(task_hint=task_hint, schema=RISKS_SCHEMA, context=context)
        out = parse_json_or_repair(raw)
        self.log("risks_llm", {"n": len(out.get('risks', [])) if isinstance(out, dict) else 0})
        return out.get("risks", []) if isinstance(out, dict) else []
