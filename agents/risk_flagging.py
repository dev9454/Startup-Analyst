from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair

RISKS_SCHEMA = '{"risks":[{"code":"","severity":"low|medium|high","message":"","evidence_excerpt":""}]}'

class RiskFlaggingAgent(BaseAgent):
    def __init__(self): super().__init__(name="risk_flagging")

    def run(self, facts: dict, verification: dict):
        context = f"FACTS:\n{facts}\n\nVERIFICATION:\n{verification}"
        user_prompt = (
            "Scan for anomalies (TAM inflation, runway, compliance/licensing, metric inconsistencies, litigation/PR). "
            "Cite brief evidence excerpts.\n"
            f"Return EXACT JSON: {RISKS_SCHEMA}\n"
            "Only JSON."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=context)
        out = parse_json_or_repair(raw)
        self.log("risks_llm", {"n": len(out.get("risks", [])) if isinstance(out, dict) else 0})
        return out.get("risks", []) if isinstance(out, dict) else []
