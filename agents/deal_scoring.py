from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair

SCORING_SCHEMA = '{"breakdown":{"founders":0,"traction":0,"unit_econ":0,"market":0},"total":0.0,"bullets":["..."]}'

class DealScoringAgent(BaseAgent):
    def __init__(self, weights=None):
        super().__init__(name="deal_scoring")
        self.weights = weights or {"founders":0.30,"traction":0.25,"unit_econ":0.25,"market":0.20}

    def score(self, facts: dict, verification: dict):
        context = f"FACTS:\n{facts}\n\nVERIFICATION:\n{verification}\n\nWEIGHTS:\n{self.weights}"
        user_prompt = (
            "Score the deal using the weights; conservative for unknowns. "
            "Round subscores to integers, total to 1 decimal.\n"
            f"Return EXACT JSON: {SCORING_SCHEMA}\n"
            "Only JSON."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=context)
        out = parse_json_or_repair(raw)
        self.log("score_llm", {"total": out.get("total") if isinstance(out, dict) else None})
        return out
