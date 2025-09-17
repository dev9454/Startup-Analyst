from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair

SCORING_SCHEMA = '{"breakdown":{"founders":0,"traction":0,"unit_econ":0,"market":0},"total":0.0,"bullets":["..."]}'

class DealScoringAgent(BaseAgent):
    def __init__(self, weights=None):
        super().__init__(name="deal_scoring")
        self.weights = weights or {"founders":0.30,"traction":0.25,"unit_econ":0.25,"market":0.20}

    def score(self, facts: dict, verification: dict):
        context = f"FACTS:\n{facts}\n\nVERIFICATION:\n{verification}\n\nWEIGHTS:\n{self.weights}"
        task_hint = (
            "Score the deal using the provided weights. Be conservative when evidence is unknown or mixed. "
            "Round sub-scores (founders, traction, unit_econ, market) to integers in [0..100]. "
            "Compute total as a weighted average, rounded to one decimal. Include 3–6 short bullets."
        )
        raw = call_llm_json(task_hint=task_hint, schema=SCORING_SCHEMA, context=context)
        out = parse_json_or_repair(raw)
        self.log("score_llm", {"total": out.get("total") if isinstance(out, dict) else None})
        return out
