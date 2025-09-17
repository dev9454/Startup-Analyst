# agents/competitive_positioning.py
from agents.base import BaseAgent
from tools.llm_router import call_llm_json   # <- use the router (Gemini)
from tools.jsonio import parse_json_or_repair

COMP_SCHEMA = """{
  "table":[{"name":"","stage":"","arr":null,"growth_yoy":null,"cac":null,"ltv":null,"ev_arr_band":""}],
  "positioning_bullets":["..."],
  "data_quality_notes":["..."]
}"""

class CompetitivePositioningAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="competitive_positioning")

    def rank(self, peers: dict, company: str):
        # Minimal, token-safe context
        context = f"PEERS(JSON): {peers}\nTARGET: {company}"

        task_hint = (
            "Rank the target vs peers by ARR growth and CAC/LTV where available. "
            "If exact values are missing, infer stage-based EV/ARR bands and note data quality. "
            "Output a table plus 3–5 positioning bullets and any data_quality_notes."
        )

        raw = call_llm_json(task_hint=task_hint, schema=COMP_SCHEMA, context=context)
        out = parse_json_or_repair(raw)
        self.log("comp_llm", {"rows": len(out.get("table", [])) if isinstance(out, dict) else 0})
        return out
