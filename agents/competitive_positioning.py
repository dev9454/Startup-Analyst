from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair

COMP_SCHEMA = """{
  "table":[{"name":"","stage":"","arr":null,"growth_yoy":null,"cac":null,"ltv":null,"ev_arr_band":""}],
  "positioning_bullets":["..."],
  "data_quality_notes":["..."]
}"""

class CompetitivePositioningAgent(BaseAgent):
    def __init__(self): super().__init__(name="competitive_positioning")

    def rank(self, peers: dict, company: str):
        ctx = f"PEERS:\n{peers}\n\nTarget: {company}"
        user_prompt = (
            "Rank peers vs target by ARR growth, CAC/LTV, multiples. "
            "Use proxy bands if exacts missing. "
            f"Return EXACT JSON: {COMP_SCHEMA}\nOnly JSON."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=ctx)
        out = parse_json_or_repair(raw)
        self.log("comp_llm", {"rows": len(out.get("table", [])) if isinstance(out, dict) else 0})
        return out
