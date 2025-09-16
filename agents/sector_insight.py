# agents/sector_insight.py
from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair
from tools.json_prompt import build_json_prompt
from tools.json_sanitize import sanitize_currency_percent

# Use numeric targets with explicit unit/period to avoid $ and % in numbers
SECTOR_SCHEMA = """{
  "kpis":["..."],
  "baselines":[{"kpi":"","target":0.0,"unit":"","period":""}],
  "snippets":["..."]
}"""

class SectorInsightAgent(BaseAgent):
    def __init__(self): super().__init__(name="sector_insight")

    def load(self, sector: str, extra_context: str = ""):
        context = f"SECTOR={sector}\n{extra_context}"
        user_prompt = build_json_prompt(
            SECTOR_SCHEMA,
            task_hint=("Output sector KPIs, numeric baselines, and slide-ready snippets. "
                       "For each baseline provide: kpi, target (number only), unit (e.g., 'USD','%','mo'), period (e.g., 'monthly').")
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=context)
        raw = sanitize_currency_percent(raw)           # <- fix $ and %
        out = parse_json_or_repair(raw)                # <- robust parse
        self.log("sector_llm", {"sector": sector})
        return out
