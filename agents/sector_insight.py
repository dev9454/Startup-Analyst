from agents.base import BaseAgent
from tools.jsonio import parse_json_or_repair
from tools.json_sanitize import sanitize_currency_percent
from tools.llm_router import call_llm_json

SECTOR_SCHEMA = """{
  "kpis":["..."],
  "baselines":[{"kpi":"","target":0.0,"unit":"","period":""}],
  "snippets":["..."]
}"""

class SectorInsightAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="sector_insight")

    def load(self, sector: str, extra_context: str = ""):
        context = f"SECTOR={sector}\n{extra_context}"
        task_hint = (
            "Output sector KPIs, numeric baselines (target as number only), and slide-ready snippets. "
            "For each baseline include: kpi, target, unit (USD/%/mo), period (monthly/quarterly)."
        )
        raw = call_llm_json(task_hint=task_hint, schema=SECTOR_SCHEMA, context=context)
        raw = sanitize_currency_percent(raw)
        out = parse_json_or_repair(raw)
        self.log("sector_llm", {"sector": sector})
        return out
