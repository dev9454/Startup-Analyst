from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair

FIN_SCHEMA = """{
  "metrics":{"net_burn_mo":0,"runway_months":0,"zero_cash_date":"YYYY-MM-DD","cac_payback_months":null},
  "assumptions":["..."],
  "sensitivities":[{"case":"base","runway_months":0},{"case":"conservative","runway_months":0}],
  "flags":["..."]
}"""

class FinancialDeepDiveAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="financial_deepdive")

    def analyze(self, tables_or_text: str):
        task_hint = (
            "From the provided financial tables/text, estimate monthly net burn, runway (months), and CAC payback if possible. "
            "State key assumptions, produce base and conservative runway sensitivities, and list notable flags."
        )
        raw = call_llm_json(task_hint=task_hint, schema=FIN_SCHEMA, context=tables_or_text)
        out = parse_json_or_repair(raw)
        self.log("finance_llm", {"keys": list(out.keys()) if isinstance(out, dict) else []})
        return out
