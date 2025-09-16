from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair

FIN_SCHEMA = """{
  "metrics":{"net_burn_mo":0,"runway_months":0,"zero_cash_date":"YYYY-MM-DD","cac_payback_months":null},
  "assumptions":["..."],
  "sensitivities":[{"case":"base","runway_months":0},{"case":"conservative","runway_months":0}],
  "flags":["..."]
}"""

class FinancialDeepDiveAgent(BaseAgent):
    def __init__(self): super().__init__(name="financial_deepdive")

    def analyze(self, tables_or_text: str):
        user_prompt = (
            "Analyze financial tables or text to estimate burn, runway, and CAC payback. "
            f"Return EXACT JSON: {FIN_SCHEMA}\nOnly JSON."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=tables_or_text)
        out = parse_json_or_repair(raw)
        self.log("finance_llm", {"keys": list(out.keys()) if isinstance(out, dict) else []})
        return out
