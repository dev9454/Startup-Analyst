from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair

TERMS_SCHEMA = '{"clauses":["..."],"dilution":[{"holder":"","post_pct":0.0}],"plain":["..."]}'

class TermSheetSimulatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="term_sheet")

    def advise(self, pre_money_cr: float, raise_cr: float, esop_refresh: float):
        context = f"pre_money_cr={pre_money_cr}; raise_cr={raise_cr}; esop_refresh={esop_refresh}"
        task_hint = (
            "Suggest seed/Series-A appropriate clauses and compute post % for Founders, ESOP, and NewInvestor "
            "(percentage of fully-diluted post-money). Keep plain explanations concise."
        )
        raw = call_llm_json(task_hint=task_hint, schema=TERMS_SCHEMA, context=context)
        out = parse_json_or_repair(raw)
        self.log("terms_llm", {"clauses": len(out.get('clauses', [])) if isinstance(out, dict) else 0})
        return out
