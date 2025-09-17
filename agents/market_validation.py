from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair
from tools.search_multi import multi_search

MARKET_SCHEMA = """{
  "tam":{"value":0,"currency":"USD","year":2025,"method":"","sources":[{"url":"","quote":""}]},
  "sam":{"value":0,"currency":"USD","assumptions":["..."]},
  "som":{"value":0,"currency":"USD","assumptions":["..."]},
  "delta_vs_deck":{"tam_pct":0.0,"notes":""}
}"""

class MarketValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="market_validation")

    def size(self, company: str, sector: str):
        results = []
        for q in [f"{sector} market size 2025 site:statista.com",
                  f"{sector} industry India report",
                  f"{sector} TAM SAM SOM"]:
            results += multi_search(q, k_total=6)

        context = "\n\n".join(
            f"{r.get('title','')}\n{r.get('url','')}\n{r.get('snippet','')}" for r in results[:25]
        ) or f"COMPANY={company}\nSECTOR={sector}"

        task_hint = (
            f"Estimate TAM/SAM/SOM for {company} in {sector}. Provide method and assumptions; include citations in TAM. "
            "Compare against deck claims (if any) and report delta as tam_pct."
        )
        raw = call_llm_json(task_hint=task_hint, schema=MARKET_SCHEMA, context=context)
        out = parse_json_or_repair(raw)
        self.log("market_llm", {"tam": out.get("tam") if isinstance(out, dict) else None})
        return out
