# agents/customer_sentiment.py
from agents.base import BaseAgent
from tools.llm_router import call_llm_json
from tools.jsonio import parse_json_or_repair
from tools.search_multi import multi_search

SENT_SCHEMA = """{
  "app":{"platform":"","rating":0.0,"reviews_sample":[{"quote":"","date":""}]},
  "themes":{"pros":["..."],"cons":["..."]},
  "social_snippets":[{"url":"","quote":""}],
  "alerts":["..."]
}"""

class CustomerSentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="customer_sentiment")

    def listen(self, brand: str, product: str):
        # keep web hits small to save API quota; if backends are disabled, multi_search returns []
        results = []
        for q in [f"{brand} reviews", f"{product} app store rating", f"{brand} complaints site:reddit.com"]:
            results += multi_search(q, k_total=4)

        context = "\n\n".join(
            f"{r.get('title','')}\n{r.get('url','')}\n{r.get('snippet','')}" for r in results[:18]
        ) or f"BRAND={brand}\nPRODUCT={product}"

        task_hint = (
            "Summarize customer sentiment: app ratings (if any), three pros and three cons, social buzz snippets, "
            "and any notable alerts. Use nulls where data is missing. Return exactly one JSON object."
        )

        raw = call_llm_json(task_hint=task_hint, schema=SENT_SCHEMA, context=context)
        out = parse_json_or_repair(raw)
        self.log("sentiment_llm", {"ok": True})
        return out
