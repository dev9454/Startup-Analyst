from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair
from tools.search_multi import multi_search
from tools.json_prompt import build_json_prompt

SENT_SCHEMA = """{
  "app":{"platform":"","rating":0.0,"reviews_sample":[{"quote":"","date":""}]},
  "themes":{"pros":["..."],"cons":["..."]},
  "social_snippets":[{"url":"","quote":""}],
  "alerts":["..."]
}"""

class CustomerSentimentAgent(BaseAgent):
    def __init__(self): super().__init__(name="customer_sentiment")

    def listen(self, brand: str, product: str):
        # Keep web hits small to save API quota; respect no-web by missing keys
        results = []
        for q in [f"{brand} reviews", f"{product} app store rating", f"{brand} complaints site:reddit.com"]:
            results += multi_search(q, k_total=4)
        ctx = "\n\n".join([f"{r.get('title','')}\n{r.get('url','')}\n{r.get('snippet','')}" for r in results[:18]])

        user_prompt = build_json_prompt(
            SENT_SCHEMA,
            task_hint="Summarize customer sentiment: app ratings, 3 pros & 3 cons, social buzz, and any alerts."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=ctx)
        out = parse_json_or_repair(raw)
        self.log("sentiment_llm", {"ok": True})
        return out
