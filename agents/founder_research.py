from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair
from tools.search_multi import multi_search

FOUNDER_SCHEMA = """{
  "founders":[
    {"name":"","timeline":[{"role":"","org":"","start":"","end":""}],
     "gaps":[{"months":0,"between":["",""]}],
     "signals":{"positive":["..."],"concerns":["..."]},
     "litigation":[{"url":"","summary":""}]}
  ]
}"""

class FounderResearchAgent(BaseAgent):
    def __init__(self): super().__init__(name="founder_research")

    def profile(self, founder_names: list[str]):
        # gather evidence
        results = []
        for name in founder_names:
            for q in [f"{name} lawsuit", f"{name} controversy", f"{name} employment history", f"{name} profile"]:
                results += multi_search(q, k_total=5)
        ctx = "\n\n".join([f"{r.get('title','')}\n{r.get('url','')}\n{r.get('snippet','')}" for r in results[:30]])
        user_prompt = (
            "Build founder profiles with employment timeline, gaps, positive/negative signals, and litigation mentions. "
            f"Return EXACT JSON: {FOUNDER_SCHEMA}\nOnly JSON."
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=ctx)
        out = parse_json_or_repair(raw)
        self.log("founder_llm", {"n": len(out.get("founders", [])) if isinstance(out, dict) else 0})
        return out
