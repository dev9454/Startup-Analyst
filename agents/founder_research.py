from agents.base import BaseAgent
from tools.llm_router import call_llm_json
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
    def __init__(self):
        super().__init__(name="founder_research")

    def profile(self, founder_names: list[str]):
        # Gather evidence
        results = []
        for name in founder_names:
            for q in [f"{name} lawsuit", f"{name} controversy", f"{name} employment history", f"{name} profile"]:
                results += multi_search(q, k_total=5)
        context = "\n\n".join(
            f"{r.get('title','')}\n{r.get('url','')}\n{r.get('snippet','')}" for r in results[:30]
        ) or ("FOUNDERS:\n" + ", ".join(founder_names))

        task_hint = (
            "Build founder profiles with employment timeline, identify gaps >6 months, list positive vs concern signals, "
            "and include any litigation mentions (url + short summary). Use nulls/empties when unknown."
        )
        raw = call_llm_json(task_hint=task_hint, schema=FOUNDER_SCHEMA, context=context)
        out = parse_json_or_repair(raw)
        self.log("founder_llm", {"n": len(out.get('founders', [])) if isinstance(out, dict) else 0})
        return out
