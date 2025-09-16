from agents.base import BaseAgent
from tools.llm import call_bedrock_llm
from tools.jsonio import parse_json_or_repair
from tools.search_multi import multi_search
from tools.json_prompt import build_json_prompt

PEERS_SCHEMA = """{
  "peers":[{"name":"","brief_reason":"","url":null,"stage":null,"revenue_or_arr":null,"ev":null,"notes":null}],
  "insights":["..."]
}"""

class PeerBenchmarkAgent(BaseAgent):
    def __init__(self): super().__init__(name="peer_benchmark")

    def run(self, company: str, sector: str):
        queries = [
            f"{company} competitors {sector}",
            f"{sector} startups India revenue",
            f"{sector} similar companies",
            f"{sector} EV/ARR multiples private"
        ]
        ctx_lines = []
        for q in queries:
            for r in multi_search(q, k_total=5):
                ctx_lines.append(f"{r.get('title','')}\n{r.get('url','')}\n{r.get('snippet','')}")
                if len(ctx_lines) >= 25:
                    break
            if len(ctx_lines) >= 25:
                break
        context = "\n\n".join(ctx_lines)

        user_prompt = build_json_prompt(
            PEERS_SCHEMA,
            task_hint=("Identify 5–10 comparables (same sector/geo/stage). For each include name, reason, url, stage, "
                       "revenue_or_arr (if any), ev (if any), notes. Add 3–5 insights.")
        )
        raw = call_bedrock_llm(user_prompt=user_prompt, context=context)
        out = parse_json_or_repair(raw)
        self.log("peers_llm", {"n": len(out.get('peers', [])) if isinstance(out, dict) else 0})
        return out
