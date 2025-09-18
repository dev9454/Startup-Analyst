# agents/narrative.py
from agents.base import BaseAgent
from tools.llm_router import call_llm_json, call_llm_text
from tools.jsonio import parse_json_or_repair

NARRATIVE_SCHEMA = """{
  "scalability":{
    "summary":"",
    "drivers":["..."],
    "unit_cost_levers":["..."]
  },
  "fundraising":{
    "total_raised_to_date": null,
    "current_round": null,
    "target_raise": null,
    "pre_money": null,
    "post_money": null,
    "dilution_target_pct": null,
    "deal_rationale": ""
  },
  "funding_ask":{
    "ask": null,
    "milestones":["..."],
    "spend_breakdown":[],
    "runway_months": null,
    "commitments_or_lois":[]
  },
  "key_problem_solved":{
    "customer_pain_points":["..."],
    "solution_how":["..."]
  },
  "why_now":{
    "thesis":"",
    "catalysts":["..."],
    "proof_points":["..."]
  },
  "data_quality_notes":["..."]
}"""

class NarrativeAgent(BaseAgent):
    def __init__(self): 
        super().__init__(name="narrative")

    def extract(self, vs, company: str):
        """Extract structured narrative (scalability, fundraising, etc.)."""
        ctx = vs.similarity_search(
            f"{company} scalability fundraising funding ask problem solved why now", k=12
        )
        context = "\n\n".join(d.page_content[:1200] for d in ctx) or f"COMPANY={company}"
        task_hint = (
            "Extract narrative aspects of the startup: scalability, fundraising history & targets, "
            "funding ask, key problem solved, and why now. Preserve numbers with units (USD, INR, %, months)."
        )
        raw = call_llm_json(task_hint=task_hint, schema=NARRATIVE_SCHEMA, context=context)
        data = parse_json_or_repair(raw)
        self.log("narrative_llm", {"ok": bool(data)})
        return data or {}

    def brief(self, vs, company: str):
        """Generate a crisp 1–2 sentence description for the company."""
        ctx = vs.similarity_search(f"{company} about overview mission what we do product pricing customers", k=6)
        context = "\n\n".join(d.page_content[:1000] for d in ctx) or f"COMPANY={company}"
        try:
            txt = call_llm_text(
                prompt="Write a crisp 1–2 sentence company description in plain English, no fluff.",
                context=context
            )
            desc = (txt or "").strip()
        except Exception:
            desc = ""
        if not desc:
            # fallback if LLM fails
            lines = [l.strip() for l in context.splitlines() if len(l.strip()) > 20]
            desc = " ".join(lines[:2])[:400] if lines else f"{company} — early-stage startup; details pending."
        return {"brief_1_2_sentences": desc}
