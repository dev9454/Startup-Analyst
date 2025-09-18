# orchestration/orchestrator.py
import time, uuid, pathlib, json
from typing import List, Dict, Any, Iterable

from agents.ingestion import IngestionAgent
from agents.deep_research import DeepResearchAgent
from agents.deal_scoring import DealScoringAgent
from agents.peer_benchmark import PeerBenchmarkAgent
from agents.risk_flagging import RiskFlaggingAgent
from agents.term_sheet import TermSheetSimulatorAgent
from agents.sector_insight import SectorInsightAgent
from agents.learning_loop import LearningLoopAgent

# NEW agents
from agents.financial_deepdive import FinancialDeepDiveAgent
from agents.founder_research import FounderResearchAgent
from agents.market_validation import MarketValidationAgent
from agents.competitive_positioning import CompetitivePositioningAgent
from agents.customer_sentiment import CustomerSentimentAgent
from agents.legal_compliance import LegalComplianceAgent
from agents.narrative import NarrativeAgent
from agents.brief import BriefAgent   # <-- NEW

OUT_NOTES = pathlib.Path("outputs/notes"); OUT_NOTES.mkdir(parents=True, exist_ok=True)


def _collect_urls(items: Iterable[Any], keys: List[str] = None) -> List[str]:
    """
    Walk a nested structure and collect anything that looks like a URL
    from commonly used keys. Very forgiving.
    """
    if keys is None:
        keys = ["url", "sourceUrl", "link", "source"]
    urls = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keys and isinstance(v, str) and v:
                    urls.add(v)
                walk(v)
        elif isinstance(x, list):
            for e in x:
                walk(e)

    walk(items)
    return [u for u in urls if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://") or u == "local")]


class Orchestrator:
    def __init__(self, sector: str = "saas", weights: dict | None = None):
        self.run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.sector = sector

        # Core
        self.ingest = IngestionAgent()
        self.verify = DeepResearchAgent()
        self.score = DealScoringAgent(weights)
        self.bench = PeerBenchmarkAgent()
        self.risk = RiskFlaggingAgent()
        self.terms = TermSheetSimulatorAgent()
        self.learn = LearningLoopAgent()
        self.sector_agent = SectorInsightAgent()

        # New
        self.finance = FinancialDeepDiveAgent()
        self.founders = FounderResearchAgent()
        self.marketval = MarketValidationAgent()
        self.competitive = CompetitivePositioningAgent()
        self.sentiment = CustomerSentimentAgent()
        self.legal = LegalComplianceAgent()
        self.narrative = NarrativeAgent()
        self.brief = BriefAgent()  # <-- NEW

    def run(self, company: str, inputs: List[str]):
        # 1) Ingest & vector index
        ig = self.ingest.run(inputs)
        vs = ig["vs"]

        # 1b) 1–2 sentence brief (cheap + early so it’s available everywhere)
        try:
            brief = self.brief.summarize(vs, company)
        except Exception as e:
            brief = {"brief_1_2_sentences": f"(brief_error: {e})"}

        # 2) LLM-driven facts + claims
        facts = self.ingest.llm_extract_facts(vs, company)
        claims = self.ingest.llm_mine_claims(vs, company)

        # 3) Build local evidence for verification
        ev_docs = vs.similarity_search(f"{company} metrics market", k=12)
        evidence = [{"url": d.metadata.get("source", "local"), "quote": d.page_content[:280]} for d in ev_docs]

        # 4) Verify claims
        try:
            verification = self.verify.verify(claims, evidence)
        except Exception as e:
            verification = {"checks": [], "error": str(e)}

        # 5) Core scoring
        try:
            deal_score = self.score.score(facts, verification)
        except Exception as e:
            deal_score = {"breakdown": {"founders": 0, "traction": 0, "unit_econ": 0, "market": 0}, "total": 0.0, "bullets": [f"scoring_error: {e}"]}

        # 6) Peer benchmarks
        try:
            benches = self.bench.run(company, sector=self.sector)
        except Exception as e:
            benches = {"peers": [], "insights": [f"benchmark_error: {e}"]}

        # 7) Narrative (Scalability, Fundraising, Funding Ask, Key Problem, Why Now)
        try:
            narrative = self.narrative.extract(vs, company)
        except Exception as e:
            narrative = {"error": f"narrative_error: {e}"}

        # 8) New agents (deeper coverage)
        try:
            fin = self.finance.analyze(json.dumps(facts))  # If you have table text, pass that instead
        except Exception as e:
            fin = {"error": f"finance_error: {e}"}

        try:
            founder_names = [f.get("name", "") for f in facts.get("founders", []) if isinstance(f, dict)]
            founders = self.founders.profile(founder_names) if founder_names else {"founders": []}
        except Exception as e:
            founders = {"founders": [], "error": f"founder_error: {e}"}

        try:
            marketv = self.marketval.size(company, self.sector)
        except Exception as e:
            marketv = {"error": f"market_validation_error: {e}"}

        try:
            comp = self.competitive.rank(benches, company)
        except Exception as e:
            comp = {"table": [], "positioning_bullets": [], "data_quality_notes": [f"competitive_error: {e}"]}

        try:
            sent = self.sentiment.listen(company, f"{company} product")
        except Exception as e:
            sent = {"error": f"sentiment_error: {e}"}

        try:
            # Adjust geo as needed; default to India for demo
            legal = self.legal.check(self.sector, "India", company)
        except Exception as e:
            legal = {"error": f"legal_error: {e}"}

        # 9) Risks (now with much richer context)
        try:
            risks = self.risk.run(
                {
                    "facts": facts,
                    "finance": fin,
                    "founders": founders,
                    "market_validation": marketv,
                    "benchmarks": benches,
                    "competitive": comp,
                    "sentiment": sent,
                    "legal": legal,
                    "narrative": narrative,
                },
                verification
            )
        except Exception as e:
            risks = [{"code": "risk_error", "severity": "low", "message": str(e), "evidence_excerpt": ""}]

        # 10) Term sheet & sector notes
        try:
            term = self.terms.advise(pre_money_cr=25, raise_cr=4, esop_refresh=0.1)
        except Exception as e:
            term = {"error": f"terms_error: {e}"}

        try:
            sector_notes = self.sector_agent.load(self.sector)
        except Exception as e:
            sector_notes = {"error": f"sector_error: {e}"}

        # 11) Aggregate sources (from local evidence + agent outputs)
        source_urls = set()
        source_urls.update([e["url"] for e in evidence if e.get("url")])
        for block in [verification, benches, founders, marketv, comp, sent, legal, narrative, brief]:
            try:
                source_urls.update(_collect_urls(block))
            except Exception:
                pass

        # 12) Build final note
        note = {
            "run_id": self.run_id,
            "company": company,
            "sector": self.sector,
            "brief": brief,                     # <-- NEW: short blurb
            "facts": facts,
            "claims": claims,
            "verification": verification,
            "score": deal_score,
            "benchmarks": benches,
            "narrative": narrative,
            "finance": fin,
            "founder_research": founders,
            "market_validation": marketv,
            "competitive_positioning": comp,
            "customer_sentiment": sent,
            "legal_compliance": legal,
            "risks": risks,
            "term_sheet": term,
            "sector_insights": sector_notes,
            "sources": sorted(source_urls),
        }

        out = OUT_NOTES / f"deal_note_{company}_{self.run_id}.json"
        out.write_text(json.dumps(note, indent=2), encoding="utf-8")

        # 13) Flush all agent logs
        for agent in [
            self.ingest, self.verify, self.score, self.bench, self.risk,
            self.terms, self.learn, self.sector_agent,
            self.finance, self.founders, self.marketval, self.competitive,
            self.sentiment, self.legal, self.narrative, self.brief,   # <-- include brief
        ]:
            try:
                agent.flush(self.run_id)
            except Exception:
                pass

        return out, note
