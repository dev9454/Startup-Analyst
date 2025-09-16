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

OUT_NOTES = pathlib.Path("outputs/notes"); OUT_NOTES.mkdir(parents=True, exist_ok=True)


def _collect_urls(items: Iterable[Any], keys: List[str] = None) -> List[str]:
    """
    Walk a nested structure and collect anything that looks like a URL
    from commonly used keys. Very forgiving.
    """
    if keys is None:
        keys = ["url", "sourceUrl", "link"]
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
    return list(urls)


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

    def run(self, company: str, inputs: List[str]):
        # 1) Ingest & vector index
        ig = self.ingest.run(inputs)
        vs = ig["vs"]

        # 2) LLM-driven facts + claims
        facts = self.ingest.llm_extract_facts(vs, company)
        claims = self.ingest.llm_mine_claims(vs, company)

        # 3) Build local evidence for verification
        ev_docs = vs.similarity_search(f"{company} metrics market", k=12)
        evidence = [{"url": d.metadata.get("source", "local"), "quote": d.page_content[:280]} for d in ev_docs]

        # 4) Verify claims (DeepResearchAgent may also use web backends)
        verification = self.verify.verify(claims, evidence)

        # 5) Core scoring
        deal_score = self.score.score(facts, verification)

        # 6) Peer benchmarks
        benches = self.bench.run(company, sector=self.sector)

        # 7) New agents (deeper coverage)
        fin = self.finance.analyze(json.dumps(facts))  # pass facts text; swap for parsed tables if you have them
        founder_names = [f.get("name", "") for f in facts.get("founders", []) if isinstance(f, dict)]
        founders = self.founders.profile(founder_names) if founder_names else {"founders": []}
        marketv = self.marketval.size(company, self.sector)
        comp = self.competitive.rank(benches, company)
        sent = self.sentiment.listen(company, f"{company} product")
        # Adjust geo as needed; default to India for demo
        legal = self.legal.check(self.sector, "India", company)

        # 8) Risks (now with much richer context)
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
            },
            verification
        )

        # 9) Term sheet & sector notes
        term = self.terms.advise(pre_money_cr=25, raise_cr=4, esop_refresh=0.1)
        sector_notes = self.sector_agent.load(self.sector)

        # 10) Aggregate sources (from local evidence + agent outputs)
        source_urls = set()
        source_urls.update([e["url"] for e in evidence if e.get("url")])
        source_urls.update(_collect_urls(verification))
        source_urls.update(_collect_urls(benches))
        source_urls.update(_collect_urls(founders))
        source_urls.update(_collect_urls(marketv))
        source_urls.update(_collect_urls(comp))
        source_urls.update(_collect_urls(sent))
        source_urls.update(_collect_urls(legal))

        # 11) Build final note
        note = {
            "run_id": self.run_id,
            "company": company,
            "sector": self.sector,
            "facts": facts,
            "claims": claims,
            "verification": verification,
            "score": deal_score,
            "benchmarks": benches,
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

        # 12) Flush all agent logs
        for agent in [
            self.ingest, self.verify, self.score, self.bench, self.risk,
            self.terms, self.learn, self.sector_agent,
            self.finance, self.founders, self.marketval, self.competitive,
            self.sentiment, self.legal,
        ]:
            agent.flush(self.run_id)

        return out, note
