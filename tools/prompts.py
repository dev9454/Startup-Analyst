# tools/prompts.py
EXTRACT_FACTS_PROMPT = """
You ingest mixed founder materials and public snippets and output structured facts.

Fields:
- founders: [{name, education, prior_companies, notable_achievements}]
- traction: [{metric, value, period, evidence_excerpt}]
- unit_economics: [{metric, value, formula, evidence_excerpt}]
- market: [{metric, value, geo, source_citation}]
- product: [{feature, price, plan, notes}]
- legal: [{topic, detail, risk_level, evidence_excerpt}]

Return JSON with exactly these top-level keys: 
{"founders":[],"traction":[],"unit_economics":[],"market":[],"product":[],"legal":[]}
"""

CLAIM_MINER_PROMPT = """
Extract 10 concise, checkable claims from the startup materials and/or retrieved context.
Each claim must be a single sentence about TAM/SAM/SOM, growth, CAC, LTV, churn, GM, revenue, runway, or founder background.
Return: {"claims":[ "...", "...", ... ]}
"""

VERIFY_PROMPT = """
You are verifying startup claims using the provided snippets/links in Context.
For each claim, set status: "supported" | "mixed" | "refuted" | "unknown".
Include rationale (1-2 sentences) and up to 5 evidence items from Context.
Return: {"checks":[{"claim":"...","status":"...","rationale":"...","evidence":[{"url":null,"quote":null}]}]}
"""

SCORING_PROMPT = """
Score this deal on 0-100 subscales with justification bullets, then compute weighted total.

Inputs:
- weights: {"founders":float,"traction":float,"unit_econ":float,"market":float} (sum≈1)
- facts + verification in Context.

Rules:
- Be conservative when evidence is weak ("unknown" counts against).
- Round to integers for subscores, 1 decimal for total.

Return:
{"breakdown":{"founders":0,"traction":0,"unit_econ":0,"market":0},
 "total":0.0,
 "bullets":["...","...","..."]}
"""

PEERS_PROMPT = """
Identify 5-10 comparable startups (same sector/geo stage). For each peer include:
{name, brief_reason, url, stage, revenue_or_arr:null|number, ev:null|number, notes}
Add 3-5 insights comparing the target vs peers (EV/ARR bands if available).
Return: {"peers":[...], "insights":[ "...", "..."]}
"""

RISKS_PROMPT = """
Scan the facts and checks for anomalies or red flags:
- TAM inflation / outdated market data
- Revenue vs pipeline mismatch
- Compliance/licensing risks
- Unusual churn/CAC/LTV inconsistencies
- Litigation or PR risk
Return: {"risks":[{"code":"...","severity":"low|medium|high","message":"...","evidence_excerpt":"..."}]}
"""

TERMS_PROMPT = """
Suggest term-sheet clauses tailored to this profile (seed to Series A).
Also propose a dilution table (post-%) for holders given inputs:
- pre_money_cr
- raise_cr
- esop_refresh (0-0.2)
Assume initial Founders=100% pre. Compute post percents.

Return:
{"clauses":["..."], "dilution":[{"holder":"Founders","post_pct":0.0},{"holder":"ESOP","post_pct":0.0},{"holder":"NewInvestor","post_pct":0.0}],
 "plain":["...","..."]}
"""

SECTOR_PROMPT = """
Given the sector name and context, output:
{"kpis":["..."], "baselines":{"kpi":"target"}, "snippets":["slide-ready 1-liner","..."]}
Prefer KPIs relevant to the sector and data you saw.
"""
