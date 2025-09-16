# tools/search_multi.py
from __future__ import annotations
import os, requests, time
from typing import List, Dict, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib.parse import urlparse
from config import (
    TAVILY_API_KEY, SERPER_API_KEY, EXA_API_KEY,
    SEARCH_TIMEOUT, SEARCH_TOPK_PER_BACKEND, SEARCH_MERGED_TOPK
)

def _norm(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": (item.get("title") or item.get("name") or "").strip(),
        "url": (item.get("url") or item.get("link") or item.get("id") or "").strip(),
        "snippet": (item.get("snippet") or item.get("content") or item.get("text") or "").strip(),
        "source": item.get("source") or "",
        "score": float(item.get("score", 0.0)),
        "published": item.get("published") or item.get("date") or "",
    }

def _domain(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

# --- Tavily ---
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
def _tavily(q: str, k=SEARCH_TOPK_PER_BACKEND) -> List[Dict]:
    if not TAVILY_API_KEY: return []
    r = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": q, "max_results": k, "search_depth": "advanced"},
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    res = [{"source":"tavily", **x} for x in r.json().get("results", [])][:k]
    return [_norm(x) for x in res]

# --- Serper (Google SERP) ---
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
def _serper(q: str, k=SEARCH_TOPK_PER_BACKEND) -> List[Dict]:
    if not SERPER_API_KEY: return []
    r = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": q, "num": k},
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    items = []
    data = r.json()
    for block in ("knowledgeGraph","organic","news"):
        for it in data.get(block, []) or []:
            items.append({"title": it.get("title") or it.get("name"),
                          "url": it.get("link") or it.get("url"),
                          "snippet": it.get("snippet") or it.get("description"),
                          "source": f"serper:{block}"})
    return [_norm(x) for x in items][:k]

# --- Exa (semantic) ---
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6))
def _exa(q: str, k=SEARCH_TOPK_PER_BACKEND) -> List[Dict]:
    if not EXA_API_KEY: return []
    r = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
        json={"query": q, "numResults": k, "useAutoprompt": True},
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    items = []
    for it in r.json().get("results", [])[:k]:
        items.append({"title": it.get("title"),
                      "url": it.get("url"),
                      "snippet": it.get("text"),
                      "source": "exa",
                      "score": it.get("score", 0.0),
                      "published": it.get("publishedDate")})
    return [_norm(x) for x in items]

# --- Merge/rank utilities ---
HIGH_AUTH = {
    # boost official / high-signal domains
    "www.mca.gov.in", "www.sec.gov", "www.sebi.gov.in", "rbi.org.in",
    "www.bseindia.com", "www.nseindia.com", "www.ft.com", "www.wsj.com",
    "www.bloomberg.com", "www.reuters.com", "www.economist.com",
    "www.crunchbase.com", "www.pitchbook.com", "tracxn.com",
    "www.statista.com", "www.oecd.org", "www.worldbank.org",
}

def _rank(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Deduplicate by URL (closest to canonical)
    seen, dedup = set(), []
    for x in items:
        u = x.get("url","")
        if not u or u in seen: 
            continue
        seen.add(u)
        dedup.append(x)
    # Score heuristic: backend score + domain authority + freshness
    ranked = []
    now = time.time()
    for x in dedup:
        base = float(x.get("score", 0.0))
        dom = _domain(x.get("url",""))
        auth = 1.0 if dom in HIGH_AUTH else 0.0
        # crude recency: prefer if date-like string present
        recent = 0.3 if any(ch.isdigit() for ch in (x.get("published") or "")) else 0.0
        backend = 0.2 if x.get("source","").startswith("serper") else 0.1 if x.get("source")=="tavily" else 0.15 if x.get("source")=="exa" else 0.0
        ranked.append((base + auth + recent + backend, x))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return [x for _, x in ranked][:SEARCH_MERGED_TOPK]

# --- Public facade ---
def multi_search(query: str, k_total: int = SEARCH_MERGED_TOPK) -> List[Dict]:
    results = []
    try:   results += _tavily(query)
    except Exception: pass
    try:   results += _serper(query)
    except Exception: pass
    try:   results += _exa(query)
    except Exception: pass
    ranked = _rank(results)
    return ranked[:k_total]
