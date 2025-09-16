from __future__ import annotations
import requests
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from config import TAVILY_API_KEY

def _norm(item) -> Dict:
    return {
        "title": item.get("title") or "",
        "url": item.get("url") or "",
        "snippet": item.get("snippet") or item.get("content") or "",
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def web_search(query: str, k: int = 6) -> List[Dict]:
    if not TAVILY_API_KEY:
        return []
    r = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": k, "search_depth": "advanced"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    return [_norm(x) for x in data.get("results", [])][:k]
