# tools/webtext.py
import re, html, requests
from bs4 import BeautifulSoup
import os
USER_AGENT = os.getenv("USER_AGENT", "StartupAnalystBot/1.0 (+https://example.com)")
TIMEOUT = 20

def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception:
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    parts = []
    # title + meta description
    if soup.title and soup.title.text:
        parts.append(soup.title.text.strip())
    for m in soup.find_all("meta"):
        if m.get("name","").lower() in ("description", "og:description") and m.get("content"):
            parts.append(m["content"].strip())
    # visible text
    for tag in soup.find_all(["h1","h2","h3","p","li","span","strong","em"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = "\n".join(parts)
    # de-dupe lines
    seen = set(); out = []
    for line in text.splitlines():
        L = line.strip()
        if L and L not in seen:
            out.append(L); seen.add(L)
    return "\n".join(out)[:200000]  # cap
