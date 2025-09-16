# tools/json_sanitize.py
import re

JSON_START_RE = re.compile(r'[\{\[]')

def sanitize_currency_percent(raw: str) -> str:
    """Make LLM JSON-ish output parseable: remove preface text, $ and % in numeric positions."""
    if not isinstance(raw, str):
        return raw

    # 1) strip anything before the first { or [
    m = JSON_START_RE.search(raw)
    if m:
        raw = raw[m.start():]

    # 2) remove $ preceding numbers (both quoted and unquoted values)
    #   : $100   -> : 100
    raw = re.sub(r':\s*\$([0-9][0-9,\.]*)', r': \1', raw)
    #   "$100"   -> "100"
    raw = re.sub(r'"\$([0-9][0-9,\.]*)"', r'"\1"', raw)

    # 3) remove % after numbers (both quoted and unquoted values)
    #   : 20%    -> : 20
    raw = re.sub(r':\s*([0-9][0-9,\.]*)\s*%', r': \1', raw)
    #   "20%"    -> "20"
    raw = re.sub(r'"([0-9][0-9,\.]*)%"', r'"\1"', raw)

    return raw
