# tools/jsonio.py
from __future__ import annotations
import json, re

try:
    import json5
except Exception:
    json5 = None

_TRAILING_COMMAS = re.compile(r",(\s*[}\]])")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Keys we expect to be arrays across your schemas
_ARRAY_KEYS = {
    "evidence","reviews_sample","social_snippets","alerts",
    "pros","cons",
    "requirements","matched","missing","citations",
    "kpis","baselines","snippets",
    "founders","traction","unit_economics","market","product","legal",
    "claims","risks","table","positioning_bullets","data_quality_notes",
    "peers","insights","dilution","clauses","plain","assumptions","sensitivities","sources","gaps","timeline"
}

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = "\n".join(s.splitlines()[1:])
    return s.strip()

def _ensure_arrays(s: str) -> str:
    """
    If a known array key is followed immediately by a closing brace/bracket,
    e.g., "evidence":}  ->  "evidence":[ ]}
    """
    for k in _ARRAY_KEYS:
        # "k" : }   or   "k":]   or "k":,} (rare) etc.
        s = re.sub(rf'"{re.escape(k)}"\s*:\s*(?=[}}\]])', f'"{k}":[]', s)
    return s

def _extract_balanced_json(s: str) -> str | None:
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i; break
    if start is None:
        return None
    stack = []
    for j in range(start, len(s)):
        ch = s[j]
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                continue
            stack.pop()
            if not stack:
                return s[start:j+1]
    closes = "".join(reversed(stack))
    return s[start:] + closes if stack else None

def _basic_sanitizers(s: str) -> str:
    s = _CONTROL_CHARS.sub("", s)
    s = _TRAILING_COMMAS.sub(r"\1", s)
    s = _ensure_arrays(s)
    return s.strip()

def parse_json_or_repair(raw: str):
    """
    Robust JSON parse with repairs:
      - try json / json5
      - strip code fences
      - remove control chars, trailing commas
      - auto-insert [] for known array keys when missing
      - balance braces/brackets if truncated
    """
    if raw is None:
        raise ValueError("Empty LLM response")

    # fast path
    try:
        return json.loads(raw)
    except Exception:
        pass
    if json5:
        try:
            return json5.loads(raw)
        except Exception:
            pass

    s = _strip_code_fences(raw)
    s = _basic_sanitizers(s)
    cand = _extract_balanced_json(s) or s
    cand = _basic_sanitizers(cand)

    try:
        return json.loads(cand)
    except Exception:
        if json5:
            try:
                return json5.loads(cand)
            except Exception:
                pass

    preview = cand[:800]
    raise ValueError(f"LLM did not return valid JSON even after repair.\nRAW START:\n{preview}")
