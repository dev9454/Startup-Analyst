# tools/jsonio.py
import json, json5, re

JSON_BLOCK_RE = re.compile(r'(\{.*\}|\[.*\])', flags=re.DOTALL)

def parse_json_or_repair(raw: str):
    """Salvage a valid JSON object/array from raw LLM output."""
    if not isinstance(raw, str):
        raise ValueError("LLM output is not a string")

    raw = raw.strip()

    # 1) Try strict
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Try JSON5
    try:
        return json5.loads(raw)
    except Exception:
        pass

    # 3) Extract the first {...} or [...] block
    matches = JSON_BLOCK_RE.findall(raw)
    for block in matches:
        try:
            return json.loads(block)
        except Exception:
            try:
                return json5.loads(block)
            except Exception:
                continue

    # 4) Remove trailing commas and newlines, try again
    fixed = raw.replace("\r", " ").replace("\n", " ")
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    try:
        return json.loads(fixed)
    except Exception:
        try:
            return json5.loads(fixed)
        except Exception:
            # Final helpful message
            preview = raw[:800]
            raise ValueError(f"LLM did not return valid JSON even after repair.\nRAW START:\n{preview}")
