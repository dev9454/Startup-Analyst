# tools/founder_miner.py
import re

NAME_RE = r"[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}"
ROLE_HINTS = r"(Founder|Co[- ]?Founder|CEO|CTO|CPO|COO|Chief|Co\s*founder)"

def mine_founders(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out = []
    # section hints
    sections = []
    buf = []
    for ln in lines:
        if re.search(r"\b(Founders?|Team)\b", ln, re.I):
            if buf:
                sections.append("\n".join(buf)); buf=[]
            buf=[ln]
        else:
            if buf:
                buf.append(ln)
    if buf:
        sections.append("\n".join(buf))
    # if no section captured, fallback to whole text
    if not sections: sections=[text]

    seen = set()
    for sec in sections:
        # pattern: Name - Role ; Name, Role
        for m in re.finditer(rf"({NAME_RE})[,–-]\s*({ROLE_HINTS}.*)", sec):
            name = m.group(1).strip()
            role = m.group(2).strip()
            key = (name.lower(), role.lower())
            if key not in seen:
                out.append({"name": name, "role": role}); seen.add(key)
        # pattern: Role: Name
        for m in re.finditer(rf"({ROLE_HINTS})[:\s]+({NAME_RE})", sec):
            role = m.group(1).strip()
            name = m.group(2).strip()
            key = (name.lower(), role.lower())
            if key not in seen:
                out.append({"name": name, "role": role}); seen.add(key)
        # bare names line under founders header
        for m in re.finditer(rf"\b({NAME_RE})\b", sec):
            name = m.group(1).strip()
            if len(name.split())>=2 and name.lower() not in (n['name'].lower() for n in out):
                # only keep if near role hints in same section
                if re.search(ROLE_HINTS, sec, re.I):
                    out.append({"name": name, "role": ""})
    return out[:10]
