# agents/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import time, json, pathlib

LOG_DIR = pathlib.Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class BaseAgent:
    name: str
    logs: List[dict] = field(default_factory=list)

    def log(self, event: str, payload: Dict[str, Any]):
        self.logs.append({
            "ts": time.time(),
            "agent": self.name,
            "event": event,
            **(payload or {})
        })

    def flush(self, run_id: str):
        fp = LOG_DIR / f"{run_id}_{self.name}.json"
        fp.write_text(json.dumps(self.logs, indent=2), encoding="utf-8")
