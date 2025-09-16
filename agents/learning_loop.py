# agents/learning_loop.py
import json, os
from agents.base import BaseAgent

STATE_PATH = "outputs/learning_state.json"

class LearningLoopAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="learning_loop")
        self.state = self._load()

    def _load(self):
        if os.path.exists(STATE_PATH):
            return json.load(open(STATE_PATH,"r",encoding="utf-8"))
        return {"weights":{"founders":0.30,"traction":0.25,"unit_econ":0.25,"market":0.20}}

    def feedback(self, thumbs_up: bool, reason: str = ""):
        # For hackathon: record feedback; weights can be re-optimized by LLM later
        self.state.setdefault("feedback", []).append({"ok":thumbs_up, "reason":reason})
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        json.dump(self.state, open(STATE_PATH,"w",encoding="utf-8"), indent=2)
        self.log("feedback", {"ok": thumbs_up})
        return self.state
