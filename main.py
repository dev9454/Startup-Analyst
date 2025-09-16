# main.py
import argparse
from orchestration.orchestrator import Orchestrator

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", required=True)
    ap.add_argument("--sector", default="saas")
    ap.add_argument("--inputs", nargs="*", required=True)
    args = ap.parse_args()

    orch = Orchestrator(sector=args.sector)
    out_path, note = orch.run(args.company, args.inputs)
    print("Saved:", out_path)
    print("Score:", note["score"]["total"], "- Peers:", [p.get("name") for p in note["benchmarks"]["peers"]])
