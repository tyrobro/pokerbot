"""Run manually: python scripts/run_calibration.py
Not part of the pytest suite -- L3/L4/L5's Monte Carlo cost across 90
legs makes this a multi-minute run."""

import sys
from pathlib import Path

# Running this file directly only puts scripts/ on sys.path, not the
# repo root -- pytest gets pyproject.toml's pythonpath=["."] for free,
# this script needs the equivalent done manually.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.calibration import run_calibration

if __name__ == "__main__":
    print("Starting calibration round-robin: 10 agents, 45 pairings, 90 legs...", flush=True)
    print("(L3/L4/L5 Monte Carlo cost makes this slow -- progress prints per pairing below)", flush=True)
    results = run_calibration()
    print(f"Completed {len(results)} pairings.")
    busts = 0
    for r in results:
        for leg in r["leg_results"]:
            if leg["hands_played"] < 200:
                busts += 1
    print(f"Legs ending in bust before hand cap: {busts}/{len(results) * 2}")
    for r in results:
        print(f"{r['agent_a_id']:20s} vs {r['agent_b_id']:20s} -> {r['pairing_score_a']:+.2f} / {r['pairing_score_b']:+.2f}")