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
    bust_hand_numbers = []
    completed = 0
    for r in results:
        for leg in r["leg_results"]:
            if leg["hands_played"] < config["max_hands"] if False else leg["hands_played"] < 200:
                bust_hand_numbers.append(leg["hands_played"])
            else:
                completed += 1
    print(f"Legs ending in bust before hand cap: {len(bust_hand_numbers)}/{len(results) * 2}")
    print(f"Legs reaching the hand cap unbusted: {completed}/{len(results) * 2}")
    if bust_hand_numbers:
        avg_bust_hand = sum(bust_hand_numbers) / len(bust_hand_numbers)
        pct_of_cap = [h / 200 * 100 for h in bust_hand_numbers]
        print(f"Average bust hand: {avg_bust_hand:.1f} / 200 ({avg_bust_hand/200*100:.0f}% of cap)")
        print(f"Bust hand distribution (% of cap): min={min(pct_of_cap):.0f}%, max={max(pct_of_cap):.0f}%")
        early_busts = sum(1 for p in pct_of_cap if p < 40)
        print(f"Busts before 40% of cap (concerning per Section 7's ~60-70% target): {early_busts}/{len(bust_hand_numbers)}")    
        print("\n--- Suspicious-tie diagnostic (pairing_score exactly 0.0 for both sides) ---")
    
    for r in results:
        if r["pairing_score_a"] == 0.0 and r["pairing_score_b"] == 0.0:
            for leg_num, leg in enumerate(r["leg_results"], start=1):
                print(
                    f"  {r['agent_a_id']} vs {r['agent_b_id']}, leg {leg_num}: "
                    f"hands_played={leg['hands_played']}, "
                    f"final_stack_a={leg['final_stack_a']}, final_stack_b={leg['final_stack_b']}"
                )
    for r in results:
        print(f"{r['agent_a_id']:20s} vs {r['agent_b_id']:20s} -> {r['pairing_score_a']:+.2f} / {r['pairing_score_b']:+.2f}")