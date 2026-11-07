"""Sprint 11 top-up: L3/L4/L5-involving pairings produced far fewer
HANDS than L1/L2-involving ones at the same MATCH count (stronger/more-
aggressive agents bust each other faster). This requests MORE matches
specifically for the thin pairings, additively on top of the existing
750-unit corpus -- already-completed units are untouched (resumable, by
unit_id, same as the original run).

Run manually: python scripts/topup_corpus.py --confirm-release-build
REQUIRES the release build, same as scripts/generate_corpus.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.corpus_generation import run_corpus_generation
from orchestrator.corpus_split import split_corpus_by_match

# Thin pairings from the first run (hand counts, not match counts):
#   l3 vs l3:  617 hands  (50 matches)
#   l3 vs l4: 2,436 hands (50 matches)
#   l4 vs l4: 5,804 hands (50 matches)
#   l3 vs l5: 2,521 hands (50 matches)
#   l4 vs l5: 8,000 hands (50 matches)
# Bumping these five pairings' match counts well above the uniform 50 so
# their HAND volume moves closer to parity with the L1/L2-heavy pairings
# (tens of thousands of hands each). Matches/hands don't scale 1:1 here
# since bust timing varies by pairing, so these multipliers are sized
# generously (not minimally) against the original per-pairing hand counts.
TOPUP_OVERRIDES = {
    ("l3", "l3"): 50 + 400,   # was 617 hands -> aim for roughly 10x more matches
    ("l3", "l4"): 50 + 250,
    ("l4", "l4"): 50 + 150,
    ("l3", "l5"): 50 + 250,
    ("l4", "l5"): 50 + 100,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-release-build", action="store_true", required=True)
    parser.add_argument("--output-dir", default="logs/corpus")
    parser.add_argument("--max-hands", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()

    summary = run_corpus_generation(
        output_dir=args.output_dir,
        matches_per_pairing=50,  # unchanged for everything NOT in the override dict
        max_hands=args.max_hands,
        processes=args.processes,
        per_pairing_overrides=TOPUP_OVERRIDES,
    )
    print(summary)

    split_summary = split_corpus_by_match(args.output_dir)
    print(split_summary)