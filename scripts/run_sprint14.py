"""Run manually: python scripts/run_sprint14.py --confirm-release-build
REQUIRES the release build (maturin develop --release) -- 870 legs of
L3-L5 Monte Carlo cost is intolerable in debug mode.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.full_integration import run_full_integration_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-release-build", action="store_true", required=True)
    parser.add_argument("--output-dir", default="logs/sprint14")
    parser.add_argument("--max-hands", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=1)
    args = parser.parse_args()

    start = time.perf_counter()
    summary = run_full_integration_test(
        logs_dir=args.output_dir,
        max_hands=args.max_hands,
        base_seed=args.base_seed,
        processes=args.processes,
    )
    elapsed = time.perf_counter() - start

    print(f"\nDone. Completed {summary['completed_this_run']} pairings this run.")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")