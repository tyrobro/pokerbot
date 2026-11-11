"""Sprint 15: the FINAL tournament run. Fully random seeds (no fixed
seeding) -- Sprint 14 was the fixed-seed validation pass; this is the
dataset any final writeup/report references.

Run manually: python scripts/run_sprint15_final.py --confirm-release-build
REQUIRES the release build (maturin develop --release).

Pre-flight check (enforced, not just a manual glance, per explicit
request): refuses to run if any fixed-seed artifact is detected in the
config this script is about to use.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.full_integration import run_full_integration_test


def preflight_check_no_fixed_seed_artifacts(randomize_seeds: bool, base_seed_arg) -> None:
    """Fails loudly (not silently) if this run would use ANY fixed/
    deterministic seeding -- enforced, per explicit request, not a
    checklist item left to a human glance."""
    if not randomize_seeds:
        raise RuntimeError(
            "PRE-FLIGHT CHECK FAILED: randomize_seeds is False. Sprint 15's final run "
            "MUST use fully random seeds (Section 9's explicit DoD requirement: "
            "'explicitly re-confirm no fixed-seed artifacts remain in the configuration "
            "before launching'). Pass --randomize-seeds (it is NOT optional for this script)."
        )
    if base_seed_arg is not None:
        raise RuntimeError(
            "PRE-FLIGHT CHECK FAILED: a --base-seed value was supplied alongside "
            "--randomize-seeds. base_seed is ignored when randomize_seeds=True, but "
            "its presence here suggests a leftover fixed-seed artifact in how this "
            "script was invoked -- remove --base-seed entirely for the final run."
        )
    print("Pre-flight check PASSED: no fixed-seed artifacts detected. Proceeding with fully random seeds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-release-build", action="store_true", required=True)
    parser.add_argument("--randomize-seeds", action="store_true", required=True)
    parser.add_argument("--output-dir", default="logs/final")
    parser.add_argument("--max-hands", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=None, help="Number of worker processes (default: all cores)")
    # Deliberately NOT exposing --base-seed as a usable option here --
    # only present so the pre-flight check can catch someone trying to
    # smuggle one in; always None for this script's real invocations.
    parser.add_argument("--base-seed", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    preflight_check_no_fixed_seed_artifacts(args.randomize_seeds, args.base_seed)

    start = time.perf_counter()
    summary = run_full_integration_test(
        logs_dir=args.output_dir,
        max_hands=args.max_hands,
        processes=args.processes,
        randomize_seeds=True,
    )
    elapsed = time.perf_counter() - start

    print(f"\nDone. Completed {summary['completed_this_run']} pairings this run.")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")