"""Run manually: python scripts/generate_corpus.py --confirm-release-build
REQUIRES the Rust extension to be built in release mode first:
    cd bindings && maturin develop --release && cd ..
This script refuses to run without --confirm-release-build, since it
cannot verify the build mode itself and debug-build Monte Carlo (L3/L4/L5)
is intolerably slow at this volume.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.corpus_generation import run_corpus_generation
from orchestrator.corpus_split import split_corpus_by_match

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-release-build", action="store_true", required=True)
    parser.add_argument("--output-dir", default="logs/corpus")
    parser.add_argument("--matches-per-pairing", type=int, default=50)
    parser.add_argument("--max-hands", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()

    summary = run_corpus_generation(
        output_dir=args.output_dir,
        matches_per_pairing=args.matches_per_pairing,
        max_hands=args.max_hands,
        processes=args.processes,
    )
    print(summary)

    split_summary = split_corpus_by_match(args.output_dir)
    print(split_summary)