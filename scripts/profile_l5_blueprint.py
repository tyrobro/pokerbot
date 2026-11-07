"""Profiles ONE full CFRConfig::default() training run in isolation,
before Sprint 14 commits to using it at scale. CFRConfig::default()
(10 buckets, 3 raises/street) has never been run -- this answers
whether it's tractable at all, and how long it takes, before the full
30-agent round-robin depends on it.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pokerbot_core

if __name__ == "__main__":
    # Matches CFRConfig::default() exactly: 10 buckets, 3 raises/street
    # (the PyCFRSolver constructor's own defaults).
    train_iterations_to_try = [30, 100, 300, 1000]

    for iterations in train_iterations_to_try:
        start = time.perf_counter()
        solver = pokerbot_core.PyCFRSolver(
            num_buckets=10,
            bucketing_trials=50,
            max_raises_per_street=2,
            seed=1,
            train_iterations=iterations,
        )
        elapsed = time.perf_counter() - start
        print(f"train_iterations={iterations}: {elapsed:.2f}s wall time")