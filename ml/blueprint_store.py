"""Sprint 14: the L5 CFR blueprint is trained ONCE (expensive, offline)
and shared READ-ONLY across every GridAgent(5,*) and AlwaysCooperateAgent
instance -- Section 4.4's "fresh per pairing" requirement is about
opponent-MEMORY, not about the trained model itself (PRD-author-
confirmed). Retraining a fresh CFRSolver per pairing was a real
misimplementation caught before Sprint 14's full run depended on it.

Config locked in from Sprint 14 calibration (see
core_engine::math_engine::level5::abstraction::CFRConfig::default):
max_raises_per_street=2 (3 OOMs at 1000 iterations), train_iterations=500
(between the 300-iter/~3.3min and 1000-iter/~26min profiled data points).
"""

import pokerbot_core

_shared_solver = None


def get_shared_l5_blueprint(
    num_buckets: int = 10,
    bucketing_trials: int = 50,
    max_raises_per_street: int = 2,
    seed: int = 1,
    train_iterations: int = 500,
):
    """Builds the blueprint on first call, returns the SAME instance on
    every subsequent call -- module-level cache, not per-process-pool-
    worker-safe on its own (see corpus_generation-style workers: each
    worker process gets its own fresh Python interpreter and therefore
    its own cache miss on first use within that process, which is
    correct -- the blueprint trains once PER WORKER PROCESS, not once
    per pairing, which is still a massive reduction from the original
    per-pairing-instance bug)."""
    global _shared_solver
    if _shared_solver is None:
        _shared_solver = pokerbot_core.PyCFRSolver(
            num_buckets, bucketing_trials, max_raises_per_street, seed, train_iterations
        )
    return _shared_solver