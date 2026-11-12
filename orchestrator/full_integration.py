"""Sprint 14: full 30-agent integration test -- all 435 pairings x 2
legs (870 match legs), fixed seed, production config (max_hands=1000,
CFRConfig::default() at max_raises_per_street=2 per Sprint 14
calibration). Per-pairing parallel workers (process pool), resumable
manifest -- same validated pattern as Sprint 11's corpus generation.

Testable without the full 30-agent/435-pairing scale: pass a smaller
`agent_ids` list and/or a different `build_agent_fn` (must be a
top-level, picklable function -- not a lambda, same Windows
multiprocessing constraint as Sprint 11).
"""

import itertools
import multiprocessing
from pathlib import Path

from agents.roster import all_agent_ids, build_agent
from orchestrator.round_robin import play_pairing

# Real concurrency bug found during Sprint 15's actual run: multiple
# worker processes calling append_pairing_summary (read-entire-file,
# modify, overwrite-entire-file) with NO locking corrupted the shared
# pairings_summary.parquet when two workers' writes raced -- the file
# became unreadable (Parquet thrift deserialize error) partway through
# a run that had already logged hundreds of hands correctly. Fixed with
# a cross-process Lock, shared via the pool's initializer (a plain
# threading.Lock would NOT work here -- these are separate OS processes
# on Windows' spawn model, not threads).
_summary_write_lock = None


def _init_worker(lock):
    global _summary_write_lock
    _summary_write_lock = lock


def generate_all_pairings(agent_ids: list = None) -> list:
    ids = agent_ids if agent_ids is not None else all_agent_ids()
    return list(itertools.combinations(ids, 2))


def _load_completed(manifest_path: Path) -> set:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _append_completed(manifest_path: Path, pairing_id: str) -> None:
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(pairing_id + "\n")
        f.flush()


def _pairing_id(a_id: str, b_id: str) -> str:
    return f"{a_id}__vs__{b_id}"


def run_one_pairing(args: tuple) -> dict:
    """Worker function (top-level, picklable). Builds fresh agents for
    THIS pairing (Section 4.4 anonymity); each worker process's
    get_shared_l5_blueprint() cache trains the L5 blueprint at most once
    per process, not once per pairing (the original per-pairing-instance
    bug this sprint fixed).

    base_seed=None (Sprint 15's fully-random path): each agent gets its
    OWN fresh os-entropy seed via random.SystemRandom, independently --
    not derived from a shared base, so no two agents' internal RNG
    streams are correlated even indirectly."""
    a_id, b_id, config, logs_dir, base_seed, pairing_idx, build_agent_fn = args

    if base_seed is None:
        import random

        sys_random = random.SystemRandom()
        seed_a = sys_random.getrandbits(63)
        seed_b = sys_random.getrandbits(63)
    else:
        seed_a = base_seed + pairing_idx * 2
        seed_b = base_seed + pairing_idx * 2 + 1

    agent_a = build_agent_fn(a_id, seed=seed_a)
    agent_b = build_agent_fn(b_id, seed=seed_b)
    # The pairings_summary.parquet write inside play_pairing is a shared-
    # file read-modify-write across all worker processes -- MUST be
    # serialized with the cross-process lock, or concurrent writes can
    # corrupt the file (the real bug this fixes; see module-level note).
    result = play_pairing(
        agent_a, agent_b, a_id, b_id, config, logs_dir, summary_lock=_summary_write_lock
    )
    return {
        "pairing_id": _pairing_id(a_id, b_id),
        "agent_a_id": a_id,
        "agent_b_id": b_id,
        "pairing_score_a": result["pairing_score_a"],
        "pairing_score_b": result["pairing_score_b"],
        "hands_leg1": result["leg_results"][0]["hands_played"],
        "hands_leg2": result["leg_results"][1]["hands_played"],
    }


def run_full_integration_test(
    logs_dir: str = "logs/sprint14",
    max_hands: int = 1000,
    base_seed: int = 1,
    processes: int = None,
    agent_ids: list = None,
    build_agent_fn=None,
    randomize_seeds: bool = False,
) -> dict:
    """randomize_seeds=True (Sprint 15's final run): EVERY RNG --
    match-level (deck/button, via MatchConfig.seed) AND each agent's own
    internal Monte Carlo seed (L3/L4/L5's draws) -- is seeded from fresh
    OS entropy per pairing, not deterministically derived from
    `base_seed`/pairing index as Sprint 14's fixed-seed validation run
    did. `base_seed` is ignored entirely when this is True; it exists
    only for the fixed-seed (Sprint 14-style) path.
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(logs_dir) / "manifest.txt"

    ids = agent_ids if agent_ids is not None else all_agent_ids()
    fn = build_agent_fn if build_agent_fn is not None else build_agent
    pairings = generate_all_pairings(ids)
    completed = _load_completed(manifest_path)

    config = {
        "starting_stack": 100_000,
        "initial_small_blind": 50,
        "initial_big_blind": 100,
        "blind_escalation_interval_hands": 20,
        "max_hands": max_hands,
        # Sprint 15: `seed=None` makes MatchConfig draw a fresh random
        # deck/button every match (core_engine::match_engine already
        # supports this -- see `seed: Option<u64>`, `None` branch).
        "seed": None if randomize_seeds else base_seed,
    }

    pending = []
    for idx, (a_id, b_id) in enumerate(pairings):
        pid = _pairing_id(a_id, b_id)
        if pid not in completed:
            # Sprint 15: pass seed=None through to the worker so IT draws
            # fresh entropy for each agent's own seed at build time,
            # rather than this process pre-computing a seed value now
            # (which would itself be a single random draw shared, in
            # effect, across however many agent-seed slots derive from
            # it -- see run_one_pairing's handling of agent_seed=None).
            agent_seed_base = None if randomize_seeds else base_seed
            pending.append((a_id, b_id, config, logs_dir, agent_seed_base, idx, fn))

    print(
        f"Total pairings: {len(pairings)}. Already completed: {len(completed)}. Pending: {len(pending)}."
    )
    print(
        "NOTE: each worker process trains its own shared L5 blueprint ONCE on first "
        f"use (not once per pairing) -- with --processes N, expect N blueprint builds "
        f"total, not {len(pending)}.",
        flush=True,
    )

    if not pending:
        return {"total_pairings": len(pairings), "completed_this_run": 0, "results": []}

    completed_this_run = 0
    results = []

    print(f"Starting pool with {processes or 'default'} processes...", flush=True)

    manager = multiprocessing.Manager()
    summary_lock = manager.Lock()

    with multiprocessing.Pool(
        processes=processes, initializer=_init_worker, initargs=(summary_lock,)
    ) as pool:
        for result in pool.imap_unordered(run_one_pairing, pending, chunksize=1):
            _append_completed(manifest_path, result["pairing_id"])
            completed_this_run += 1
            results.append(result)
            print(
                f"  [{completed_this_run}/{len(pending)}] {result['pairing_id']} done "
                f"(score: {result['pairing_score_a']:+.2f}/{result['pairing_score_b']:+.2f}, "
                f"hands: {result['hands_leg1']}+{result['hands_leg2']})...",
                flush=True,
            )

    return {
        "total_pairings": len(pairings),
        "completed_this_run": completed_this_run,
        "results": results,
    }
