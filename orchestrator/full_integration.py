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
    bug this sprint fixed)."""
    a_id, b_id, config, logs_dir, base_seed, pairing_idx, build_agent_fn = args
    agent_a = build_agent_fn(a_id, seed=base_seed + pairing_idx * 2)
    agent_b = build_agent_fn(b_id, seed=base_seed + pairing_idx * 2 + 1)
    result = play_pairing(agent_a, agent_b, a_id, b_id, config, logs_dir)
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
) -> dict:
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
        "seed": base_seed,
    }

    pending = []
    for idx, (a_id, b_id) in enumerate(pairings):
        pid = _pairing_id(a_id, b_id)
        if pid not in completed:
            pending.append((a_id, b_id, config, logs_dir, base_seed, idx, fn))

    print(f"Total pairings: {len(pairings)}. Already completed: {len(completed)}. Pending: {len(pending)}.")
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

    with multiprocessing.Pool(processes=processes) as pool:
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

    return {"total_pairings": len(pairings), "completed_this_run": completed_this_run, "results": results}