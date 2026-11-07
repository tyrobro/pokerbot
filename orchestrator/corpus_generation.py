"""Sprint 11, Stage 0 (Section 5): self-play corpus generation from the
5 math-engine-only agents, at volume, via a multiprocessing worker pool.

Hard requirements (confirmed, not optional):
  - Rust MUST be built in release mode (`maturin develop --release`)
    before running this. Debug-build Monte Carlo (L3/L4/L5) is often an
    order of magnitude slower -- intolerable at hundreds-of-thousands-
    of-hands volume. This module cannot detect debug vs. release from
    Python, so `scripts/generate_corpus.py` requires an explicit
    `--confirm-release-build` flag rather than silently assuming it.
  - Multiprocessing: every match is fully independent (embarrassingly
    parallel), so work is distributed across a process pool, one
    complete match per work unit, each worker writing its own shard.
  - Resumable: a manifest (one completed work-unit id per line, flushed
    immediately after each shard write succeeds) is read at startup;
    already-completed units are skipped. A crash/interrupt loses at
    most the in-flight units, not prior progress.
"""

import itertools
import json
import multiprocessing
from pathlib import Path

import pokerbot_core
from agents.math_agents import MathLevel1Agent, MathLevel2Agent, MathLevel3Agent, MathLevel4Agent, MathLevel5Agent
from orchestrator.logging import build_hand_rows

MATH_LEVELS = ["l1", "l2", "l3", "l4", "l5"]

_AGENT_FACTORIES = {
    "l1": lambda seed: MathLevel1Agent(),
    "l2": lambda seed: MathLevel2Agent(),
    "l3": lambda seed: MathLevel3Agent(seed=seed),
    "l4": lambda seed: MathLevel4Agent(seed=seed),
    "l5": lambda seed: MathLevel5Agent(seed=seed, train_iterations=30),
}


def build_work_units(
    matches_per_pairing: int,
    base_seed: int,
    max_hands: int,
    starting_stack: int,
    per_pairing_overrides: dict = None,
) -> list:
    """One unit per (pairing, match index). Pairings include mirror
    matchups (e.g. l3 vs l3) -- a level playing itself is still useful
    self-play signal, not excluded by Section 5's wording.

    `per_pairing_overrides`: optional dict mapping (level_a, level_b) ->
    match count, overriding `matches_per_pairing` for just that pairing.
    Used for the Sprint 11 top-up (L3/L4/L5-involving pairings produced
    far fewer HANDS than L1/L2-involving ones at equal MATCH counts,
    since stronger/more-aggressive levels bust each other much faster --
    this lets us request more matches specifically for the thin
    pairings without regenerating the already-adequate ones. unit_ids
    here are identical in scheme to a plain uniform run, so increasing a
    pairing's count is purely additive against the existing manifest --
    already-completed lower-index units for that pairing are unaffected
    and skipped as usual by run_corpus_generation's resumability check."""
    units = []
    overrides = per_pairing_overrides or {}
    pairings = list(itertools.combinations_with_replacement(MATH_LEVELS, 2))
    for pairing_idx, (a, b) in enumerate(pairings):
        count = overrides.get((a, b), matches_per_pairing)
        for match_idx in range(count):
            unit_id = f"{a}_vs_{b}__m{match_idx}"
            units.append(
                {
                    "unit_id": unit_id,
                    "agent_a": a,
                    "agent_b": b,
                    "seed": base_seed + pairing_idx * 10_000 + match_idx,
                    "max_hands": max_hands,
                    "starting_stack": starting_stack,
                }
            )
    return units

def _append_completed(manifest_path: Path, unit_id: str) -> None:
    # Append-only, flushed immediately -- this line being written IS the
    # durable record that the shard write below it succeeded.
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(unit_id + "\n")
        f.flush()


def generate_one_match(unit: dict, shards_dir: str) -> dict:
    """Worker function (top-level, picklable): plays one full match,
    writes its hand rows as one shard. Returns a small result dict; the
    CALLER (single process, imap_unordered's consumer loop) appends to
    the manifest -- writing the manifest from multiple workers
    concurrently would risk interleaved/corrupted lines."""
    agent_a = _AGENT_FACTORIES[unit["agent_a"]](unit["seed"])
    agent_b = _AGENT_FACTORIES[unit["agent_b"]](unit["seed"] + 1)

    result = pokerbot_core.play_match_py(
        agent_a,
        agent_b,
        starting_stack=unit["starting_stack"],
        seed=unit["seed"],
        max_hands=unit["max_hands"],
    )

    rows = build_hand_rows(result, unit["starting_stack"])
    for row in rows:
        row["match_id"] = unit["unit_id"]
        row["agent_a_level"] = unit["agent_a"]
        row["agent_b_level"] = unit["agent_b"]

    shard_path = Path(shards_dir) / f"{unit['unit_id']}.parquet"
    import pandas as pd

    pd.DataFrame(rows).to_parquet(shard_path, index=False)

    return {"unit_id": unit["unit_id"], "hands": len(rows), "shard_path": str(shard_path)}

def _generate_one_match_star(args: tuple) -> dict:
    unit, shards_dir = args
    return generate_one_match(unit, shards_dir)

def _load_completed(manifest_path: Path) -> set:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def run_corpus_generation(
    output_dir: str,
    matches_per_pairing: int = 50,
    max_hands: int = 1000,
    starting_stack: int = 100_000,
    base_seed: int = 1,
    processes: int = None,
    per_pairing_overrides: dict = None,
) -> dict:
    """Builds work units, skips already-completed ones (manifest-based
    resumability), distributes the rest across a process pool. Returns a
    summary dict. Call AFTER confirming the release build (see module
    doc) -- this function itself does not verify that."""
    shards_dir = Path(output_dir) / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(output_dir) / "manifest.txt"

    all_units = build_work_units(matches_per_pairing, base_seed, max_hands, starting_stack, per_pairing_overrides)
    completed = _load_completed(manifest_path)
    pending = [u for u in all_units if u["unit_id"] not in completed]

    print(f"Total work units: {len(all_units)}. Already completed: {len(completed)}. Pending: {len(pending)}.")

    if not pending:
        return {"total_units": len(all_units), "completed_this_run": 0, "total_hands": "see manifest"}

    worker_args = [(unit, str(shards_dir)) for unit in pending]
    completed_this_run = 0
    total_hands_this_run = 0

    print(f"Starting pool with {processes or 'default'} processes...", flush=True)

    with multiprocessing.Pool(processes=processes) as pool:
        for result in pool.imap_unordered(_generate_one_match_star, worker_args, chunksize=1):
            _append_completed(manifest_path, result["unit_id"])
            completed_this_run += 1
            total_hands_this_run += result["hands"]
            print(
                f"  [{completed_this_run}/{len(pending)}] {result['unit_id']} done "
                f"({result['hands']} hands, {total_hands_this_run} total this run)...",
                flush=True,
            )

    return {
        "total_units": len(all_units),
        "completed_this_run": completed_this_run,
        "total_hands_this_run": total_hands_this_run,
    }