"""Sprint 11 DoD: corpus generated/validated (no malformed hands,
reasonable label distribution), train/val split by match implemented.
Kept small-scale for pytest speed -- real-volume generation is the
manual script above, not part of the test suite."""

from orchestrator.corpus_generation import build_work_units, generate_one_match, run_corpus_generation
from orchestrator.corpus_split import split_corpus_by_match


def test_build_work_units_includes_mirror_matchups():
    units = build_work_units(matches_per_pairing=2, base_seed=1, max_hands=5, starting_stack=100_000)
    mirror_units = [u for u in units if u["agent_a"] == u["agent_b"]]
    assert len(mirror_units) > 0  # l1vl1, l2vl2, ... included


def test_build_work_units_count_matches_combinations_with_replacement():
    # C(5,2) + 5 mirrors = 10 + 5 = 15 pairings, x matches_per_pairing
    units = build_work_units(matches_per_pairing=3, base_seed=1, max_hands=5, starting_stack=100_000)
    assert len(units) == 15 * 3


def test_generate_one_match_writes_a_well_formed_shard(tmp_path):
    units = build_work_units(matches_per_pairing=1, base_seed=1, max_hands=5, starting_stack=100_000)
    unit = next(u for u in units if u["agent_a"] == "l1" and u["agent_b"] == "l2")

    result = generate_one_match(unit, str(tmp_path))
    assert result["hands"] > 0

    import pandas as pd

    df = pd.read_parquet(result["shard_path"])
    assert len(df) == result["hands"]
    assert (df["match_id"] == unit["unit_id"]).all()
    assert not df["hole_cards_agent_a"].isna().any()


def test_corpus_generation_is_resumable(tmp_path):
    output_dir = str(tmp_path)
    # First run: generate a small corpus.
    summary_1 = run_corpus_generation(
        output_dir, matches_per_pairing=1, max_hands=5, processes=1, base_seed=1
    )
    assert summary_1["completed_this_run"] == 15  # all 15 pairings x 1 match

    # Second run, SAME params: everything should already be in the
    # manifest -- nothing new should be generated.
    summary_2 = run_corpus_generation(
        output_dir, matches_per_pairing=1, max_hands=5, processes=1, base_seed=1
    )
    assert summary_2["completed_this_run"] == 0


def test_split_by_match_covers_every_completed_unit_exactly_once(tmp_path):
    output_dir = str(tmp_path)
    run_corpus_generation(output_dir, matches_per_pairing=2, max_hands=5, processes=1, base_seed=1)

    split = split_corpus_by_match(output_dir, val_fraction=0.2, seed=7)
    assert split["train_matches"] + split["val_matches"] == 15 * 2
    assert split["val_matches"] > 0  # non-trivial split at this size
    
def test_per_pairing_overrides_only_affects_specified_pairings():
    overrides = {("l3", "l3"): 5}
    units = build_work_units(matches_per_pairing=2, base_seed=1, max_hands=5, starting_stack=100_000, per_pairing_overrides=overrides)

    l3_l3_units = [u for u in units if u["agent_a"] == "l3" and u["agent_b"] == "l3"]
    other_units = [u for u in units if not (u["agent_a"] == "l3" and u["agent_b"] == "l3")]

    assert len(l3_l3_units) == 5  # overridden
    assert all(len([u for u in other_units if u["agent_a"] == a and u["agent_b"] == b]) == 2
               for a, b in {(u["agent_a"], u["agent_b"]) for u in other_units})