"""Sprint 15: fully-random-seed path + the enforced pre-flight check.
Kept fast via a tiny roster -- the real 870-leg final run is the manual
script above.
"""

import pytest

from agents.roster import build_agent
from orchestrator.full_integration import run_full_integration_test

def _tiny_agent_ids():
    return ["grid_m1_k1", "grid_m2_k2", "pd_always_defect"]


def test_randomized_run_completes_and_conserves_chips(tmp_path):
    summary = run_full_integration_test(
        logs_dir=str(tmp_path),
        max_hands=5,
        processes=1,
        agent_ids=_tiny_agent_ids(),
        build_agent_fn=build_agent,
        randomize_seeds=True,
    )
    assert summary["completed_this_run"] == 3
    for r in summary["results"]:
        assert r["hands_leg1"] > 0
        assert r["hands_leg2"] > 0


def test_randomized_run_produces_different_results_across_runs(tmp_path_factory):
    # Two separate randomized runs of the SAME tiny roster should (with
    # overwhelming probability) NOT produce byte-identical results --
    # confirms randomize_seeds=True actually varies outcomes, unlike
    # Sprint 14's fixed-seed path which is deliberately reproducible.
    dir1 = tmp_path_factory.mktemp("run1")
    dir2 = tmp_path_factory.mktemp("run2")

    summary_1 = run_full_integration_test(
        logs_dir=str(dir1), max_hands=20, processes=1,
        agent_ids=_tiny_agent_ids(), build_agent_fn=build_agent, randomize_seeds=True,
    )
    summary_2 = run_full_integration_test(
        logs_dir=str(dir2), max_hands=20, processes=1,
        agent_ids=_tiny_agent_ids(), build_agent_fn=build_agent, randomize_seeds=True,
    )

    scores_1 = sorted((r["pairing_score_a"], r["pairing_score_b"]) for r in summary_1["results"])
    scores_2 = sorted((r["pairing_score_a"], r["pairing_score_b"]) for r in summary_2["results"])
    assert scores_1 != scores_2, "two randomized runs produced identical scores -- seeding may not be random"


def test_preflight_check_rejects_non_randomized_run():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_sprint15_final import preflight_check_no_fixed_seed_artifacts

    with pytest.raises(RuntimeError, match="randomize_seeds is False"):
        preflight_check_no_fixed_seed_artifacts(randomize_seeds=False, base_seed_arg=None)


def test_preflight_check_rejects_leftover_base_seed():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_sprint15_final import preflight_check_no_fixed_seed_artifacts

    with pytest.raises(RuntimeError, match="base_seed"):
        preflight_check_no_fixed_seed_artifacts(randomize_seeds=True, base_seed_arg=42)


def test_preflight_check_passes_for_correct_invocation(capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_sprint15_final import preflight_check_no_fixed_seed_artifacts

    preflight_check_no_fixed_seed_artifacts(randomize_seeds=True, base_seed_arg=None)
    assert "PASSED" in capsys.readouterr().out