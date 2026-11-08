"""Sprint 14 DoD, kept test-suite-fast via a tiny 3-agent roster (3
pairings, not 435): pairing generation, resumability, and (separately)
the literal 30-agent/435-pairing count as a pure arithmetic check. The
real 870-leg production run is the manual script above.
"""

from agents.roster import all_agent_ids, build_agent
from orchestrator.full_integration import generate_all_pairings, run_full_integration_test


def _tiny_agent_ids():
    return ["grid_m1_k1", "grid_m2_k2", "pd_always_defect"]


def test_full_roster_has_30_agents():
    ids = all_agent_ids()
    assert len(ids) == 30
    assert len(set(ids)) == 30  # no duplicate ids


def test_full_roster_pairing_count_is_435():
    pairings = generate_all_pairings()
    assert len(pairings) == 435  # C(30,2), Section 9's DoD number


def test_small_roster_integration_completes(tmp_path):
    summary = run_full_integration_test(
        logs_dir=str(tmp_path),
        max_hands=5,
        base_seed=1,
        processes=1,
        agent_ids=_tiny_agent_ids(),
        build_agent_fn=build_agent,
    )
    assert summary["completed_this_run"] == 3  # C(3,2)
    for r in summary["results"]:
        assert r["hands_leg1"] > 0
        assert r["hands_leg2"] > 0


def test_small_roster_integration_is_resumable(tmp_path):
    output_dir = str(tmp_path)
    summary_1 = run_full_integration_test(
        logs_dir=output_dir, max_hands=5, base_seed=1, processes=1,
        agent_ids=_tiny_agent_ids(), build_agent_fn=build_agent,
    )
    assert summary_1["completed_this_run"] == 3

    summary_2 = run_full_integration_test(
        logs_dir=output_dir, max_hands=5, base_seed=1, processes=1,
        agent_ids=_tiny_agent_ids(), build_agent_fn=build_agent,
    )
    assert summary_2["completed_this_run"] == 0  # already-completed pairings skipped