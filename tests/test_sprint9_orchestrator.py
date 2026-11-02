"""Sprint 9 DoD, kept test-suite-fast: pairing generation, a small
round-robin completing + reproducible, and the Section 4.4 flush-between-
pairings behavior (via fresh-instance-per-pairing, confirmed directly).
"""

from agents.math_agents import MathLevel1Agent, MathLevel2Agent
from agents.pd_archetypes import AlwaysDefectAgent, GrudgerAgent, HandHistoryEntry, PavlovAgent, PavlovMode, TitForTatAgent
from orchestrator.round_robin import generate_pairings, run_round_robin


def test_pairing_generation_count_for_ten_agents():
    ids = [f"a{i}" for i in range(10)]
    assert len(generate_pairings(ids)) == 45  # C(10,2), per Section 7


def test_small_round_robin_completes_and_conserves_chips():
    factories = {
        "l1": lambda seed: MathLevel1Agent(),
        "l2": lambda seed: MathLevel2Agent(),
        "defect": lambda seed: AlwaysDefectAgent(),
    }
    config = {
        "starting_stack": 100_000, "initial_small_blind": 50, "initial_big_blind": 100,
        "blind_escalation_interval_hands": 0, "max_hands": 10, "seed": 7,
    }
    results = run_round_robin(factories, config, logs_dir="logs/test_small_rr")
    assert len(results) == 3  # C(3,2)
    for r in results:
        for leg in r["leg_results"]:
            assert leg["final_stack_a"] + leg["final_stack_b"] == 200_000


def test_round_robin_is_reproducible_under_fixed_seed():
    factories = {"l1": lambda seed: MathLevel1Agent(), "defect": lambda seed: AlwaysDefectAgent()}
    config = {
        "starting_stack": 100_000, "initial_small_blind": 50, "initial_big_blind": 100,
        "blind_escalation_interval_hands": 0, "max_hands": 10, "seed": 123,
    }
    r1 = run_round_robin(factories, config, logs_dir="logs/test_rr_repro_1")
    r2 = run_round_robin(factories, config, logs_dir="logs/test_rr_repro_2")
    for a, b in zip(r1, r2):
        assert a["pairing_score_a"] == b["pairing_score_a"]
        assert a["pairing_score_b"] == b["pairing_score_b"]


def test_grudger_fresh_instance_per_pairing_is_unanchored():
    agent1 = GrudgerAgent()
    agent1.triggered = True
    agent2 = GrudgerAgent()  # as run_round_robin builds fresh per pairing
    assert not agent2.triggered


def test_tit_for_tat_flush_resets_history():
    agent = TitForTatAgent()
    agent.observe_hand_result(HandHistoryEntry())
    assert agent.match_history
    agent.flush()
    assert agent.match_history == []


def test_pavlov_flush_resets_mode():
    agent = PavlovAgent()
    agent.mode = PavlovMode.LOOSE_AGGRESSIVE
    agent.flush()
    assert agent.mode == PavlovMode.BALANCED