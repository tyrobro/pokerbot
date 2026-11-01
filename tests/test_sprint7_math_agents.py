"""Sprint 7 close-out: each math-engine level, wrapped as an Agent, can
complete a full match via the Sprint-5 orchestrator (play_match_py) --
Section 9's Sprint 7 Definition of Done, read literally.

L1/L2 are deterministic (no RNG anywhere) and tested for reproducibility.
L3/L4/L5 are NOT currently reproducible under a fixed match seed (see
agents/math_agents.py module doc) -- only completion/well-formedness is
asserted for them here.
"""

from agents.math_agents import (
    MathLevel1Agent,
    MathLevel2Agent,
    MathLevel3Agent,
    MathLevel4Agent,
    MathLevel5Agent,
)
from agents.stub_agents import AlwaysCallAgent
from orchestrator.match_runner import run_match


def _assert_well_formed(result: dict, max_hands: int) -> None:
    assert result["hands_played"] <= max_hands
    assert result["final_stack_a"] + result["final_stack_b"] == 200_000
    assert len(result["hand_summaries"]) == result["hands_played"]


def test_level1_agent_completes_a_full_match():
    result = run_match(
        MathLevel1Agent(), AlwaysCallAgent(), seed=1, max_hands=20, blind_escalation_interval_hands=0
    )
    _assert_well_formed(result, 20)


def test_level1_agent_is_reproducible_under_a_fixed_seed():
    kwargs = dict(seed=11, max_hands=15, blind_escalation_interval_hands=0)
    r1 = run_match(MathLevel1Agent(), MathLevel1Agent(), **kwargs)
    r2 = run_match(MathLevel1Agent(), MathLevel1Agent(), **kwargs)
    assert r1 == r2  # L1 has no RNG anywhere


def test_level2_agent_completes_a_full_match():
    result = run_match(
        MathLevel2Agent(), AlwaysCallAgent(), seed=2, max_hands=20, blind_escalation_interval_hands=0
    )
    _assert_well_formed(result, 20)


def test_level2_agent_is_reproducible_under_a_fixed_seed():
    kwargs = dict(seed=22, max_hands=15, blind_escalation_interval_hands=0)
    r1 = run_match(MathLevel2Agent(), MathLevel2Agent(), **kwargs)
    r2 = run_match(MathLevel2Agent(), MathLevel2Agent(), **kwargs)
    assert r1 == r2  # L2 has no RNG anywhere


def test_level3_agent_completes_a_full_match():
    result = run_match(
        MathLevel3Agent(), AlwaysCallAgent(), seed=3, max_hands=10, blind_escalation_interval_hands=0
    )
    _assert_well_formed(result, 10)


def test_level4_agent_completes_a_full_match():
    result = run_match(
        MathLevel4Agent(), AlwaysCallAgent(), seed=4, max_hands=10, blind_escalation_interval_hands=0
    )
    _assert_well_formed(result, 10)


def test_level5_agent_completes_a_full_match():
    agent = MathLevel5Agent(train_iterations=30)
    result = run_match(agent, AlwaysCallAgent(), seed=5, max_hands=5, blind_escalation_interval_hands=0)
    _assert_well_formed(result, 5)


def test_all_five_levels_complete_a_match_against_a_common_opponent():
    # Sprint 7's DoD, read literally: "each level is wrapped as a
    # standalone Agent and can complete a full match via the Sprint-5
    # orchestrator." One pass over all 5 as a single integration check.
    agents = [
        MathLevel1Agent(),
        MathLevel2Agent(),
        MathLevel3Agent(),
        MathLevel4Agent(),
        MathLevel5Agent(train_iterations=20),
    ]
    for agent in agents:
        result = run_match(agent, AlwaysCallAgent(), seed=99, max_hands=5, blind_escalation_interval_hands=0)
        _assert_well_formed(result, 5)