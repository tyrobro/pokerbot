"""Sprint 5 DoD: a full match between two Python-defined stub agents runs
via the Python orchestrator and produces a correctly formed, inspectable
result."""

from agents.stub_agents import AlwaysCallAgent, AlwaysFoldAgent
from orchestrator.match_runner import run_match


def test_full_match_runs_and_produces_well_formed_raw_result():
    result = run_match(
        AlwaysCallAgent(),
        AlwaysFoldAgent(),
        seed=123,
        max_hands=50,
        blind_escalation_interval_hands=0,  # isolate from bust-by-escalation
    )

    assert isinstance(result, dict)
    for key in ("hands_played", "final_stack_a", "final_stack_b", "busted", "hand_summaries"):
        assert key in result

    assert result["hands_played"] <= 50
    assert len(result["hand_summaries"]) == result["hands_played"]

    # Chip conservation (Section 8-style invariant), regardless of outcome.
    assert result["final_stack_a"] + result["final_stack_b"] == 200_000

    first = result["hand_summaries"][0]
    for key in ("hand_number", "button", "small_blind", "big_blind", "stack_a_after", "stack_b_after"):
        assert key in first


def test_match_is_reproducible_under_a_fixed_seed():
    kwargs = dict(seed=55, max_hands=30, blind_escalation_interval_hands=0)
    result_1 = run_match(AlwaysCallAgent(), AlwaysCallAgent(), **kwargs)
    result_2 = run_match(AlwaysCallAgent(), AlwaysCallAgent(), **kwargs)

    assert result_1 == result_2