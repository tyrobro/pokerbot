"""Sprint 13 DoD: all 25 grid-agent combinations instantiate and
complete a full match; corner-case sanity checks at (5,1)/(1,5)/(5,5)/(1,1)
-- exploratory, documented, not asserted as strictly "correct"."""

import pytest

from agents.confidence_weight import confidence_weight
from agents.grid_agent import GridAgent
from agents.stub_agents import AlwaysCallAgent
from ml.exploit_signal import action_to_scalar, scalar_to_action, AGGRESSION_SCALE
from agents.base import Action, ActionType, PotFraction
from orchestrator.match_runner import run_match


def test_confidence_weight_level_1_is_always_zero():
    assert confidence_weight(1) == 0.0


def test_confidence_weight_scales_linearly_with_level():
    assert confidence_weight(5) == 1.0
    assert confidence_weight(3) == pytest.approx(0.6)


def test_action_scalar_round_trip_for_each_tier():
    for action in [
        Action(action_type=ActionType.FOLD),
        Action(action_type=ActionType.CHECK),
        Action(action_type=ActionType.RAISE, size=PotFraction.HALF),
        Action(action_type=ActionType.ALL_IN),
    ]:
        scalar = action_to_scalar(action)
        recovered = scalar_to_action(scalar, bet_to_call=0)
        assert action_to_scalar(recovered) == pytest.approx(scalar)


@pytest.fixture(scope="module")
def shared_l5_solver_for_grid_tests():
    import pokerbot_core
    return pokerbot_core.PyCFRSolver(
        num_buckets=4, bucketing_trials=20, max_raises_per_street=1, seed=1, train_iterations=20
    )


@pytest.mark.parametrize("math_level", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("ml_level", [1, 2, 3, 4, 5])  # full grid -- ONNX models now exist (Sprint 12 trained)
def test_grid_agent_instantiates_and_completes_a_match_ml_rule_based(math_level, ml_level, shared_l5_solver_for_grid_tests):
    kwargs = {"shared_l5_solver": shared_l5_solver_for_grid_tests} if math_level == 5 else {}
    agent = GridAgent(math_level=math_level, ml_level=ml_level, seed=math_level * 10 + ml_level, **kwargs)
    result = run_match(agent, AlwaysCallAgent(), seed=1, max_hands=5, blind_escalation_interval_hands=0)
    assert result["final_stack_a"] + result["final_stack_b"] == 200_000

def test_corner_cases_produce_well_formed_results(shared_l5_solver_for_grid_tests):
    # (5,1), (1,5), (5,5), (1,1) -- Sprint 13 DoD's named corners.
    # Exploratory per the DoD's own wording ("documented, not necessarily
    # correct") -- this just confirms each one runs and returns sane data.
    corners = [(5, 1), (1, 5), (5, 5), (1, 1)]
    for math_level, ml_level in corners:
        kwargs = {"shared_l5_solver": shared_l5_solver_for_grid_tests} if math_level == 5 else {}
        agent = GridAgent(math_level=math_level, ml_level=ml_level, seed=1, **kwargs)
        result = run_match(agent, AlwaysCallAgent(), seed=2, max_hands=5, blind_escalation_interval_hands=0)
        assert result["hands_played"] > 0
        assert result["final_stack_a"] + result["final_stack_b"] == 200_000
        
def test_exploit_signal_and_confidence_are_independent_quantities():
    from ml.exploit_signal import ml_distribution_to_exploit_signal

    # Peaked distribution, heavily predicting FOLD: high confidence,
    # exploit_signal should push toward AGGRESSION (positive).
    fold_heavy = [0.9, 0.025, 0.025, 0.025, 0.025]  # FOLD, CHECK, CALL, RAISE, ALL_IN
    signal_a, conf_a = ml_distribution_to_exploit_signal(fold_heavy)
    assert signal_a > 0.5
    assert conf_a == pytest.approx(0.9)

    # Peaked distribution, heavily predicting RAISE: same high
    # confidence, but exploit_signal should now push toward CAUTION
    # (negative) -- proves confidence doesn't determine signal's sign.
    raise_heavy = [0.025, 0.025, 0.025, 0.9, 0.025]
    signal_b, conf_b = ml_distribution_to_exploit_signal(raise_heavy)
    assert signal_b < -0.5
    assert conf_b == pytest.approx(0.9)

    # Same confidence, opposite signal -- the two are independent.
    assert conf_a == pytest.approx(conf_b)
    assert signal_a > 0 > signal_b