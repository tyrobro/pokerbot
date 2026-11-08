"""GridAgent(math_level, ml_level) (Sprint 13, Section 4.1 combination
rule): wraps one math engine + one ML engine, blends via
confidence_weight(k) on a common aggression scale (see
ml/exploit_signal.py for the scale definition and the reasoning behind
it -- an interpretation of an underspecified mechanism, flagged there).
"""

import pokerbot_core
from agents.base import Agent
from agents.confidence_weight import confidence_weight
from agents.math_agents import _translate
from ml.exploit_signal import action_to_scalar, scalar_to_action
from ml.ml_engine import build_ml_engine

_MATH_RECOMMEND = {
    1: lambda state: pokerbot_core.l1_recommend_py(state),
    2: lambda state: pokerbot_core.l2_recommend_py(state),
}


class GridAgent(Agent):
    def __init__(
        self,
        math_level: int,
        ml_level: int,
        seed: int = 1,
        onnx_dir: str = "ml/onnx",
        shared_l5_solver=None,
    ):
        """shared_l5_solver: REQUIRED when math_level == 5. Pass a single
        pretrained PyCFRSolver (see ml/blueprint_store.py) shared across
        every GridAgent(5,*) and AlwaysCooperateAgent instance -- Section
        4.4's fresh-per-pairing requirement is about opponent memory, not
        the trained blueprint itself (Sprint 14 fix; the original
        per-instance training here was a real bug caught before the full
        30-agent round-robin depended on it)."""
        if not 1 <= math_level <= 5:
            raise ValueError("math_level must be 1-5")
        if not 1 <= ml_level <= 5:
            raise ValueError("ml_level must be 1-5")
        self.math_level = math_level
        self.ml_level = ml_level
        self._next_seed = seed
        self._ml_engine = build_ml_engine(ml_level, onnx_dir)
        self._opponent_actions: list = []  # current-hand opponent actions seen so far

        if math_level in (3, 4):
            from agents.math_agents import MathLevel3Agent, MathLevel4Agent

            self._math_agent = (MathLevel3Agent if math_level == 3 else MathLevel4Agent)(seed=seed)
        elif math_level == 5:
            if shared_l5_solver is None:
                raise ValueError("GridAgent(math_level=5, ...) requires shared_l5_solver")
            self._l5_solver = shared_l5_solver
        else:
            self._math_agent = None  # L1/L2 use the stateless functions above
            
    def observe_hand_result(self, entry) -> None:
        self._opponent_actions = []  # Section 4.4: fresh per hand within a match is NOT right --
        # actually this resets per HAND; cross-hand memory for ML read-in is Sprint 9's
        # opponent-memory object, which ml_engine.read_opponent does not yet consume
        # (flagged below in the class docstring / Sprint 13 close-out notes).

    def decide_action(self, state):
        if self.math_level in (1, 2):
            kind, size = _MATH_RECOMMEND[self.math_level](state)
            baseline = _translate(kind, size, state.bet_to_call)
        elif self.math_level == 5:
            kind, size = self._l5_solver.recommend(state, self._next_seed)
            self._next_seed += 1
            baseline = _translate(kind, size, state.bet_to_call)
        else:
            baseline = self._math_agent.decide_action(state)

        # exploit_signal and confidence are now genuinely separate
        # quantities (PRD-author-corrected, see ml/exploit_signal.py's
        # module doc): exploit_signal is a DIRECTIONAL recommendation
        # (fold-likely -> more aggression, continue-likely -> caution),
        # confidence is the distribution's peakedness. Neither is reused
        # for the other.
        exploit_signal, confidence = self._ml_engine.read_opponent(self._opponent_actions)
        weight = confidence_weight(self.ml_level) * confidence

        baseline_scalar = action_to_scalar(baseline)
        blended_scalar = baseline_scalar + weight * (exploit_signal - baseline_scalar)
        # scalar_to_action snaps to the nearest discrete tier; the
        # engine's own resolve_raise (Section 3.3) applies downstream to
        # whatever RAISE/ALL_IN comes out of this, exactly as for every
        # other agent level -- no separate bet-sizing path here.
        final_action = scalar_to_action(blended_scalar, state.bet_to_call)

        for street, player, action_type, amount, pot_after in state.actions_so_far:
            if player != state.own_player:
                self._opponent_actions.append(
                    {"street": street, "action_type": action_type, "amount": amount, "pot_before": pot_after}
                )

        return final_action