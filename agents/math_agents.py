"""Agent wrappers around the Rust math-engine levels. Sprint 9: L3/L4/L5
now carry a `seed` constructor param and derive a fresh, deterministic
seed per decision from it (self._next_seed, incremented each call) --
closes the Sprint 7 precondition that L3/L4/L5 agents were not
reproducible under a fixed match seed. L1/L2 remain unseeded since they
have no RNG anywhere.
"""

import pokerbot_core
from agents.base import Action, ActionType, Agent, PotFraction


def _translate(kind: str, size, bet_to_call: int) -> Action:
    if kind == "FOLD":
        return Action(action_type=ActionType.FOLD)
    if kind == "CHECKORCALL":
        if bet_to_call == 0:
            return Action(action_type=ActionType.CHECK)
        return Action(action_type=ActionType.CALL)
    if kind == "RAISE":
        return Action(action_type=ActionType.RAISE, size=PotFraction(size))
    raise ValueError(f"unknown recommendation kind {kind!r}")


class MathLevel1Agent(Agent):
    """Deterministic, no RNG -- reproducible under any match seed."""

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l1_recommend_py(state)
        return _translate(kind, size, state.bet_to_call)


class MathLevel2Agent(Agent):
    """Deterministic, no RNG -- reproducible under any match seed."""

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l2_recommend_py(state)
        return _translate(kind, size, state.bet_to_call)


class MathLevel3Agent(Agent):
    """Monte Carlo equity vs. uniform random range. Reproducible given
    the SAME `seed` constructor arg and the same sequence of decisions."""

    def __init__(self, seed: int = 1):
        self._next_seed = seed

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l3_recommend_py(state, self._next_seed)
        self._next_seed += 1
        return _translate(kind, size, state.bet_to_call)


class MathLevel4Agent(Agent):
    """Range-vs-range equity, now reading real current-hand opponent
    actions (Sprint 9). Reproducible given the same `seed`."""

    def __init__(self, seed: int = 1):
        self._next_seed = seed

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l4_recommend_py(state, self._next_seed)
        self._next_seed += 1
        return _translate(kind, size, state.bet_to_call)


class MathLevel5Agent(Agent):
    """Wraps a PyCFRSolver blueprint, trained ONCE at construction (small,
    fast Sprint-7-scope config -- NOT the production CFRConfig::default()
    blueprint, Sprint 11-12's offline work). Decision-level seed is now
    explicit and incrementing (Sprint 9), and hand_log is now real
    (closes the L5 "confidently wrong InfoSetKey" precondition)."""

    def __init__(
        self,
        num_buckets: int = 6,
        bucketing_trials: int = 50,
        max_raises_per_street: int = 1,
        seed: int = 1,
        train_iterations: int = 50,
    ):
        self._solver = pokerbot_core.PyCFRSolver(
            num_buckets, bucketing_trials, max_raises_per_street, seed, train_iterations
        )
        self._next_seed = seed + train_iterations  # avoid reusing training seeds

    def decide_action(self, state) -> Action:
        kind, size = self._solver.recommend(state, self._next_seed)
        self._next_seed += 1
        return _translate(kind, size, state.bet_to_call)