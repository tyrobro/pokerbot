"""Agent wrappers around the Rust math-engine levels (Section 4.1),
closing out Sprint 7's Definition of Done: each level completes a full
match via the Sprint-5 orchestrator (play_match_py).

Known gaps, carried forward rather than silently resolved:

1. PyHandView does not yet carry the full current-hand action sequence or
   the opponent match-history object (Section 4.3's full state contract;
   Section 4.4's opponent-memory object is Sprint 8-9 work). L4 and L5's
   within-hand/match-history reads therefore run against empty/default
   inputs here.

   L4 degenerates to its own "no observed actions" case -- a DESIGNED,
   TESTED fallback (empty_observed_actions_means_no_narrowing), identical
   in kind to L3's uniform-range assumption. This is a known, bounded
   approximation.

   L5 is NOT the same kind of gap and must not be treated as such: an
   always-empty hand_log means recommend()'s InfoSetKey lookups silently
   undercount raises_this_street whenever real raises occurred earlier in
   the street, producing a CONFIDENTLY WRONG lookup that often still
   matches a real trained entry -- just the wrong one -- rather than
   tripping the (correctly implemented) unseen-key fallback chain. See
   core_engine/src/math_engine/level5/recommend.rs's "CALLER CONTRACT"
   doc comment for the full explanation. Practical consequence:
   MathLevel5Agent's decisions are UNVERIFIED past the first action of
   each street. This is a named precondition for trusting any Sprint 9
   calibration run that includes MathLevel5Agent, tracked separately from
   gap 2 below -- not a generic "state-threading is future work" note.
   
2. L3/L4/L5 use MonteCarloConfig::default() (seed=None) on the Rust side,
   and L5's recommend() draws its own unseeded RNG internally. This means
   a match's `seed` parameter (passed to play_match_py) controls the deck
   shuffle and button only -- NOT these agents' own Monte Carlo draws.
   Matches using MathLevel3Agent/4/5 are therefore NOT currently
   reproducible under a fixed seed, unlike MathLevel1Agent/2Agent (no RNG
   at all). This is a real prerequisite for Sprint 9's fixed-seed
   calibration phase, not yet built -- flagged explicitly rather than
   assumed fixed.
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
    """Crude heuristics (Chen formula + made-hand category), tiered
    thresholds, fixed raise size. No randomness anywhere -- fully
    deterministic and reproducible under a fixed match seed."""

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l1_recommend_py(state)
        return _translate(kind, size, state.bet_to_call)


class MathLevel2Agent(Agent):
    """L1 + basic pot-odds (fold->call widening only). No randomness --
    deterministic and reproducible under a fixed match seed."""

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l2_recommend_py(state)
        return _translate(kind, size, state.bet_to_call)


class MathLevel3Agent(Agent):
    """Monte Carlo equity vs. a uniform random opponent range. Uses
    unseeded randomness internally (see module doc) -- not reproducible
    under a fixed match seed."""

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l3_recommend_py(state)
        return _translate(kind, size, state.bet_to_call)


class MathLevel4Agent(Agent):
    """Range-vs-range equity (ObservedActions degenerates to empty here,
    see module doc), position-aware bet sizing. Uses unseeded randomness
    internally -- not reproducible under a fixed match seed."""

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l4_recommend_py(state)
        return _translate(kind, size, state.bet_to_call)


class MathLevel5Agent(Agent):
    """Wraps a PyCFRSolver blueprint, trained ONCE at construction. This
    is a small, fast Sprint-7-scope blueprint to prove the wrapping
    works -- NOT the production-scale CFRConfig::default() blueprint
    (Sprint 11-12's offline curriculum training, Section 5). Uses
    unseeded randomness internally -- not reproducible under a fixed
    match seed."""

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

    def decide_action(self, state) -> Action:
        kind, size = self._solver.recommend(state)
        return _translate(kind, size, state.bet_to_call)