"""PD archetype agents (Section 4.2): Tit-for-Tat, Always-Defect,
Always-Cooperate, Pavlov, Grudger.

Known tension (flagged, resolution applied pending confirmation): Section
4.2 describes Grudger as triggering "after it observes a clear
bluff/overbet revealed at a showdown it WON." Section 4.4's card-
visibility rule says the opposite -- an agent sees the opponent's hole
cards only in hands it LOST (the winner must show to claim the pot; in
hands the agent won, the opponent's cards are never revealed). These two
sections contradict each other as written. Resolution used here: detect
the trigger from BETTING PATTERN alone (a showdown the agent won, where
the opponent made at least one overbet-sized action that still lost) --
this needs no hole-card visibility at all, so it sidesteps the
contradiction while matching the spirit of "a bluff/overbet that didn't
pay off." See `_is_bluff_catch`.

Opponent-memory gap (same class of gap as L4/L5, Sprint 7): none of these
agents' cross-hand reads are wired into a live match yet -- PyHandView
carries no action history (Section 4.3), and Section 4.4's real
opponent-memory object (persisting across a pairing's two legs, flushing
between pairings) is Sprint 9 orchestrator territory. Each archetype
below exposes an explicit `observe_hand_result(entry)` method so this
sprint's tests can feed scripted fixtures directly (per Sprint 7 DoD's
own wording: "a unit test confirming its core behavioral rule triggers
correctly on a scripted opponent-history fixture") -- Sprint 9 wires the
real orchestrator to call this after every hand, and calls `flush()`
(where present) at pairing boundaries.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import pokerbot_core
from agents.base import Action, ActionType, Agent, PotFraction
from agents.math_agents import MathLevel5Agent, _translate


@dataclass
class ActionRecord:
    street: str
    action_type: str
    amount: int
    pot_size_at_time: int


@dataclass
class HandHistoryEntry:
    """One completed hand vs this opponent.

    actions: the OPPONENT's actions only, street-tagged (same convention
        as Section 4.4's object / Sprint 7 L4's ObservedActions).
    opponent_hole_cards: per Section 4.4, visible only when self_won is
        False. Kept for fixture realism; NOT used by Grudger's trigger
        (see module docstring).
    self_won: True/False/None(split), from THIS agent's own perspective.
    self_stack_delta: this agent's own net chip change this hand
        (positive = gained).
    """

    actions: List[ActionRecord] = field(default_factory=list)
    opponent_hole_cards: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    showdown_reached: bool = False
    self_won: Optional[bool] = None
    self_stack_delta: int = 0


# ---- Tit-for-Tat: mirror aggression ----

def _opponent_aggression_frequency(
    history: List[HandHistoryEntry], decay: float = 0.85
) -> Optional[float]:
    """Recency-weighted fraction of the opponent's actions that were
    aggressive (RAISE/ALL_IN), most recent hand weighted highest, older
    hands decayed ("decays old reads over time", Section 4.2). `None` if
    there's no data yet (first hands of a pairing)."""
    weight = 1.0
    total_w = 0.0
    agg_w = 0.0
    for entry in reversed(history):
        for action in entry.actions:
            total_w += weight
            if action.action_type in ("RAISE", "ALL_IN"):
                agg_w += weight
        weight *= decay
    if total_w == 0:
        return None
    return agg_w / total_w


class TitForTatAgent(Agent):
    """Starts at a standard/GTO-baseline aggression level (L2, Sprint
    7); shifts up/down based on a decayed rolling read of the opponent's
    aggression. Thresholds below are placeholders, Sprint 9 calibration
    targets like every other level's thresholds."""

    ESCALATE_ABOVE = 0.55
    DEESCALATE_BELOW = 0.25

    def __init__(self, decay: float = 0.85):
        self.decay = decay
        self.match_history: List[HandHistoryEntry] = []
    
        
    def flush(self) -> None:
        self.match_history = []

    def observe_hand_result(self, entry: HandHistoryEntry) -> None:
        self.match_history.append(entry)

    def decide_action(self, state) -> Action:
        kind, size = pokerbot_core.l2_recommend_py(state)
        baseline = _translate(kind, size, state.bet_to_call)

        freq = _opponent_aggression_frequency(self.match_history, self.decay)
        if freq is None:
            return baseline

        if freq >= self.ESCALATE_ABOVE and baseline.action_type in (
            ActionType.CHECK,
            ActionType.CALL,
        ):
            return Action(action_type=ActionType.RAISE, size=PotFraction.HALF)
        if freq <= self.DEESCALATE_BELOW and baseline.action_type == ActionType.RAISE:
            if state.bet_to_call == 0:
                return Action(action_type=ActionType.CHECK)
            return Action(action_type=ActionType.CALL)
        return baseline


# ---- Always-Defect: fixed near-maximum aggression ----

class AlwaysDefectAgent(Agent):
    """Fixed near-maximum aggression, no opponent-read module (Section
    4.2). Never reads `state` -- always raises full pot."""

    def decide_action(self, state) -> Action:
        return Action(action_type=ActionType.RAISE, size=PotFraction.FULL)


# ---- Always-Cooperate: GTO-pure, direct L5 reuse ----

class AlwaysCooperateAgent(MathLevel5Agent):
    """Section 4.2: "reuses the math-engine level-5 solver standalone,
    never adapts." Direct reuse of MathLevel5Agent (Sprint 7) -- see that
    class's docstring for the Sprint-7-scope blueprint caveat (a small,
    fast training config, not the production-scale CFRConfig::default()
    blueprint, Sprint 11-12's offline work)."""


# ---- Pavlov: win-stay / lose-shift ----

class PavlovMode(str, Enum):
    TIGHT_PASSIVE = "TIGHT_PASSIVE"
    BALANCED = "BALANCED"
    LOOSE_AGGRESSIVE = "LOOSE_AGGRESSIVE"


# Section 4.2 doesn't pin down which mode a loss switches TO, only that
# it switches. Placeholder cycle, Sprint 9-tunable like everything else.
_MODE_CYCLE_ON_LOSS = {
    PavlovMode.BALANCED: PavlovMode.LOOSE_AGGRESSIVE,
    PavlovMode.LOOSE_AGGRESSIVE: PavlovMode.TIGHT_PASSIVE,
    PavlovMode.TIGHT_PASSIVE: PavlovMode.BALANCED,
}

# (raise_above_percentile, fold_below_percentile) per mode. BALANCED
# matches L1's own Sprint-7 defaults.
_MODE_THRESHOLDS = {
    PavlovMode.TIGHT_PASSIVE: (0.92, 0.65),
    PavlovMode.BALANCED: (0.85, 0.50),
    PavlovMode.LOOSE_AGGRESSIVE: (0.65, 0.20),
}


def _pavlov_next_mode(
    current_mode: PavlovMode,
    self_stack_delta: int,
    starting_stack: int,
    loss_threshold_pct: float = 0.1,
) -> PavlovMode:
    """Switches mode after a loss above `loss_threshold_pct` of the
    starting stack; stays in mode after a win (Section 4.2)."""
    if starting_stack <= 0:
        return current_mode
    loss_frac = -self_stack_delta / starting_stack
    if loss_frac > loss_threshold_pct:
        return _MODE_CYCLE_ON_LOSS[current_mode]
    return current_mode


class PavlovAgent(Agent):
    def __init__(self, starting_stack: int = 100_000, loss_threshold_pct: float = 0.1):
        self.mode = PavlovMode.BALANCED
        self.starting_stack = starting_stack
        self.loss_threshold_pct = loss_threshold_pct
        
    def flush(self) -> None:
        self.mode = PavlovMode.BALANCED

    def observe_hand_result(self, entry: HandHistoryEntry) -> None:
        self.mode = _pavlov_next_mode(
            self.mode, entry.self_stack_delta, self.starting_stack, self.loss_threshold_pct
        )

    def decide_action(self, state) -> Action:
        raise_above, fold_below = _MODE_THRESHOLDS[self.mode]
        percentile = pokerbot_core.estimate_hand_strength_percentile_py(state)

        if percentile >= raise_above:
            return Action(action_type=ActionType.RAISE, size=PotFraction.HALF)
        if state.bet_to_call == 0:
            return Action(action_type=ActionType.CHECK)
        if percentile < fold_below:
            return Action(action_type=ActionType.FOLD)
        return Action(action_type=ActionType.CALL)


# ---- Grudger: permanent punishment ----

def _is_bluff_catch(entry: HandHistoryEntry, overbet_ratio: float = 0.75) -> bool:
    """Grudger's trigger, detected from betting pattern alone -- see
    module docstring for why hole cards aren't used. True iff this hand
    reached showdown, the agent (self) won it, and the opponent made at
    least one action sized at or above `overbet_ratio` of the pot at the
    time (a large bet/raise that still lost)."""
    if not entry.showdown_reached or entry.self_won is not True:
        return False
    for action in entry.actions:
        if action.pot_size_at_time <= 0:
            continue
        if action.amount / action.pot_size_at_time >= overbet_ratio:
            return True
    return False


class GrudgerAgent(Agent):
    """Plays standard (L2, Sprint 7) until it catches a bluff (see
    `_is_bluff_catch`), then permanently switches to max-aggression for
    the rest of the pairing (both legs -- Section 4.2: "the grudge
    persists across the pairing's shared opponent memory, then flushes
    with everything else between pairings"). Flushing between pairings
    is Sprint 9 orchestrator territory; `flush()` is exposed for it to
    call, not yet auto-invoked anywhere."""

    def __init__(self, overbet_ratio: float = 0.75):
        self.overbet_ratio = overbet_ratio
        self.triggered = False

    def observe_hand_result(self, entry: HandHistoryEntry) -> None:
        if not self.triggered and _is_bluff_catch(entry, self.overbet_ratio):
            self.triggered = True

    def flush(self) -> None:
        self.triggered = False

    def decide_action(self, state) -> Action:
        if self.triggered:
            return Action(action_type=ActionType.RAISE, size=PotFraction.FULL)
        kind, size = pokerbot_core.l2_recommend_py(state)
        return _translate(kind, size, state.bet_to_call)