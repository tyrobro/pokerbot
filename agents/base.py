"""Agent interface contract (Section 4.3). A Python-defined agent is any
object with a `decide_action(state) -> Action` method, where `state` is a
pokerbot_core.PyHandView (attribute access: state.own_hole_cards, etc).

Not yet included in `state` (grows in later sprints): current blind level
and hand number (Sprint 6 logging), full betting-action sequence for the
current hand, and the opponent match-history object (Section 4.4,
Sprint 8-9). Agents built before those sprints land should not assume
those fields exist yet.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    RAISE = "RAISE"
    ALL_IN = "ALL_IN"


class PotFraction(str, Enum):
    QUARTER = "QUARTER"
    HALF = "HALF"
    THREE_QUARTER = "THREE_QUARTER"
    FULL = "FULL"


@dataclass
class Action:
    """Section 4.3: action_type, size (required iff RAISE — the
    pre-resolution pot-fraction choice; the engine applies Section 3.3's
    resolution), and metadata (logging-only, never consumed by engine
    logic)."""

    action_type: ActionType
    size: Optional[PotFraction] = None
    metadata: dict = field(default_factory=dict)


class Agent(ABC):
    """Every agent type (math-only, PD archetypes, hybrid grid) implements
    this. `state` is a pokerbot_core.PyHandView passed in by the Rust
    engine for every decision point."""

    @abstractmethod
    def decide_action(self, state) -> Action:
        ...