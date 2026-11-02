"""Agent interface implementations: math-only, PD archetypes, hybrid grid
(Section 4). Sprint 5: Agent ABC + stubs. Sprint 7 close-out: all 5 math
levels wrapped as Agents. Sprint 8: the 5 PD archetype agents.
"""

from agents.base import Action, ActionType, Agent, PotFraction
from agents.math_agents import (
    MathLevel1Agent,
    MathLevel2Agent,
    MathLevel3Agent,
    MathLevel4Agent,
    MathLevel5Agent,
)
from agents.pd_archetypes import (
    AlwaysCooperateAgent,
    AlwaysDefectAgent,
    GrudgerAgent,
    HandHistoryEntry,
    PavlovAgent,
    PavlovMode,
    TitForTatAgent,
    ActionRecord as PdActionRecord,
)
from agents.stub_agents import AlwaysCallAgent, AlwaysFoldAgent

__all__ = [
    "Action",
    "ActionType",
    "Agent",
    "PotFraction",
    "AlwaysCallAgent",
    "AlwaysFoldAgent",
    "MathLevel1Agent",
    "MathLevel2Agent",
    "MathLevel3Agent",
    "MathLevel4Agent",
    "MathLevel5Agent",
    "TitForTatAgent",
    "AlwaysDefectAgent",
    "AlwaysCooperateAgent",
    "PavlovAgent",
    "PavlovMode",
    "GrudgerAgent",
    "HandHistoryEntry",
    "PdActionRecord",
]