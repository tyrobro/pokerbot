"""Agent interface implementations: math-only, PD archetypes, hybrid grid
(Section 4). Sprint 5: the Agent ABC (base.py) and two trivial stubs
(stub_agents.py) proving the interface contract end to end.
"""

from agents.base import Action, ActionType, Agent, PotFraction
from agents.stub_agents import AlwaysCallAgent, AlwaysFoldAgent

__all__ = [
    "Action",
    "ActionType",
    "Agent",
    "PotFraction",
    "AlwaysCallAgent",
    "AlwaysFoldAgent",
]