"""Trivial stub agents (Sprint 5 DoD): always-fold, always-call."""

from agents.base import Action, ActionType, Agent


class AlwaysFoldAgent(Agent):
    def decide_action(self, state) -> Action:
        return Action(action_type=ActionType.FOLD)


class AlwaysCallAgent(Agent):
    def decide_action(self, state) -> Action:
        if state.bet_to_call == 0:
            return Action(action_type=ActionType.CHECK)
        return Action(action_type=ActionType.CALL)