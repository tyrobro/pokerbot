"""Thin driver calling the Rust match loop (Sprint 5). Formal Parquet/CSV
logging per Section 6 lands in Sprint 6 — this just returns the raw
MatchResult dict so it's inspectable (Sprint 5 DoD).
"""

import pokerbot_core
from agents.base import Agent


def run_match(agent_a: Agent, agent_b: Agent, **config_kwargs) -> dict:
    """config_kwargs: starting_stack, initial_small_blind,
    initial_big_blind, blind_escalation_interval_hands,
    blind_escalation_multiplier_numerator,
    blind_escalation_multiplier_denominator, max_hands, seed."""
    return pokerbot_core.play_match_py(agent_a, agent_b, **config_kwargs)