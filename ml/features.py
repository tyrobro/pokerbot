"""Corpus row -> training-example extraction. Player identity per action
is NOT stored explicitly in the Section 6.2 schema -- reconstructed here
from the fact that heads-up actions strictly alternate between the two
players, with the first actor per street fixed by Section 3.1's acting
order (SB/button first preflop, BB/non-button first postflop; already
engine-tested, Sprint 3)."""

import pandas as pd

from ml.lstm_model import ACTION_TYPES, STREETS, FEATURE_DIM


def _street_first_actor(street: str, button_position: str) -> str:
    if street == "PREFLOP":
        return button_position
    return "B" if button_position == "A" else "A"


def _other(player: str) -> str:
    return "B" if player == "A" else "A"


def parse_action_sequence(action_sequence: str, button_position: str) -> list:
    if not action_sequence:
        return []
    entries = []
    current_street = None
    next_actor = None
    for part in action_sequence.split(";"):
        fields = part.split(":")
        street, action_type = fields[0], fields[1]
        amount = int(fields[2]) if len(fields) > 2 else 0
        if street != current_street:
            current_street = street
            next_actor = _street_first_actor(street, button_position)
        entries.append({"player": next_actor, "street": street, "action_type": action_type, "amount": amount})
        next_actor = _other(next_actor)
    return entries


def encode_action(entry: dict, pot_before: float) -> list:
    vec = [0.0] * FEATURE_DIM
    vec[STREETS.index(entry["street"])] = 1.0
    vec[len(STREETS) + ACTION_TYPES.index(entry["action_type"])] = 1.0
    size_ratio = entry["amount"] / pot_before if pot_before > 0 else 0.0
    vec[-1] = min(size_ratio, 5.0)  # clamp, keep features bounded
    return vec


def build_opponent_sequences(row: pd.Series, target_player: str) -> list:
    """Extracts (context-so-far, next-action-label) pairs for
    TARGET_PLAYER's own action sequence within one hand -- i.e. examples
    of predicting that player's next action from their own history,
    which is exactly what an opponent-read model needs when the OTHER
    player is doing the reading."""
    entries = parse_action_sequence(row["action_sequence"], row["button_position"])
    pot_progression = [int(p) for p in row["pot_size_progression"].split(";")] if row["pot_size_progression"] else []

    examples = []
    seq_so_far = []
    pot_before = row["small_blind"] + row["big_blind"]
    for i, entry in enumerate(entries):
        pot_at_point = pot_progression[i - 1] if i > 0 else pot_before
        if entry["player"] == target_player and seq_so_far:
            label = ACTION_TYPES.index(entry["action_type"])
            examples.append((list(seq_so_far), label))
        seq_so_far.append(encode_action(entry, pot_at_point))
    return examples