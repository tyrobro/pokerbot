"""Parquet + CSV logging (Section 6.2). One row per hand; Parquet is the
primary analysis format, CSV is a human-readable companion generated from
the SAME in-memory MatchResult so the two are guaranteed consistent.
"""

import csv
from pathlib import Path

import pandas as pd

from orchestrator.scoring import leg_score

RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}
# Suit indices are an unordered 0-3 label internal to the engine (Section
# 3.1 never assigns suit identity -- only flush-matching cares they're
# equal). These letters are a display convenience only.
SUIT_LETTERS = {0: "c", 1: "d", 2: "h", 3: "s"}


def card_str(card: tuple) -> str:
    rank, suit = card
    return f"{RANK_NAMES.get(rank, str(rank))}{SUIT_LETTERS.get(suit, '?')}"


def cards_str(cards: list) -> str:
    return ",".join(card_str(c) for c in cards)


def action_sequence_str(actions: list) -> str:
    parts = []
    for a in actions:
        if a["action_type"] in ("CHECK", "FOLD"):
            parts.append(f"{a['street']}:{a['action_type']}")
        else:
            parts.append(f"{a['street']}:{a['action_type']}:{a['amount']}")
    return ";".join(parts)


def pot_progression_str(pot_progression: list) -> str:
    return ";".join(str(p) for p in pot_progression)


def build_hand_rows(match_result: dict, starting_stack: int) -> list:
    """Converts a raw pokerbot_core match_result dict into one row per
    hand matching Section 6.2's schema (board split into flop/turn/river;
    action_sequence and pot_size_progression as semicolon-delimited
    strings, matching the CSV spec exactly; Parquet stores the same
    strings so the two formats are guaranteed to agree row-for-row)."""
    rows = []
    for s in match_result["hand_summaries"]:
        board = s["board"]
        flop = board[0:3]
        turn = board[3:4]
        river = board[4:5]

        winner = s["winner"]
        amount_won = s["pot_awarded"] if winner != "SPLIT" else s["pot_awarded"] // 2

        rows.append(
            {
                "hand_number": s["hand_number"],
                "blind_level": s["blind_level"],
                "small_blind": s["small_blind"],
                "big_blind": s["big_blind"],
                "button_position": s["button"],
                "hole_cards_agent_a": cards_str(s["hole_a"]),
                "hole_cards_agent_b": cards_str(s["hole_b"]),
                "board_flop": cards_str(flop),
                "board_turn": cards_str(turn),
                "board_river": cards_str(river),
                "action_sequence": action_sequence_str(s["actions"]),
                "pot_size_progression": pot_progression_str(s["pot_progression"]),
                "hand_ended_by": "FOLD" if s["ended_by_fold"] else "SHOWDOWN",
                "winner": winner,
                "amount_won": amount_won,
                "stack_agent_a_after": s["stack_a_after"],
                "stack_agent_b_after": s["stack_b_after"],
                "score_agent_a_this_hand": leg_score(s["stack_a_after"], starting_stack),
                "score_agent_b_this_hand": leg_score(s["stack_b_after"], starting_stack),
            }
        )
    return rows


CSV_FIELDNAMES = [
    "hand_number",
    "blind_level",
    "small_blind",
    "big_blind",
    "button_position",
    "hole_cards_agent_a",
    "hole_cards_agent_b",
    "board_flop",
    "board_turn",
    "board_river",
    "action_sequence",
    "pot_size_progression",
    "hand_ended_by",
    "winner",
    "amount_won",
    "stack_agent_a_after",
    "stack_agent_b_after",
    "score_agent_a_this_hand",
    "score_agent_b_this_hand",
]


def write_leg_logs(
    rows: list, agent_a_id: str, agent_b_id: str, leg: int, logs_dir: str = "logs"
) -> dict:
    """Writes logs/{agent_a_id}_vs_{agent_b_id}/leg_{1|2}.parquet AND the
    matching .csv (Section 6.2), generated from the same `rows`. Returns
    both paths."""
    if leg not in (1, 2):
        raise ValueError("leg must be 1 or 2")

    pair_dir = Path(logs_dir) / f"{agent_a_id}_vs_{agent_b_id}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = pair_dir / f"leg_{leg}.parquet"
    csv_path = pair_dir / f"leg_{leg}.csv"

    df = pd.DataFrame(rows, columns=CSV_FIELDNAMES)
    df.to_parquet(parquet_path, index=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {"parquet": str(parquet_path), "csv": str(csv_path)}


def append_pairing_summary(
    agent_a_id: str,
    agent_b_id: str,
    leg_1_score_a: float,
    leg_1_score_b: float,
    leg_2_score_a: float,
    leg_2_score_b: float,
    pairing_score_a: float,
    pairing_score_b: float,
    logs_dir: str = "logs",
) -> str:
    """Upserts one row into the top-level pairings_summary.parquet
    (Section 6.2): replaces any existing row for this (agent_a_id,
    agent_b_id) pairing rather than appending a duplicate, so re-running
    a round-robin (e.g. during debugging) doesn't silently accumulate
    multiple rows per pairing. Found via a Sprint 10 dashboard bug: bars
    of very different, non-9 lengths for a 10-agent field, caused by
    exactly this duplication across repeated calibration runs."""
    path = Path(logs_dir) / "pairings_summary.parquet"
    row = {
        "agent_a_id": agent_a_id,
        "agent_b_id": agent_b_id,
        "leg_1_score_a": leg_1_score_a,
        "leg_1_score_b": leg_1_score_b,
        "leg_2_score_a": leg_2_score_a,
        "leg_2_score_b": leg_2_score_b,
        "pairing_score_a": pairing_score_a,
        "pairing_score_b": pairing_score_b,
    }
    new_df = pd.DataFrame([row])

    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[
            ~((existing["agent_a_id"] == agent_a_id) & (existing["agent_b_id"] == agent_b_id))
        ]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df

    combined.to_parquet(path, index=False)
    return str(path)