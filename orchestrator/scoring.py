"""Scoring functions (Section 6.1)."""


def leg_score(final_stack: int, starting_stack: int) -> float:
    """score = (final_stack - starting_stack) / starting_stack, bounded
    [-1, +1] (table-stakes rules guarantee the lower bound; equal starting
    stacks guarantee the upper bound)."""
    score = (final_stack - starting_stack) / starting_stack
    return max(-1.0, min(1.0, score))


def pairing_score(leg_1_score: float, leg_2_score: float) -> float:
    """Legs are SUMMED, not averaged (Section 6.1) -> bounded [-2, +2]."""
    return leg_1_score + leg_2_score


def pairing_outcome(score: float) -> str:
    """win: score > 0, loss: score < 0, draw: score == 0 (Section 6.1)."""
    if score > 0:
        return "WIN"
    if score < 0:
        return "LOSS"
    return "DRAW"