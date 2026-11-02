"""Section 4.4 opponent memory: builds HandHistoryEntry objects from a
play_hand_py result and drives observe_hand_result/flush on agents that
implement them (duck-typed, per Sprint 8's archetypes)."""

from agents.pd_archetypes import ActionRecord, HandHistoryEntry


def build_hand_history_entries(hand_result: dict):
    """Returns (entry_for_a, entry_for_b), each carrying the OTHER
    player's actions only, from that side's own perspective.
    self_stack_delta is left at 0 -- the caller fills it in once
    before/after stacks are known."""
    actions_for_a = []  # B's actions, as seen by A
    actions_for_b = []  # A's actions, as seen by B
    for action, pot_after in zip(hand_result["actions"], hand_result["pot_progression"]):
        record = ActionRecord(action["street"], action["action_type"], action["amount"], pot_after)
        if action["player"] == "A":
            actions_for_b.append(record)
        else:
            actions_for_a.append(record)

    winner = hand_result["winner"]
    showdown_reached = not hand_result["ended_by_fold"]

    def self_won(player):
        return None if winner == "SPLIT" else winner == player

    entry_a = HandHistoryEntry(
        actions=actions_for_a, showdown_reached=showdown_reached, self_won=self_won("A")
    )
    entry_b = HandHistoryEntry(
        actions=actions_for_b, showdown_reached=showdown_reached, self_won=self_won("B")
    )
    return entry_a, entry_b


def observe(agent, entry) -> None:
    fn = getattr(agent, "observe_hand_result", None)
    if callable(fn):
        fn(entry)


def flush(agent) -> None:
    fn = getattr(agent, "flush", None)
    if callable(fn):
        fn()