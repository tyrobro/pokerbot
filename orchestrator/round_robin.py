"""Sprint 9: pairing generation (Section 7), full round-robin driver,
opponent-memory persist-then-flush (Section 4.4), Section 6 logging and
scoring. Hands are driven one at a time from Python (play_hand_py) so
observe_hand_result fires after every hand, not just at leg end.
"""

import itertools

import pokerbot_core
from orchestrator.blinds import blinds_for_hand
from orchestrator.logging import build_hand_rows, write_leg_logs, append_pairing_summary
from orchestrator.opponent_memory import build_hand_history_entries, observe, flush
from orchestrator.scoring import leg_score, pairing_score


def generate_pairings(agent_ids):
    """All C(n,2) unique pairings (Section 7)."""
    return list(itertools.combinations(sorted(agent_ids), 2))


def play_leg(agent_a, agent_b, leg: int, config: dict) -> dict:
    """Plays one full leg (Section 3.1: up to max_hands or bust). Legs
    start button-flipped relative to each other."""
    starting_stack = config["starting_stack"]
    stack_a, stack_b = starting_stack, starting_stack
    button = "A" if leg == 1 else "B"
    hand_summaries = []
    seed_base = config.get("seed")
    hand_number = 0

    while hand_number < config["max_hands"] and stack_a > 0 and stack_b > 0:
        if hand_number % 10 == 0:
            print(f"      hand {hand_number}/{config['max_hands']} (stacks: {stack_a}, {stack_b})...", flush=True)
        sb, bb = blinds_for_hand(
            config["initial_small_blind"], config["initial_big_blind"], hand_number,
            config["blind_escalation_interval_hands"],
        )
        hand_seed = None if seed_base is None else seed_base + hand_number * 7919 + leg

        stack_a_before, stack_b_before = stack_a, stack_b
        try:
            result = pokerbot_core.play_hand_py(agent_a, agent_b, button, sb, bb, stack_a, stack_b, hand_seed)
        except Exception as e:
            print(f"      !! hand {hand_number} raised: {type(e).__name__}: {e}", flush=True)
            raise
        stack_a, stack_b = result["final_stack_a"], result["final_stack_b"]

        entry_a, entry_b = build_hand_history_entries(result)
        entry_a.self_stack_delta = stack_a - stack_a_before
        entry_b.self_stack_delta = stack_b - stack_b_before

        hand_summaries.append({
            "hand_number": hand_number,
            "blind_level": 0 if config["blind_escalation_interval_hands"] == 0
                else hand_number // config["blind_escalation_interval_hands"],
            "small_blind": sb, "big_blind": bb, "button": button,
            "hole_a": result["hole_a"], "hole_b": result["hole_b"], "board": result["board"],
            "actions": result["actions"], "pot_progression": result["pot_progression"],
            "ended_by_fold": result["ended_by_fold"], "winner": result["winner"],
            "pot_awarded": result["pot_awarded"],
            "stack_a_after": stack_a, "stack_b_after": stack_b,
        })

        observe(agent_a, entry_a)
        observe(agent_b, entry_b)

        button = "B" if button == "A" else "A"
        hand_number += 1

    return {
        "hands_played": hand_number,
        "final_stack_a": stack_a,
        "final_stack_b": stack_b,
        "hand_summaries": hand_summaries,
    }


def play_pairing(agent_a, agent_b, agent_a_id: str, agent_b_id: str, config: dict, logs_dir="logs") -> dict:
    starting_stack = config["starting_stack"]
    leg_results = []
    for leg in (1, 2):
        result = play_leg(agent_a, agent_b, leg, config)
        rows = build_hand_rows(
            {"final_stack_a": result["final_stack_a"], "final_stack_b": result["final_stack_b"],
             "hand_summaries": result["hand_summaries"]},
            starting_stack,
        )
        write_leg_logs(rows, agent_a_id, agent_b_id, leg, logs_dir)
        leg_results.append(result)

    # Section 4.4: opponent memory flushed to zero once both legs complete.
    flush(agent_a)
    flush(agent_b)

    leg1_a = leg_score(leg_results[0]["final_stack_a"], starting_stack)
    leg1_b = leg_score(leg_results[0]["final_stack_b"], starting_stack)
    leg2_a = leg_score(leg_results[1]["final_stack_a"], starting_stack)
    leg2_b = leg_score(leg_results[1]["final_stack_b"], starting_stack)
    pa = pairing_score(leg1_a, leg2_a)
    pb = pairing_score(leg1_b, leg2_b)

    append_pairing_summary(agent_a_id, agent_b_id, leg1_a, leg1_b, leg2_a, leg2_b, pa, pb, logs_dir)

    return {
        "agent_a_id": agent_a_id, "agent_b_id": agent_b_id,
        "pairing_score_a": pa, "pairing_score_b": pb,
        "leg_results": leg_results,
    }


def run_round_robin(agent_factories: dict, config: dict, logs_dir="logs") -> list:
    """agent_factories: dict[str, Callable[[int], Agent]] -- a FRESH
    agent instance is built per pairing (Section 4.4: "fully anonymous to
    its opponent" at pairing start), seeded deterministically from
    config['seed'] + the pairing's index, for reproducibility."""
    agent_ids = sorted(agent_factories.keys())
    pairings = generate_pairings(agent_ids)
    base_seed = config.get("seed") or 0
    results = []
    for idx, (a_id, b_id) in enumerate(pairings):
        print(f"  [{idx + 1}/{len(pairings)}] {a_id} vs {b_id}...", flush=True)
        agent_a = agent_factories[a_id](base_seed + idx * 2)
        agent_b = agent_factories[b_id](base_seed + idx * 2 + 1)
        results.append(play_pairing(agent_a, agent_b, a_id, b_id, config, logs_dir))
    return results