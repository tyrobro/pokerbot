"""Python mirror of match_engine's blinds_for_hand (Section 3.2) --
needed because Sprint 9's round-robin drives hands one at a time from
Python (play_hand_py), not via the Rust play_match_py loop."""


def blinds_for_hand(initial_sb, initial_bb, hand_number, interval, mult_num=6, mult_den=5):
    tiers = 0 if interval == 0 else hand_number // interval
    sb, bb = initial_sb, initial_bb
    for _ in range(tiers):
        sb = (sb * mult_num + mult_den // 2) // mult_den
        bb = (bb * mult_num + mult_den // 2) // mult_den
    return sb, bb