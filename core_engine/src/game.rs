//! Single-hand game loop (Sprint 3): blinds, hole cards, 4 betting rounds,
//! showdown or fold resolution, pot awarded. Sprint 6 adds action-sequence
//! and pot-progression logging (Section 6.2 schema) to HandResult.
//!
//! Raise sizing uses the pot-after-call convention (see betting.rs doc
//! comment). Uncalled-excess settlement is handled uniformly by
//! `settle_uncalled_excess`, used by both fold and showdown resolution.

use crate::betting::{resolve_raise, ActionType, PotFraction};
use crate::card::{Card, Deck};
use crate::hand_eval::evaluate_hand;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Player {
    A,
    B,
}

impl Player {
    pub fn other(self) -> Player {
        match self {
            Player::A => Player::B,
            Player::B => Player::A,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

#[derive(Debug, Clone)]
pub struct HandView {
    pub own_hole_cards: [Card; 2],
    pub board: Vec<Card>,
    pub pot_size: u64,
    pub own_stack: u64,
    pub opponent_stack: u64,
    pub bet_to_call: u64,
    pub is_button: bool,
    pub street: Street,
}

#[derive(Debug, Clone, Copy)]
pub struct Action {
    pub action_type: ActionType,
    pub size: Option<PotFraction>,
}

#[derive(Debug, Clone, Copy)]
pub struct HandConfig {
    pub button: Player,
    pub small_blind: u64,
    pub big_blind: u64,
    pub stack_a: u64,
    pub stack_b: u64,
}

/// One entry in the hand's street-tagged action sequence (Section 6.2).
/// `amount` is the chips this specific action added to the pot (0 for
/// Fold/Check).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionLogEntry {
    pub street: Street,
    pub player: Player,
    pub action_type: ActionType,
    pub amount: u64,
}

#[derive(Debug, Clone)]
pub struct HandResult {
    pub ended_by_fold: bool,
    pub winner: Option<Player>, // None => split pot
    pub final_stack_a: u64,
    pub final_stack_b: u64,
    pub board: Vec<Card>,
    pub hole_a: [Card; 2],
    pub hole_b: [Card; 2],
    pub actions: Vec<ActionLogEntry>,
    /// Pot size (committed_a + committed_b) after each action in `actions`,
    /// index-aligned (Section 6.2: "pot-size progression").
    pub pot_progression: Vec<u64>,
    /// The matched pot actually awarded to the winner (excludes any
    /// uncalled excess returned via `settle_uncalled_excess`). 0 on a
    /// split pot.
    pub pot_awarded: u64,
}

fn player_idx(p: Player) -> usize {
    match p {
        Player::A => 0,
        Player::B => 1,
    }
}

fn read_stacks(p: Player, stack_a: u64, stack_b: u64, sc_a: u64, sc_b: u64) -> (u64, u64) {
    match p {
        Player::A => (stack_a, sc_a),
        Player::B => (stack_b, sc_b),
    }
}

fn stacks_mut<'a>(
    p: Player,
    stack_a: &'a mut u64,
    stack_b: &'a mut u64,
    committed_a: &'a mut u64,
    committed_b: &'a mut u64,
) -> (&'a mut u64, &'a mut u64) {
    match p {
        Player::A => (stack_a, committed_a),
        Player::B => (stack_b, committed_b),
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_commit(
    p: Player,
    amount: u64,
    stack_a: &mut u64,
    stack_b: &mut u64,
    committed_a: &mut u64,
    committed_b: &mut u64,
    street_committed_a: &mut u64,
    street_committed_b: &mut u64,
) {
    match p {
        Player::A => {
            *stack_a -= amount;
            *committed_a += amount;
            *street_committed_a += amount;
        }
        Player::B => {
            *stack_b -= amount;
            *committed_b += amount;
            *street_committed_b += amount;
        }
    }
}

/// Settlement correction for a short all-in: the portion of the larger
/// commitment beyond what the opponent could ever match was never at risk
/// and is returned to whoever committed more, BEFORE the matched portion
/// (`2 × min(committed_a, committed_b)`) is awarded to the hand's winner.
fn settle_uncalled_excess(
    committed_a: u64,
    committed_b: u64,
    stack_a: u64,
    stack_b: u64,
) -> (u64, u64, u64) {
    let awarded_pot = 2 * committed_a.min(committed_b);
    let excess = committed_a.abs_diff(committed_b);
    let mut sa = stack_a;
    let mut sb = stack_b;
    if committed_a > committed_b {
        sa += excess;
    } else if committed_b > committed_a {
        sb += excess;
    }
    (awarded_pot, sa, sb)
}

#[allow(clippy::too_many_arguments)]
fn finish_by_fold(
    winner: Player,
    stack_a: u64,
    stack_b: u64,
    committed_a: u64,
    committed_b: u64,
    board: Vec<Card>,
    hole_a: [Card; 2],
    hole_b: [Card; 2],
    actions: Vec<ActionLogEntry>,
    pot_progression: Vec<u64>,
) -> HandResult {
    let (pot, mut stack_a, mut stack_b) =
        settle_uncalled_excess(committed_a, committed_b, stack_a, stack_b);
    match winner {
        Player::A => stack_a += pot,
        Player::B => stack_b += pot,
    }
    HandResult {
        ended_by_fold: true,
        winner: Some(winner),
        final_stack_a: stack_a,
        final_stack_b: stack_b,
        board,
        hole_a,
        hole_b,
        actions,
        pot_progression,
        pot_awarded: pot,
    }
}

#[allow(clippy::too_many_arguments)]
fn showdown(
    stack_a: u64,
    stack_b: u64,
    committed_a: u64,
    committed_b: u64,
    button: Player,
    board: Vec<Card>,
    hole_a: [Card; 2],
    hole_b: [Card; 2],
    actions: Vec<ActionLogEntry>,
    pot_progression: Vec<u64>,
) -> HandResult {
    let (pot, mut stack_a, mut stack_b) =
        settle_uncalled_excess(committed_a, committed_b, stack_a, stack_b);

    let seven_a: [Card; 7] = [
        hole_a[0], hole_a[1], board[0], board[1], board[2], board[3], board[4],
    ];
    let seven_b: [Card; 7] = [
        hole_b[0], hole_b[1], board[0], board[1], board[2], board[3], board[4],
    ];
    let rank_a = evaluate_hand(&seven_a);
    let rank_b = evaluate_hand(&seven_b);

    let winner = match rank_a.cmp(&rank_b) {
        std::cmp::Ordering::Greater => {
            stack_a += pot;
            Some(Player::A)
        }
        std::cmp::Ordering::Less => {
            stack_b += pot;
            Some(Player::B)
        }
        std::cmp::Ordering::Equal => {
            let half = pot / 2;
            let remainder = pot % 2;
            stack_a += half;
            stack_b += half;
            match button.other() {
                Player::A => stack_a += remainder,
                Player::B => stack_b += remainder,
            }
            None
        }
    };

    HandResult {
        ended_by_fold: false,
        winner,
        final_stack_a: stack_a,
        final_stack_b: stack_b,
        board,
        hole_a,
        hole_b,
        actions,
        pot_progression,
        pot_awarded: pot,
    }
}

/// Plays one full hand end-to-end. `deck` is passed in so callers control
/// the shuffle: `Deck::new()` for real play, `Deck::new_seeded(seed)` for
/// deterministic reproducibility (Section 8).
pub fn play_hand(
    config: &HandConfig,
    mut deck: Deck,
    agent_a: &mut impl FnMut(&HandView) -> Action,
    agent_b: &mut impl FnMut(&HandView) -> Action,
) -> HandResult {
    let hole_a = [deck.deal().unwrap(), deck.deal().unwrap()];
    let hole_b = [deck.deal().unwrap(), deck.deal().unwrap()];

    let mut stack_a = config.stack_a;
    let mut stack_b = config.stack_b;
    let (sb_player, bb_player) = (config.button, config.button.other());

    let mut committed_a: u64 = 0;
    let mut committed_b: u64 = 0;

    let mut action_log: Vec<ActionLogEntry> = Vec::new();
    let mut pot_progression: Vec<u64> = Vec::new();

    {
        let (sb_stack, sb_committed) = stacks_mut(
            sb_player,
            &mut stack_a,
            &mut stack_b,
            &mut committed_a,
            &mut committed_b,
        );
        let post = config.small_blind.min(*sb_stack);
        *sb_stack -= post;
        *sb_committed += post;
    }
    {
        let (bb_stack, bb_committed) = stacks_mut(
            bb_player,
            &mut stack_a,
            &mut stack_b,
            &mut committed_a,
            &mut committed_b,
        );
        let post = config.big_blind.min(*bb_stack);
        *bb_stack -= post;
        *bb_committed += post;
    }

    let mut board: Vec<Card> = Vec::new();
    let streets = [Street::Preflop, Street::Flop, Street::Turn, Street::River];

    let mut street_committed_a = committed_a;
    let mut street_committed_b = committed_b;

    for (i, &street) in streets.iter().enumerate() {
        match street {
            Street::Flop => {
                for _ in 0..3 {
                    board.push(deck.deal().unwrap());
                }
            }
            Street::Turn | Street::River => {
                board.push(deck.deal().unwrap());
            }
            Street::Preflop => {}
        }

        if i > 0 {
            street_committed_a = 0;
            street_committed_b = 0;
        }

        let both_all_in = stack_a == 0 && stack_b == 0;
        let one_all_in_no_action_needed =
            (stack_a == 0 || stack_b == 0) && street_committed_a == street_committed_b;

        if !(both_all_in || one_all_in_no_action_needed) {
            let order = if street == Street::Preflop {
                [sb_player, bb_player]
            } else {
                [bb_player, sb_player]
            };

            let mut acted: [bool; 2] = [false, false];
            let mut last_raise_increment = config.big_blind;
            let mut idx = 0usize;

            loop {
                let acting = order[idx % 2];
                let (acting_stack, acting_committed) = read_stacks(
                    acting,
                    stack_a,
                    stack_b,
                    street_committed_a,
                    street_committed_b,
                );
                let (opp_stack, opp_committed) = read_stacks(
                    acting.other(),
                    stack_a,
                    stack_b,
                    street_committed_a,
                    street_committed_b,
                );

                if acting_stack == 0 {
                    acted[player_idx(acting)] = true;
                    if acted[0] && acted[1] && street_committed_a == street_committed_b {
                        break;
                    }
                    idx += 1;
                    continue;
                }

                let bet_to_call = opp_committed.saturating_sub(acting_committed);
                let pot_before_action = committed_a + committed_b;

                let view = HandView {
                    own_hole_cards: if acting == Player::A { hole_a } else { hole_b },
                    board: board.clone(),
                    pot_size: pot_before_action,
                    own_stack: acting_stack,
                    opponent_stack: opp_stack,
                    bet_to_call,
                    is_button: acting == config.button,
                    street,
                };

                let action = if acting == Player::A {
                    agent_a(&view)
                } else {
                    agent_b(&view)
                };

                match action.action_type {
                    ActionType::Fold => {
                        action_log.push(ActionLogEntry {
                            street,
                            player: acting,
                            action_type: ActionType::Fold,
                            amount: 0,
                        });
                        pot_progression.push(committed_a + committed_b);
                        return finish_by_fold(
                            acting.other(),
                            stack_a,
                            stack_b,
                            committed_a,
                            committed_b,
                            board,
                            hole_a,
                            hole_b,
                            action_log,
                            pot_progression,
                        );
                    }
                    ActionType::Check => {
                        debug_assert_eq!(bet_to_call, 0, "illegal check facing a bet");
                        acted[player_idx(acting)] = true;
                        action_log.push(ActionLogEntry {
                            street,
                            player: acting,
                            action_type: ActionType::Check,
                            amount: 0,
                        });
                        pot_progression.push(committed_a + committed_b);
                    }
                    ActionType::Call => {
                        let call_amount = bet_to_call.min(acting_stack); // short all-in call
                        apply_commit(
                            acting,
                            call_amount,
                            &mut stack_a,
                            &mut stack_b,
                            &mut committed_a,
                            &mut committed_b,
                            &mut street_committed_a,
                            &mut street_committed_b,
                        );
                        acted[player_idx(acting)] = true;
                        action_log.push(ActionLogEntry {
                            street,
                            player: acting,
                            action_type: ActionType::Call,
                            amount: call_amount,
                        });
                        pot_progression.push(committed_a + committed_b);
                    }
                    ActionType::Raise | ActionType::AllIn => {
                        let fraction = action.size.unwrap_or(PotFraction::Full);
                        let resolved = resolve_raise(
                            fraction,
                            pot_before_action,
                            bet_to_call,
                            last_raise_increment,
                            acting_stack,
                        );
                        let delta = resolved.amount;

                        let min_total_for_full_raise = bet_to_call + last_raise_increment;
                        let reopens_betting =
                            !(resolved.is_all_in && delta < min_total_for_full_raise);

                        if reopens_betting && delta > bet_to_call {
                            last_raise_increment = delta - bet_to_call;
                        }

                        apply_commit(
                            acting,
                            delta,
                            &mut stack_a,
                            &mut stack_b,
                            &mut committed_a,
                            &mut committed_b,
                            &mut street_committed_a,
                            &mut street_committed_b,
                        );

                        if reopens_betting {
                            acted = [false, false];
                        }
                        acted[player_idx(acting)] = true;
                        action_log.push(ActionLogEntry {
                            street,
                            player: acting,
                            action_type: action.action_type,
                            amount: delta,
                        });
                        pot_progression.push(committed_a + committed_b);
                    }
                }

                let street_matched = street_committed_a == street_committed_b;
                let either_all_in = stack_a == 0 || stack_b == 0;

                if acted[0] && acted[1] && (street_matched || either_all_in) {
                    break;
                }
                idx += 1;
            }
        }
    }

    showdown(
        stack_a,
        stack_b,
        committed_a,
        committed_b,
        config.button,
        board,
        hole_a,
        hole_b,
        action_log,
        pot_progression,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    #[test]
    fn settle_returns_uncalled_excess_to_larger_committer() {
        let (pot, sa, sb) = settle_uncalled_excess(1000, 600, 0, 0);
        assert_eq!(pot, 1200);
        assert_eq!(sa, 400);
        assert_eq!(sb, 0);
    }

    #[test]
    fn settle_is_a_no_op_when_commitments_match() {
        let (pot, sa, sb) = settle_uncalled_excess(500, 500, 0, 0);
        assert_eq!(pot, 1000);
        assert_eq!(sa, 0);
        assert_eq!(sb, 0);
    }

    #[test]
    fn showdown_does_not_award_uncalled_excess_to_a_winning_short_stack() {
        let board = vec![c(2, 0), c(5, 1), c(9, 2), c(11, 3), c(7, 0)];
        let hole_a = [c(3, 1), c(4, 2)];
        let hole_b = [c(13, 1), c(13, 2)];

        let result = showdown(
            0,
            0,
            1000,
            600,
            Player::A,
            board,
            hole_a,
            hole_b,
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(result.winner, Some(Player::B));
        assert_eq!(result.final_stack_a, 400);
        assert_eq!(result.final_stack_b, 1200);
        assert_eq!(result.pot_awarded, 1200);
        assert_eq!(result.final_stack_a + result.final_stack_b, 1600);
    }

    #[test]
    fn showdown_tie_splits_only_the_matched_pot() {
        let board = vec![c(2, 0), c(5, 1), c(9, 2), c(11, 3), c(7, 0)];
        let hole_a = [c(3, 1), c(4, 2)];
        let hole_b = [c(3, 2), c(4, 3)];

        let result = showdown(
            0,
            0,
            1000,
            600,
            Player::A,
            board,
            hole_a,
            hole_b,
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(result.winner, None);
        assert_eq!(result.final_stack_a, 400 + 600);
        assert_eq!(result.final_stack_b, 600);
        assert_eq!(result.final_stack_a + result.final_stack_b, 1600);
    }
}
