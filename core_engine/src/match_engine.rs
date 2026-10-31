//! Full match loop (Sprint 4): wraps `play_hand` into up to 1000 hands or
//! bust, with blind escalation and button alternation (Section 3.1, 3.2).
//! Sprint 6: HandSummary grows to carry everything Section 6.2's logging
//! schema needs (hole cards, board, action sequence, pot progression,
//! outcome) so the Python side can write it out without recomputation.

use crate::card::{Card, Deck};
use crate::game::{play_hand, Action, ActionLogEntry, HandConfig, HandView, Player};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

#[derive(Debug, Clone)]
pub struct MatchConfig {
    pub starting_stack: u64,
    pub initial_small_blind: u64,
    pub initial_big_blind: u64,
    pub blind_escalation_interval_hands: u32,
    pub blind_escalation_multiplier_numerator: u64,
    pub blind_escalation_multiplier_denominator: u64,
    pub max_hands: u32,
    pub seed: Option<u64>,
}

impl Default for MatchConfig {
    fn default() -> Self {
        MatchConfig {
            starting_stack: 100_000,
            initial_small_blind: 50,
            initial_big_blind: 100,
            blind_escalation_interval_hands: 20,
            blind_escalation_multiplier_numerator: 6,
            blind_escalation_multiplier_denominator: 5,
            max_hands: 1_000,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandSummary {
    pub hand_number: u32,
    pub blind_level: u32,
    pub button: Player,
    pub small_blind: u64,
    pub big_blind: u64,
    pub stack_a_after: u64,
    pub stack_b_after: u64,
    pub hole_a: [Card; 2],
    pub hole_b: [Card; 2],
    pub board: Vec<Card>,
    pub actions: Vec<ActionLogEntry>,
    pub pot_progression: Vec<u64>,
    pub ended_by_fold: bool,
    pub winner: Option<Player>,
    pub pot_awarded: u64,
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub hands_played: u32,
    pub final_stack_a: u64,
    pub final_stack_b: u64,
    pub busted: bool,
    pub hand_summaries: Vec<HandSummary>,
}

fn round_mul(value: u64, num: u64, den: u64) -> u64 {
    (value * num + den / 2) / den
}

pub fn blinds_for_hand(
    initial_small_blind: u64,
    initial_big_blind: u64,
    hand_number: u32,
    escalation_interval_hands: u32,
    multiplier_numerator: u64,
    multiplier_denominator: u64,
) -> (u64, u64) {
    let tiers = if escalation_interval_hands == 0 {
        0
    } else {
        hand_number / escalation_interval_hands
    };

    let mut sb = initial_small_blind;
    let mut bb = initial_big_blind;
    for _ in 0..tiers {
        sb = round_mul(sb, multiplier_numerator, multiplier_denominator);
        bb = round_mul(bb, multiplier_numerator, multiplier_denominator);
    }
    (sb, bb)
}

fn blind_tier(hand_number: u32, escalation_interval_hands: u32) -> u32 {
    if escalation_interval_hands == 0 {
        0
    } else {
        hand_number / escalation_interval_hands
    }
}

pub fn play_match(
    config: &MatchConfig,
    mut agent_a: impl FnMut(&HandView) -> Action,
    mut agent_b: impl FnMut(&HandView) -> Action,
) -> MatchResult {
    let mut rng_stream = config.seed.map(StdRng::seed_from_u64);

    let mut stack_a = config.starting_stack;
    let mut stack_b = config.starting_stack;

    let mut button = match &mut rng_stream {
        Some(rng) => {
            if rng.next_u32() % 2 == 0 {
                Player::A
            } else {
                Player::B
            }
        }
        None => {
            if rand::random::<bool>() {
                Player::A
            } else {
                Player::B
            }
        }
    };

    let mut hand_summaries = Vec::new();
    let mut busted = false;
    let mut hand_number = 0u32;

    while hand_number < config.max_hands && stack_a > 0 && stack_b > 0 {
        let (small_blind, big_blind) = blinds_for_hand(
            config.initial_small_blind,
            config.initial_big_blind,
            hand_number,
            config.blind_escalation_interval_hands,
            config.blind_escalation_multiplier_numerator,
            config.blind_escalation_multiplier_denominator,
        );

        let hand_config = HandConfig {
            button,
            small_blind,
            big_blind,
            stack_a,
            stack_b,
        };

        let deck = match &mut rng_stream {
            Some(rng) => Deck::new_seeded(rng.next_u64()),
            None => Deck::new(),
        };

        let result = play_hand(&hand_config, deck, &mut agent_a, &mut agent_b);
        stack_a = result.final_stack_a;
        stack_b = result.final_stack_b;

        hand_summaries.push(HandSummary {
            hand_number,
            blind_level: blind_tier(hand_number, config.blind_escalation_interval_hands),
            button,
            small_blind,
            big_blind,
            stack_a_after: stack_a,
            stack_b_after: stack_b,
            hole_a: result.hole_a,
            hole_b: result.hole_b,
            board: result.board,
            actions: result.actions,
            pot_progression: result.pot_progression,
            ended_by_fold: result.ended_by_fold,
            winner: result.winner,
            pot_awarded: result.pot_awarded,
        });

        if stack_a == 0 || stack_b == 0 {
            busted = true;
        }

        button = button.other();
        hand_number += 1;
    }

    MatchResult {
        hands_played: hand_number,
        final_stack_a: stack_a,
        final_stack_b: stack_b,
        busted,
        hand_summaries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blinds_unchanged_within_first_tier() {
        assert_eq!(blinds_for_hand(50, 100, 0, 20, 6, 5), (50, 100));
        assert_eq!(blinds_for_hand(50, 100, 19, 20, 6, 5), (50, 100));
    }

    #[test]
    fn blinds_escalate_by_1_2x_at_hand_20() {
        assert_eq!(blinds_for_hand(50, 100, 20, 20, 6, 5), (60, 120));
    }

    #[test]
    fn blinds_compound_across_multiple_tiers() {
        assert_eq!(blinds_for_hand(50, 100, 40, 20, 6, 5), (72, 144));
    }

    #[test]
    fn zero_interval_never_escalates() {
        assert_eq!(blinds_for_hand(50, 100, 10_000, 0, 6, 5), (50, 100));
    }
}
