//! Full match loop (Sprint 4): wraps `play_hand` into up to 1000 hands or
//! bust, with blind escalation and button alternation (Section 3.1, 3.2).
//!
//! Blind-rounding decision (not pinned down by Section 3.2, flagged for
//! Sprint 9 calibration review): escalation rounds to the nearest chip at
//! each tier, compounding from the previous (already-rounded) tier's
//! value — the way real tournament structures round blind levels. Uses
//! integer numerator/denominator arithmetic (not floats) so results stay
//! bit-for-bit reproducible across platforms.

use crate::card::Deck;
use crate::game::{play_hand, Action, HandConfig, HandView, Player};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

#[derive(Debug, Clone)]
pub struct MatchConfig {
    pub starting_stack: u64,
    pub initial_small_blind: u64,
    pub initial_big_blind: u64,
    /// Escalate every N hands (Section 3.2). 0 disables escalation.
    pub blind_escalation_interval_hands: u32,
    /// Multiplier as a fraction (default 6/5 = ×1.2) — integer arithmetic,
    /// no floats near money.
    pub blind_escalation_multiplier_numerator: u64,
    pub blind_escalation_multiplier_denominator: u64,
    pub max_hands: u32,
    /// `Some(seed)` for deterministic calibration-phase runs (Section 7);
    /// `None` for fully random final-run seeding.
    pub seed: Option<u64>,
}

impl Default for MatchConfig {
    fn default() -> Self {
        // Starting stack/blinds are placeholders pending Sprint 9
        // calibration (Section 3.2) — not treated as final values.
        MatchConfig {
            starting_stack: 100_000,
            initial_small_blind: 50,
            initial_big_blind: 100,
            blind_escalation_interval_hands: 20,
            blind_escalation_multiplier_numerator: 6, // 1.2 = 6/5
            blind_escalation_multiplier_denominator: 5,
            max_hands: 1_000,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandSummary {
    pub hand_number: u32,
    pub button: Player,
    pub small_blind: u64,
    pub big_blind: u64,
    pub stack_a_after: u64,
    pub stack_b_after: u64,
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub hands_played: u32,
    pub final_stack_a: u64,
    pub final_stack_b: u64,
    pub busted: bool,
    /// Per-hand snapshot — proves reproducibility under a fixed seed
    /// (Section 8); formalized into full logging at Sprint 6.
    pub hand_summaries: Vec<HandSummary>,
}

/// Rounds `value * (num/den)` to the nearest integer (round-half-up style).
fn round_mul(value: u64, num: u64, den: u64) -> u64 {
    (value * num + den / 2) / den
}

/// Blinds at a given 0-indexed hand number, per the escalation schedule
/// (Section 3.2). Exposed standalone so it's independently unit-testable.
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

/// Plays a full match: up to `max_hands` hands or until either stack hits
/// 0 (Section 3.1), with blind escalation and strict button alternation
/// (randomized coin flip for hand 0, Section 3.1).
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
            button,
            small_blind,
            big_blind,
            stack_a_after: stack_a,
            stack_b_after: stack_b,
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

    // ---- blind escalation schedule correctness (Section 8) ----

    #[test]
    fn blinds_unchanged_within_first_tier() {
        assert_eq!(blinds_for_hand(50, 100, 0, 20, 6, 5), (50, 100));
        assert_eq!(blinds_for_hand(50, 100, 19, 20, 6, 5), (50, 100));
    }

    #[test]
    fn blinds_escalate_by_1_2x_at_hand_20() {
        // 50 * 1.2 = 60, 100 * 1.2 = 120.
        assert_eq!(blinds_for_hand(50, 100, 20, 20, 6, 5), (60, 120));
    }

    #[test]
    fn blinds_compound_across_multiple_tiers() {
        // tier 2 (hand 40): 60 * 1.2 = 72, 120 * 1.2 = 144.
        assert_eq!(blinds_for_hand(50, 100, 40, 20, 6, 5), (72, 144));
    }

    #[test]
    fn zero_interval_never_escalates() {
        assert_eq!(blinds_for_hand(50, 100, 10_000, 0, 6, 5), (50, 100));
    }
}
