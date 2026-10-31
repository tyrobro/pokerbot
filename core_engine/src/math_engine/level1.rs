//! Math engine Level 1 (Section 4.1): "Crude heuristics — hand-strength
//! percentile thresholds only, no pot odds."
//!
//! Open design decisions (flagged, not silently assumed):
//!   - "Percentile" at L1 is context-free: preflop uses the Chen formula
//!     over the 169 starting-hand types (no opponent-range modeling,
//!     which only enters at L3+); postflop uses made-hand category alone
//!     (ignores kickers/draws -- PRD-author-confirmed as exactly the
//!     right crudeness to separate L1 from L2's pot-odds and L3's
//!     Monte-Carlo equity).
//!   - Percentile maps to action+size via a tiered cutoff structure
//!     (PRD-author-requested): top tier -> raise at one FIXED size
//!     (no sizing reasoning at L1, by design), middle tier -> check/call,
//!     bottom tier -> fold. Both cutoffs AND the fixed raise size live in
//!     `ThresholdConfig`, tunable without touching this file's logic.
//!   - All percentile thresholds below are placeholders pending Sprint 9
//!     calibration (Section 7).

use super::{MathEngineState, RecommendedAction};
use crate::betting::PotFraction;
use crate::card::Card;
use crate::game::Street;
use crate::hand_eval::{evaluate_hand, HandCategory};

#[derive(Debug, Clone, Copy)]
pub struct ThresholdConfig {
    /// Hands at or above this percentile (0.0-1.0, higher = stronger)
    /// raise, at the fixed `raise_size`.
    pub raise_above_percentile: f64,
    /// Fixed sizing used for every L1 raise -- deliberately not
    /// hand-strength-dependent; "crude" L1 does no sizing reasoning.
    pub raise_size: PotFraction,
    /// Hands below this percentile fold (if facing a bet) or check (if
    /// not). Hands in [fold_below_percentile, raise_above_percentile)
    /// check/call.
    pub fold_below_percentile: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        // Placeholders (Sprint 9 calibration target, Section 7): raise
        // the best ~15% of hands at half pot, fold the worst ~50%, check
        // or call the middle tier.
        ThresholdConfig {
            raise_above_percentile: 0.85,
            raise_size: PotFraction::Half,
            fold_below_percentile: 0.50,
        }
    }
}

/// Chen formula: a well-known, deterministic, context-free preflop
/// starting-hand strength heuristic (0-20+ points). Used here only as an
/// ordinal ranking source, not reproduced as poker advice/literature text.
fn chen_score(hole_cards: [Card; 2]) -> f64 {
    let mut ranks = [hole_cards[0].rank, hole_cards[1].rank];
    ranks.sort_unstable();
    let (lo, hi) = (ranks[0], ranks[1]);

    let card_points = |rank: u8| -> f64 {
        match rank {
            14 => 10.0,
            13 => 8.0,
            12 => 7.0,
            11 => 6.0,
            r => (r as f64) / 2.0, // 2-10 -> 1.0-5.0
        }
    };

    let mut score = card_points(hi);
    let pair = lo == hi;
    let suited = hole_cards[0].suit == hole_cards[1].suit;
    let gap = (hi - lo).saturating_sub(1);

    if pair {
        score = (card_points(hi) * 2.0).max(5.0);
    }
    if suited {
        score += 2.0;
    }
    if !pair {
        score -= match gap {
            0 => 0.0,
            1 => 1.0,
            2 => 2.0,
            3 => 4.0,
            _ => 5.0,
        };
        // "Straight-capable" bonus: both cards below a Queen and gap <= 1.
        if gap <= 1 && hi < 12 {
            score += 1.0;
        }
    }

    score.max(0.0)
}

/// Approximate percentile (0.0-1.0) of a Chen score among the 169 distinct
/// starting-hand types. Chen scores range roughly 0-20; this is a simple,
/// deterministic monotonic mapping, not a literature-sourced table.
fn chen_score_to_percentile(score: f64) -> f64 {
    (score / 20.0).clamp(0.0, 1.0)
}

/// Postflop percentile: ordinal rank of the made-hand category alone,
/// mapped onto 0.0-1.0 across the 9 categories.
fn category_to_percentile(category: HandCategory) -> f64 {
    (category as u8 as f64) / (HandCategory::StraightFlush as u8 as f64)
}

pub(crate) fn hand_strength_percentile_pub(state: &MathEngineState) -> f64 {
    match state.street {
        Street::Preflop => chen_score_to_percentile(chen_score(state.hole_cards)),
        _ => {
            let board = &state.board;
            if board.len() < 5 {
                // Sprint 7 scope: evaluate_hand expects exactly 7 cards
                // (2 hole + 5 board). Flop/turn (3-4 board cards) aren't
                // evaluable yet without a partial-board equity method --
                // that's Monte Carlo territory (L3). L1 falls back to the
                // preflop heuristic on those streets as a crude stand-in.
                chen_score_to_percentile(chen_score(state.hole_cards))
            } else {
                let seven: [Card; 7] = [
                    state.hole_cards[0],
                    state.hole_cards[1],
                    board[0],
                    board[1],
                    board[2],
                    board[3],
                    board[4],
                ];
                category_to_percentile(evaluate_hand(&seven).category)
            }
        }
    }
}

/// Level 1 recommendation (Section 4.1): tiered percentile thresholds,
/// fixed raise size, no pot odds, no opponent modeling.
pub fn recommend(state: &MathEngineState, config: &ThresholdConfig) -> RecommendedAction {
    let percentile = hand_strength_percentile_pub(state);

    if percentile >= config.raise_above_percentile {
        RecommendedAction::Raise(config.raise_size)
    } else if percentile < config.fold_below_percentile {
        if state.bet_to_call == 0 {
            RecommendedAction::CheckOrCall // nothing to fold to
        } else {
            RecommendedAction::Fold
        }
    } else {
        RecommendedAction::CheckOrCall
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    fn base_state(hole: [Card; 2]) -> MathEngineState {
        MathEngineState {
            hole_cards: hole,
            board: Vec::new(),
            street: Street::Preflop,
            pot_size: 150,
            own_stack: 100_000,
            opponent_stack: 100_000,
            bet_to_call: 100,
            is_button: true,
        }
    }

    // ---- Chen formula sanity: known strong/weak hands rank correctly ----

    #[test]
    fn pocket_aces_scores_highest_among_pairs() {
        let aa = chen_score([c(14, 0), c(14, 1)]);
        let kk = chen_score([c(13, 0), c(13, 1)]);
        assert!(aa > kk);
    }

    #[test]
    fn suited_connector_beats_offsuit_same_ranks() {
        let suited = chen_score([c(9, 0), c(10, 0)]);
        let offsuit = chen_score([c(9, 0), c(10, 1)]);
        assert!(suited > offsuit);
    }

    #[test]
    fn seven_deuce_offsuit_is_among_the_worst_hands() {
        // The canonical "worst starting hand" in poker literature.
        let worst = chen_score([c(7, 0), c(2, 1)]);
        let premium = chen_score([c(14, 0), c(14, 1)]);
        assert!(worst < premium / 2.0);
    }

    #[test]
    fn pair_score_is_never_negative() {
        let low_pair = chen_score([c(2, 0), c(2, 1)]);
        assert!(low_pair >= 0.0);
    }

    // ---- recommend(): threshold branches, each hand's score/percentile
    // computed explicitly in the comment so a future threshold change
    // (Sprint 9) can't silently invalidate the expectation unnoticed ----

    #[test]
    fn premium_pair_raises() {
        // AA: pair, card_points(14)=10 -> score=max(20,5)=20 -> percentile 1.0.
        let state = base_state([c(14, 0), c(14, 1)]);
        let config = ThresholdConfig::default();
        assert_eq!(
            recommend(&state, &config),
            RecommendedAction::Raise(config.raise_size)
        );
    }

    #[test]
    fn trash_hand_folds_facing_a_bet() {
        // 72o: hi=7 -> 3.5, not suited, gap=(7-2-1)=4 -> deduct 5,
        // gap>1 so no straight bonus -> score=max(0, 3.5-5)=0 -> percentile 0.0.
        let state = base_state([c(7, 0), c(2, 1)]);
        let config = ThresholdConfig::default();
        assert_eq!(recommend(&state, &config), RecommendedAction::Fold);
    }

    #[test]
    fn trash_hand_checks_when_nothing_to_call() {
        let mut state = base_state([c(7, 0), c(2, 1)]);
        state.bet_to_call = 0;
        let config = ThresholdConfig::default();
        assert_eq!(recommend(&state, &config), RecommendedAction::CheckOrCall);
    }

    #[test]
    fn medium_hand_checks_or_calls() {
        // AKo: hi=14 -> 10.0, not suited, gap=(14-13-1)=0 -> deduct 0,
        // hi=14 is NOT <12 so no straight bonus -> score=10.0 ->
        // percentile 0.5. fold_below(0.50) is NOT > 0.5 (strict <
        // required to fold), and 0.5 < raise_above(0.85) -> CheckOrCall.
        let state = base_state([c(14, 0), c(13, 1)]);
        let config = ThresholdConfig::default();
        assert_eq!(recommend(&state, &config), RecommendedAction::CheckOrCall);
    }

    // ---- postflop: made-hand category drives percentile ----

    #[test]
    fn postflop_quads_raises() {
        let state = MathEngineState {
            hole_cards: [c(9, 0), c(9, 1)],
            board: vec![c(9, 2), c(9, 3), c(2, 0), c(5, 1), c(7, 2)],
            street: Street::River,
            pot_size: 1000,
            own_stack: 100_000,
            opponent_stack: 100_000,
            bet_to_call: 500,
            is_button: true,
        };
        let config = ThresholdConfig::default();
        assert_eq!(
            recommend(&state, &config),
            RecommendedAction::Raise(config.raise_size)
        );
    }

    #[test]
    fn postflop_high_card_folds_facing_a_bet() {
        let state = MathEngineState {
            hole_cards: [c(3, 0), c(4, 1)],
            board: vec![c(9, 2), c(11, 3), c(2, 0), c(13, 1), c(7, 2)],
            street: Street::River,
            pot_size: 1000,
            own_stack: 100_000,
            opponent_stack: 100_000,
            bet_to_call: 500,
            is_button: true,
        };
        let config = ThresholdConfig::default();
        assert_eq!(recommend(&state, &config), RecommendedAction::Fold);
    }

    #[test]
    fn custom_threshold_config_changes_the_decision() {
        let state = base_state([c(10, 0), c(9, 0)]); // suited -> score 8.0, percentile 0.4
        let lenient = ThresholdConfig {
            raise_above_percentile: 0.3, // much lower bar
            raise_size: PotFraction::Half,
            fold_below_percentile: 0.1,
        };
        assert_eq!(
            recommend(&state, &lenient),
            RecommendedAction::Raise(lenient.raise_size)
        );
    }

    #[test]
    fn raise_size_is_configurable_not_hardcoded() {
        // Same premium hand as the default-config test, but a different
        // fixed raise size -- proves recommend() reads raise_size from
        // config rather than hardcoding a literal.
        let state = base_state([c(14, 0), c(14, 1)]);
        let config = ThresholdConfig {
            raise_above_percentile: 0.85,
            raise_size: PotFraction::Full,
            fold_below_percentile: 0.50,
        };
        assert_eq!(
            recommend(&state, &config),
            RecommendedAction::Raise(PotFraction::Full)
        );
    }
}
