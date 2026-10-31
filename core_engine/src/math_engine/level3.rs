//! Math engine Level 3 (Section 4.1): "Monte Carlo equity estimation vs.
//! an assumed random opponent range."
//!
//! Confirmed with PRD author:
//!   - "Random opponent range" = uniform sampling over all 2-card
//!     combinations from the remaining undealt deck (52 minus own hole
//!     cards minus board cards dealt so far) for the opponent's hand, and
//!     uniform sampling over the same remaining deck for any yet-undealt
//!     board cards, per trial. This is deliberately the same simulation
//!     machinery L4 will reuse with a non-uniform, action-estimated range
//!     swapped in for the opponent-hand sampling step.
//!   - Sample count is a tunable config (`MonteCarloConfig`), 2,000
//!     trials as the Sprint 9-revisable default, same treatment as
//!     ThresholdConfig elsewhere.
//!   - Action mapping is TOTALLY upgraded to real equity, not partially:
//!     raise threshold is now equity-driven (not L1's percentile cutoff),
//!     while bet sizing per tier stays fixed/config-driven (real sizing
//!     reasoning is explicitly L4's job).
//!       if equity >= raise_cutoff      -> raise (fixed size, config)
//!       elif equity >= required_equity -> call  (pot odds, Section 3.3)
//!       else                            -> fold

use super::{MathEngineState, RecommendedAction};
use crate::betting::PotFraction;
use crate::card::Card;
use crate::hand_eval::evaluate_hand;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Debug, Clone, Copy)]
pub struct MonteCarloConfig {
    pub trials: u32,
    /// `Some(seed)` for deterministic/reproducible equity estimates
    /// (useful for tests and the calibration phase, Section 7);
    /// `None` draws a fresh seed each call.
    pub seed: Option<u64>,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        MonteCarloConfig {
            trials: 2_000,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThresholdConfig {
    /// Equity (0.0-1.0) at or above which L3 raises, at the fixed
    /// `raise_size` -- real win-probability now, not L1's ordinal
    /// percentile proxy.
    pub raise_above_equity: f64,
    /// Fixed sizing used for every L3 raise -- still no sizing reasoning
    /// at this level; that's explicitly L4's job ("basic bet-sizing
    /// theory").
    pub raise_size: PotFraction,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        // Placeholder (Sprint 9 calibration target, Section 7).
        ThresholdConfig {
            raise_above_equity: 0.65,
            raise_size: PotFraction::Half,
        }
    }
}

/// All cards from a fresh 52-card deck not already accounted for by the
/// agent's own hole cards or the board dealt so far.
fn remaining_deck(hole: [Card; 2], board: &[Card]) -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for suit in 0..4u8 {
        for rank in 2..=14u8 {
            let card = Card { rank, suit };
            if card == hole[0] || card == hole[1] || board.contains(&card) {
                continue;
            }
            deck.push(card);
        }
    }
    deck
}

/// Monte Carlo equity estimate: fraction of `trials` the hand wins
/// (ties count as 0.5), sampling a uniformly random 2-card opponent hand
/// and uniformly random remaining board cards each trial.
pub fn estimate_equity(state: &MathEngineState, config: &MonteCarloConfig) -> f64 {
    let remaining = remaining_deck(state.hole_cards, &state.board);
    let board_needed = 5 - state.board.len();
    let trials = config.trials.max(1);

    let mut rng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(rand::random::<u64>()),
    };

    let mut wins = 0.0f64;

    for _ in 0..trials {
        let mut shuffled = remaining.clone();
        shuffled.shuffle(&mut rng);

        let opp_hole = [shuffled[0], shuffled[1]];
        let mut full_board = state.board.clone();
        full_board.extend_from_slice(&shuffled[2..2 + board_needed]);

        let seven_own: [Card; 7] = [
            state.hole_cards[0],
            state.hole_cards[1],
            full_board[0],
            full_board[1],
            full_board[2],
            full_board[3],
            full_board[4],
        ];
        let seven_opp: [Card; 7] = [
            opp_hole[0],
            opp_hole[1],
            full_board[0],
            full_board[1],
            full_board[2],
            full_board[3],
            full_board[4],
        ];

        let own_rank = evaluate_hand(&seven_own);
        let opp_rank = evaluate_hand(&seven_opp);

        match own_rank.cmp(&opp_rank) {
            std::cmp::Ordering::Greater => wins += 1.0,
            std::cmp::Ordering::Equal => wins += 0.5,
            std::cmp::Ordering::Less => {}
        }
    }

    wins / trials as f64
}

fn required_equity(amount_to_call: u64, pot_after_call: u64) -> f64 {
    if pot_after_call == 0 {
        0.0
    } else {
        amount_to_call as f64 / pot_after_call as f64
    }
}

/// Level 3 recommendation (Section 4.1): real Monte Carlo equity drives
/// BOTH the raise threshold and the pot-odds call/fold boundary -- no
/// remaining use of L1's percentile proxy anywhere in this level.
pub fn recommend(
    state: &MathEngineState,
    threshold_config: &ThresholdConfig,
    mc_config: &MonteCarloConfig,
) -> RecommendedAction {
    let equity = estimate_equity(state, mc_config);

    if equity >= threshold_config.raise_above_equity {
        return RecommendedAction::Raise(threshold_config.raise_size);
    }

    if state.bet_to_call == 0 {
        return RecommendedAction::CheckOrCall;
    }

    let pot_after_call = state.pot_size.saturating_add(state.bet_to_call);
    let req_equity = required_equity(state.bet_to_call, pot_after_call);

    if equity >= req_equity {
        RecommendedAction::CheckOrCall
    } else {
        RecommendedAction::Fold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Street;

    fn c(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    fn preflop_state(hole: [Card; 2], pot_size: u64, bet_to_call: u64) -> MathEngineState {
        MathEngineState {
            hole_cards: hole,
            board: Vec::new(),
            street: Street::Preflop,
            pot_size,
            own_stack: 100_000,
            opponent_stack: 100_000,
            bet_to_call,
            is_button: true,
        }
    }

    // ---- remaining_deck: excludes own hole + board, nothing else ----

    #[test]
    fn remaining_deck_excludes_hole_and_board_cards() {
        let hole = [c(14, 0), c(14, 1)];
        let board = vec![c(2, 2), c(5, 3)];
        let remaining = remaining_deck(hole, &board);

        assert_eq!(remaining.len(), 52 - 2 - 2);
        assert!(!remaining.contains(&c(14, 0)));
        assert!(!remaining.contains(&c(14, 1)));
        assert!(!remaining.contains(&c(2, 2)));
        assert!(!remaining.contains(&c(5, 3)));
    }

    // ---- equity estimate: sane bounds, reproducibility ----

    #[test]
    fn equity_is_always_between_zero_and_one() {
        let state = preflop_state([c(7, 0), c(2, 1)], 150, 100);
        let mc = MonteCarloConfig {
            trials: 500,
            seed: Some(1),
        };
        let equity = estimate_equity(&state, &mc);
        assert!((0.0..=1.0).contains(&equity));
    }

    #[test]
    fn pocket_aces_beats_seven_deuce_in_expected_equity() {
        // Famous, well-established result: AA is a big preflop favorite
        // over 72o. Generous trial count + wide margin keeps this robust
        // to Monte Carlo variance rather than asserting a precise number.
        let aa = preflop_state([c(14, 0), c(14, 1)], 150, 100);
        let seven_deuce = preflop_state([c(7, 2), c(2, 3)], 150, 100);
        let mc = MonteCarloConfig {
            trials: 3_000,
            seed: Some(42),
        };

        let aa_equity = estimate_equity(&aa, &mc);
        let seven_deuce_equity = estimate_equity(&seven_deuce, &mc);

        assert!(aa_equity > 0.70, "AA equity too low: {aa_equity}");
        assert!(
            seven_deuce_equity < 0.50,
            "72o equity too high: {seven_deuce_equity}"
        );
        assert!(aa_equity > seven_deuce_equity);
    }

    #[test]
    fn same_seed_gives_identical_equity_estimate() {
        let state = preflop_state([c(11, 0), c(11, 1)], 150, 100);
        let mc = MonteCarloConfig {
            trials: 1_000,
            seed: Some(7),
        };

        let e1 = estimate_equity(&state, &mc);
        let e2 = estimate_equity(&state, &mc);

        assert_eq!(e1, e2);
    }

    #[test]
    fn river_with_full_board_still_estimates_cleanly() {
        // No board cards left to sample -- trials vary only the
        // opponent's hole cards.
        let state = MathEngineState {
            hole_cards: [c(9, 0), c(9, 1)],
            board: vec![c(9, 2), c(9, 3), c(2, 0), c(5, 1), c(7, 2)], // quad 9s
            street: Street::River,
            pot_size: 1000,
            own_stack: 100_000,
            opponent_stack: 100_000,
            bet_to_call: 500,
            is_button: true,
        };
        let mc = MonteCarloConfig {
            trials: 1_000,
            seed: Some(3),
        };
        let equity = estimate_equity(&state, &mc);

        // Quad 9s on the river loses to essentially nothing possible from
        // the remaining deck (only quad kings/aces etc. aren't possible
        // given the board) -- equity should be at or extremely close to 1.0.
        assert!(equity > 0.99, "quads equity unexpectedly low: {equity}");
    }

    // ---- action mapping: total equity-driven upgrade, raise + call/fold ----

    #[test]
    fn raise_branch_is_equity_driven_not_percentile_driven() {
        let state = preflop_state([c(14, 0), c(14, 1)], 150, 100);
        let mc = MonteCarloConfig {
            trials: 1_000,
            seed: Some(9),
        };
        // Deliberately low cutoff so AA's real equity clears it --
        // proves the branch reads `equity`, not L1's percentile.
        let threshold = ThresholdConfig {
            raise_above_equity: 0.05,
            raise_size: PotFraction::Half,
        };

        assert_eq!(
            recommend(&state, &threshold, &mc),
            RecommendedAction::Raise(threshold.raise_size)
        );
    }

    #[test]
    fn pot_odds_widening_still_present_at_l3_using_real_equity() {
        // 72o: real equity is realistically ~0.25-0.40 against a random
        // hand, well above a near-zero required_equity from a tiny bet
        // into a huge pot -- so even a raise cutoff of 0.99 (never
        // triggers raise) should still resolve to a call, not a fold.
        let state = preflop_state([c(7, 2), c(2, 3)], 10_000, 10);
        let mc = MonteCarloConfig {
            trials: 2_000,
            seed: Some(11),
        };
        let threshold = ThresholdConfig {
            raise_above_equity: 0.99,
            raise_size: PotFraction::Half,
        };

        assert_eq!(
            recommend(&state, &threshold, &mc),
            RecommendedAction::CheckOrCall
        );
    }

    #[test]
    fn weak_hand_folds_when_required_equity_exceeds_real_equity() {
        // Same weak hand, but now facing a massive overbet relative to
        // the pot -- required_equity should exceed 72o's real equity.
        let state = preflop_state([c(7, 2), c(2, 3)], 100, 5_000);
        let mc = MonteCarloConfig {
            trials: 2_000,
            seed: Some(11),
        };
        let threshold = ThresholdConfig {
            raise_above_equity: 0.99,
            raise_size: PotFraction::Half,
        };

        assert_eq!(recommend(&state, &threshold, &mc), RecommendedAction::Fold);
    }

    #[test]
    fn no_bet_to_call_short_circuits_to_check() {
        let state = preflop_state([c(7, 2), c(2, 3)], 500, 0);
        let mc = MonteCarloConfig {
            trials: 500,
            seed: Some(1),
        };
        let threshold = ThresholdConfig::default();

        assert_eq!(
            recommend(&state, &threshold, &mc),
            RecommendedAction::CheckOrCall
        );
    }
}
