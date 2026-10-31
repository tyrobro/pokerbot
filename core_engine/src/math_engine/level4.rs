//! Math engine Level 4 (Section 4.1): "Range-vs-range equity (opponent
//! range estimated from observed action, not assumed random),
//! position-aware, basic bet-sizing theory."
//!
//! Confirmed with PRD author (dependency-inversion approach): range
//! estimation reads a generic `ObservedActions` interface rather than the
//! real Section 4.4 opponent-memory object, which doesn't exist until
//! Sprint 9. Sprint 7 tests this against hand-built fixtures; Sprint 9
//! re-points it at the real object with no behavior change at this
//! call site, then re-validates against a real multi-hand pairing
//! (PRD author's explicit Sprint 9 DoD addition).
//!
//! Range-narrowing design note (Sprint 7 scope, not independently
//! PRD-reviewed -- flagged like other implementation-level choices): this
//! reuses L1's Chen-score heuristic on raw hole cards, regardless of
//! street, as the range-strength filter. Real postflop range x
//! board-texture reasoning is solver-level work Section 4.1 reserves for
//! L5 (CFR) -- filtering on Chen score alone keeps L4 "basic" rather than
//! smuggling in L5's job, same principle the PRD author used to scope L2
//! and L3.
//!
//! With empty `ObservedActions` (no current-hand opponent raise, no match
//! history), range-narrowing is a no-op and L4 behaves identically to
//! L3's uniform-random-range assumption -- the correct degenerate case
//! for a pairing's very first hands, before any read exists yet.

use super::level1::chen_score;
use super::{
    remaining_deck, ActionRecord, MatchHistoryEntry, MathEngineState, ObservedActions,
    RecommendedAction,
};
use crate::betting::{ActionType, PotFraction};
use crate::card::Card;
use crate::hand_eval::evaluate_hand;
use crate::math_engine::level3::MonteCarloConfig;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Debug, Clone, Copy)]
pub struct RangeConfig {
    /// Fraction of weakest combos (by Chen percentile) excluded from the
    /// opponent's range when THIS hand shows an opponent raise -- a
    /// strong, recent signal.
    pub tight_exclude_percentile: f64,
    /// Narrower exclusion applied when there's some match history but no
    /// current-hand raise signal, and the opponent hasn't shown an
    /// overall loose/aggressive tendency.
    pub passive_exclude_percentile: f64,
    /// Opponent's overall raise frequency (across current hand + match
    /// history) at or above this is treated as loose/aggressive -> no
    /// narrowing (their range is already wide, same as uniform).
    pub loose_aggression_frequency_threshold: f64,
}

impl Default for RangeConfig {
    fn default() -> Self {
        // Placeholders (Sprint 9 calibration target, Section 7).
        RangeConfig {
            tight_exclude_percentile: 0.50,
            passive_exclude_percentile: 0.20,
            loose_aggression_frequency_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThresholdConfig {
    pub raise_above_equity: f64,
    /// Equity margin above `raise_above_equity` at or beyond which the
    /// strong (larger) raise size is used instead of the moderate one --
    /// basic bet-sizing theory: respond to how far equity clears the bar,
    /// not just whether it clears it.
    pub strong_margin: f64,
    pub moderate_raise_size: PotFraction,
    pub strong_raise_size: PotFraction,
    /// Position-aware (Section 4.1): on the button, bump the chosen size
    /// up one tier (Quarter->Half->ThreeQuarter->Full, capped at Full).
    pub button_size_bump: bool,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        // Placeholders (Sprint 9 calibration target, Section 7).
        ThresholdConfig {
            raise_above_equity: 0.60,
            strong_margin: 0.15,
            moderate_raise_size: PotFraction::Half,
            strong_raise_size: PotFraction::Full,
            button_size_bump: true,
        }
    }
}

fn bump_size(size: PotFraction) -> PotFraction {
    match size {
        PotFraction::Quarter => PotFraction::Half,
        PotFraction::Half => PotFraction::ThreeQuarter,
        PotFraction::ThreeQuarter => PotFraction::Full,
        PotFraction::Full => PotFraction::Full,
    }
}

pub(crate) fn choose_raise_size(
    equity: f64,
    config: &ThresholdConfig,
    is_button: bool,
) -> PotFraction {
    let margin = equity - config.raise_above_equity;
    let mut size = if margin >= config.strong_margin {
        config.strong_raise_size
    } else {
        config.moderate_raise_size
    };
    if config.button_size_bump && is_button {
        size = bump_size(size);
    }
    size
}

fn is_aggressive(action_type: ActionType) -> bool {
    matches!(action_type, ActionType::Raise | ActionType::AllIn)
}

fn current_hand_has_opponent_raise(observed: &ObservedActions) -> bool {
    observed
        .current_hand_actions
        .iter()
        .any(|a: &ActionRecord| is_aggressive(a.action_type))
}

/// Opponent's raise frequency across everything observed (this hand +
/// match history). `None` if there's no data at all yet.
fn aggression_frequency(observed: &ObservedActions) -> Option<f64> {
    let mut raises = 0u32;
    let mut total = 0u32;

    for a in &observed.current_hand_actions {
        total += 1;
        if is_aggressive(a.action_type) {
            raises += 1;
        }
    }
    for entry in &observed.match_history {
        for a in &entry.actions {
            total += 1;
            if is_aggressive(a.action_type) {
                raises += 1;
            }
        }
    }

    if total == 0 {
        None
    } else {
        Some(raises as f64 / total as f64)
    }
}

/// How much of the weakest end of the range to exclude, given what's been
/// observed. 0.0 = no narrowing (full uniform range, same as L3).
fn exclude_percentile_for(observed: &ObservedActions, config: &RangeConfig) -> f64 {
    if current_hand_has_opponent_raise(observed) {
        return config.tight_exclude_percentile;
    }
    match aggression_frequency(observed) {
        Some(freq) if freq >= config.loose_aggression_frequency_threshold => 0.0,
        Some(_) => config.passive_exclude_percentile,
        None => 0.0, // no data at all -> fall back to L3's uniform assumption
    }
}

/// All 2-card combos from `remaining`, filtered to the top
/// `(1 - exclude_percentile)` by Chen-score percentile. Never returns an
/// empty set -- falls back to the full uniform combo set if filtering
/// would otherwise eliminate everything (a degenerate-but-safe guard).
fn opponent_range_combos(remaining: &[Card], exclude_percentile: f64) -> Vec<(Card, Card)> {
    let mut all_combos = Vec::new();
    for i in 0..remaining.len() {
        for j in (i + 1)..remaining.len() {
            all_combos.push((remaining[i], remaining[j]));
        }
    }

    if exclude_percentile <= 0.0 {
        return all_combos;
    }

    let filtered: Vec<(Card, Card)> = all_combos
        .iter()
        .copied()
        .filter(|&(a, b)| {
            let percentile = (chen_score([a, b]) / 20.0).clamp(0.0, 1.0);
            percentile >= exclude_percentile
        })
        .collect();

    if filtered.is_empty() {
        all_combos
    } else {
        filtered
    }
}

/// Monte Carlo equity against the NARROWED opponent range (built from
/// `observed`), rather than L3's fully uniform range -- same simulation
/// machinery, different input distribution (per PRD author's confirmed
/// L3->L4 relationship).
pub fn estimate_range_equity(
    state: &MathEngineState,
    observed: &ObservedActions,
    range_config: &RangeConfig,
    mc_config: &MonteCarloConfig,
) -> f64 {
    let remaining = remaining_deck(state.hole_cards, &state.board);
    let exclude_percentile = exclude_percentile_for(observed, range_config);
    let combos = opponent_range_combos(&remaining, exclude_percentile);
    let board_needed = 5 - state.board.len();
    let trials = mc_config.trials.max(1);

    let mut rng = match mc_config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(rand::random::<u64>()),
    };

    let mut wins = 0.0f64;

    for _ in 0..trials {
        let &(opp1, opp2) = combos.choose(&mut rng).expect("combos is never empty");

        let mut board_pool: Vec<Card> = remaining
            .iter()
            .copied()
            .filter(|&c| c != opp1 && c != opp2)
            .collect();
        board_pool.shuffle(&mut rng);

        let mut full_board = state.board.clone();
        full_board.extend_from_slice(&board_pool[0..board_needed]);

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
            opp1,
            opp2,
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

/// Level 4 recommendation (Section 4.1): range-vs-range equity (not
/// assumed-random), position-aware basic bet-sizing. Fold/call boundary
/// still pot-odds-driven, same shape as L3 -- only the equity SOURCE and
/// the raise SIZING changed at this level.
pub fn recommend(
    state: &MathEngineState,
    observed: &ObservedActions,
    threshold_config: &ThresholdConfig,
    range_config: &RangeConfig,
    mc_config: &MonteCarloConfig,
) -> RecommendedAction {
    let equity = estimate_range_equity(state, observed, range_config, mc_config);

    if equity >= threshold_config.raise_above_equity {
        let size = choose_raise_size(equity, threshold_config, state.is_button);
        return RecommendedAction::Raise(size);
    }

    if state.bet_to_call == 0 {
        return RecommendedAction::CheckOrCall;
    }

    let pot_after_call = state.pot_size.saturating_add(state.bet_to_call);
    let req_equity = if pot_after_call == 0 {
        0.0
    } else {
        state.bet_to_call as f64 / pot_after_call as f64
    };

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

    fn preflop_state(
        hole: [Card; 2],
        pot_size: u64,
        bet_to_call: u64,
        is_button: bool,
    ) -> MathEngineState {
        MathEngineState {
            hole_cards: hole,
            board: Vec::new(),
            street: Street::Preflop,
            pot_size,
            own_stack: 100_000,
            opponent_stack: 100_000,
            bet_to_call,
            is_button,
        }
    }

    fn opponent_raise(street: Street, amount: u64, pot_at_time: u64) -> ActionRecord {
        ActionRecord {
            street,
            action_type: ActionType::Raise,
            amount,
            pot_size_at_time: pot_at_time,
        }
    }
    fn opponent_call(street: Street, amount: u64, pot_at_time: u64) -> ActionRecord {
        ActionRecord {
            street,
            action_type: ActionType::Call,
            amount,
            pot_size_at_time: pot_at_time,
        }
    }

    // ---- degenerate case: empty data -> no narrowing (same as L3) ----

    #[test]
    fn empty_observed_actions_means_no_narrowing() {
        let observed = ObservedActions::default();
        let config = RangeConfig::default();
        assert_eq!(exclude_percentile_for(&observed, &config), 0.0);

        let remaining = remaining_deck([c(14, 0), c(14, 1)], &[]);
        let combos = opponent_range_combos(&remaining, 0.0);
        let total_combos = remaining.len() * (remaining.len() - 1) / 2;
        assert_eq!(combos.len(), total_combos);
    }

    // ---- range narrowing responds to observed action ----

    #[test]
    fn current_hand_raise_triggers_tight_narrowing() {
        let observed = ObservedActions {
            current_hand_actions: vec![opponent_raise(Street::Preflop, 300, 150)],
            match_history: vec![],
        };
        let config = RangeConfig::default();
        assert_eq!(
            exclude_percentile_for(&observed, &config),
            config.tight_exclude_percentile
        );
    }

    #[test]
    fn loose_match_history_overrides_default_narrowing() {
        // 4 of 5 past actions were raises -> frequency 0.8, above the
        // default 0.5 loose threshold -> no narrowing, despite having
        // history (a loose/aggressive opponent's range is already wide).
        let history_actions = vec![
            opponent_raise(Street::Preflop, 300, 150),
            opponent_raise(Street::Flop, 400, 600),
            opponent_raise(Street::Turn, 500, 1200),
            opponent_raise(Street::River, 600, 2000),
            opponent_call(Street::Preflop, 100, 150),
        ];
        let observed = ObservedActions {
            current_hand_actions: vec![], // no raise THIS hand
            match_history: vec![MatchHistoryEntry {
                actions: history_actions,
                opponent_hole_cards: None,
                showdown_reached: true,
                opponent_won: Some(true),
                amount_won_or_lost: 500,
            }],
        };
        let config = RangeConfig::default();
        assert_eq!(exclude_percentile_for(&observed, &config), 0.0);
    }

    #[test]
    fn passive_match_history_with_no_current_raise_applies_mild_narrowing() {
        let history_actions = vec![
            opponent_call(Street::Preflop, 100, 150),
            opponent_call(Street::Flop, 100, 350),
        ];
        let observed = ObservedActions {
            current_hand_actions: vec![],
            match_history: vec![MatchHistoryEntry {
                actions: history_actions,
                opponent_hole_cards: None,
                showdown_reached: false,
                opponent_won: None,
                amount_won_or_lost: 0,
            }],
        };
        let config = RangeConfig::default();
        assert_eq!(
            exclude_percentile_for(&observed, &config),
            config.passive_exclude_percentile
        );
    }

    #[test]
    fn tight_narrowing_excludes_the_weakest_combos() {
        let remaining = remaining_deck([c(14, 0), c(14, 1)], &[]);
        let narrowed = opponent_range_combos(&remaining, 0.50);
        let uniform = opponent_range_combos(&remaining, 0.0);

        assert!(narrowed.len() < uniform.len());
        for &(a, b) in &narrowed {
            let percentile = (chen_score([a, b]) / 20.0).clamp(0.0, 1.0);
            assert!(percentile >= 0.50);
        }
    }

    // ---- equity estimation: bounds + the actual proof that narrowing
    // changes the result ----

    #[test]
    fn range_equity_is_always_between_zero_and_one() {
        let state = preflop_state([c(12, 0), c(11, 1)], 150, 100, true);
        let observed = ObservedActions::default();
        let equity = estimate_range_equity(
            &state,
            &observed,
            &RangeConfig::default(),
            &MonteCarloConfig {
                trials: 500,
                seed: Some(1),
            },
        );
        assert!((0.0..=1.0).contains(&equity));
    }

    #[test]
    fn tight_opponent_range_lowers_a_mediocre_hands_equity_vs_uniform() {
        // KQo is a decent-but-not-great hand. Against a uniform random
        // range (L3's assumption), it does reasonably. Against a range
        // narrowed to only strong hands (L4's response to an observed
        // opponent raise), its equity should be measurably LOWER --
        // excluding weak villain combos raises the opponent's average
        // strength, which is the whole point of range-vs-range equity.
        let hole = [c(14, 0), c(13, 1)];
        let state = preflop_state(hole, 150, 100, false);

        let uniform_equity = crate::math_engine::level3::estimate_equity(
            &state,
            &MonteCarloConfig {
                trials: 3_000,
                seed: Some(5),
            },
        );

        let observed = ObservedActions {
            current_hand_actions: vec![opponent_raise(Street::Preflop, 300, 150)],
            match_history: vec![],
        };
        let tight_equity = estimate_range_equity(
            &state,
            &observed,
            &RangeConfig::default(),
            &MonteCarloConfig {
                trials: 3_000,
                seed: Some(5),
            },
        );

        assert!(
            tight_equity < uniform_equity,
            "tight range equity ({tight_equity}) should be lower than uniform ({uniform_equity})"
        );
    }

    // ---- basic bet-sizing theory: margin-driven, position-aware ----

    #[test]
    fn strong_margin_selects_the_strong_raise_size() {
        let config = ThresholdConfig {
            raise_above_equity: 0.60,
            strong_margin: 0.15,
            moderate_raise_size: PotFraction::Quarter,
            strong_raise_size: PotFraction::Full,
            button_size_bump: false,
        };
        // equity 0.80, margin = 0.80-0.60 = 0.20 >= strong_margin(0.15).
        assert_eq!(choose_raise_size(0.80, &config, false), PotFraction::Full);
    }

    #[test]
    fn moderate_margin_selects_the_moderate_raise_size() {
        let config = ThresholdConfig {
            raise_above_equity: 0.60,
            strong_margin: 0.15,
            moderate_raise_size: PotFraction::Quarter,
            strong_raise_size: PotFraction::Full,
            button_size_bump: false,
        };
        // equity 0.65, margin = 0.05 < strong_margin(0.15).
        assert_eq!(
            choose_raise_size(0.65, &config, false),
            PotFraction::Quarter
        );
    }

    #[test]
    fn button_bump_steps_the_size_up_one_tier() {
        let config = ThresholdConfig {
            raise_above_equity: 0.60,
            strong_margin: 0.15,
            moderate_raise_size: PotFraction::Quarter,
            strong_raise_size: PotFraction::Full,
            button_size_bump: true,
        };
        // Moderate-margin case: off the button -> Quarter; on the button
        // -> bumped to Half. Isolates the position-aware behavior.
        assert_eq!(
            choose_raise_size(0.65, &config, false),
            PotFraction::Quarter
        );
        assert_eq!(choose_raise_size(0.65, &config, true), PotFraction::Half);
    }

    #[test]
    fn button_bump_caps_at_full_pot() {
        let config = ThresholdConfig {
            raise_above_equity: 0.60,
            strong_margin: 0.15,
            moderate_raise_size: PotFraction::Full,
            strong_raise_size: PotFraction::Full,
            button_size_bump: true,
        };
        assert_eq!(choose_raise_size(0.65, &config, true), PotFraction::Full);
    }

    // ---- recommend(): end-to-end threshold branching ----

    #[test]
    fn premium_hand_raises() {
        let state = preflop_state([c(14, 0), c(14, 1)], 150, 100, true);
        let observed = ObservedActions::default();
        let threshold = ThresholdConfig::default();
        let range = RangeConfig::default();
        let mc = MonteCarloConfig {
            trials: 1_000,
            seed: Some(21),
        };

        let result = recommend(&state, &observed, &threshold, &range, &mc);
        assert!(matches!(result, RecommendedAction::Raise(_)));
    }

    #[test]
    fn weak_hand_with_great_pot_odds_calls_not_folds() {
        let state = preflop_state([c(7, 2), c(2, 3)], 10_000, 10, false);
        let observed = ObservedActions::default();
        let threshold = ThresholdConfig {
            raise_above_equity: 0.99,
            ..ThresholdConfig::default()
        };
        let range = RangeConfig::default();
        let mc = MonteCarloConfig {
            trials: 2_000,
            seed: Some(22),
        };

        assert_eq!(
            recommend(&state, &observed, &threshold, &range, &mc),
            RecommendedAction::CheckOrCall
        );
    }

    #[test]
    fn weak_hand_facing_a_huge_overbet_folds() {
        let state = preflop_state([c(7, 2), c(2, 3)], 100, 5_000, false);
        let observed = ObservedActions::default();
        let threshold = ThresholdConfig {
            raise_above_equity: 0.99,
            ..ThresholdConfig::default()
        };
        let range = RangeConfig::default();
        let mc = MonteCarloConfig {
            trials: 2_000,
            seed: Some(22),
        };

        assert_eq!(
            recommend(&state, &observed, &threshold, &range, &mc),
            RecommendedAction::Fold
        );
    }

    #[test]
    fn no_bet_to_call_short_circuits_to_check() {
        let state = preflop_state([c(7, 2), c(2, 3)], 500, 0, false);
        let observed = ObservedActions::default();
        let threshold = ThresholdConfig::default();
        let range = RangeConfig::default();
        let mc = MonteCarloConfig {
            trials: 500,
            seed: Some(1),
        };

        assert_eq!(
            recommend(&state, &observed, &threshold, &range, &mc),
            RecommendedAction::CheckOrCall
        );
    }
}
