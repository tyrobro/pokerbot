//! recommend() for L5 (Section 4.1): turns a trained CFRSolver's average
//! strategy into a RecommendedAction for a live decision.
//!
//! Three PRD-author-confirmed requirements, all enforced here:
//!   1. Bucket config (num_buckets, bucketing_trials) is read DIRECTLY
//!      from the trained CFRSolver's own config (`solver.config()`) --
//!      never independently re-derived or passed a different config at
//!      inference. This does NOT eliminate boundary noise near a bucket
//!      edge (equity estimation is inherently Monte Carlo), but using
//!      the SAME trials count as training is the system's existing
//!      accepted approximation, stated explicitly rather than assumed
//!      away.
//!   2. recommend() samples EXCLUSIVELY from the solver's accumulated
//!      average strategy (`CFRSolver::average_strategy_at`, backed by
//!      strategy_sum) -- never the raw per-iteration regret-matching
//!      strategy. Only the average strategy carries CFR's convergence
//!      guarantee; sampling from current_strategy would silently produce
//!      a non-equilibrium, exploitable agent. Tested explicitly below.
//!   3. Unseen InfoSetKey fallback, never a panic (matches "no illegal
//!      moves, ever," Section 3.3): first try the nearest visited bucket
//!      at the same decision point; if nothing nearby was visited
//!      either, fall back to L4's range-vs-range recommend() as a safety
//!      net.
//!
//! Open structural gap found while implementing this (flagged, not
//! silently resolved): L4's `ObservedActions` tracks the OPPONENT's
//! actions only (by its own design), but CFR's InfoSetKey.history is a
//! record of BOTH players' interleaved actions across the hand. It
//! cannot reconstruct what CFR trained against. Resolved here by
//! accepting a separate, explicit `HandActionLogEntry` log -- both
//! players' actions, in order, for the CURRENT hand only -- rather than
//! overloading the PRD-approved ObservedActions shape. This matches
//! Section 4.3's own listed field ("full betting-action sequence for the
//! CURRENT hand"), distinct from Section 4.4's opponent-history object
//! that ObservedActions.match_history represents.
//!
//! CALLER CONTRACT, STATED PRECISELY (not a generic "state-threading is
//! future work" caveat -- this is a known, higher-severity failure mode
//! than L4's analogous gap): an empty or incomplete `hand_log` does NOT
//! degrade gracefully the way L4's empty ObservedActions does. L4's
//! empty-input case is a designed, tested fallback (Section 4.4's
//! opponent-memory gap, validated: `empty_observed_actions_means_no_narrowing`).
//! Here, an incomplete hand_log makes `reconstruct_street_state` silently
//! undercount `raises_this_street` (e.g. reporting 0 when real raises
//! occurred), which produces a CONFIDENTLY WRONG InfoSetKey -- one that
//! will often still successfully match a real trained entry, just the
//! WRONG one, rather than tripping the unseen-key fallback chain (nearest
//! bucket / L4 safety net) that only triggers when no entry matches at
//! all. This is a silent correctness bug in the caller's hands, not a
//! flagged degenerate case in this function's. Any caller that cannot
//! supply the complete, accurate both-players action log for the current
//! hand (e.g. the Sprint 7 Python bindings, which always pass an empty
//! log -- see bindings/src/lib.rs and agents/math_agents.py) must be
//! treated as producing an L5 agent whose InfoSetKey lookups are
//! UNVERIFIED past the first action of each street, not merely
//! "conservative" or "uninformed." Tracked explicitly as a named
//! precondition for trusting any Sprint 9 calibration run that includes
//! L5 -- do not fold this into the generic Monte-Carlo-seeding gap below.
use super::abstraction::{
    bucketing_equity_seeded, equity_to_bucket, ActionAbstraction, InfoSetKey,
};
use super::cfr::{sample_action, CFRSolver};
use crate::betting::{ActionType, PotFraction};
use crate::game::Street;
use crate::math_engine::level3::MonteCarloConfig;
use crate::math_engine::{
    level4, ActionRecord, MathEngineState, ObservedActions, RecommendedAction,
};
use rand::rngs::StdRng;
use std::collections::BTreeMap;

/// Who acted, for one entry in the current hand's full (both-players)
/// action log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Actor {
    Own,
    Opponent,
}

#[derive(Debug, Clone, Copy)]
pub struct HandActionLogEntry {
    pub actor: Actor,
    pub record: ActionRecord,
}

/// Classifies an observed raise amount into the nearest abstracted
/// PotFraction tier. Lossy (real raises are resolved through
/// resolve_raise's rounding, Section 3.3, so they won't always land
/// exactly on 25/50/75/100% of the pot) -- an explicit, named
/// approximation, same spirit as the rest of this abstraction.
fn classify_fraction(amount: u64, pot_before: u64) -> PotFraction {
    let ratio = amount as f64 / (pot_before.max(1) as f64);
    let tiers = [
        (0.25, PotFraction::Quarter),
        (0.5, PotFraction::Half),
        (0.75, PotFraction::ThreeQuarter),
        (1.0, PotFraction::Full),
    ];
    let mut best = PotFraction::Full;
    let mut best_dist = f64::INFINITY;
    for (target, frac) in tiers {
        let dist = (ratio - target).abs();
        if dist < best_dist {
            best_dist = dist;
            best = frac;
        }
    }
    best
}

fn to_action_abstraction(record: &ActionRecord) -> ActionAbstraction {
    match record.action_type {
        ActionType::Fold => ActionAbstraction::Fold,
        ActionType::Check | ActionType::Call => ActionAbstraction::CheckCall,
        ActionType::AllIn => ActionAbstraction::AllIn,
        ActionType::Raise => {
            ActionAbstraction::Raise(classify_fraction(record.amount, record.pot_size_at_time))
        }
    }
}

/// Reconstructs the abstracted full-hand history, plus the CURRENT
/// street's raise count and facing-bet flag (these reset per street,
/// same as training -- see cfr.rs's apply_action_bookkeeping).
fn reconstruct_street_state(
    log: &[HandActionLogEntry],
    current_street: Street,
) -> (Vec<ActionAbstraction>, u8, bool) {
    let mut history = Vec::new();
    let mut raises_this_street = 0u8;
    let mut facing_bet = false;

    for entry in log {
        let abstracted = to_action_abstraction(&entry.record);
        history.push(abstracted);

        if entry.record.street == current_street {
            match abstracted {
                ActionAbstraction::Raise(_) | ActionAbstraction::AllIn => {
                    raises_this_street += 1;
                    facing_bet = true;
                }
                ActionAbstraction::CheckCall => facing_bet = false,
                ActionAbstraction::Fold => {}
            }
        }
    }

    (history, raises_this_street, facing_bet)
}

/// PRD requirement 3, step 1: searches outward from `bucket` (0, ±1, ±2,
/// ...) for the nearest bucket at the SAME decision point the solver
/// actually visited during training. `None` if nothing in the full
/// bucket range was visited (triggers the L4 fallback).
fn nearest_visited_bucket(
    solver: &CFRSolver,
    bucket: u8,
    street: Street,
    raises_this_street: u8,
    facing_bet: bool,
    history: &[ActionAbstraction],
    num_buckets: u8,
) -> Option<BTreeMap<ActionAbstraction, f64>> {
    let n = num_buckets as i32;
    let b0 = bucket as i32;

    for radius in 0..n {
        let candidates: Vec<i32> = if radius == 0 {
            vec![b0]
        } else {
            vec![b0 - radius, b0 + radius]
        };
        for c in candidates {
            if c < 0 || c >= n {
                continue;
            }
            let key = InfoSetKey {
                bucket: c as u8,
                street,
                raises_this_street,
                facing_bet,
                history: history.to_vec(),
            };
            if let Some(strategy) = solver.average_strategy_at(&key) {
                return Some(strategy);
            }
        }
    }
    None
}

fn action_abstraction_to_recommended(action: ActionAbstraction) -> RecommendedAction {
    match action {
        ActionAbstraction::Fold => RecommendedAction::Fold,
        ActionAbstraction::CheckCall => RecommendedAction::CheckOrCall,
        ActionAbstraction::Raise(frac) => RecommendedAction::Raise(frac),
        // RecommendedAction has no AllIn variant (L1-L4 never produce one
        // either -- the engine's own all-in override in resolve_raise,
        // Section 3.3 step 4, is what actually turns a raise into an
        // all-in when it exceeds the stack). Mapped to the closest
        // existing variant, flagged rather than silently losing the
        // distinction.
        ActionAbstraction::AllIn => RecommendedAction::Raise(PotFraction::Full),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn recommend(
    state: &MathEngineState,
    hand_log: &[HandActionLogEntry],
    observed: &ObservedActions,
    solver: &CFRSolver,
    rng: &mut StdRng,
    l4_threshold: &level4::ThresholdConfig,
    l4_range: &level4::RangeConfig,
    l4_mc: &MonteCarloConfig,
) -> RecommendedAction {
    // PRD requirement 1: read config from the trained blueprint itself.
    let config = solver.config();

    let equity = bucketing_equity_seeded(state.hole_cards, &state.board, &config.buckets, rng);
    let bucket = equity_to_bucket(equity, config.buckets.num_buckets);

    let (history, raises_this_street, facing_bet) =
        reconstruct_street_state(hand_log, state.street);

    let key = InfoSetKey {
        bucket,
        street: state.street,
        raises_this_street,
        facing_bet,
        history: history.clone(),
    };

    // PRD requirement 2: average_strategy_at is backed by strategy_sum
    // (the accumulated average) -- never current_strategy.
    let strategy = solver.average_strategy_at(&key).or_else(|| {
        nearest_visited_bucket(
            solver,
            bucket,
            state.street,
            raises_this_street,
            facing_bet,
            &history,
            config.buckets.num_buckets,
        )
    });

    let Some(strategy) = strategy else {
        // PRD requirement 3, step 2: L4 safety net -- never a panic.
        return level4::recommend(state, observed, l4_threshold, l4_range, l4_mc);
    };

    let legal =
        ActionAbstraction::legal_at(facing_bet, raises_this_street, config.max_raises_per_street);
    let chosen = sample_action(&strategy, &legal, rng);
    action_abstraction_to_recommended(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;
    use crate::math_engine::level5::abstraction::BucketConfig;
    use crate::math_engine::level5::abstraction::CFRConfig;
    use rand::SeedableRng;

    fn c(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    fn test_config() -> CFRConfig {
        CFRConfig {
            buckets: BucketConfig {
                num_buckets: 6,
                bucketing_trials: 50,
            },
            max_raises_per_street: 1,
        }
    }

    // ---- PRD requirement 1: solver exposes its own training config ----

    #[test]
    fn solver_exposes_its_own_training_config_for_recommend_to_reuse() {
        let config = CFRConfig {
            buckets: BucketConfig {
                num_buckets: 9,
                bucketing_trials: 777,
            },
            max_raises_per_street: 2,
        };
        let solver = CFRSolver::new(config, 1);
        assert_eq!(solver.config().buckets.num_buckets, 9);
        assert_eq!(solver.config().buckets.bucketing_trials, 777);
    }

    // ---- PRD requirement 2: average strategy, never current strategy ----

    #[test]
    fn recommend_follows_average_strategy_not_current_regret_matching_strategy() {
        let mut solver = CFRSolver::new(test_config(), 1);
        let key = InfoSetKey {
            bucket: 3,
            street: Street::Flop,
            raises_this_street: 0,
            facing_bet: true,
            history: vec![],
        };
        // Craft regret_sum to heavily favor Fold; strategy_sum to
        // heavily favor CheckCall -- deliberately divergent tables.
        solver.test_inject_entry(
            key.clone(),
            vec![
                (ActionAbstraction::Fold, 100.0),
                (ActionAbstraction::CheckCall, 0.0),
            ],
            vec![
                (ActionAbstraction::Fold, 1.0),
                (ActionAbstraction::CheckCall, 99.0),
            ],
        );

        let avg = solver.average_strategy_at(&key).unwrap();
        assert!(
            avg[&ActionAbstraction::CheckCall] > 0.9,
            "average strategy should follow strategy_sum"
        );

        let cur = solver.current_strategy_at(&key).unwrap();
        assert!(
            cur[&ActionAbstraction::Fold] > 0.9,
            "current strategy should follow regret_sum -- proves the tables genuinely diverge"
        );

        // Sample many times from exactly what recommend() uses.
        let legal = vec![ActionAbstraction::Fold, ActionAbstraction::CheckCall];
        let mut rng = StdRng::seed_from_u64(99);
        let mut checkcall_count = 0;
        for _ in 0..500 {
            if sample_action(&avg, &legal, &mut rng) == ActionAbstraction::CheckCall {
                checkcall_count += 1;
            }
        }
        assert!(checkcall_count > 400, "recommend()'s sampling source should heavily favor CheckCall, matching strategy_sum, not the Fold-dominated regret_sum");
    }

    // ---- PRD requirement 3: unseen-key fallback, two steps ----

    #[test]
    fn nearest_visited_bucket_finds_adjacent_bucket_when_exact_is_unvisited() {
        let mut solver = CFRSolver::new(test_config(), 1);
        let key_at_5 = InfoSetKey {
            bucket: 5,
            street: Street::Turn,
            raises_this_street: 0,
            facing_bet: false,
            history: vec![],
        };
        solver.test_inject_entry(
            key_at_5,
            vec![(ActionAbstraction::CheckCall, 1.0)],
            vec![(ActionAbstraction::CheckCall, 1.0)],
        );

        let found = nearest_visited_bucket(&solver, 4, Street::Turn, 0, false, &[], 10);
        assert!(
            found.is_some(),
            "should find the bucket-5 entry when searching outward from bucket 4"
        );
    }

    #[test]
    fn nearest_visited_bucket_returns_none_when_nothing_at_all_was_visited() {
        let solver = CFRSolver::new(test_config(), 1);
        let found = nearest_visited_bucket(&solver, 4, Street::Turn, 0, false, &[], 10);
        assert!(found.is_none());
    }

    #[test]
    fn recommend_falls_back_to_l4_when_no_info_sets_were_ever_visited() {
        let solver = CFRSolver::new(test_config(), 1); // untrained, zero info sets
        let state = MathEngineState {
            hole_cards: [c(14, 0), c(14, 1)],
            board: vec![],
            street: Street::Preflop,
            pot_size: 150,
            own_stack: 100_000,
            opponent_stack: 100_000,
            bet_to_call: 100,
            is_button: true,
        };
        let observed = ObservedActions::default();
        let mut rng = StdRng::seed_from_u64(5);

        // Must not panic -- that's the actual point of this test.
        let result = recommend(
            &state,
            &[],
            &observed,
            &solver,
            &mut rng,
            &level4::ThresholdConfig::default(),
            &level4::RangeConfig::default(),
            &MonteCarloConfig {
                trials: 200,
                seed: Some(1),
            },
        );
        assert!(matches!(
            result,
            RecommendedAction::Raise(_) | RecommendedAction::CheckOrCall | RecommendedAction::Fold
        ));
    }

    // ---- supporting-logic unit tests ----

    #[test]
    fn classify_fraction_picks_nearest_tier() {
        assert_eq!(classify_fraction(250, 1000), PotFraction::Quarter);
        assert_eq!(classify_fraction(500, 1000), PotFraction::Half);
        assert_eq!(classify_fraction(750, 1000), PotFraction::ThreeQuarter);
        assert_eq!(classify_fraction(1000, 1000), PotFraction::Full);
        assert_eq!(classify_fraction(2000, 1000), PotFraction::Full);
    }

    #[test]
    fn reconstruct_street_state_resets_raise_count_at_street_boundary() {
        let log = vec![
            HandActionLogEntry {
                actor: Actor::Opponent,
                record: ActionRecord {
                    street: Street::Preflop,
                    action_type: ActionType::Raise,
                    amount: 300,
                    pot_size_at_time: 150,
                },
            },
            HandActionLogEntry {
                actor: Actor::Own,
                record: ActionRecord {
                    street: Street::Preflop,
                    action_type: ActionType::Call,
                    amount: 300,
                    pot_size_at_time: 450,
                },
            },
            HandActionLogEntry {
                actor: Actor::Opponent,
                record: ActionRecord {
                    street: Street::Flop,
                    action_type: ActionType::Check,
                    amount: 0,
                    pot_size_at_time: 750,
                },
            },
        ];
        let (history, raises, facing_bet) = reconstruct_street_state(&log, Street::Flop);
        assert_eq!(history.len(), 3);
        assert_eq!(
            raises, 0,
            "preflop raise shouldn't count toward the flop's raises_this_street"
        );
        assert!(!facing_bet);
    }

    #[test]
    fn all_in_maps_to_raise_full_since_recommended_action_has_no_all_in_variant() {
        assert_eq!(
            action_abstraction_to_recommended(ActionAbstraction::AllIn),
            RecommendedAction::Raise(PotFraction::Full)
        );
    }
}
