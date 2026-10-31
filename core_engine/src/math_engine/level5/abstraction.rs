//! Card bucketing (equity-percentile-based) and the abstracted game-tree
//! representation for L5's CFR solve.

use crate::betting::PotFraction;
use crate::card::Card;
use crate::game::Street;
use crate::hand_eval::evaluate_hand;
use crate::math_engine::level3::MonteCarloConfig;
use crate::math_engine::remaining_deck;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Debug, Clone, Copy)]
pub struct BucketConfig {
    /// Card buckets per street (PRD author default: 10).
    pub num_buckets: u8,
    /// Run-outs averaged per hand to get a potential-aware equity number
    /// for bucketing on streets with cards still to come (flop/turn).
    /// Implementation-level simplification (flagged, not PRD-specified):
    /// a cheap proxy for true potential-aware abstraction, reusing L3's
    /// Monte Carlo machinery rather than a dedicated clustering metric.
    pub bucketing_trials: u32,
}

impl Default for BucketConfig {
    fn default() -> Self {
        BucketConfig {
            num_buckets: 10,
            bucketing_trials: 200,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CFRConfig {
    pub buckets: BucketConfig,
    /// Raise-depth cap per street (PRD author default: 3) -- the actual
    /// tractability lever, NOT street-count reduction.
    pub max_raises_per_street: u8,
}

impl Default for CFRConfig {
    fn default() -> Self {
        CFRConfig {
            buckets: BucketConfig::default(),
            max_raises_per_street: 3,
        }
    }
}

/// Equity-percentile bucket index, 0 = weakest, `num_buckets - 1` =
/// strongest.
pub type Bucket = u8;

/// Estimates a potential-aware equity number for bucketing: on the river
/// (full board), this is plain showdown equity vs. a uniform random
/// opponent hand (reusing L3's exact method). On earlier streets, it's
/// averaged over `bucketing_trials` random run-outs so a hand's DRAW
/// potential (not just current made-hand strength) contributes --
/// otherwise a flush draw and bottom pair with similar current equity
/// would be bucketed identically, which is the known weakness
/// potential-aware abstraction exists to address.
pub fn bucketing_equity(hole: [Card; 2], board: &[Card], config: &BucketConfig) -> f64 {
    if board.len() == 5 {
        let mc = MonteCarloConfig {
            trials: config.bucketing_trials,
            seed: None,
        };
        let state = crate::math_engine::MathEngineState {
            hole_cards: hole,
            board: board.to_vec(),
            street: Street::River,
            pot_size: 0,
            own_stack: 0,
            opponent_stack: 0,
            bet_to_call: 0,
            is_button: false,
        };
        return crate::math_engine::level3::estimate_equity(&state, &mc);
    }

    let remaining = remaining_deck(hole, board);
    let board_needed = 5 - board.len();
    let mut rng = StdRng::seed_from_u64(rand::random::<u64>());
    let trials = config.bucketing_trials.max(1);
    let mut wins = 0.0f64;

    for _ in 0..trials {
        let mut shuffled = remaining.clone();
        shuffled.shuffle(&mut rng);

        let opp_hole = [shuffled[0], shuffled[1]];
        let mut full_board = board.to_vec();
        full_board.extend_from_slice(&shuffled[2..2 + board_needed]);

        let seven_own: [Card; 7] = [
            hole[0],
            hole[1],
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

        match evaluate_hand(&seven_own).cmp(&evaluate_hand(&seven_opp)) {
            std::cmp::Ordering::Greater => wins += 1.0,
            std::cmp::Ordering::Equal => wins += 0.5,
            std::cmp::Ordering::Less => {}
        }
    }

    wins / trials as f64
}

/// Maps an equity value (0.0-1.0) to a bucket index, 0 = weakest.
pub fn equity_to_bucket(equity: f64, num_buckets: u8) -> Bucket {
    let n = num_buckets.max(1) as f64;
    let idx = (equity * n).floor().clamp(0.0, n - 1.0);
    idx as Bucket
}

/// Per-street bucket boundaries, kept as a thin wrapper so future
/// per-street-specific bucketing logic (if ever needed) has a home
/// without changing every call site.
#[derive(Debug, Clone, Copy)]
pub struct StreetBuckets {
    pub num_buckets: u8,
}

impl StreetBuckets {
    pub fn bucket_for(&self, hole: [Card; 2], board: &[Card], config: &BucketConfig) -> Bucket {
        let equity = bucketing_equity(hole, board, config);
        equity_to_bucket(equity, self.num_buckets)
    }
}

/// The abstracted action space (Section 3.3's set exactly): fold,
/// check/call collapsed to one (legality already disambiguates which is
/// possible at a given node, same convention `game.rs` uses), and the 4
/// discrete raise/bet sizes plus all-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionAbstraction {
    Fold,
    CheckCall,
    Raise(PotFraction),
    AllIn,
}

impl ActionAbstraction {
    pub fn all() -> [ActionAbstraction; 6] {
        [
            ActionAbstraction::Fold,
            ActionAbstraction::CheckCall,
            ActionAbstraction::Raise(PotFraction::Quarter),
            ActionAbstraction::Raise(PotFraction::Half),
            ActionAbstraction::Raise(PotFraction::ThreeQuarter),
            ActionAbstraction::Raise(PotFraction::Full),
        ]
    }

    /// Legal actions at a node, given whether there's a bet to call and
    /// how many raises have already happened this street (raise-depth
    /// cap -- PRD author's confirmed tractability lever).
    pub fn legal_at(
        facing_bet: bool,
        raises_this_street: u8,
        max_raises: u8,
    ) -> Vec<ActionAbstraction> {
        let mut actions = Vec::new();
        if facing_bet {
            actions.push(ActionAbstraction::Fold);
        }
        actions.push(ActionAbstraction::CheckCall);

        if raises_this_street < max_raises {
            actions.push(ActionAbstraction::Raise(PotFraction::Quarter));
            actions.push(ActionAbstraction::Raise(PotFraction::Half));
            actions.push(ActionAbstraction::Raise(PotFraction::ThreeQuarter));
            actions.push(ActionAbstraction::Raise(PotFraction::Full));
        }
        // All-in is always available regardless of raise-depth cap --
        // matches Section 3.3: "exceeds remaining stack -> override to
        // all-in" always takes precedence, so a capped-out player can
        // still shove.
        actions.push(ActionAbstraction::AllIn);

        actions
    }
}

/// Identifies one information set for CFR: which player is deciding,
/// their card bucket, the street, and the abstracted betting history so
/// far this street (raise count + facing-bet state is enough to
/// reconstruct legality; full action-type sequence is kept for the
/// regret-table key so distinct lines aren't conflated).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InfoSetKey {
    pub bucket: Bucket,
    pub street: Street,
    pub raises_this_street: u8,
    pub facing_bet: bool,
    /// Abstracted action history across ALL streets so far this hand
    /// (not just the current street) -- distinguishes e.g. "checked flop
    /// then faces a turn bet" from "called a flop raise then faces a
    /// turn bet", which are different info sets even if `raises_this_
    /// street`/`facing_bet` happen to match.
    pub history: Vec<ActionAbstraction>,
}

/// One node in the abstracted betting tree. A thin, explicit
/// representation (rather than implicit recursion inside the CFR solver)
/// so the tree can be inspected/tested independently of the solve
/// algorithm itself.
#[derive(Debug, Clone)]
pub struct BettingNode {
    pub street: Street,
    pub raises_this_street: u8,
    pub facing_bet: bool,
    pub legal_actions: Vec<ActionAbstraction>,
}

impl BettingNode {
    pub fn new(
        street: Street,
        raises_this_street: u8,
        facing_bet: bool,
        config: &CFRConfig,
    ) -> Self {
        let legal_actions = ActionAbstraction::legal_at(
            facing_bet,
            raises_this_street,
            config.max_raises_per_street,
        );
        BettingNode {
            street,
            raises_this_street,
            facing_bet,
            legal_actions,
        }
    }
}

/// Enumerates every reachable `BettingNode` within one street under the
/// raise-depth cap -- used both to size the CFR regret tables up front
/// and to independently verify the tree stays bounded (tested below).
pub fn enumerate_street_nodes(street: Street, config: &CFRConfig) -> Vec<BettingNode> {
    let mut nodes = Vec::new();
    // raises_this_street ranges 0..=max_raises_per_street inclusive (the
    // cap itself is a valid, reachable depth -- only further raises are
    // excluded once there).
    for raises in 0..=config.max_raises_per_street {
        for &facing_bet in &[false, true] {
            if raises == 0 && facing_bet {
                // Can't be facing a bet with zero raises AND no blind-
                // equivalent wager this street in this simplified model;
                // the only "facing a bet with 0 raises" case is the
                // original blind/bet itself, which the solver treats as
                // the street's starting facing_bet=true node separately.
                // Skip the contradictory combination here.
                continue;
            }
            nodes.push(BettingNode::new(street, raises, facing_bet, config));
        }
    }
    nodes
}

/// Total abstracted info-set count across all 4 streets and all buckets
/// -- a sanity/tractability check, not used by the solver directly.
pub fn estimate_infoset_count(config: &CFRConfig) -> usize {
    let streets = [Street::Preflop, Street::Flop, Street::Turn, Street::River];
    let nodes_per_street = enumerate_street_nodes(Street::Preflop, config).len(); // same shape every street
    streets.len() * nodes_per_street * config.buckets.num_buckets as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn c(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    // ---- bucketing: river (exact equity) sanity + ordering ----

    #[test]
    fn river_bucketing_equity_matches_l3_exactly() {
        let hole = [c(14, 0), c(14, 1)];
        let board = vec![c(2, 2), c(5, 3), c(9, 0), c(11, 1), c(7, 2)];
        let config = BucketConfig {
            num_buckets: 10,
            bucketing_trials: 1_000,
        };

        let mc = MonteCarloConfig {
            trials: 1_000,
            seed: Some(1),
        };
        let state = crate::math_engine::MathEngineState {
            hole_cards: hole,
            board: board.clone(),
            street: Street::River,
            pot_size: 0,
            own_stack: 0,
            opponent_stack: 0,
            bet_to_call: 0,
            is_button: false,
        };
        let l3_equity = crate::math_engine::level3::estimate_equity(&state, &mc);

        // Not byte-identical (different RNG draws), but should be very
        // close given 1000 trials each -- same statistical process.
        let bucket_equity = bucketing_equity(hole, &board, &config);
        assert!(
            (bucket_equity - l3_equity).abs() < 0.1,
            "bucket_equity={bucket_equity}, l3_equity={l3_equity}"
        );
    }

    #[test]
    fn stronger_hand_gets_a_higher_or_equal_bucket() {
        let board = vec![c(2, 2), c(5, 3), c(9, 0)];
        let config = BucketConfig {
            num_buckets: 10,
            bucketing_trials: 500,
        };

        let strong = bucketing_equity([c(14, 0), c(14, 1)], &board, &config);
        let weak = bucketing_equity([c(3, 0), c(4, 1)], &board, &config);

        assert!(equity_to_bucket(strong, 10) >= equity_to_bucket(weak, 10));
    }

    #[test]
    fn equity_to_bucket_covers_full_range() {
        assert_eq!(equity_to_bucket(0.0, 10), 0);
        assert_eq!(equity_to_bucket(0.99, 10), 9);
        assert_eq!(equity_to_bucket(1.0, 10), 9); // clamped, not out of bounds
    }

    #[test]
    fn equity_to_bucket_never_panics_with_one_bucket() {
        assert_eq!(equity_to_bucket(0.5, 1), 0);
    }

    // ---- action abstraction: legality + raise-depth capping ----

    #[test]
    fn legal_actions_exclude_fold_when_not_facing_a_bet() {
        let actions = ActionAbstraction::legal_at(false, 0, 3);
        assert!(!actions.contains(&ActionAbstraction::Fold));
        assert!(actions.contains(&ActionAbstraction::CheckCall));
    }

    #[test]
    fn legal_actions_include_fold_when_facing_a_bet() {
        let actions = ActionAbstraction::legal_at(true, 0, 3);
        assert!(actions.contains(&ActionAbstraction::Fold));
    }

    #[test]
    fn raises_disappear_once_depth_cap_is_reached() {
        let under_cap = ActionAbstraction::legal_at(true, 2, 3);
        let at_cap = ActionAbstraction::legal_at(true, 3, 3);

        assert!(under_cap
            .iter()
            .any(|a| matches!(a, ActionAbstraction::Raise(_))));
        assert!(!at_cap
            .iter()
            .any(|a| matches!(a, ActionAbstraction::Raise(_))));
    }

    #[test]
    fn all_in_is_always_legal_even_past_the_raise_cap() {
        let at_cap = ActionAbstraction::legal_at(true, 3, 3);
        assert!(at_cap.contains(&ActionAbstraction::AllIn));
    }

    // ---- tree enumeration: bounded, matches the raise-depth cap ----

    #[test]
    fn street_node_count_is_bounded_by_raise_cap() {
        let config = CFRConfig {
            buckets: BucketConfig::default(),
            max_raises_per_street: 3,
        };
        let nodes = enumerate_street_nodes(Street::Flop, &config);
        // (max_raises+1) depths x 2 facing_bet states, minus the 1
        // contradictory (0 raises, facing_bet=true) combo excluded.
        assert_eq!(nodes.len(), (3 + 1) * 2 - 1);
    }

    #[test]
    fn higher_raise_cap_strictly_increases_node_count() {
        let low = CFRConfig {
            buckets: BucketConfig::default(),
            max_raises_per_street: 1,
        };
        let high = CFRConfig {
            buckets: BucketConfig::default(),
            max_raises_per_street: 5,
        };

        let low_nodes = enumerate_street_nodes(Street::Turn, &low).len();
        let high_nodes = enumerate_street_nodes(Street::Turn, &high).len();

        assert!(high_nodes > low_nodes);
    }

    #[test]
    fn infoset_count_is_finite_and_scales_with_buckets() {
        let config_10 = CFRConfig::default(); // 10 buckets
        let config_20 = CFRConfig {
            buckets: BucketConfig {
                num_buckets: 20,
                ..BucketConfig::default()
            },
            ..config_10
        };

        let count_10 = estimate_infoset_count(&config_10);
        let count_20 = estimate_infoset_count(&config_20);

        assert!(count_10 > 0);
        assert_eq!(count_20, count_10 * 2); // linear in bucket count, by construction
    }

    // ---- InfoSetKey: distinct histories produce distinct keys ----

    #[test]
    fn infoset_keys_with_different_histories_are_not_equal() {
        let key_a = InfoSetKey {
            bucket: 5,
            street: Street::Turn,
            raises_this_street: 0,
            facing_bet: true,
            history: vec![ActionAbstraction::CheckCall],
        };
        let key_b = InfoSetKey {
            bucket: 5,
            street: Street::Turn,
            raises_this_street: 0,
            facing_bet: true,
            history: vec![
                ActionAbstraction::Raise(PotFraction::Half),
                ActionAbstraction::CheckCall,
            ],
        };
        assert_ne!(key_a, key_b);

        // Also confirm they hash differently (required for correct
        // HashMap-keyed regret tables in the solver, built next message).
        let mut map: HashMap<InfoSetKey, u8> = HashMap::new();
        map.insert(key_a.clone(), 1);
        map.insert(key_b.clone(), 2);
        assert_eq!(map.len(), 2);
        assert_eq!(map[&key_a], 1);
        assert_eq!(map[&key_b], 2);
    }
}
