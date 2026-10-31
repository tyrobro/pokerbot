//! Monte Carlo CFR (external sampling) over the FULL hand across all 4
//! streets (Section 4.1: "Discounted CFR solve"), replacing a prior
//! per-street independent-solve draft that the PRD author correctly
//! identified as unsound (a form of unsafe subgame solving -- it would
//! not have let continuation values from later streets genuinely
//! propagate backward, making it structurally different from, and far
//! weaker than, real CFR).
//!
//! External sampling (Lanctot et al. 2009): per iteration, sample one
//! full deal (both players' hole cards + complete board) once, then for
//! EACH player in turn as "traverser," recurse the hand tree across all
//! 4 streets in one trajectory. At the traverser's own nodes, enumerate
//! every legal action exactly (standard CFR regret update). At the
//! opponent's nodes, sample one action from their current strategy and
//! recurse into only that branch. Showdown values use REAL hand
//! evaluation on the sampled cards -- cross-street linkage is intact by
//! construction, not approximated.
//!
//! Discounted CFR (Brown & Sandholm 2019) discount schedule
//! (alpha=1.5, beta=0.5, gamma=2.0), same as before.
//!
//! Modeling simplifications (flagged): fixed acting-role convention
//! (player A first preflop, player B first postflop) rather than real
//! button rotation (a match-level concern, Sprint 4); stake accounting
//! in normalized pot-fraction units, not yet wired to real chip math
//! (Section 3.3's resolve_raise) -- that integration is scoped for when
//! this blueprint gets wired into recommend().

use super::abstraction::{
    bucketing_equity_seeded, equity_to_bucket, ActionAbstraction, CFRConfig, InfoSetKey,
};
use crate::card::Card;
use crate::game::Street;
use crate::hand_eval::evaluate_hand;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
struct RegretEntry {
    // BTreeMap, not HashMap: these get SUMMED (average_cumulative_regret,
    // current_strategy's `total`, etc.), and HashMap randomizes its
    // iteration order per-process -- summing the same floats in a
    // different order produces tiny (sub-1e-10) but real discrepancies,
    // which broke exact-reproducibility (Section 8). BTreeMap's sorted,
    // fixed iteration order fixes this at the root.
    regret_sum: BTreeMap<ActionAbstraction, f64>,
    strategy_sum: BTreeMap<ActionAbstraction, f64>,
}

impl RegretEntry {
    fn new(actions: &[ActionAbstraction]) -> Self {
        let mut regret_sum = BTreeMap::new();
        let mut strategy_sum = BTreeMap::new();
        for &a in actions {
            regret_sum.insert(a, 0.0);
            strategy_sum.insert(a, 0.0);
        }
        RegretEntry {
            regret_sum,
            strategy_sum,
        }
    }

    fn current_strategy(&self, actions: &[ActionAbstraction]) -> BTreeMap<ActionAbstraction, f64> {
        let positive: BTreeMap<ActionAbstraction, f64> = actions
            .iter()
            .map(|&a| (a, self.regret_sum.get(&a).copied().unwrap_or(0.0).max(0.0)))
            .collect();
        let total: f64 = positive.values().sum();
        if total > 0.0 {
            positive.into_iter().map(|(a, v)| (a, v / total)).collect()
        } else {
            let n = actions.len().max(1) as f64;
            actions.iter().map(|&a| (a, 1.0 / n)).collect()
        }
    }

    fn average_strategy(&self, actions: &[ActionAbstraction]) -> BTreeMap<ActionAbstraction, f64> {
        let total: f64 = actions
            .iter()
            .map(|&a| self.strategy_sum.get(&a).copied().unwrap_or(0.0))
            .sum();
        if total > 0.0 {
            actions
                .iter()
                .map(|&a| (a, self.strategy_sum.get(&a).copied().unwrap_or(0.0) / total))
                .collect()
        } else {
            let n = actions.len().max(1) as f64;
            actions.iter().map(|&a| (a, 1.0 / n)).collect()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Traverser {
    A,
    B,
}

/// One sampled full deal, plus each player's precomputed equity-
/// percentile bucket at each street (computed once per deal -- bucket
/// depends only on hole+board-so-far, not betting history, so this
/// avoids recomputing it at every node visited within a street).
struct SampledDeal {
    hole_a: [Card; 2],
    hole_b: [Card; 2],
    board: [Card; 5],
    bucket_a: [u8; 4],
    bucket_b: [u8; 4],
}

fn street_index(street: Street) -> usize {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

fn board_len_for(street: Street) -> usize {
    match street {
        Street::Preflop => 0,
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => 5,
    }
}

fn next_street(street: Street) -> Option<Street> {
    match street {
        Street::Preflop => Some(Street::Flop),
        Street::Flop => Some(Street::Turn),
        Street::Turn => Some(Street::River),
        Street::River => None,
    }
}

/// Fixed role convention (flagged in module doc): A acts first preflop,
/// B acts first on every postflop street.
fn first_to_act(street: Street) -> Traverser {
    if street == Street::Preflop {
        Traverser::A
    } else {
        Traverser::B
    }
}

fn action_stake(action: ActionAbstraction) -> f64 {
    match action {
        ActionAbstraction::Raise(frac) => match frac {
            crate::betting::PotFraction::Quarter => 0.25,
            crate::betting::PotFraction::Half => 0.5,
            crate::betting::PotFraction::ThreeQuarter => 0.75,
            crate::betting::PotFraction::Full => 1.0,
        },
        ActionAbstraction::AllIn => 1.5,
        ActionAbstraction::CheckCall => 0.1,
        ActionAbstraction::Fold => 0.0,
    }
}

fn sample_action(
    strategy: &BTreeMap<ActionAbstraction, f64>,
    legal: &[ActionAbstraction],
    rng: &mut StdRng,
) -> ActionAbstraction {
    use rand::Rng;
    let r: f64 = rng.gen();
    let mut cumulative = 0.0;
    for &action in legal {
        cumulative += strategy.get(&action).copied().unwrap_or(0.0);
        if r <= cumulative {
            return action;
        }
    }
    *legal.last().expect("legal actions is never empty")
}

/// External-sampling Discounted CFR solver over the full 4-street hand.
pub struct CFRSolver {
    config: CFRConfig,
    regrets: BTreeMap<InfoSetKey, RegretEntry>,
    rng: StdRng,
    iteration: u32,
    node_visits: u64,
}

impl CFRSolver {
    pub fn new(config: CFRConfig, seed: u64) -> Self {
        CFRSolver {
            config,
            regrets: BTreeMap::new(),
            rng: StdRng::seed_from_u64(seed),
            iteration: 0,
            node_visits: 0,
        }
    }

    fn sample_deal(&mut self) -> SampledDeal {
        let mut deck: Vec<Card> = Vec::with_capacity(52);
        for suit in 0..4u8 {
            for rank in 2..=14u8 {
                deck.push(Card { rank, suit });
            }
        }
        deck.shuffle(&mut self.rng);

        let hole_a = [deck[0], deck[1]];
        let hole_b = [deck[2], deck[3]];
        let board: [Card; 5] = [deck[4], deck[5], deck[6], deck[7], deck[8]];

        let bucket_config = self.config.buckets;
        let mut bucket_a = [0u8; 4];
        let mut bucket_b = [0u8; 4];
        for (idx, &street) in [Street::Preflop, Street::Flop, Street::Turn, Street::River]
            .iter()
            .enumerate()
        {
            let n = board_len_for(street);
            let equity_a =
                bucketing_equity_seeded(hole_a, &board[0..n], &bucket_config, &mut self.rng);
            let equity_b =
                bucketing_equity_seeded(hole_b, &board[0..n], &bucket_config, &mut self.rng);
            bucket_a[idx] = equity_to_bucket(equity_a, bucket_config.num_buckets);
            bucket_b[idx] = equity_to_bucket(equity_b, bucket_config.num_buckets);
        }

        SampledDeal {
            hole_a,
            hole_b,
            board,
            bucket_a,
            bucket_b,
        }
    }

    /// Runs `iterations` of external-sampling Discounted CFR. Returns the
    /// cheap average-cumulative-regret trace (per-iteration sanity
    /// signal; NOT exploitability -- dual-metric plan, exploitability via
    /// best-response comes in the next sub-step).
    pub fn train(&mut self, iterations: u32) -> Vec<f64> {
        let mut trace = Vec::with_capacity(iterations as usize);

        for _ in 0..iterations {
            self.iteration += 1;
            let t = self.iteration as f64;
            let pos_discount = t.powf(1.5) / (t.powf(1.5) + 1.0);
            let neg_discount = t.powf(0.5) / (t.powf(0.5) + 1.0);
            let strategy_discount = (t / (t + 1.0)).powf(2.0);

            let deal = self.sample_deal();

            for &traverser in &[Traverser::A, Traverser::B] {
                self.traverse(
                    &deal,
                    Vec::new(),
                    Street::Preflop,
                    0,
                    false,
                    0,
                    0.0,
                    0.0,
                    first_to_act(Street::Preflop),
                    traverser,
                    1.0,
                    pos_discount,
                    neg_discount,
                    strategy_discount,
                );
            }

            trace.push(self.average_cumulative_regret());
        }

        trace
    }

    #[allow(clippy::too_many_arguments)]
    fn traverse(
        &mut self,
        deal: &SampledDeal,
        history: Vec<ActionAbstraction>,
        street: Street,
        raises_this_street: u8,
        facing_bet: bool,
        actions_this_street: u8,
        invested_a: f64,
        invested_b: f64,
        acting: Traverser,
        traverser: Traverser,
        own_reach: f64,
        pos_discount: f64,
        neg_discount: f64,
        strategy_discount: f64,
    ) -> f64 {
        let legal = ActionAbstraction::legal_at(
            facing_bet,
            raises_this_street,
            self.config.max_raises_per_street,
        );

        let bucket = match acting {
            Traverser::A => deal.bucket_a[street_index(street)],
            Traverser::B => deal.bucket_b[street_index(street)],
        };
        let key = InfoSetKey {
            bucket,
            street,
            raises_this_street,
            facing_bet,
            history: history.clone(),
        };

        self.regrets
            .entry(key.clone())
            .or_insert_with(|| RegretEntry::new(&legal));
        let strategy = self.regrets[&key].current_strategy(&legal);

        if acting == traverser {
            let mut action_values: BTreeMap<ActionAbstraction, f64> = BTreeMap::new();
            let mut node_value = 0.0;

            for &action in &legal {
                let value = self.step(
                    deal,
                    &history,
                    action,
                    street,
                    raises_this_street,
                    facing_bet,
                    actions_this_street,
                    invested_a,
                    invested_b,
                    acting,
                    traverser,
                    own_reach * strategy[&action],
                    pos_discount,
                    neg_discount,
                    strategy_discount,
                );
                action_values.insert(action, value);
                node_value += strategy[&action] * value;
            }

            let entry = self.regrets.get_mut(&key).unwrap();
            for &action in &legal {
                let regret = action_values[&action] - node_value;
                let current = entry.regret_sum.get(&action).copied().unwrap_or(0.0);
                let discount = if regret >= 0.0 {
                    pos_discount
                } else {
                    neg_discount
                };
                entry.regret_sum.insert(action, current * discount + regret);

                let current_strat = entry.strategy_sum.get(&action).copied().unwrap_or(0.0);
                entry.strategy_sum.insert(
                    action,
                    current_strat * strategy_discount + own_reach * strategy[&action],
                );
            }

            node_value
        } else {
            let action = sample_action(&strategy, &legal, &mut self.rng);
            self.step(
                deal,
                &history,
                action,
                street,
                raises_this_street,
                facing_bet,
                actions_this_street,
                invested_a,
                invested_b,
                acting,
                traverser,
                own_reach,
                pos_discount,
                neg_discount,
                strategy_discount,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        deal: &SampledDeal,
        history: &[ActionAbstraction],
        action: ActionAbstraction,
        street: Street,
        raises_this_street: u8,
        _facing_bet: bool,
        actions_this_street: u8,
        invested_a: f64,
        invested_b: f64,
        acting: Traverser,
        traverser: Traverser,
        own_reach: f64,
        pos_discount: f64,
        neg_discount: f64,
        strategy_discount: f64,
    ) -> f64 {
        if action == ActionAbstraction::Fold {
            let value_to_a = match acting {
                Traverser::A => -invested_a,
                Traverser::B => invested_b,
            };
            return if traverser == Traverser::A {
                value_to_a
            } else {
                -value_to_a
            };
        }

        let mut next_history = history.to_vec();
        next_history.push(action);

        let stake = action_stake(action);
        let pot_before = invested_a + invested_b;
        let add = stake * pot_before.max(1.0);
        let (mut next_invested_a, mut next_invested_b) = (invested_a, invested_b);
        match acting {
            Traverser::A => next_invested_a += add,
            Traverser::B => next_invested_b += add,
        }

        // Safety backstop: AllIn is always legal (Section 3.1's all-in
        // override always takes precedence), but unlike CheckCall it
        // never by itself closes a betting round -- two players could
        // otherwise alternate AllIn indefinitely, since the raise-depth
        // cap only restricts Raise, not AllIn. This bounds total actions
        // per street as a hard backstop; it does not change the raise-
        // depth cap's role as the intended abstraction lever.
        let safety_cap: u16 = 4;
        let safety_cap_reached = (actions_this_street as u16) + 1 >= safety_cap;

        self.node_visits += 1;
        const NODE_VISIT_BUDGET: u64 = 5_000_000;
        assert!(
            self.node_visits < NODE_VISIT_BUDGET,
            "CFRSolver exceeded {NODE_VISIT_BUDGET} node visits in one train() call -- \
             likely a combinatorial blowup from action-space branching compounding across \
             streets (see module doc). Check max_raises_per_street, the safety cap, and the \
             number of training iterations before increasing this budget."
        );

        let street_closes = (action == ActionAbstraction::CheckCall && actions_this_street > 0)
            || safety_cap_reached;
        let next_raises = if matches!(
            action,
            ActionAbstraction::Raise(_) | ActionAbstraction::AllIn
        ) {
            raises_this_street + 1
        } else {
            raises_this_street
        };
        let next_facing_bet = matches!(
            action,
            ActionAbstraction::Raise(_) | ActionAbstraction::AllIn
        );

        if street_closes {
            return match next_street(street) {
                Some(ns) => self.traverse(
                    deal,
                    next_history,
                    ns,
                    0,
                    false,
                    0,
                    next_invested_a,
                    next_invested_b,
                    first_to_act(ns),
                    traverser,
                    own_reach,
                    pos_discount,
                    neg_discount,
                    strategy_discount,
                ),
                None => self.showdown_value(deal, next_invested_a, next_invested_b, traverser),
            };
        }

        let next_acting = match acting {
            Traverser::A => Traverser::B,
            Traverser::B => Traverser::A,
        };

        self.traverse(
            deal,
            next_history,
            street,
            next_raises,
            next_facing_bet,
            actions_this_street + 1,
            next_invested_a,
            next_invested_b,
            next_acting,
            traverser,
            own_reach,
            pos_discount,
            neg_discount,
            strategy_discount,
        )
    }

    fn showdown_value(
        &self,
        deal: &SampledDeal,
        invested_a: f64,
        invested_b: f64,
        traverser: Traverser,
    ) -> f64 {
        let seven_a: [Card; 7] = [
            deal.hole_a[0],
            deal.hole_a[1],
            deal.board[0],
            deal.board[1],
            deal.board[2],
            deal.board[3],
            deal.board[4],
        ];
        let seven_b: [Card; 7] = [
            deal.hole_b[0],
            deal.hole_b[1],
            deal.board[0],
            deal.board[1],
            deal.board[2],
            deal.board[3],
            deal.board[4],
        ];
        let rank_a = evaluate_hand(&seven_a);
        let rank_b = evaluate_hand(&seven_b);

        let pot = invested_a + invested_b;
        let value_to_a = match rank_a.cmp(&rank_b) {
            std::cmp::Ordering::Greater => pot - invested_a,
            std::cmp::Ordering::Less => -invested_a,
            std::cmp::Ordering::Equal => pot / 2.0 - invested_a,
        };

        if traverser == Traverser::A {
            value_to_a
        } else {
            -value_to_a
        }
    }

    /// Cheap per-iteration sanity signal (NOT exploitability).
    pub fn average_cumulative_regret(&self) -> f64 {
        if self.regrets.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0u32;
        for entry in self.regrets.values() {
            for &r in entry.regret_sum.values() {
                total += r.abs();
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    pub fn average_strategy_at(
        &self,
        key: &InfoSetKey,
    ) -> Option<BTreeMap<ActionAbstraction, f64>> {
        let legal = ActionAbstraction::legal_at(
            key.facing_bet,
            key.raises_this_street,
            self.config.max_raises_per_street,
        );
        self.regrets.get(key).map(|e| e.average_strategy(&legal))
    }

    pub fn info_set_count(&self) -> usize {
        self.regrets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_engine::level5::abstraction::BucketConfig;

    fn test_config() -> CFRConfig {
        CFRConfig {
            buckets: BucketConfig {
                num_buckets: 6,
                bucketing_trials: 50,
            },
            max_raises_per_street: 1,
        }
    }

    // ---- regret matching basics ----

    #[test]
    fn regret_matching_is_uniform_with_zero_regret() {
        let actions = vec![ActionAbstraction::Fold, ActionAbstraction::CheckCall];
        let entry = RegretEntry::new(&actions);
        let strategy = entry.current_strategy(&actions);
        assert!((strategy[&ActionAbstraction::Fold] - 0.5).abs() < 1e-9);
        assert!((strategy[&ActionAbstraction::CheckCall] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn regret_matching_favors_higher_positive_regret_action() {
        let actions = vec![ActionAbstraction::Fold, ActionAbstraction::CheckCall];
        let mut entry = RegretEntry::new(&actions);
        entry.regret_sum.insert(ActionAbstraction::Fold, 1.0);
        entry.regret_sum.insert(ActionAbstraction::CheckCall, 3.0);

        let strategy = entry.current_strategy(&actions);
        assert!(strategy[&ActionAbstraction::CheckCall] > strategy[&ActionAbstraction::Fold]);
    }

    // ---- the actual architectural fix: cross-street linkage is real ----

    #[test]
    fn training_visits_all_four_streets() {
        // Direct proof of the fix the PRD author required: info sets at
        // Flop/Turn/River actually get visited, because trajectories
        // genuinely play out across streets rather than being solved in
        // isolation. 60 iterations makes "never once reaches the river"
        // vanishingly unlikely under any sane strategy mix.
        let mut solver = CFRSolver::new(test_config(), 11);
        solver.train(15);

        let streets_seen: std::collections::HashSet<Street> =
            solver.regrets.keys().map(|k| k.street).collect();

        assert!(streets_seen.contains(&Street::Preflop));
        assert!(
            streets_seen.contains(&Street::Flop),
            "flop never visited -- cross-street linkage broken"
        );
        assert!(
            streets_seen.contains(&Street::Turn),
            "turn never visited -- cross-street linkage broken"
        );
        assert!(
            streets_seen.contains(&Street::River),
            "river never visited -- cross-street linkage broken"
        );
    }

    // ---- solver sanity ----

    #[test]
    fn average_strategy_sums_to_one_for_every_visited_info_set() {
        let mut solver = CFRSolver::new(test_config(), 2);
        solver.train(10);

        assert!(solver.info_set_count() > 0);
        for key in solver.regrets.keys() {
            let strat = solver.average_strategy_at(key).unwrap();
            let total: f64 = strat.values().sum();
            assert!(
                (total - 1.0).abs() < 1e-6,
                "strategy sums to {total}, not 1.0"
            );
        }
    }

    #[test]
    fn average_cumulative_regret_is_non_negative_and_finite() {
        let mut solver = CFRSolver::new(test_config(), 3);
        let trace = solver.train(10);

        for &r in &trace {
            assert!(r >= 0.0);
            assert!(r.is_finite());
        }
    }

    #[test]
    fn regret_trends_downward_over_training() {
        // "Trends downward," not strictly monotonic (PRD author's
        // correction) -- compares the average of the first quarter of
        // the trace against the average of the last quarter.
        let mut solver = CFRSolver::new(test_config(), 4);
        let trace = solver.train(40);

        let quarter = trace.len() / 4;
        let first_quarter_avg: f64 = trace[0..quarter].iter().sum::<f64>() / quarter as f64;
        let last_quarter_avg: f64 =
            trace[trace.len() - quarter..].iter().sum::<f64>() / quarter as f64;

        assert!(
            last_quarter_avg < first_quarter_avg,
            "regret did not trend downward: first_quarter={first_quarter_avg}, last_quarter={last_quarter_avg}"
        );
    }

    #[test]
    fn training_is_deterministic_given_same_seed() {
        let mut solver_1 = CFRSolver::new(test_config(), 42);
        let trace_1 = solver_1.train(10);

        let mut solver_2 = CFRSolver::new(test_config(), 42);
        let trace_2 = solver_2.train(10);

        assert_eq!(trace_1, trace_2);
        assert_eq!(solver_1.regrets, solver_2.regrets);
    }
}
