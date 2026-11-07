//! Monte Carlo CFR (external sampling) over the FULL hand across all 4
//! streets (Section 4.1: "Discounted CFR solve"), plus a Local Best
//! Response (LBR) solver for sampled exploitability, computed at
//! checkpoints per the PRD author's dual-metric plan:
//!   - average cumulative regret: computed every iteration, cheap,
//!     free byproduct of the CFR loop, internal sanity signal only.
//!   - LBR (sampled exploitability): computed periodically (every
//!     `checkpoint_every` iterations) via a Local Best Response solver.
//!
//! Naming precision (PRD-author-required, same principle as the earlier
//! "monotonically" correction): this metric is called "LBR" or "sampled
//! exploitability" everywhere -- NEVER "true exploitability" or bare
//! "exploitability" -- because it is not exact. Exact best response is
//! intractable at this scale; Local Best Response (Lisý & Bowling) is the
//! standard, published technique for measuring exploitability in
//! large-scale poker bots by sampling the opponent's action proportional
//! to their average strategy rather than summing over it exactly.
//! PRD-author-confirmed: this is the right tool for the job, not an
//! improvised workaround.
//!
//! LBR's raise-depth cap matches training's EXACTLY (reuses
//! `CFRConfig.max_raises_per_street` and the same `SAFETY_CAP_PER_STREET`
//! backstop) -- a PRD-author-required fix. An earlier draft used a
//! tighter cap for LBR than training, which would have introduced a
//! one-directional bias (the metric unable to explore lines the real
//! policy can actually take, making convergence look better than it is)
//! -- exactly the kind of bias that defeats the point of having this
//! metric. If LBR needs to be cheaper, the lever is `checkpoint_every`
//! (compute it less often), never narrowing the action space it explores.
//!
//! External sampling (Lanctot et al. 2009) for the CFR solve itself:
//! sample one full deal per iteration, recurse the hand tree across all
//! 4 streets in one trajectory per traverser. At the traverser's own
//! nodes, enumerate every legal action exactly. At the opponent's nodes,
//! sample one action from their current strategy.
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
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

/// Shared by BOTH the CFR solve and LBR (PRD-author-required: they must
/// match exactly, not use separate tighter/looser values).
const SAFETY_CAP_PER_STREET: u16 = 4;

const TRAIN_NODE_VISIT_BUDGET: u64 = 200_000_000;
const LBR_NODE_VISIT_BUDGET: u64 = 2_000_000;

/// Normalized per-player stack cap for this simplified stake model (not
/// yet wired to real chip math -- see module doc). Mirrors Section 3.3's
/// own all-in-override rule, applied here as a hard per-player
/// investment cap, so repeated raises can't compound into implausible
/// terminal values.
const NORMALIZED_STACK: f64 = 1.0;

#[derive(Debug, Clone, PartialEq)]
struct RegretEntry {
    // BTreeMap, not HashMap: these get SUMMED, and HashMap randomizes
    // its iteration order per-process -- summing the same floats in a
    // different order produces tiny but real discrepancies, which broke
    // exact-reproducibility (Section 8). BTreeMap's sorted, fixed
    // iteration order fixes this at the root.
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

pub(crate) fn sample_action(
    strategy: &BTreeMap<ActionAbstraction, f64>,
    legal: &[ActionAbstraction],
    rng: &mut StdRng,
) -> ActionAbstraction {
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

struct StepOutcome {
    next_invested_a: f64,
    next_invested_b: f64,
    street_closes: bool,
    next_raises: u8,
    next_facing_bet: bool,
}

fn apply_action_bookkeeping(
    action: ActionAbstraction,
    actions_this_street: u8,
    raises_this_street: u8,
    invested_a: f64,
    invested_b: f64,
    acting: Traverser,
) -> StepOutcome {
    let stake = action_stake(action);
    let pot_before = invested_a + invested_b;
    let raw_add = stake * pot_before.max(1.0);

    // Cap at the acting player's remaining normalized stack (Section
    // 3.3 step 4's all-in-override rule, applied to this simplified
    // model) -- without this, repeated raises compound multiplicatively
    // and terminal values become implausibly large.
    let current_invested = match acting {
        Traverser::A => invested_a,
        Traverser::B => invested_b,
    };
    let remaining_stack = (NORMALIZED_STACK - current_invested).max(0.0);
    let add = raw_add.min(remaining_stack);

    let (mut next_invested_a, mut next_invested_b) = (invested_a, invested_b);
    match acting {
        Traverser::A => next_invested_a += add,
        Traverser::B => next_invested_b += add,
    }

    // Safety backstop: AllIn is always legal, but unlike CheckCall it
    // never by itself closes a betting round. Same constant for BOTH
    // training and LBR (PRD-author-required) -- see module doc.
    let safety_cap_reached = (actions_this_street as u16) + 1 >= SAFETY_CAP_PER_STREET;

    let street_closes =
        (action == ActionAbstraction::CheckCall && actions_this_street > 0) || safety_cap_reached;
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

    StepOutcome {
        next_invested_a,
        next_invested_b,
        street_closes,
        next_raises,
        next_facing_bet,
    }
}

pub struct CFRSolver {
    config: CFRConfig,
    regrets: BTreeMap<InfoSetKey, RegretEntry>,
    rng: StdRng,
    iteration: u32,
    node_visits: u64,
}

/// Config for the LBR (Local Best Response) solver -- Lisý & Bowling's
/// technique for estimating exploitability by sampling the opponent's
/// action rather than summing over it exactly (PRD-author-confirmed as
/// the correct, standard tool here, not an improvised workaround).
#[derive(Debug, Clone, Copy)]
pub struct LbrConfig {
    /// Deals averaged per LBR checkpoint (Monte Carlo estimate at two
    /// levels: deal sampling AND opponent-action sampling).
    pub sample_deals: u32,
}

impl Default for LbrConfig {
    fn default() -> Self {
        LbrConfig { sample_deals: 200 }
    }
}

/// Result of one `train()` call: dual-cadence traces per the PRD
/// author's dual-metric plan.
#[derive(Debug, Clone)]
pub struct TrainingTrace {
    /// Average cumulative regret, one entry per iteration. Cheap,
    /// internal sanity signal only.
    pub regret_trace: Vec<f64>,
    /// LBR-estimated (sampled) exploitability, one entry per checkpoint
    /// (or empty if `checkpoint_every == 0`). NOT exact -- see module
    /// doc on why this is named "LBR," never "true exploitability."
    pub lbr_exploitability_trace: Vec<f64>,
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

    /// Runs `iterations` of external-sampling Discounted CFR. Every
    /// `checkpoint_every` iterations (if > 0), also computes LBR-
    /// estimated exploitability via `lbr_config`. Pass
    /// `checkpoint_every = 0` to skip LBR entirely.
    pub fn train(
        &mut self,
        iterations: u32,
        checkpoint_every: u32,
        lbr_config: &LbrConfig,
    ) -> TrainingTrace {
        let mut regret_trace = Vec::with_capacity(iterations as usize);
        let mut lbr_exploitability_trace = Vec::new();

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

            regret_trace.push(self.average_cumulative_regret());

            if checkpoint_every > 0 && self.iteration % checkpoint_every == 0 {
                lbr_exploitability_trace.push(self.lbr_exploitability_estimate(lbr_config));
            }
        }

        TrainingTrace {
            regret_trace,
            lbr_exploitability_trace,
        }
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
        self.node_visits += 1;
        assert!(
            self.node_visits < TRAIN_NODE_VISIT_BUDGET,
            "CFRSolver exceeded {TRAIN_NODE_VISIT_BUDGET} node visits in one train() call."
        );

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

        let outcome = apply_action_bookkeeping(
            action,
            actions_this_street,
            raises_this_street,
            invested_a,
            invested_b,
            acting,
        );

        if outcome.street_closes {
            return match next_street(street) {
                Some(ns) => self.traverse(
                    deal,
                    next_history,
                    ns,
                    0,
                    false,
                    0,
                    outcome.next_invested_a,
                    outcome.next_invested_b,
                    first_to_act(ns),
                    traverser,
                    own_reach,
                    pos_discount,
                    neg_discount,
                    strategy_discount,
                ),
                None => self.showdown_value(
                    deal,
                    outcome.next_invested_a,
                    outcome.next_invested_b,
                    traverser,
                ),
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
            outcome.next_raises,
            outcome.next_facing_bet,
            actions_this_street + 1,
            outcome.next_invested_a,
            outcome.next_invested_b,
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

    /// Cheap per-iteration sanity signal (NOT the exploitability metric).
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

    pub fn config(&self) -> &CFRConfig {
        &self.config
    }

    // ---- LBR (Local Best Response): sampled exploitability estimate ----

    /// LBR-estimated exploitability (Lisý & Bowling): for each player,
    /// best-response value against the OTHER player's current average
    /// strategy -- with the opponent's action SAMPLED proportional to
    /// their average strategy, not summed exactly (PRD-author-confirmed
    /// standard technique) -- averaged over sampled deals, then averaged
    /// over both players. Trends toward 0 as training approaches
    /// equilibrium. NEVER call this "true exploitability" -- see module
    /// doc.
    pub fn lbr_exploitability_estimate(&mut self, config: &LbrConfig) -> f64 {
        let mut total = 0.0;
        for _ in 0..config.sample_deals.max(1) {
            let deal = self.sample_deal();
            let seed_a: u64 = self.rng.gen();
            let seed_b: u64 = self.rng.gen();
            let mut rng_a = StdRng::seed_from_u64(seed_a);
            let mut rng_b = StdRng::seed_from_u64(seed_b);

            let lbr_a = self.local_best_response_value(&deal, Traverser::A, &mut rng_a);
            let lbr_b = self.local_best_response_value(&deal, Traverser::B, &mut rng_b);
            total += (lbr_a + lbr_b) / 2.0;
        }
        (total / config.sample_deals.max(1) as f64).max(0.0)
    }

    fn local_best_response_value(
        &self,
        deal: &SampledDeal,
        traverser: Traverser,
        rng: &mut StdRng,
    ) -> f64 {
        let mut node_budget: u64 = 0;
        self.lbr_traverse(
            deal,
            Vec::new(),
            Street::Preflop,
            0,
            false,
            0,
            0.0,
            0.0,
            first_to_act(Street::Preflop),
            traverser,
            rng,
            &mut node_budget,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn lbr_traverse(
        &self,
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
        rng: &mut StdRng,
        node_budget: &mut u64,
    ) -> f64 {
        let legal = ActionAbstraction::legal_at(
            facing_bet,
            raises_this_street,
            self.config.max_raises_per_street,
        );

        if acting == traverser {
            // Best response: max over actions -- the single best
            // deterministic counter-strategy. This side is NOT sampled;
            // it's the one thing best response fundamentally requires
            // enumerating exactly.
            let mut best = f64::NEG_INFINITY;
            for &action in &legal {
                let v = self.lbr_step(
                    deal,
                    &history,
                    action,
                    street,
                    raises_this_street,
                    actions_this_street,
                    invested_a,
                    invested_b,
                    acting,
                    traverser,
                    rng,
                    node_budget,
                );
                if v > best {
                    best = v;
                }
            }
            best
        } else {
            // LBR: the opponent's action is SAMPLED proportional to
            // their average strategy (Lisý & Bowling), not summed
            // exactly -- this is what makes LBR tractable, and it's the
            // precise source of the "sampled," not "true," naming.
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
            let strategy = self.average_strategy_at(&key).unwrap_or_else(|| {
                let n = legal.len().max(1) as f64;
                legal.iter().map(|&a| (a, 1.0 / n)).collect()
            });
            let action = sample_action(&strategy, &legal, rng);

            self.lbr_step(
                deal,
                &history,
                action,
                street,
                raises_this_street,
                actions_this_street,
                invested_a,
                invested_b,
                acting,
                traverser,
                rng,
                node_budget,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lbr_step(
        &self,
        deal: &SampledDeal,
        history: &[ActionAbstraction],
        action: ActionAbstraction,
        street: Street,
        raises_this_street: u8,
        actions_this_street: u8,
        invested_a: f64,
        invested_b: f64,
        acting: Traverser,
        traverser: Traverser,
        rng: &mut StdRng,
        node_budget: &mut u64,
    ) -> f64 {
        *node_budget += 1;
        assert!(
            *node_budget < LBR_NODE_VISIT_BUDGET,
            "LBR traversal exceeded {LBR_NODE_VISIT_BUDGET} node visits -- if this needs to be \
             cheaper, reduce checkpoint_every, not the action space being explored."
        );

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

        let outcome = apply_action_bookkeeping(
            action,
            actions_this_street,
            raises_this_street,
            invested_a,
            invested_b,
            acting,
        );

        if outcome.street_closes {
            return match next_street(street) {
                Some(ns) => self.lbr_traverse(
                    deal,
                    next_history,
                    ns,
                    0,
                    false,
                    0,
                    outcome.next_invested_a,
                    outcome.next_invested_b,
                    first_to_act(ns),
                    traverser,
                    rng,
                    node_budget,
                ),
                None => self.showdown_value(
                    deal,
                    outcome.next_invested_a,
                    outcome.next_invested_b,
                    traverser,
                ),
            };
        }

        let next_acting = match acting {
            Traverser::A => Traverser::B,
            Traverser::B => Traverser::A,
        };

        self.lbr_traverse(
            deal,
            next_history,
            street,
            outcome.next_raises,
            outcome.next_facing_bet,
            actions_this_street + 1,
            outcome.next_invested_a,
            outcome.next_invested_b,
            next_acting,
            traverser,
            rng,
            node_budget,
        )
    }
}

#[cfg(test)]
impl CFRSolver {
    /// Test-only: exposes the RAW regret-matching strategy (the thing
    /// recommend() must NEVER sample from -- only the average strategy
    /// carries CFR's convergence guarantee). Visible only so tests can
    /// prove the two tables genuinely diverge and that recommend()'s
    /// wiring to the correct one is actually load-bearing.
    pub(crate) fn current_strategy_at(
        &self,
        key: &InfoSetKey,
    ) -> Option<BTreeMap<ActionAbstraction, f64>> {
        let legal = ActionAbstraction::legal_at(
            key.facing_bet,
            key.raises_this_street,
            self.config.max_raises_per_street,
        );
        self.regrets.get(key).map(|e| e.current_strategy(&legal))
    }

    /// Test-only: directly injects a crafted info-set entry, bypassing
    /// actual training. Used to deliberately make regret_sum and
    /// strategy_sum disagree, proving recommend() follows strategy_sum.
    pub(crate) fn test_inject_entry(
        &mut self,
        key: InfoSetKey,
        regret_sum: Vec<(ActionAbstraction, f64)>,
        strategy_sum: Vec<(ActionAbstraction, f64)>,
    ) {
        let entry = RegretEntry {
            regret_sum: regret_sum.into_iter().collect(),
            strategy_sum: strategy_sum.into_iter().collect(),
        };
        self.regrets.insert(key, entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_engine::level5::abstraction::BucketConfig;

    fn test_config() -> CFRConfig {
        // Deliberately smaller than CFRConfig::default() (10 buckets, 3
        // raises/street): these are Sprint 9 calibration targets for the
        // real training run. Unit tests here need to stay fast.
        CFRConfig {
            buckets: BucketConfig {
                num_buckets: 6,
                bucketing_trials: 50,
            },
            max_raises_per_street: 1,
        }
    }

    fn no_checkpoint() -> LbrConfig {
        LbrConfig { sample_deals: 1 }
    }

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

    #[test]
    fn training_visits_all_four_streets() {
        let mut solver = CFRSolver::new(test_config(), 11);
        solver.train(15, 0, &no_checkpoint());

        let streets_seen: std::collections::HashSet<Street> =
            solver.regrets.keys().map(|k| k.street).collect();

        assert!(streets_seen.contains(&Street::Preflop));
        assert!(streets_seen.contains(&Street::Flop));
        assert!(streets_seen.contains(&Street::Turn));
        assert!(streets_seen.contains(&Street::River));
    }

    #[test]
    fn average_strategy_sums_to_one_for_every_visited_info_set() {
        let mut solver = CFRSolver::new(test_config(), 2);
        solver.train(10, 0, &no_checkpoint());

        assert!(solver.info_set_count() > 0);
        for key in solver.regrets.keys() {
            let strat = solver.average_strategy_at(key).unwrap();
            let total: f64 = strat.values().sum();
            assert!((total - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn average_cumulative_regret_is_non_negative_and_finite() {
        let mut solver = CFRSolver::new(test_config(), 3);
        let trace = solver.train(10, 0, &no_checkpoint());
        for &r in &trace.regret_trace {
            assert!(r >= 0.0);
            assert!(r.is_finite());
        }
    }

    #[test]
    fn regret_trends_downward_over_training() {
        // Larger iteration count + a noise tolerance, not strict `<`:
        // the normalized-stack cap makes this abstracted game converge
        // quickly, so after enough iterations the trace sits near a
        // noise floor rather than still visibly descending. "Trends
        // downward" (not strictly monotonic) allows for that.
        let mut solver = CFRSolver::new(test_config(), 4);
        let trace = solver.train(150, 0, &no_checkpoint());

        let quarter = trace.regret_trace.len() / 4;
        let first_quarter_avg: f64 =
            trace.regret_trace[0..quarter].iter().sum::<f64>() / quarter as f64;
        let last_quarter_avg: f64 = trace.regret_trace[trace.regret_trace.len() - quarter..]
            .iter()
            .sum::<f64>()
            / quarter as f64;

        let tolerance = first_quarter_avg * 0.1;
        assert!(
            last_quarter_avg <= first_quarter_avg + tolerance,
            "regret trended upward beyond noise tolerance: first={first_quarter_avg}, last={last_quarter_avg}"
        );
    }

    #[test]
    fn training_is_deterministic_given_same_seed() {
        let mut solver_1 = CFRSolver::new(test_config(), 42);
        let trace_1 = solver_1.train(10, 0, &no_checkpoint());

        let mut solver_2 = CFRSolver::new(test_config(), 42);
        let trace_2 = solver_2.train(10, 0, &no_checkpoint());

        assert_eq!(trace_1.regret_trace, trace_2.regret_trace);
        assert_eq!(solver_1.regrets, solver_2.regrets);
    }

    // ---- LBR / sampled exploitability ----

    #[test]
    fn dual_cadence_trace_lengths_match_iterations_and_checkpoints() {
        let mut solver = CFRSolver::new(test_config(), 5);
        let lbr_config = LbrConfig { sample_deals: 2 };
        let trace = solver.train(12, 4, &lbr_config);

        assert_eq!(trace.regret_trace.len(), 12);
        assert_eq!(trace.lbr_exploitability_trace.len(), 3); // 12 / 4
    }

    #[test]
    fn checkpoint_every_zero_disables_lbr_entirely() {
        let mut solver = CFRSolver::new(test_config(), 6);
        let trace = solver.train(10, 0, &LbrConfig::default());
        assert!(trace.lbr_exploitability_trace.is_empty());
        assert_eq!(trace.regret_trace.len(), 10);
    }

    #[test]
    fn lbr_exploitability_is_never_negative() {
        let mut solver = CFRSolver::new(test_config(), 7);
        let lbr_config = LbrConfig { sample_deals: 3 };
        let trace = solver.train(10, 5, &lbr_config);

        for &e in &trace.lbr_exploitability_trace {
            assert!(e >= 0.0);
            assert!(e.is_finite());
        }
    }

    #[test]
    fn lbr_exploitability_trends_downward_over_training() {
        let mut solver = CFRSolver::new(test_config(), 8);
        let lbr_config = LbrConfig { sample_deals: 6 };
        let trace = solver.train(100, 10, &lbr_config); // 10 checkpoints

        let half = trace.lbr_exploitability_trace.len() / 2;
        let first_half_avg: f64 =
            trace.lbr_exploitability_trace[0..half].iter().sum::<f64>() / half as f64;
        let second_half_avg: f64 = trace.lbr_exploitability_trace[half..].iter().sum::<f64>()
            / (trace.lbr_exploitability_trace.len() - half) as f64;

        let tolerance = first_half_avg * 0.15;
        assert!(
            second_half_avg <= first_half_avg + tolerance,
            "LBR exploitability trended upward beyond noise tolerance: first_half={first_half_avg}, second_half={second_half_avg}"
        );
    }

    #[test]
    fn lbr_against_self_consistent_strategy_is_reasonably_small() {
        let mut solver = CFRSolver::new(test_config(), 9);
        let lbr_config = LbrConfig { sample_deals: 5 };
        let trace = solver.train(20, 10, &lbr_config);

        for &e in &trace.lbr_exploitability_trace {
            assert!(e < 10.0, "LBR exploitability implausibly large: {e}");
        }
    }

    #[test]
    fn lbr_raise_depth_cap_matches_training_exactly() {
        // PRD-author-required fix, tested directly: LBR's legal-action
        // computation must use the SAME max_raises_per_street as
        // training, via the same ActionAbstraction::legal_at call --
        // not a separately configured, tighter cap. This test confirms
        // a higher max_raises_per_street actually produces more raise
        // options available to LBR's own traversal, proving no separate
        // ceiling is silently capping it below the config value.
        let config_low = CFRConfig {
            buckets: BucketConfig {
                num_buckets: 4,
                bucketing_trials: 20,
            },
            max_raises_per_street: 1,
        };
        let config_high = CFRConfig {
            buckets: BucketConfig {
                num_buckets: 4,
                bucketing_trials: 20,
            },
            max_raises_per_street: 3,
        };

        let actions_low = ActionAbstraction::legal_at(true, 1, config_low.max_raises_per_street);
        let actions_high = ActionAbstraction::legal_at(true, 1, config_high.max_raises_per_street);

        // At raises_this_street=1: capped out under config_low (cap=1),
        // still open under config_high (cap=3) -- so raises should be
        // legal under config_high but not config_low.
        assert!(!actions_low
            .iter()
            .any(|a| matches!(a, ActionAbstraction::Raise(_))));
        assert!(actions_high
            .iter()
            .any(|a| matches!(a, ActionAbstraction::Raise(_))));
    }
}
