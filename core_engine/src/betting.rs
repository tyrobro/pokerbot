//! Betting legality and raise resolution (Section 3.3). The 4-step
//! resolution order is implemented as a single pure function, with every
//! branch unit-tested per the Sprint 2 DoD.
//!
//! Pot-fraction convention (per PRD author clarification): the fraction
//! sizes the RAISE INCREMENT relative to the pot AFTER calling, not the
//! player's total resulting street commitment — the industry-standard
//! convention used by PioSolver/GTOWizard/Simple Postflop.
//!   amount_to_call = opponent's street bet − your street commitment
//!   pot_after_call = pot_before_action + amount_to_call
//!   raise_increment = fraction × pot_after_call
//!   total_chips_this_action = amount_to_call + raise_increment
//! When amount_to_call == 0 (opening bet) this collapses to
//! fraction × pot_before_action.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PotFraction {
    Quarter,
    Half,
    ThreeQuarter,
    Full,
}

impl PotFraction {
    /// Returns (numerator, denominator) so raw-target math stays in
    /// integer arithmetic — no floats anywhere near money.
    fn as_fraction(self) -> (u64, u64) {
        match self {
            PotFraction::Quarter => (1, 4),
            PotFraction::Half => (1, 2),
            PotFraction::ThreeQuarter => (3, 4),
            PotFraction::Full => (1, 1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedAmount {
    /// Total chips this action adds to the pot (call portion + raise
    /// portion combined, or just the call/bet amount if not raising).
    pub amount: u64,
    pub is_all_in: bool,
}

/// Denomination tiers for raise rounding (Section 3.3 table). Fixed for the
/// entire match — does not scale with blind level. Keyed by the pot size
/// BEFORE this action.
pub fn denomination_for_pot(pot_before_action: u64) -> u64 {
    if pot_before_action < 1_000 {
        1
    } else if pot_before_action < 10_000 {
        5
    } else if pot_before_action < 50_000 {
        10
    } else {
        50
    }
}

/// Rounds `value` UP to the nearest multiple of `multiple` (never down,
/// per Section 3.3 step 2).
fn round_up_to_multiple(value: u64, multiple: u64) -> u64 {
    if multiple <= 1 {
        return value;
    }
    let remainder = value % multiple;
    if remainder == 0 {
        value
    } else {
        value + (multiple - remainder)
    }
}

/// Implements Section 3.3's exact 4-step raise resolution order:
///   1. raw target = amount_to_call + fraction × pot_after_call
///   2. round UP to the nearest legal multiple for the active denom tier
///   3. if still below (amount_to_call + min_raise_increment), bump to the
///      next legal multiple that satisfies the minimum-raise rule
///   4. if the result exceeds the agent's remaining stack, override to
///      all-in for the full remaining stack — this step always takes
///      precedence over steps 2-3.
pub fn resolve_raise(
    pot_fraction: PotFraction,
    pot_before_action: u64,
    amount_to_call: u64,
    min_raise_increment: u64,
    stack_size: u64,
) -> ResolvedAmount {
    // Step 1: raw target, pot-after-call convention.
    let pot_after_call = pot_before_action.saturating_add(amount_to_call);
    let (num, den) = pot_fraction.as_fraction();
    let raise_increment = pot_after_call.saturating_mul(num) / den;
    let raw_total = amount_to_call.saturating_add(raise_increment);

    // Step 2: round up to the active denomination tier.
    let denom = denomination_for_pot(pot_before_action);
    let mut total = round_up_to_multiple(raw_total, denom);

    // Step 3: bump to satisfy the minimum-raise rule if still short.
    let min_total = amount_to_call.saturating_add(min_raise_increment);
    if total < min_total {
        total = round_up_to_multiple(min_total, denom);
    }

    // Step 4: stack override — takes precedence over steps 2-3 outright.
    if total > stack_size {
        return ResolvedAmount {
            amount: stack_size,
            is_all_in: true,
        };
    }

    ResolvedAmount {
        amount: total,
        is_all_in: total == stack_size,
    }
}

/// Action-legality filter (Section 4.3 action types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    Fold,
    Check,
    Call,
    Raise,
    AllIn,
}

pub fn legal_actions(bet_to_call: u64, stack_size: u64) -> Vec<ActionType> {
    let mut actions = Vec::new();
    actions.push(ActionType::Fold);

    if bet_to_call == 0 {
        actions.push(ActionType::Check);
    } else {
        actions.push(ActionType::Call);
    }

    if stack_size > 0 {
        actions.push(ActionType::Raise);
        actions.push(ActionType::AllIn);
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- denomination tiers ----

    #[test]
    fn denomination_tier_boundaries() {
        assert_eq!(denomination_for_pot(0), 1);
        assert_eq!(denomination_for_pot(999), 1);
        assert_eq!(denomination_for_pot(1_000), 5);
        assert_eq!(denomination_for_pot(9_999), 5);
        assert_eq!(denomination_for_pot(10_000), 10);
        assert_eq!(denomination_for_pot(49_999), 10);
        assert_eq!(denomination_for_pot(50_000), 50);
        assert_eq!(denomination_for_pot(1_000_000), 50);
    }

    // ---- opening-bet cases (amount_to_call = 0): formula collapses to
    // fraction × pot_before_action, same numbers as Sprint 2 originally ----

    #[test]
    fn opening_bet_baseline_case_no_rounding_needed() {
        // pot=500, quarter pot opening bet = 125, already a legal multiple
        // of 1 -> no change from any step.
        let r = resolve_raise(PotFraction::Quarter, 500, 0, 1, 1_000_000);
        assert_eq!(r.amount, 125);
        assert!(!r.is_all_in);
    }

    #[test]
    fn opening_bet_step2_rounds_up_to_nearest_legal_multiple() {
        // pot=2003 (tier: multiple of 5), half pot = 1001 (integer div),
        // NOT a multiple of 5 -> must round UP to 1005 (never down).
        let r = resolve_raise(PotFraction::Half, 2_003, 0, 1, 1_000_000);
        assert_eq!(r.amount, 1_005);
        assert!(!r.is_all_in);
    }

    #[test]
    fn opening_bet_step3_bumps_up_when_below_min_raise() {
        // pot=500, quarter pot = 125 (legal multiple of 1), but
        // min_raise_increment is 200 -> bump to 200.
        let r = resolve_raise(PotFraction::Quarter, 500, 0, 200, 1_000_000);
        assert_eq!(r.amount, 200);
        assert!(!r.is_all_in);
    }

    #[test]
    fn opening_bet_step3_bump_also_respects_the_denomination_tier() {
        // pot=2003 (tier: multiple of 5), quarter pot = 500 (legal), but
        // min_raise_increment=503 is not itself a multiple of 5 -> bump
        // target is round_up(503, 5) = 505, not the raw 503.
        let r = resolve_raise(PotFraction::Quarter, 2_003, 0, 503, 1_000_000);
        assert_eq!(r.amount, 505);
    }

    #[test]
    fn opening_bet_step4_overrides_to_all_in_when_exceeding_stack() {
        // pot=100,000 (tier: multiple of 50), full pot = 100,000 (already
        // legal), but stack is only 3,000 -> override to all-in for 3,000.
        let r = resolve_raise(PotFraction::Full, 100_000, 0, 1, 3_000);
        assert_eq!(r.amount, 3_000);
        assert!(r.is_all_in);
    }

    #[test]
    fn opening_bet_step4_overrides_even_after_a_step3_bump() {
        // pot=500, quarter pot = 125, min_raise_increment=9000 bumps target
        // to 9000, but stack=50 -> step 4 wins outright: amount is 50.
        let r = resolve_raise(PotFraction::Quarter, 500, 0, 9_000, 50);
        assert_eq!(r.amount, 50);
        assert!(r.is_all_in);
    }

    #[test]
    fn opening_bet_exact_stack_match_is_marked_all_in() {
        let r = resolve_raise(PotFraction::Full, 500, 0, 1, 500);
        assert_eq!(r.amount, 500);
        assert!(r.is_all_in);
    }

    // ---- facing-a-bet cases: the actual point of this rework ----

    #[test]
    fn facing_bet_half_pot_uses_pot_after_call() {
        // Opponent bet 100 into a pot that was 100 before their bet, so
        // pot_before_action = 200 (includes their bet), amount_to_call =
        // 100. pot_after_call = 300. Half pot increment = 150.
        // total_chips_this_action = 100 (call) + 150 (raise) = 250.
        let r = resolve_raise(PotFraction::Half, 200, 100, 100, 1_000_000);
        assert_eq!(r.amount, 250);
        assert!(!r.is_all_in);
    }

    #[test]
    fn facing_bet_full_pot_raise() {
        // Same spot, but a "full pot" raise: increment = 1 × 300 = 300.
        // total = 100 (call) + 300 (raise) = 400.
        let r = resolve_raise(PotFraction::Full, 200, 100, 100, 1_000_000);
        assert_eq!(r.amount, 400);
    }

    #[test]
    fn facing_bet_vs_opening_bet_differ_for_the_same_fraction() {
        // Same pot_before_action and fraction, but one faces a bet and one
        // doesn't -- the two must NOT produce the same total, because the
        // pot-after-call convention explicitly depends on amount_to_call.
        let opening = resolve_raise(PotFraction::Half, 1_000, 0, 1, 1_000_000);
        let facing = resolve_raise(PotFraction::Half, 1_000, 500, 1, 1_000_000);
        assert_ne!(opening.amount, facing.amount);
    }

    // ---- action legality filter ----

    #[test]
    fn check_legal_only_when_nothing_to_call() {
        let actions = legal_actions(0, 10_000);
        assert!(actions.contains(&ActionType::Check));
        assert!(!actions.contains(&ActionType::Call));
    }

    #[test]
    fn call_legal_only_when_something_to_call() {
        let actions = legal_actions(500, 10_000);
        assert!(actions.contains(&ActionType::Call));
        assert!(!actions.contains(&ActionType::Check));
    }

    #[test]
    fn raise_illegal_with_zero_stack() {
        let actions = legal_actions(500, 0);
        assert!(!actions.contains(&ActionType::Raise));
        assert!(!actions.contains(&ActionType::AllIn));
    }

    #[test]
    fn fold_always_legal() {
        assert!(legal_actions(0, 0).contains(&ActionType::Fold));
        assert!(legal_actions(500, 10_000).contains(&ActionType::Fold));
    }
}
