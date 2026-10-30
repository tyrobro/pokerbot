//! Betting legality and raise resolution (Section 3.3). The 4-step
//! resolution order is implemented as a single pure function, with every
//! branch unit-tested per the Sprint 2 DoD.

/// Pre-resolution pot-fraction choice an agent picks (Section 4.3: "size:
/// required iff action_type == RAISE, pre-resolution pot-fraction choice").
/// `AllIn` as an explicit *choice* of raise sizing is distinct from the
/// agent's `ALL_IN` action_type (Section 4.3) — this variant exists so a
/// raise explicitly sized "all-in" also runs through the same resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub amount: u64,
    pub is_all_in: bool,
}

/// Denomination tiers for raise rounding (Section 3.3 table). Fixed for the
/// entire match — does not scale with blind level.
pub fn denomination_for_pot(pot_size: u64) -> u64 {
    if pot_size < 1_000 {
        1
    } else if pot_size < 10_000 {
        5
    } else if pot_size < 50_000 {
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
///   1. raw target = pot_fraction × pot_size
///   2. round UP to the nearest legal multiple for the active denom tier
///   3. if still below min_raise, bump to the next legal multiple that
///      satisfies the minimum-raise rule
///   4. if the result exceeds the agent's stack, override to all-in for
///      the full remaining stack — this step always takes precedence over
///      steps 2-3.
pub fn resolve_raise(
    pot_fraction: PotFraction,
    pot_size: u64,
    min_raise: u64,
    stack_size: u64,
) -> ResolvedAmount {
    // Step 1: raw target.
    let (num, den) = pot_fraction.as_fraction();
    let raw_target = pot_size.saturating_mul(num) / den;

    // Step 2: round up to the active denomination tier.
    let denom = denomination_for_pot(pot_size);
    let mut amount = round_up_to_multiple(raw_target, denom);

    // Step 3: bump to satisfy the minimum-raise rule if still short.
    if amount < min_raise {
        amount = round_up_to_multiple(min_raise, denom);
    }

    // Step 4: stack override — takes precedence over steps 2-3 outright.
    if amount > stack_size {
        return ResolvedAmount {
            amount: stack_size,
            is_all_in: true,
        };
    }

    ResolvedAmount {
        amount,
        is_all_in: amount == stack_size,
    }
}

/// Action-legality filter (Section 4.3 action types): which action types
/// are legal given what's owed and what's left in the stack. Check is only
/// legal when nothing is owed; call only when something is owed; raising
/// requires chips behind to raise with.
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

    // Folding is always legal (even facing no bet — a no-op fold is still
    // a legal, if strange, choice from the engine's point of view).
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

    // ---- step 1+2: raw target + round-up, no other steps engaged ----

    #[test]
    fn baseline_case_no_rounding_needed() {
        // pot=500 (tier: multiple of 1), quarter pot = 125, already a
        // legal multiple of 1 -> no change from any step.
        let r = resolve_raise(PotFraction::Quarter, 500, 1, 1_000_000);
        assert_eq!(r.amount, 125);
        assert!(!r.is_all_in);
    }

    #[test]
    fn step2_rounds_up_to_nearest_legal_multiple() {
        // pot=2003 (tier: multiple of 5), half pot = 1001 (integer div),
        // NOT a multiple of 5 -> must round UP to 1005 (never down).
        let r = resolve_raise(PotFraction::Half, 2_003, 1, 1_000_000);
        assert_eq!(r.amount, 1_005);
        assert!(!r.is_all_in);
    }

    // ---- step 3: below min-raise gets bumped ----

    #[test]
    fn step3_bumps_up_when_below_min_raise() {
        // pot=500, quarter pot = 125 (legal multiple of 1), but min_raise
        // (size of the previous bet this round) is 200 -> bump to 200.
        let r = resolve_raise(PotFraction::Quarter, 500, 200, 1_000_000);
        assert_eq!(r.amount, 200);
        assert!(!r.is_all_in);
    }

    #[test]
    fn step3_bump_also_respects_the_denomination_tier() {
        // pot=2003 (tier: multiple of 5), quarter pot = 500 (legal), but
        // min_raise=503 is not itself a multiple of 5 -> bump target is
        // round_up(503, 5) = 505, not the raw 503.
        let r = resolve_raise(PotFraction::Quarter, 2_003, 503, 1_000_000);
        assert_eq!(r.amount, 505);
    }

    // ---- step 4: stack override, precedence over steps 2-3 ----

    #[test]
    fn step4_overrides_to_all_in_when_exceeding_stack() {
        // pot=100,000 (tier: multiple of 50), full pot = 100,000 (already
        // legal), but stack is only 3,000 -> override to all-in for 3,000.
        let r = resolve_raise(PotFraction::Full, 100_000, 1, 3_000);
        assert_eq!(r.amount, 3_000);
        assert!(r.is_all_in);
    }

    #[test]
    fn step4_overrides_even_after_a_step3_bump() {
        // pot=500, quarter pot = 125, min_raise=9000 bumps target to 9000,
        // but stack=50 -> step 4 wins outright: amount is 50, not 9000.
        let r = resolve_raise(PotFraction::Quarter, 500, 9_000, 50);
        assert_eq!(r.amount, 50);
        assert!(r.is_all_in);
    }

    #[test]
    fn exact_stack_match_is_marked_all_in() {
        // Resolved amount lands exactly on the remaining stack even
        // without triggering the ">" override branch.
        let r = resolve_raise(PotFraction::Full, 500, 1, 500);
        assert_eq!(r.amount, 500);
        assert!(r.is_all_in);
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
