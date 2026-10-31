use core_engine::betting::{resolve_raise, PotFraction};
use proptest::prelude::*;

fn any_pot_fraction() -> impl Strategy<Value = PotFraction> {
    prop_oneof![
        Just(PotFraction::Quarter),
        Just(PotFraction::Half),
        Just(PotFraction::ThreeQuarter),
        Just(PotFraction::Full),
    ]
}

proptest! {
    /// "Raise resolution never produces a bet exceeding the agent's
    /// stack" AND "never below min-raise unless overridden by all-in"
    /// (Section 8), now against the pot-after-call formula.
    #[test]
    fn resolved_amount_is_always_legal(
        fraction in any_pot_fraction(),
        pot_before_action in 0u64..200_000,
        amount_to_call in 0u64..50_000,
        min_raise_increment in 0u64..50_000,
        stack_size in 0u64..200_000,
    ) {
        let r = resolve_raise(fraction, pot_before_action, amount_to_call, min_raise_increment, stack_size);

        prop_assert!(r.amount <= stack_size);

        let min_total = amount_to_call.saturating_add(min_raise_increment);
        if !r.is_all_in {
            prop_assert!(r.amount >= min_total);
        }

        if r.is_all_in {
            prop_assert_eq!(r.amount, stack_size);
        }
    }
}
