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
    /// (Section 8).
    #[test]
    fn resolved_amount_is_always_legal(
        fraction in any_pot_fraction(),
        pot_size in 0u64..200_000,
        min_raise in 0u64..50_000,
        stack_size in 0u64..200_000,
    ) {
        let r = resolve_raise(fraction, pot_size, min_raise, stack_size);

        // Never exceeds the agent's stack.
        prop_assert!(r.amount <= stack_size);

        // Never below the minimum raise UNLESS it's an all-in override
        // (a short all-in is explicitly allowed by Section 3.1).
        if !r.is_all_in {
            prop_assert!(r.amount >= min_raise);
        }

        // An all-in resolution always equals the full remaining stack.
        if r.is_all_in {
            prop_assert_eq!(r.amount, stack_size);
        }
    }
}
