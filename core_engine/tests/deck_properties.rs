use core_engine::Deck;
use proptest::prelude::*;
use std::collections::HashSet;

proptest! {
    /// "Deck shuffle produces all 52 unique cards" (Section 8).
    /// The `_run` input just forces proptest to repeat this many times
    /// across its default number of random cases.
    #[test]
    fn shuffled_deck_always_has_52_unique_cards(_run in 0u32..200) {
        let deck = Deck::new();
        prop_assert_eq!(deck.remaining(), 52);

        let set: HashSet<_> = deck.all_cards().iter().collect();
        prop_assert_eq!(set.len(), 52);
    }
}
