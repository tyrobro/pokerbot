//! Card representation and a fresh-shuffle-per-hand Deck (Section 3.1).

use rand::seq::SliceRandom;
use rand::thread_rng;

/// rank: 2..=14 (11=J, 12=Q, 13=K, 14=A). suit: 0..=3 (no suit ordering in poker).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card {
    pub rank: u8,
    pub suit: u8,
}

pub struct Deck {
    cards: Vec<Card>,
}

impl Deck {
    /// Fresh, cryptographically-seeded 52-card shuffle (Section 3.1: "fresh
    /// shuffle every hand, no continuous shoe").
    pub fn new() -> Self {
        let mut cards = Vec::with_capacity(52);
        for suit in 0..4u8 {
            for rank in 2..=14u8 {
                cards.push(Card { rank, suit });
            }
        }
        cards.shuffle(&mut thread_rng());
        Deck { cards }
    }

    /// Deals one card off the top. `None` once empty.
    pub fn deal(&mut self) -> Option<Card> {
        self.cards.pop()
    }

    pub fn remaining(&self) -> usize {
        self.cards.len()
    }

    /// Exposed for tests: the full current contents of the deck.
    pub fn all_cards(&self) -> &[Card] {
        &self.cards
    }
}

impl Default for Deck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn fresh_deck_has_52_cards() {
        let deck = Deck::new();
        assert_eq!(deck.remaining(), 52);
    }

    #[test]
    fn fresh_deck_has_52_unique_cards() {
        let deck = Deck::new();
        let set: HashSet<Card> = deck.all_cards().iter().copied().collect();
        assert_eq!(set.len(), 52);
    }

    #[test]
    fn dealing_reduces_remaining_count() {
        let mut deck = Deck::new();
        let dealt = deck.deal();
        assert!(dealt.is_some());
        assert_eq!(deck.remaining(), 51);
    }
}
