//! Best-5-of-7 hand evaluation (Section 3.1).

use crate::card::Card;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandCategory {
    HighCard = 0,
    Pair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    Straight = 4,
    Flush = 5,
    FullHouse = 6,
    FourOfAKind = 7,
    StraightFlush = 8,
}

/// Comparable hand strength: category first, then rank-based tiebreakers
/// (kickers), both compared highest-first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandRank {
    pub category: HandCategory,
    pub tiebreak: [u8; 5],
}

impl PartialOrd for HandRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HandRank {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.category as u8, self.tiebreak).cmp(&(other.category as u8, other.tiebreak))
    }
}

/// Evaluates exactly 5 cards.
pub fn evaluate_5(cards: &[Card; 5]) -> HandRank {
    let mut ranks: Vec<u8> = cards.iter().map(|c| c.rank).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a)); // descending

    let is_flush = cards.iter().all(|c| c.suit == cards[0].suit);

    let mut unique_ranks = ranks.clone();
    unique_ranks.dedup();
    let is_wheel = unique_ranks == [14, 5, 4, 3, 2];
    let is_straight_normal = unique_ranks.len() == 5 && unique_ranks[0] - unique_ranks[4] == 4;
    let is_straight = is_straight_normal || is_wheel;
    let straight_high = if is_wheel {
        5
    } else {
        *unique_ranks.first().unwrap_or(&0)
    };

    let mut count_map: HashMap<u8, u8> = HashMap::new();
    for &r in &ranks {
        *count_map.entry(r).or_insert(0) += 1;
    }
    // (count, rank) tuples, sorted descending => highest count first, ties
    // broken by highest rank first.
    let mut groups: Vec<(u8, u8)> = count_map.into_iter().map(|(r, c)| (c, r)).collect();
    groups.sort_unstable_by(|a, b| b.cmp(a));

    let (category, tiebreak) = if is_straight && is_flush {
        (HandCategory::StraightFlush, [straight_high, 0, 0, 0, 0])
    } else if groups[0].0 == 4 {
        (
            HandCategory::FourOfAKind,
            [groups[0].1, groups[1].1, 0, 0, 0],
        )
    } else if groups[0].0 == 3 && groups[1].0 == 2 {
        (HandCategory::FullHouse, [groups[0].1, groups[1].1, 0, 0, 0])
    } else if is_flush {
        (
            HandCategory::Flush,
            [ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]],
        )
    } else if is_straight {
        (HandCategory::Straight, [straight_high, 0, 0, 0, 0])
    } else if groups[0].0 == 3 {
        let k: Vec<u8> = groups[1..].iter().map(|g| g.1).collect();
        (HandCategory::ThreeOfAKind, [groups[0].1, k[0], k[1], 0, 0])
    } else if groups[0].0 == 2 && groups[1].0 == 2 {
        let kicker = groups[2].1;
        (
            HandCategory::TwoPair,
            [groups[0].1, groups[1].1, kicker, 0, 0],
        )
    } else if groups[0].0 == 2 {
        let k: Vec<u8> = groups[1..].iter().map(|g| g.1).collect();
        (HandCategory::Pair, [groups[0].1, k[0], k[1], k[2], 0])
    } else {
        (
            HandCategory::HighCard,
            [ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]],
        )
    };

    HandRank { category, tiebreak }
}

/// Best 5-of-7 (Section 3.1): tries all C(7,5)=21 subsets, keeps the max.
pub fn evaluate_hand(cards: &[Card; 7]) -> HandRank {
    let mut best: Option<HandRank> = None;
    for (i, j) in exclude_pairs() {
        let combo: Vec<Card> = (0..7)
            .filter(|x| *x != i && *x != j)
            .map(|x| cards[x])
            .collect();
        let combo: [Card; 5] = combo.try_into().unwrap();
        let rank = evaluate_5(&combo);
        best = Some(match best {
            Some(b) if b >= rank => b,
            _ => rank,
        });
    }
    best.unwrap()
}

/// All (i, j) index pairs to exclude from 7, i.e. all C(7,2)=21 ways to
/// pick which 2 of the 7 cards are left out of a 5-card subset.
fn exclude_pairs() -> impl Iterator<Item = (usize, usize)> {
    (0..7).flat_map(|i| (i + 1..7).map(move |j| (i, j)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    #[test]
    fn high_card() {
        let h = evaluate_5(&[c(14, 0), c(10, 1), c(7, 2), c(4, 3), c(2, 0)]);
        assert_eq!(h.category, HandCategory::HighCard);
    }

    #[test]
    fn pair() {
        let h = evaluate_5(&[c(9, 0), c(9, 1), c(7, 2), c(4, 3), c(2, 0)]);
        assert_eq!(h.category, HandCategory::Pair);
    }

    #[test]
    fn two_pair() {
        let h = evaluate_5(&[c(9, 0), c(9, 1), c(4, 2), c(4, 3), c(2, 0)]);
        assert_eq!(h.category, HandCategory::TwoPair);
    }

    #[test]
    fn three_of_a_kind() {
        let h = evaluate_5(&[c(9, 0), c(9, 1), c(9, 2), c(4, 3), c(2, 0)]);
        assert_eq!(h.category, HandCategory::ThreeOfAKind);
    }

    #[test]
    fn straight_ace_high() {
        let h = evaluate_5(&[c(14, 0), c(13, 1), c(12, 2), c(11, 3), c(10, 0)]);
        assert_eq!(h.category, HandCategory::Straight);
        assert_eq!(h.tiebreak[0], 14);
    }

    #[test]
    fn straight_wheel_is_five_high() {
        let h = evaluate_5(&[c(14, 0), c(2, 1), c(3, 2), c(4, 3), c(5, 0)]);
        assert_eq!(h.category, HandCategory::Straight);
        assert_eq!(h.tiebreak[0], 5); // wheel ranks as 5-high, not ace-high
    }

    #[test]
    fn flush() {
        let h = evaluate_5(&[c(2, 0), c(5, 0), c(9, 0), c(11, 0), c(13, 0)]);
        assert_eq!(h.category, HandCategory::Flush);
    }

    #[test]
    fn full_house() {
        let h = evaluate_5(&[c(9, 0), c(9, 1), c(9, 2), c(4, 3), c(4, 0)]);
        assert_eq!(h.category, HandCategory::FullHouse);
    }

    #[test]
    fn four_of_a_kind() {
        let h = evaluate_5(&[c(9, 0), c(9, 1), c(9, 2), c(9, 3), c(4, 0)]);
        assert_eq!(h.category, HandCategory::FourOfAKind);
    }

    #[test]
    fn straight_flush() {
        let h = evaluate_5(&[c(5, 0), c(6, 0), c(7, 0), c(8, 0), c(9, 0)]);
        assert_eq!(h.category, HandCategory::StraightFlush);
    }

    #[test]
    fn royal_flush() {
        let h = evaluate_5(&[c(10, 0), c(11, 0), c(12, 0), c(13, 0), c(14, 0)]);
        assert_eq!(h.category, HandCategory::StraightFlush);
        assert_eq!(h.tiebreak[0], 14);
    }

    #[test]
    fn best_5_of_7_picks_the_strongest_subset() {
        // Board: 7-7-7-2-2 ; hole: K-3 -> best hand is full house 7s over 2s,
        // not just the trips.
        let seven_cards: [Card; 7] = [
            c(13, 0),
            c(3, 1),
            c(7, 2),
            c(7, 3),
            c(7, 0),
            c(2, 1),
            c(2, 2),
        ];
        let h = evaluate_hand(&seven_cards);
        assert_eq!(h.category, HandCategory::FullHouse);
        assert_eq!(h.tiebreak[0], 7);
        assert_eq!(h.tiebreak[1], 2);
    }

    #[test]
    fn tricky_tiebreak_two_pair_vs_two_pair() {
        // Both two pair (Aces & Kings), but hand B's kicker (Queen) beats
        // hand A's kicker (Jack).
        let a: [Card; 5] = [c(14, 0), c(14, 1), c(13, 2), c(13, 3), c(11, 0)];
        let b: [Card; 5] = [c(14, 2), c(14, 3), c(13, 0), c(13, 1), c(12, 0)];
        assert!(evaluate_5(&b) > evaluate_5(&a));
    }

    #[test]
    fn tricky_tiebreak_full_house_rank_matters_more_than_pair_rank() {
        // Trip Jacks over a pair of Aces beats trip Tens over a pair of Kings.
        let jacks_full: [Card; 5] = [c(11, 0), c(11, 1), c(11, 2), c(14, 3), c(14, 0)];
        let tens_full: [Card; 5] = [c(10, 0), c(10, 1), c(10, 2), c(13, 3), c(13, 0)];
        assert!(evaluate_5(&jacks_full) > evaluate_5(&tens_full));
    }

    #[test]
    fn tricky_tiebreak_straight_flush_beats_quads() {
        let sf: [Card; 5] = [c(4, 0), c(5, 0), c(6, 0), c(7, 0), c(8, 0)];
        let quads: [Card; 5] = [c(14, 0), c(14, 1), c(14, 2), c(14, 3), c(2, 0)];
        assert!(evaluate_5(&sf) > evaluate_5(&quads));
    }
}
