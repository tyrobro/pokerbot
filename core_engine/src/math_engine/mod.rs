//! Math engine levels 1-5 (Section 4.1). Each level implements
//! `recommend(state) -> RecommendedAction` deterministically.

pub mod level1;
pub mod level2;
pub mod level3;
pub mod level4;

use crate::betting::{ActionType, PotFraction};
use crate::card::Card;
use crate::game::Street;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    Fold,
    CheckOrCall,
    Raise(PotFraction),
}

#[derive(Debug, Clone)]
pub struct MathEngineState {
    pub hole_cards: [Card; 2],
    pub board: Vec<Card>,
    pub street: Street,
    pub pot_size: u64,
    pub own_stack: u64,
    pub opponent_stack: u64,
    pub bet_to_call: u64,
    pub is_button: bool,
}

/// One OPPONENT action (never the agent's own), with enough context for
/// range-narrowing (Sprint 7 L4, PRD-author-specified shape): street,
/// action type, size, and the pot size at the time of the action -- a
/// raise into a small pot signals differently than the same raise into a
/// large one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionRecord {
    pub street: Street,
    pub action_type: ActionType,
    pub amount: u64,
    pub pot_size_at_time: u64,
}

/// One completed hand vs this opponent, mirroring what Section 4.4's
/// opponent-history object will eventually track (actions, showdown
/// result per the card-visibility rule, win/loss amount). Sprint 7 tests
/// this against hand-built fixtures; Sprint 9 re-points `match_history` at
/// the real object once the orchestrator/opponent-memory infrastructure
/// exists (straight parameter substitution per the PRD author's plan).
#[derive(Debug, Clone)]
pub struct MatchHistoryEntry {
    pub actions: Vec<ActionRecord>,
    /// Hole cards are only ever visible in hands the AGENT lost at
    /// showdown (Section 4.4's muck-when-beaten rule) -- `None` otherwise
    /// (folded, or a hand the agent won).
    pub opponent_hole_cards: Option<[Card; 2]>,
    pub showdown_reached: bool,
    /// `Some(true)` opponent won, `Some(false)` opponent lost, `None` split.
    pub opponent_won: Option<bool>,
    pub amount_won_or_lost: i64,
}

/// L4's range-estimation input (PRD-author-specified two-field shape):
/// this hand's in-progress action sequence (Section 4.3) plus the
/// opponent's completed-hand history (Section 4.4). `match_history` is
/// empty in Sprint 7's fixtures -- populated once Sprint 9 wires in the
/// real opponent-memory object.
#[derive(Debug, Clone, Default)]
pub struct ObservedActions {
    pub current_hand_actions: Vec<ActionRecord>,
    pub match_history: Vec<MatchHistoryEntry>,
}

/// All cards from a fresh 52-card deck not already accounted for by the
/// agent's own hole cards or the board dealt so far. Shared by L3 and L4
/// (both need the "remaining undealt deck" concept for Monte Carlo
/// sampling).
pub(crate) fn remaining_deck(hole: [Card; 2], board: &[Card]) -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for suit in 0..4u8 {
        for rank in 2..=14u8 {
            let card = Card { rank, suit };
            if card == hole[0] || card == hole[1] || board.contains(&card) {
                continue;
            }
            deck.push(card);
        }
    }
    deck
}