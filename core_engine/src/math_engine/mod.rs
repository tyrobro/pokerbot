//! Math engine levels 1-5 (Section 4.1). Each level implements
//! `recommend(state) -> RecommendedAction` deterministically. Sprint 7
//! builds these incrementally, level by level.

pub mod level1;
pub mod level2;

use crate::betting::PotFraction;
use crate::card::Card;
use crate::game::Street;

/// A math engine's pre-blend recommendation (Section 4.1's
/// `baseline_action` in the combination-rule formula). Deliberately
/// mirrors `game::Action`'s shape but kept as its own type: math engines
/// reason in this vocabulary before any ML blending (Section 4.1
/// combination rule) is applied on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    Fold,
    CheckOrCall,
    Raise(PotFraction),
}

/// The subset of hand/game state a math engine needs to decide. A
/// deliberately narrower view than `game::HandView` — math engines never
/// see anything beyond what's legal for them to use at their level (L1-2:
/// no opponent-read signal at all; that's what the ML engine is for).
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
