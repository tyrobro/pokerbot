//! core_engine: Rust game engine for the Hold'em tournament simulation.

pub mod betting;
pub mod card;
pub mod game;
pub mod hand_eval;
pub mod match_engine;

pub use betting::{legal_actions, resolve_raise, ActionType, PotFraction, ResolvedAmount};
pub use card::{Card, Deck};
pub use game::{play_hand, Action, HandConfig, HandResult, HandView, Player, Street};
pub use hand_eval::{evaluate_5, evaluate_hand, HandCategory, HandRank};
pub use match_engine::{blinds_for_hand, play_match, HandSummary, MatchConfig, MatchResult};

/// Trivial sanity function exposed through the Python binding in `bindings/`.
pub fn ping() -> i64 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_returns_one() {
        assert_eq!(ping(), 1);
    }
}
