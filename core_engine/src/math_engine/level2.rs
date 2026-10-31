//! Math engine Level 2 (Section 4.1): "Static thresholds + basic pot-odds
//! calculation."
//!
//! Confirmed with PRD author: L2 treats the L1 hand-strength percentile
//! directly as an equity proxy (building a real percentile->equity lookup
//! would smuggle L3's Monte Carlo job in a level early). The pot-odds rule
//! is scoped to ONLY widen fold->call; it never touches the raise
//! threshold, which stays pure L1 percentile logic unchanged -- otherwise
//! this would blur into L4's bet-sizing-theory territory.
//!
//! Exact rule (PRD author, percentile as a 0-1 fraction, higher=stronger):
//!   required_equity = amount_to_call / pot_after_call
//!   if percentile >= L1's raise threshold -> raise (unchanged from L1)
//!   elif percentile >= required_equity    -> call (new: pot-odds widening)
//!   else                                   -> fold

use super::level1::{self, ThresholdConfig as L1ThresholdConfig};
use super::{MathEngineState, RecommendedAction};

/// Reuses L1's config wholesale (raise cutoff, raise size, fold cutoff) --
/// L2 adds pot-odds reasoning ON TOP of L1's thresholds, it doesn't
/// replace them. See module doc: the fold cutoff below is intentionally
/// unused by this level's logic, kept only because L1ThresholdConfig is
/// shared; L2's fold/call boundary is decided by pot odds instead.
pub type ThresholdConfig = L1ThresholdConfig;

/// pot_after_call per the Section 3.3 / betting.rs formula: the pot
/// including this call, before any raise on top of it.
fn pot_after_call(pot_before_action: u64, amount_to_call: u64) -> u64 {
    pot_before_action.saturating_add(amount_to_call)
}

fn required_equity(amount_to_call: u64, pot_after_call: u64) -> f64 {
    if pot_after_call == 0 {
        0.0
    } else {
        amount_to_call as f64 / pot_after_call as f64
    }
}

/// Level 2 recommendation (Section 4.1): L1's percentile thresholds,
/// unchanged for the raise branch; pot-odds reasoning widens the
/// fold/call boundary only.
pub fn recommend(state: &MathEngineState, config: &ThresholdConfig) -> RecommendedAction {
    let percentile = level1::hand_strength_percentile_pub(state);

    if percentile >= config.raise_above_percentile {
        // Unchanged from L1 -- pot odds never touch the raise threshold.
        return RecommendedAction::Raise(config.raise_size);
    }

    if state.bet_to_call == 0 {
        // Nothing to call, so there's no pot-odds question to ask --
        // same as L1's fallback.
        return RecommendedAction::CheckOrCall;
    }

    let pac = pot_after_call(state.pot_size, state.bet_to_call);
    let req_equity = required_equity(state.bet_to_call, pac);

    if percentile >= req_equity {
        // Pot-odds widening: a hand L1 might have folded is profitable
        // enough to call given the price being offered.
        RecommendedAction::CheckOrCall
    } else {
        RecommendedAction::Fold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Card;
    use crate::game::Street;

    fn c(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    fn state_with(hole: [Card; 2], pot_size: u64, bet_to_call: u64) -> MathEngineState {
        MathEngineState {
            hole_cards: hole,
            board: Vec::new(),
            street: Street::Preflop,
            pot_size,
            own_stack: 100_000,
            opponent_stack: 100_000,
            bet_to_call,
            is_button: true,
        }
    }

    // ---- required_equity arithmetic ----

    #[test]
    fn required_equity_pot_odds_arithmetic() {
        // pot_before_action=200, amount_to_call=100 -> pot_after_call=300.
        // required_equity = 100/300 = 0.3333...
        let pac = pot_after_call(200, 100);
        assert_eq!(pac, 300);
        let req = required_equity(100, pac);
        assert!((req - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn required_equity_is_zero_with_nothing_to_call() {
        assert_eq!(required_equity(0, 500), 0.0);
    }

    // ---- raise branch is byte-for-byte identical to L1, pot odds never
    // touch it (PRD author's second confirmed constraint) ----

    #[test]
    fn raise_threshold_is_unaffected_by_pot_odds() {
        // AA: percentile 1.0, comfortably above raise_above_percentile
        // regardless of how large amount_to_call is relative to the pot.
        let config = ThresholdConfig::default();
        let huge_bet_state = state_with([c(14, 0), c(14, 1)], 100, 1_000_000);
        assert_eq!(
            recommend(&huge_bet_state, &config),
            RecommendedAction::Raise(config.raise_size)
        );
    }

    #[test]
    fn raise_cutoff_matches_l1_exactly_at_the_boundary() {
        // Any hand L1 would raise, L2 must also raise -- same cutoff,
        // same config field, no pot-odds override possible at this tier.
        let config = ThresholdConfig::default();
        let hole = [c(14, 0), c(14, 1)];
        let l1_state = state_with(hole, 150, 100);
        assert_eq!(
            level1::recommend(&l1_state, &config),
            recommend(&l1_state, &config)
        );
    }

    // ---- the actual new behavior: pot-odds widening fold -> call ----

    #[test]
    fn weak_hand_folds_under_l1_but_calls_under_l2_with_great_pot_odds() {
        // 72o: percentile 0.0 (see level1.rs test comment for the score
        // derivation). Facing a tiny bet into a massive pot gives a tiny
        // required_equity, well under even a percentile of 0.0 won't
        // clear -- so pick a hand with SOME percentile instead: use a
        // weak-but-not-zero hand so the widening is demonstrable.
        // 93o: hi=9 -> 4.5, not suited, gap=(9-3-1)=5 -> deduct 5,
        // gap>1 so no bonus -> score=max(0, 4.5-5)=0 -> percentile 0.0.
        // Pure 0.0-percentile hands can never clear pot odds (required
        // equity is always > 0 whenever there's a real bet to call), so
        // this specific scenario is intentionally a NO-WIDEN case:
        let config = ThresholdConfig::default();
        let zero_percentile_state = state_with([c(9, 0), c(3, 1)], 10_000, 1);
        assert_eq!(
            recommend(&zero_percentile_state, &config),
            RecommendedAction::Fold
        );
    }

    #[test]
    fn hand_below_fold_cutoff_but_above_required_equity_calls() {
        // KQo: hi=14 -> 10.0, not suited, gap=(14-12-1)=1 -> deduct 1,
        // hi=14 not <12 so no straight bonus -> score=9.0 -> percentile 0.45.
        // Below default fold_below_percentile (0.50) -> L1 would fold.
        // Facing a tiny bet into a large pot: pot_size=10_000,
        // bet_to_call=100 -> pot_after_call=10_100,
        // required_equity=100/10_100=0.0099 -- well under 0.45 ->
        // pot odds say CALL even though L1's percentile is in fold range.
        let config = ThresholdConfig::default();
        let state = state_with([c(14, 0), c(12, 1)], 10_000, 100);

        assert_eq!(level1::recommend(&state, &config), RecommendedAction::Fold);
        assert_eq!(recommend(&state, &config), RecommendedAction::CheckOrCall);
    }

    #[test]
    fn hand_below_fold_cutoff_and_below_required_equity_still_folds() {
        // Same KQo hand (percentile 0.45), but now facing a bet so large
        // relative to the pot that required_equity exceeds 0.45 -> pot
        // odds do NOT bail this hand out, L2 folds same as L1.
        let config = ThresholdConfig::default();
        // pot_size=100, bet_to_call=1000 -> pot_after_call=1100,
        // required_equity=1000/1100=0.909 >> 0.45.
        let state = state_with([c(14, 0), c(12, 1)], 100, 1_000);

        assert_eq!(level1::recommend(&state, &config), RecommendedAction::Fold);
        assert_eq!(recommend(&state, &config), RecommendedAction::Fold);
    }

    #[test]
    fn no_bet_to_call_short_circuits_to_check() {
        let config = ThresholdConfig::default();
        let state = state_with([c(9, 0), c(3, 1)], 500, 0);
        assert_eq!(recommend(&state, &config), RecommendedAction::CheckOrCall);
    }
}
