use core_engine::play_hand;
use core_engine::{Action, ActionType, Deck, HandConfig, HandView, Player, Street, PotFraction};
use std::cell::RefCell;
use std::rc::Rc;

fn always_check_call(view: &HandView) -> Action {
    if view.bet_to_call == 0 {
        Action {
            action_type: ActionType::Check,
            size: None,
        }
    } else {
        Action {
            action_type: ActionType::Call,
            size: None,
        }
    }
}

/// Heads-up acting order (Section 3.1): SB/button acts first preflop,
/// BB/non-button acts first on every subsequent street. Explicitly tested
/// per the Sprint 3 DoD.
#[test]
fn acting_order_is_heads_up_specific() {
    let log: Rc<RefCell<Vec<(Street, bool)>>> = Rc::new(RefCell::new(Vec::new()));
    let log_a = Rc::clone(&log);
    let log_b = Rc::clone(&log);

    let config = HandConfig {
        button: Player::A,
        small_blind: 50,
        big_blind: 100,
        stack_a: 100_000,
        stack_b: 100_000,
    };
    let deck = Deck::new_seeded(1);

    let mut agent_a = move |view: &HandView| {
        log_a.borrow_mut().push((view.street, view.is_button));
        always_check_call(view)
    };
    let mut agent_b = move |view: &HandView| {
        log_b.borrow_mut().push((view.street, view.is_button));
        always_check_call(view)
    };

    let _ = play_hand(&config, deck, &mut agent_a, &mut agent_b);

    let log = log.borrow();
    let first_preflop = log.iter().find(|(s, _)| *s == Street::Preflop).unwrap();
    assert!(
        first_preflop.1,
        "button (small blind) must act first preflop"
    );

    let first_flop = log.iter().find(|(s, _)| *s == Street::Flop).unwrap();
    assert!(
        !first_flop.1,
        "non-button (big blind) must act first postflop"
    );
}

/// Full hand, two scripted always-check/call agents, manually checked pot
/// math (Section 8 DoD). Since neither agent ever raises, the only money
/// in the pot is the two blinds (each capped to 100 once the preflop call
/// matches them up) — a result checkable without knowing the dealt cards.
#[test]
fn full_hand_pot_math_is_conserved_and_correct_with_no_raises() {
    let config = HandConfig {
        button: Player::A,
        small_blind: 50,
        big_blind: 100,
        stack_a: 100_000,
        stack_b: 100_000,
    };
    let deck = Deck::new_seeded(7);

    let result = play_hand(
        &config,
        deck,
        &mut always_check_call,
        &mut always_check_call,
    );

    // Chip conservation: no chips created or destroyed.
    assert_eq!(result.final_stack_a + result.final_stack_b, 200_000);

    // pot = 200 (both players capped at 100 committed, no raises occurred).
    let possible_outcomes = [(100_100, 99_900), (99_900, 100_100), (100_000, 100_000)];
    assert!(
        possible_outcomes.contains(&(result.final_stack_a, result.final_stack_b)),
        "got ({}, {})",
        result.final_stack_a,
        result.final_stack_b
    );
}

/// A fold ends the hand immediately and awards the whole pot to the other
/// player — also checked by chip conservation plus the exact pot value.
#[test]
fn fold_on_first_action_awards_pot_to_opponent() {
    let config = HandConfig {
        button: Player::A,
        small_blind: 50,
        big_blind: 100,
        stack_a: 100_000,
        stack_b: 100_000,
    };
    let deck = Deck::new_seeded(3);

    // Button (A) acts first preflop and immediately folds.
    let mut agent_a = |_: &HandView| Action {
        action_type: ActionType::Fold,
        size: None,
    };
    let mut agent_b = always_check_call;

    let result = play_hand(&config, deck, &mut agent_a, &mut agent_b);

    assert!(result.ended_by_fold);
    assert_eq!(result.winner, Some(Player::B));
    // B wins the blinds pot: started 99,900 (after posting BB=100), gains
    // the full pot of 150 (A's 50 + B's own 100).
    assert_eq!(result.final_stack_b, 99_900 + 150);
    assert_eq!(result.final_stack_a, 99_950); // A only lost the SB
    assert_eq!(result.final_stack_a + result.final_stack_b, 200_000);
}

/// Regression test for a real bug found during Sprint 9 calibration: once
/// one player is all-in (stack 0), the other player could keep legally
/// re-raising against an opponent who can never respond, looping the
/// betting round indefinitely. This hand forces exactly that shape: B
/// starts with a tiny stack, A always raises -- the hand must still
/// terminate promptly once B is capped.
#[test]
fn hand_terminates_when_opponent_is_capped_all_in_and_other_player_keeps_raising() {
    let config = HandConfig {
        button: Player::A,
        small_blind: 50,
        big_blind: 100,
        stack_a: 100_000,
        stack_b: 300, // tiny -- will be all-in almost immediately
    };
    let deck = Deck::new_seeded(21);

    let mut agent_a = |view: &HandView| {
        if view.bet_to_call == 0 {
            Action {
                action_type: ActionType::Raise,
                size: Some(PotFraction::Full),
            }
        } else {
            Action {
                action_type: ActionType::Raise,
                size: Some(PotFraction::Full),
            }
        }
    };
    let mut agent_b = always_check_call;

    // The test itself is the assertion: this must return at all (previously
    // it would loop past the 1000-action safety cap and panic).
    let result = play_hand(&config, deck, &mut agent_a, &mut agent_b);
    assert_eq!(result.final_stack_a + result.final_stack_b, 100_300);
}
