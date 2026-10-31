use core_engine::{play_match, Action, ActionType, HandView, MatchConfig};

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

/// "Full match to completion (cap or bust)" (Section 8 DoD). With blinds
/// escalating ×1.2 every 20 hands compounding over up to 1000 hands, a
/// bust before the hand cap is the EXPECTED outcome of Section 3.2's
/// schedule at full length (the escalation grows exponentially and will
/// eventually exceed any fixed starting stack) — so this test checks what's
/// actually guaranteed: the match terminates within the cap, and whichever
/// way it ends, chips are conserved.
#[test]
fn match_terminates_within_cap_and_conserves_chips() {
    let config = MatchConfig {
        seed: Some(7),
        ..MatchConfig::default() // max_hands = 1000
    };

    let result = play_match(&config, always_check_call, always_check_call);

    assert!(result.hands_played <= config.max_hands);
    assert_eq!(result.final_stack_a + result.final_stack_b, 200_000);
    assert_eq!(result.hand_summaries.len(), result.hands_played as usize);

    if result.busted {
        assert!(result.final_stack_a == 0 || result.final_stack_b == 0);
    } else {
        assert_eq!(result.hands_played, config.max_hands);
    }
}

/// Exercises the bust path specifically: a tiny starting stack relative
/// to the blinds forces a bust well before the hand cap, with escalation
/// disabled so the bust is attributable to blind pressure alone, not the
/// schedule's compounding.
#[test]
fn match_ends_early_when_a_stack_busts() {
    let config = MatchConfig {
        seed: Some(99),
        starting_stack: 300,
        initial_small_blind: 50,
        initial_big_blind: 100,
        blind_escalation_interval_hands: 0, // isolate the bust condition
        max_hands: 1_000,
        ..MatchConfig::default()
    };

    let result = play_match(&config, always_check_call, always_check_call);

    assert!(result.busted);
    assert!(result.hands_played < config.max_hands);
    assert!(result.final_stack_a == 0 || result.final_stack_b == 0);
    assert_eq!(result.final_stack_a + result.final_stack_b, 600);
}

/// Core reproducibility guarantee (Section 8): a fixed seed must produce
/// byte-identical results across repeated runs.
#[test]
fn fixed_seed_match_is_byte_identical_across_repeated_runs() {
    let config = MatchConfig {
        seed: Some(42),
        max_hands: 50, // short run is plenty to prove the property
        ..MatchConfig::default()
    };

    let run = || play_match(&config, always_check_call, always_check_call);

    let result_1 = run();
    let result_2 = run();

    assert_eq!(result_1.hand_summaries, result_2.hand_summaries);
    assert_eq!(result_1.final_stack_a, result_2.final_stack_a);
    assert_eq!(result_1.final_stack_b, result_2.final_stack_b);
    assert_eq!(result_1.hands_played, result_2.hands_played);
}
