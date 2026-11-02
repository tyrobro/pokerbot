//! PyO3 binding crate: exposes core_engine to Python as pokerbot_core.
//! Sprint 9: closes two Sprint 7 preconditions.
//!   1. L3/L4/L5 Rust calls now take an explicit `seed`, threaded from
//!      Python (agents/math_agents.py derives a fresh deterministic seed
//!      per decision from each agent's own base seed) -- MonteCarloConfig
//!      is never `seed: None` from these entry points anymore.
//!   2. L4's current-hand opponent actions and L5's hand_log are built
//!      from the live HandView's new `actions_so_far` field (real data),
//!      not an always-empty default/Vec::new().
//! NOT closed this sprint (flagged): L4/L5's own Rust-side
//! `match_history` field (cross-PAIRING memory inside the equity calls
//! themselves) stays empty -- only current-hand data is threaded. Cross-
//! pairing opponent memory IS correctly implemented, but on the Python
//! side only (PD archetypes' observe_hand_result/flush, orchestrator/
//! opponent_memory.py), not serialized back into these Rust calls.
//! Also added: `play_hand_py`, a single-hand binding -- needed so the
//! Python-side round-robin driver (Sprint 9) can call observe_hand_result
//! after every hand (Section 4.4), which `play_match_py`'s all-in-one-call
//! design cannot support.

use core_engine::betting::{ActionType as RustActionType, PotFraction as RustPotFraction};
use core_engine::card::{Card, Deck};
use core_engine::game::{
    play_hand, Action as RustAction, HandConfig, HandView as RustHandView, Player, Street,
};
use core_engine::match_engine::{play_match, HandSummary, MatchConfig, MatchResult};
use core_engine::math_engine::level3::MonteCarloConfig;
use core_engine::math_engine::level5::{
    Actor, BucketConfig, CFRConfig, CFRSolver, HandActionLogEntry, LbrConfig,
};
use core_engine::math_engine::{
    level1, level2, level3, level4, ActionRecord, MathEngineState as RustMathState,
    ObservedActions, RecommendedAction,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;

#[pyfunction]
fn ping() -> PyResult<i64> {
    Ok(core_engine::ping())
}

#[pyclass]
#[derive(Clone)]
struct PyHandView {
    #[pyo3(get)]
    own_hole_cards: Vec<(u8, u8)>,
    #[pyo3(get)]
    board: Vec<(u8, u8)>,
    #[pyo3(get)]
    pot_size: u64,
    #[pyo3(get)]
    own_stack: u64,
    #[pyo3(get)]
    opponent_stack: u64,
    #[pyo3(get)]
    bet_to_call: u64,
    #[pyo3(get)]
    is_button: bool,
    #[pyo3(get)]
    street: String,
    #[pyo3(get)]
    own_player: String,
    /// (street, player, action_type, amount, pot_after), in order.
    #[pyo3(get)]
    actions_so_far: Vec<(String, String, String, u64, u64)>,
}

fn cards_to_tuples(cards: &[Card]) -> Vec<(u8, u8)> {
    cards.iter().map(|c| (c.rank, c.suit)).collect()
}

fn street_to_str(street: Street) -> &'static str {
    match street {
        Street::Preflop => "PREFLOP",
        Street::Flop => "FLOP",
        Street::Turn => "TURN",
        Street::River => "RIVER",
    }
}

fn parse_street(s: &str) -> PyResult<Street> {
    match s {
        "PREFLOP" => Ok(Street::Preflop),
        "FLOP" => Ok(Street::Flop),
        "TURN" => Ok(Street::Turn),
        "RIVER" => Ok(Street::River),
        other => Err(PyValueError::new_err(format!("unknown street {other:?}"))),
    }
}

fn action_type_to_str(a: RustActionType) -> &'static str {
    match a {
        RustActionType::Fold => "FOLD",
        RustActionType::Check => "CHECK",
        RustActionType::Call => "CALL",
        RustActionType::Raise => "RAISE",
        RustActionType::AllIn => "ALL_IN",
    }
}

fn player_to_str(p: Player) -> &'static str {
    match p {
        Player::A => "A",
        Player::B => "B",
    }
}

fn parse_player(s: &str) -> PyResult<Player> {
    match s {
        "A" => Ok(Player::A),
        "B" => Ok(Player::B),
        other => Err(PyValueError::new_err(format!("unknown player {other:?}"))),
    }
}

fn parse_action_type(s: &str) -> PyResult<RustActionType> {
    match s {
        "FOLD" => Ok(RustActionType::Fold),
        "CHECK" => Ok(RustActionType::Check),
        "CALL" => Ok(RustActionType::Call),
        "RAISE" => Ok(RustActionType::Raise),
        "ALL_IN" => Ok(RustActionType::AllIn),
        other => Err(PyValueError::new_err(format!(
            "unknown action_type {other:?}; expected one of FOLD, CHECK, CALL, RAISE, ALL_IN"
        ))),
    }
}

fn parse_pot_fraction(s: &str) -> PyResult<RustPotFraction> {
    match s {
        "QUARTER" => Ok(RustPotFraction::Quarter),
        "HALF" => Ok(RustPotFraction::Half),
        "THREE_QUARTER" => Ok(RustPotFraction::ThreeQuarter),
        "FULL" => Ok(RustPotFraction::Full),
        other => Err(PyValueError::new_err(format!(
            "unknown pot fraction {other:?}; expected one of QUARTER, HALF, THREE_QUARTER, FULL"
        ))),
    }
}

fn pot_fraction_to_str(frac: RustPotFraction) -> &'static str {
    match frac {
        RustPotFraction::Quarter => "QUARTER",
        RustPotFraction::Half => "HALF",
        RustPotFraction::ThreeQuarter => "THREE_QUARTER",
        RustPotFraction::Full => "FULL",
    }
}

fn view_to_py(view: &RustHandView, own_player: Player) -> PyHandView {
    let actions_so_far = view
        .actions_so_far
        .iter()
        .zip(view.pot_progression_so_far.iter())
        .map(|(a, &pot_after)| {
            (
                street_to_str(a.street).to_string(),
                player_to_str(a.player).to_string(),
                action_type_to_str(a.action_type).to_string(),
                a.amount,
                pot_after,
            )
        })
        .collect();

    PyHandView {
        own_hole_cards: cards_to_tuples(&view.own_hole_cards),
        board: cards_to_tuples(&view.board),
        pot_size: view.pot_size,
        own_stack: view.own_stack,
        opponent_stack: view.opponent_stack,
        bet_to_call: view.bet_to_call,
        is_button: view.is_button,
        street: street_to_str(view.street).to_string(),
        own_player: player_to_str(own_player).to_string(),
        actions_so_far,
    }
}

fn call_agent(
    py: Python<'_>,
    agent: &PyObject,
    view: &RustHandView,
    own_player: Player,
) -> PyResult<RustAction> {
    let py_view = view_to_py(view, own_player);
    let result = agent.call_method1(py, "decide_action", (py_view,))?;
    let result = result.bind(py);

    let action_type_str: String = result.getattr("action_type")?.extract()?;
    let action_type = parse_action_type(&action_type_str)?;

    let size_obj = result.getattr("size")?;
    let size = if size_obj.is_none() {
        None
    } else {
        let size_str: String = size_obj.extract()?;
        Some(parse_pot_fraction(&size_str)?)
    };

    Ok(RustAction { action_type, size })
}

fn hand_summary_to_py(py: Python<'_>, s: &HandSummary) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("hand_number", s.hand_number)?;
    dict.set_item("blind_level", s.blind_level)?;
    dict.set_item("button", player_to_str(s.button))?;
    dict.set_item("small_blind", s.small_blind)?;
    dict.set_item("big_blind", s.big_blind)?;
    dict.set_item("stack_a_after", s.stack_a_after)?;
    dict.set_item("stack_b_after", s.stack_b_after)?;
    dict.set_item("hole_a", cards_to_tuples(&s.hole_a))?;
    dict.set_item("hole_b", cards_to_tuples(&s.hole_b))?;
    dict.set_item("board", cards_to_tuples(&s.board))?;

    let actions: PyResult<Vec<PyObject>> = s
        .actions
        .iter()
        .map(|a| -> PyResult<PyObject> {
            let d = PyDict::new(py);
            d.set_item("street", street_to_str(a.street))?;
            d.set_item("player", player_to_str(a.player))?;
            d.set_item("action_type", action_type_to_str(a.action_type))?;
            d.set_item("amount", a.amount)?;
            Ok(d.into_any().unbind())
        })
        .collect();
    dict.set_item("actions", actions?)?;

    dict.set_item("pot_progression", s.pot_progression.clone())?;
    dict.set_item("ended_by_fold", s.ended_by_fold)?;
    dict.set_item("winner", s.winner.map(player_to_str).unwrap_or("SPLIT"))?;
    dict.set_item("pot_awarded", s.pot_awarded)?;
    Ok(dict.into_any().unbind())
}

fn match_result_to_py(py: Python<'_>, result: MatchResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("hands_played", result.hands_played)?;
    dict.set_item("final_stack_a", result.final_stack_a)?;
    dict.set_item("final_stack_b", result.final_stack_b)?;
    dict.set_item("busted", result.busted)?;
    let summaries: PyResult<Vec<PyObject>> = result
        .hand_summaries
        .iter()
        .map(|s| hand_summary_to_py(py, s))
        .collect();
    dict.set_item("hand_summaries", summaries?)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (
    agent_a, agent_b, starting_stack = 100_000, initial_small_blind = 50,
    initial_big_blind = 100, blind_escalation_interval_hands = 20,
    blind_escalation_multiplier_numerator = 6, blind_escalation_multiplier_denominator = 5,
    max_hands = 1000, seed = None,
))]
#[allow(clippy::too_many_arguments)]
fn play_match_py(
    py: Python<'_>,
    agent_a: PyObject,
    agent_b: PyObject,
    starting_stack: u64,
    initial_small_blind: u64,
    initial_big_blind: u64,
    blind_escalation_interval_hands: u32,
    blind_escalation_multiplier_numerator: u64,
    blind_escalation_multiplier_denominator: u64,
    max_hands: u32,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    let config = MatchConfig {
        starting_stack,
        initial_small_blind,
        initial_big_blind,
        blind_escalation_interval_hands,
        blind_escalation_multiplier_numerator,
        blind_escalation_multiplier_denominator,
        max_hands,
        seed,
    };

    let error: RefCell<Option<PyErr>> = RefCell::new(None);

    let mut py_agent_a = |view: &RustHandView| -> RustAction {
        match call_agent(py, &agent_a, view, Player::A) {
            Ok(action) => action,
            Err(e) => {
                if error.borrow().is_none() {
                    *error.borrow_mut() = Some(e);
                }
                RustAction {
                    action_type: RustActionType::Fold,
                    size: None,
                }
            }
        }
    };
    let mut py_agent_b = |view: &RustHandView| -> RustAction {
        match call_agent(py, &agent_b, view, Player::B) {
            Ok(action) => action,
            Err(e) => {
                if error.borrow().is_none() {
                    *error.borrow_mut() = Some(e);
                }
                RustAction {
                    action_type: RustActionType::Fold,
                    size: None,
                }
            }
        }
    };

    let result = play_match(&config, &mut py_agent_a, &mut py_agent_b);
    if let Some(e) = error.into_inner() {
        return Err(e);
    }
    match_result_to_py(py, result)
}

/// Single-hand binding (Sprint 9): needed so the Python-side round-robin
/// driver can call observe_hand_result (Section 4.4) after every hand,
/// which play_match_py's all-in-one-call design cannot support.
#[pyfunction]
#[pyo3(signature = (agent_a, agent_b, button, small_blind, big_blind, stack_a, stack_b, seed = None))]
#[allow(clippy::too_many_arguments)]
fn play_hand_py(
    py: Python<'_>,
    agent_a: PyObject,
    agent_b: PyObject,
    button: String,
    small_blind: u64,
    big_blind: u64,
    stack_a: u64,
    stack_b: u64,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    let button_player = parse_player(&button)?;
    let config = HandConfig {
        button: button_player,
        small_blind,
        big_blind,
        stack_a,
        stack_b,
    };
    let deck = match seed {
        Some(s) => Deck::new_seeded(s),
        None => Deck::new(),
    };

    let error: RefCell<Option<PyErr>> = RefCell::new(None);
    let mut py_agent_a = |view: &RustHandView| -> RustAction {
        match call_agent(py, &agent_a, view, Player::A) {
            Ok(a) => a,
            Err(e) => {
                if error.borrow().is_none() {
                    *error.borrow_mut() = Some(e);
                }
                RustAction {
                    action_type: RustActionType::Fold,
                    size: None,
                }
            }
        }
    };
    let mut py_agent_b = |view: &RustHandView| -> RustAction {
        match call_agent(py, &agent_b, view, Player::B) {
            Ok(a) => a,
            Err(e) => {
                if error.borrow().is_none() {
                    *error.borrow_mut() = Some(e);
                }
                RustAction {
                    action_type: RustActionType::Fold,
                    size: None,
                }
            }
        }
    };

    let result = play_hand(&config, deck, &mut py_agent_a, &mut py_agent_b);
    if let Some(e) = error.into_inner() {
        return Err(e);
    }

    let dict = PyDict::new(py);
    dict.set_item("ended_by_fold", result.ended_by_fold)?;
    dict.set_item(
        "winner",
        result.winner.map(player_to_str).unwrap_or("SPLIT"),
    )?;
    dict.set_item("final_stack_a", result.final_stack_a)?;
    dict.set_item("final_stack_b", result.final_stack_b)?;
    dict.set_item("pot_awarded", result.pot_awarded)?;
    dict.set_item("hole_a", cards_to_tuples(&result.hole_a))?;
    dict.set_item("hole_b", cards_to_tuples(&result.hole_b))?;
    dict.set_item("board", cards_to_tuples(&result.board))?;
    let actions: PyResult<Vec<PyObject>> = result
        .actions
        .iter()
        .map(|a| -> PyResult<PyObject> {
            let d = PyDict::new(py);
            d.set_item("street", street_to_str(a.street))?;
            d.set_item("player", player_to_str(a.player))?;
            d.set_item("action_type", action_type_to_str(a.action_type))?;
            d.set_item("amount", a.amount)?;
            Ok(d.into_any().unbind())
        })
        .collect();
    dict.set_item("actions", actions?)?;
    dict.set_item("pot_progression", result.pot_progression.clone())?;
    Ok(dict.into_any().unbind())
}

// ---- math-engine level recommend() bindings ----

fn py_state_to_rust(view: &PyHandView) -> PyResult<RustMathState> {
    if view.own_hole_cards.len() != 2 {
        return Err(PyValueError::new_err(
            "own_hole_cards must have exactly 2 cards",
        ));
    }
    let hole_cards = [
        Card {
            rank: view.own_hole_cards[0].0,
            suit: view.own_hole_cards[0].1,
        },
        Card {
            rank: view.own_hole_cards[1].0,
            suit: view.own_hole_cards[1].1,
        },
    ];
    let board: Vec<Card> = view
        .board
        .iter()
        .map(|&(r, s)| Card { rank: r, suit: s })
        .collect();
    let street = parse_street(&view.street)?;
    Ok(RustMathState {
        hole_cards,
        board,
        street,
        pot_size: view.pot_size,
        own_stack: view.own_stack,
        opponent_stack: view.opponent_stack,
        bet_to_call: view.bet_to_call,
        is_button: view.is_button,
    })
}

/// Builds ObservedActions.current_hand_actions from the live view's
/// actions_so_far, filtered to the OPPONENT's actions only (L4's
/// ObservedActions contract). pot_size_at_time uses the pot AFTER the
/// action (an approximation, flagged -- not the pot immediately before
/// it, which would need one more field threaded through).
/// match_history stays empty -- NOT closed this sprint (see module doc).
fn build_observed_actions(view: &PyHandView) -> PyResult<ObservedActions> {
    let own_player = parse_player(&view.own_player)?;
    let mut current_hand_actions = Vec::new();
    for (street_s, player_s, action_s, amount, pot_after) in &view.actions_so_far {
        let player = parse_player(player_s)?;
        if player == own_player {
            continue;
        }
        current_hand_actions.push(ActionRecord {
            street: parse_street(street_s)?,
            action_type: parse_action_type(action_s)?,
            amount: *amount,
            pot_size_at_time: *pot_after,
        });
    }
    Ok(ObservedActions {
        current_hand_actions,
        match_history: Vec::new(),
    })
}

/// Builds L5's full-hand (both players') action log from the live view's
/// actions_so_far -- closes the "confidently wrong InfoSetKey" gap
/// flagged at the end of Sprint 7 (hand_log is no longer always empty).
fn build_hand_log(view: &PyHandView) -> PyResult<Vec<HandActionLogEntry>> {
    let own_player = parse_player(&view.own_player)?;
    let mut log = Vec::new();
    for (street_s, player_s, action_s, amount, pot_after) in &view.actions_so_far {
        let player = parse_player(player_s)?;
        let actor = if player == own_player {
            Actor::Own
        } else {
            Actor::Opponent
        };
        log.push(HandActionLogEntry {
            actor,
            record: ActionRecord {
                street: parse_street(street_s)?,
                action_type: parse_action_type(action_s)?,
                amount: *amount,
                pot_size_at_time: *pot_after,
            },
        });
    }
    Ok(log)
}

fn recommended_to_py(action: RecommendedAction) -> (String, Option<String>) {
    match action {
        RecommendedAction::Fold => ("FOLD".to_string(), None),
        RecommendedAction::CheckOrCall => ("CHECKORCALL".to_string(), None),
        RecommendedAction::Raise(frac) => (
            "RAISE".to_string(),
            Some(pot_fraction_to_str(frac).to_string()),
        ),
    }
}

#[pyfunction]
fn l1_recommend_py(view: PyHandView) -> PyResult<(String, Option<String>)> {
    let state = py_state_to_rust(&view)?;
    let config = level1::ThresholdConfig::default();
    Ok(recommended_to_py(level1::recommend(&state, &config)))
}

#[pyfunction]
fn l2_recommend_py(view: PyHandView) -> PyResult<(String, Option<String>)> {
    let state = py_state_to_rust(&view)?;
    let config = level2::ThresholdConfig::default();
    Ok(recommended_to_py(level2::recommend(&state, &config)))
}

/// Sprint 9: now takes an explicit `seed` -- threaded from Python's
/// per-agent deterministic counter, never MonteCarloConfig::default()'s
/// seed=None.
#[pyfunction]
fn l3_recommend_py(view: PyHandView, seed: u64) -> PyResult<(String, Option<String>)> {
    let state = py_state_to_rust(&view)?;
    let threshold = level3::ThresholdConfig::default();
    let mc = MonteCarloConfig {
        trials: 2000,
        seed: Some(seed),
    };
    Ok(recommended_to_py(level3::recommend(
        &state, &threshold, &mc,
    )))
}

/// Sprint 9: seeded, AND now reads real current-hand opponent actions
/// via build_observed_actions (previously always ObservedActions::default()).
#[pyfunction]
fn l4_recommend_py(view: PyHandView, seed: u64) -> PyResult<(String, Option<String>)> {
    let state = py_state_to_rust(&view)?;
    let observed = build_observed_actions(&view)?;
    let threshold = level4::ThresholdConfig::default();
    let range = level4::RangeConfig::default();
    let mc = MonteCarloConfig {
        trials: 2000,
        seed: Some(seed),
    };
    Ok(recommended_to_py(level4::recommend(
        &state, &observed, &threshold, &range, &mc,
    )))
}

#[pyfunction]
fn estimate_hand_strength_percentile_py(view: PyHandView) -> PyResult<f64> {
    let state = py_state_to_rust(&view)?;
    Ok(level1::hand_strength_percentile_pub(&state))
}

#[pyclass]
struct PyCFRSolver {
    inner: CFRSolver,
}

#[pymethods]
impl PyCFRSolver {
    #[new]
    #[pyo3(signature = (num_buckets = 6, bucketing_trials = 50, max_raises_per_street = 1, seed = 1, train_iterations = 50))]
    fn new(
        num_buckets: u8,
        bucketing_trials: u32,
        max_raises_per_street: u8,
        seed: u64,
        train_iterations: u32,
    ) -> Self {
        let config = CFRConfig {
            buckets: BucketConfig {
                num_buckets,
                bucketing_trials,
            },
            max_raises_per_street,
        };
        let mut inner = CFRSolver::new(config, seed);
        inner.train(train_iterations, 0, &LbrConfig::default());
        PyCFRSolver { inner }
    }

    /// Sprint 9: `seed` is now required (threaded from Python), and
    /// hand_log/observed are built from the live view's real
    /// actions_so_far (previously always empty).
    fn recommend(&self, view: PyHandView, seed: u64) -> PyResult<(String, Option<String>)> {
        let state = py_state_to_rust(&view)?;
        let observed = build_observed_actions(&view)?;
        let hand_log = build_hand_log(&view)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let action = core_engine::math_engine::level5::l5_recommend(
            &state,
            &hand_log,
            &observed,
            &self.inner,
            &mut rng,
            &level4::ThresholdConfig::default(),
            &level4::RangeConfig::default(),
            &MonteCarloConfig::default(),
        );
        Ok(recommended_to_py(action))
    }
}

#[pymodule]
fn pokerbot_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHandView>()?;
    m.add_class::<PyCFRSolver>()?;
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(play_match_py, m)?)?;
    m.add_function(wrap_pyfunction!(play_hand_py, m)?)?;
    m.add_function(wrap_pyfunction!(l1_recommend_py, m)?)?;
    m.add_function(wrap_pyfunction!(l2_recommend_py, m)?)?;
    m.add_function(wrap_pyfunction!(l3_recommend_py, m)?)?;
    m.add_function(wrap_pyfunction!(l4_recommend_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hand_strength_percentile_py, m)?)?;
    Ok(())
}
