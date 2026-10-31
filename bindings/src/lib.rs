//! PyO3 binding crate: exposes core_engine to Python as pokerbot_core.
//! Sprint 5: full play_match binding + the Agent decision callback loop
//! (Section 4.3's decide_action(state) -> Action contract, called from
//! Rust into Python for every decision point in every hand).

use core_engine::betting::{ActionType as RustActionType, PotFraction as RustPotFraction};
use core_engine::card::Card;
use core_engine::game::{Action as RustAction, HandView as RustHandView, Player, Street};
use core_engine::match_engine::{play_match, HandSummary, MatchConfig, MatchResult};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::RefCell;

/// Trivial sanity function (Sprint 0) — kept for the Rust<->Python round
/// trip check.
#[pyfunction]
fn ping() -> PyResult<i64> {
    Ok(core_engine::ping())
}

/// Python-visible snapshot of a HandView (Section 4.3): attribute access
/// (e.g. `state.own_hole_cards`), not a dict. Cards are (rank, suit)
/// tuples; rank 2..=14 (11=J,12=Q,13=K,14=A), suit 0..=3 (unordered).
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

fn view_to_py(view: &RustHandView) -> PyHandView {
    PyHandView {
        own_hole_cards: cards_to_tuples(&view.own_hole_cards),
        board: cards_to_tuples(&view.board),
        pot_size: view.pot_size,
        own_stack: view.own_stack,
        opponent_stack: view.opponent_stack,
        bet_to_call: view.bet_to_call,
        is_button: view.is_button,
        street: street_to_str(view.street).to_string(),
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

/// Calls `agent.decide_action(view)` in Python and parses the returned
/// Action (Section 4.3 contract) back into a Rust Action.
fn call_agent(py: Python<'_>, agent: &PyObject, view: &RustHandView) -> PyResult<RustAction> {
    let py_view = view_to_py(view);
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
    dict.set_item("button", if s.button == Player::A { "A" } else { "B" })?;
    dict.set_item("small_blind", s.small_blind)?;
    dict.set_item("big_blind", s.big_blind)?;
    dict.set_item("stack_a_after", s.stack_a_after)?;
    dict.set_item("stack_b_after", s.stack_b_after)?;
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

/// Plays a full match between two Python Agent objects (Section 4.3),
/// driving the Rust engine end to end. `agent_a`/`agent_b` must each
/// implement `decide_action(state) -> Action` per `agents.base.Agent`.
#[pyfunction]
#[pyo3(signature = (
    agent_a,
    agent_b,
    starting_stack = 100_000,
    initial_small_blind = 50,
    initial_big_blind = 100,
    blind_escalation_interval_hands = 20,
    blind_escalation_multiplier_numerator = 6,
    blind_escalation_multiplier_denominator = 5,
    max_hands = 1000,
    seed = None,
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

    // Shared via RefCell (not a plain mut capture) so both closures can
    // hold only a shared `&` borrow of it simultaneously -- play_match
    // needs two independently-callable FnMut closures alive at once.
    let error: RefCell<Option<PyErr>> = RefCell::new(None);

    let mut py_agent_a = |view: &RustHandView| -> RustAction {
        match call_agent(py, &agent_a, view) {
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
        match call_agent(py, &agent_b, view) {
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

#[pymodule]
fn pokerbot_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHandView>()?;
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(play_match_py, m)?)?;
    Ok(())
}
