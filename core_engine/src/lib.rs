//! core_engine: Rust game engine for the Hold'em tournament simulation.
//!
//! Sprint 0 scope: empty crate skeleton that builds and tests cleanly, plus a
//! trivial function used to prove the PyO3 binding round-trip works end to
//! end (Section 2.1, Sprint 0 DoD). Real game logic (deck, evaluator, betting,
//! CFR solver, etc.) is added starting in Sprint 1.

/// Trivial sanity function exposed through the Python binding in `bindings/`.
/// Used only to prove the Rust -> PyO3 -> Python call path works.
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
