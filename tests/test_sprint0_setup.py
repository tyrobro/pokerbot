"""Sprint 0 Definition of Done (Section 9):

- `cargo test` and `pytest` both run with zero errors (even on empty suites).
- A trivial Rust function is callable from Python via the PyO3 binding.

This file covers the Python-side half of that DoD.
"""

import agents
import analysis
import ml
import orchestrator


def test_python_packages_are_importable():
    """The five top-level Python packages (Section 2.1 repo layout) must be
    importable, even while empty, before any sprint-specific code lands."""
    for pkg in (orchestrator, agents, ml, analysis):
        assert pkg is not None


def test_rust_binding_round_trip():
    """A trivial Rust function (core_engine::ping, exposed via the
    pokerbot_core PyO3 module) must be callable from Python."""
    import pokerbot_core

    assert pokerbot_core.ping() == 1
