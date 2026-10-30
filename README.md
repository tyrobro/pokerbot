# Hold'em Tournament Engine

Implementation of `pokerbot.md` (the PRD). Sprint-by-sprint build; see that
file for the full spec and sprint plan (Section 9).

## Repo layout

```
/core_engine/   Rust crate: game engine, hand evaluator, CFR solver, raise logic
/bindings/      PyO3 crate exposing core_engine to Python as `pokerbot_core`
/ml/            Python: LSTM model definitions, training pipeline
/orchestrator/  Python: tournament loop, pairing generator, scoring, logging
/agents/        Python: agent interface implementations
/analysis/      Python: post-tournament analysis & plotting
/logs/          Parquet/CSV output (gitignored)
/tests/         Cross-language / integration tests (pytest)
```

## Setup

### This container (Linux/Ubuntu)

```
apt-get install -y cargo rustc            # Rust 1.75 via apt
python3 -m venv .venv
source .venv/bin/activate                 # bash, not sh
pip install maturin pytest
cd bindings && maturin develop && cd ..   # builds pokerbot_core into the venv
```

> **Pin note**: `core_engine/Cargo.toml` pins `proptest`, `getrandom`, and
> `tempfile` to older versions. The apt-packaged Cargo here is 1.75, which
> predates the `edition2024` Cargo feature that the latest releases of those
> crates (pulled in transitively) require. The pins are harmless on newer
> toolchains too, so they're left in place rather than made conditional.

### Target environment (Windows, per Section 2.1 of the PRD)

Follow Section 2.1 exactly: `rustup` (not apt) for Rust, `winget`/official
installer for Python 3.11+, `uv` or `poetry`, `maturin`, VS Build Tools (MSVC
linker), Git for Windows. A recent `rustup`-installed toolchain supports
`edition2024`, so the version pins above are optional there — but they don't
hurt.

## Verifying Sprint 0 (Definition of Done)

```
cargo test --workspace        # Rust: core_engine + bindings, 0 failures
source .venv/bin/activate
python -m pytest -v           # Python: package imports + Rust<->Python round trip
```

Both currently pass:
- `cargo test --workspace` — 1 test (`core_engine::tests::ping_returns_one`), 0 failures.
- `pytest` — 2 tests (package-import check, `pokerbot_core.ping()` round trip), 0 failures.

Sprint 0 deliverables (Section 9): repo skeleton ✅, empty Rust crate building
via `maturin develop` ✅, empty Python packages importable ✅, trivial Rust
function callable from Python via PyO3 ✅. CI config was left out (optional
per the DoD) — say the word if you want a GitHub Actions workflow added.
