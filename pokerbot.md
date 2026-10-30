# Hold'em Tournament Engine — Production Spec (PRD)

## 1. Overview & Goal

Build a heads-up No-Limit Texas Hold'em simulation engine in which 30 distinct agents compete in a full round-robin tournament. Each agent is defined by a combination of a **math engine** (deterministic probability/game-theory reasoning, 5 levels) and an **ML engine** (behavioral/opponent-modeling, 5 levels), producing a 5×5 = 25-agent grid, plus 5 additional hand-built "Prisoner's Dilemma archetype" agents (Tit-for-Tat, Always-Defect, Always-Cooperate, Pavlov, Grudger) — **30 agents total**.

The system must: run every pairing of agents against each other over two button-swapped legs, log every hand in full detail, score performance on a normalized scale, support a fixed-seed calibration phase followed by a fully random final run, and produce aggregate rankings with variance/win/loss analysis.

This document is the single source of truth. Every open design question discussed during planning has been resolved below. Nothing in this document should require further interpretation during implementation — if an implementer hits an ambiguity not covered here, that is a gap in this PRD and should be raised, not guessed.

---

## 2. Tech Stack

| Layer | Choice | Rationale |
| --- | --- | --- |
| Core game engine (dealing, hand evaluation, betting legality, pot math, denomination/raise resolution, CFR solver) | **Rust** | Performance-critical, run millions of times across training + tournament; memory safety without GC pauses matters for a long-running simulation. |
| Python bindings for the Rust core | **PyO3 + maturin** | Exposes the Rust engine as a native Python module so orchestration code stays in Python without FFI overhead per call. |
| ML engines (LSTM opponent models) | **Python + PyTorch** | Rust ML ecosystem is immature for this; PyTorch is the industry-standard choice for sequence models and has the best tooling for iterative/curriculum training. |
| ML inference in the hot loop | **ONNX Runtime** (export trained PyTorch models to ONNX) | Avoids pulling the full PyTorch runtime into the tournament loop; ONNX Runtime has Python and Rust (`ort` crate) bindings, so inference can be called from either side depending on where the orchestration loop lives. |
| Tournament orchestration (pairing generation, match loop driver, scoring, logging) | **Python** | Fast iteration, rich data/analysis ecosystem, calls into Rust engine via PyO3 and into ML models via ONNX Runtime. |
| Data storage for hand logs | **Parquet files** (via `pyarrow`), one directory per match leg | Columnar, compresses well, easy to load into pandas/polars for analysis; avoids standing up a database for a local simulation project. |
| Analysis & visualization (win/loss graphs, variance tracking) | **Python: pandas/polars + matplotlib or plotly** | Standard, sufficient for static post-hoc analysis; plotly if interactive exploration is wanted. |
| Build/dependency management | **Cargo workspaces** (Rust side), **uv or poetry** (Python side) | Standard modern tooling for each ecosystem. |
| Testing | **`cargo test`** (Rust unit/integration), **`pytest`** (Python unit/integration), **`proptest`** (Rust property-based testing for engine invariants) | See Section 8. |
| Version control / CI | **Git**, **GitHub Actions** (or local pre-commit hooks if no remote CI desired) | Each sprint's definition of done includes passing tests; CI enforces this automatically if set up. |

### 2.1 Windows Environment Setup

1. Install **Rust** via `rustup` (https://rustup.rs) — installs `cargo`, `rustc`. Use the default stable toolchain.
2. Install **Python 3.11+** (via official installer or `winget install Python.Python.3.11`).
3. Install **uv** (`pipx install uv` or the standalone installer) for fast Python dependency management, or use `poetry` if preferred — pick one and use it consistently.
4. Install **maturin** (`pip install maturin` or `uv pip install maturin`) — builds the Rust core as an installable Python wheel (`maturin develop` for local iterative dev builds).
5. Install **Visual Studio Build Tools** (C++ build tools workload) — required by Rust on Windows for the MSVC linker.
6. Install **Git for Windows**.
7. Recommended: **VS Code** with the `rust-analyzer` and Python extensions.
8. Verify setup: `cargo --version`, `python --version`, `maturin --version` all resolve; `cargo new --lib core_engine && cd core_engine && cargo test` runs a clean empty test suite with no errors.

Repository layout:

```
/core_engine/      # Rust crate: game engine, hand evaluator, CFR solver, denomination/raise logic
/bindings/         # PyO3 binding crate exposing core_engine to Python
/ml/               # Python: LSTM model definitions, training pipeline, curriculum self-play scripts
/orchestrator/     # Python: tournament loop, pairing generator, scoring, logging
/agents/           # Python: agent interface implementations (math-only, PD archetypes, hybrid grid)
/analysis/         # Python: post-tournament analysis & plotting scripts
/logs/             # Parquet output, gitignored
/tests/            # Cross-language integration tests
```

---

## 3. Locked Game Rules

### 3.1 Match Structure

- **Format**: Heads-up (2 players) No-Limit Texas Hold'em only. No side pots are ever needed (hard invariant — enforce with an assertion in code, since multiway pot-splitting logic should never be reachable).
- **Deck**: Standard 52 cards, fresh cryptographically-seeded shuffle every hand (no continuous shoe).
- **Starting stack**: 100,000 units for each agent, every match leg.
- **Match length**: ends at **1000 hands** or when either stack hits **0**, whichever comes first.
- **Hand ranking**: standard poker hand rankings; best 5-of-7 (2 hole cards + 5 board cards).
- **Ties**: split pot evenly. Odd remainder chip (from the {1,2,5,10} denomination set) goes to the **non-button player**.
- **Acting order (heads-up specific — do not use multiway ordering)**: small blind acts first preflop; big blind acts first on every subsequent street (flop, turn, river).
- **Button**: randomized coin flip for the very first hand of a match leg, then strictly alternates every hand thereafter. Each leg of a pairing starts with the button flipped relative to the other leg.
- **All-in before river**: deal out the remaining board once (single run-out). "Run it twice" is explicitly **not** implemented.
- **Short all-in** (all-in for less than a full call): treated as a call for the amount they have; does not reopen betting (moot in heads-up but state explicitly for correctness).

### 3.2 Blinds

- Escalate by **×1.2 every 20 hands**, starting from an initial small/big blind value to be fixed during the calibration sprint (Sprint 9).
- This rate is a **starting point**, explicitly subject to recalibration after pilot/calibration runs (Sprint 9) — do not hard-code it as immutable; expose it as a configurable schedule parameter.

### 3.3 Chips & Raise Resolution

- Chip denominations: **{1, 2, 5, 10}** units.
- Denomination tiers for raise rounding, **fixed for the entire match** (do not scale with blind level):

| Pot size range | Raise must be a multiple of |
| --- | --- |
| 0 – 1,000 | 1 |
| 1,000 – 10,000 | 5 |
| 10,000 – 50,000 | 10 |
| 50,000+ | 50 |

- **Allowed raise sizes** (before rounding/clamping): fold, check/call, or raise sized at **¼, ½, ¾, or full pot**, or **all-in**. No other bet sizes are legal.
- **Raise resolution — exact order of operations (implement as a single pure function, unit-test every branch):**

1. Compute raw target raise = chosen pot-fraction × current pot size.
2. Round **up** to the nearest legal multiple for the active denomination tier (never round down).
3. If the result is still below the minimum legal raise (≥ size of the previous bet/raise this betting round), bump up to the next legal multiple that satisfies the minimum-raise rule.
4. If the result exceeds the agent's remaining stack, override to **all-in** for the full remaining stack. This step always takes precedence over steps 2–3.

- **No illegal moves are ever possible** — every agent-chosen action is passed through this resolution function before being applied to game state.

---

## 4. Agent Architecture

### 4.1 The 25-Agent Grid

Each grid agent is defined by `(math_level, ml_level)`, each ∈ {1,2,3,4,5}.

**Math engine levels** (deterministic, Rust):

1. Crude heuristics — hand-strength percentile thresholds only, no pot odds.
2. Static thresholds + basic pot-odds calculation.
3. Monte Carlo equity estimation vs. an assumed random opponent range.
4. Range-vs-range equity (opponent range estimated from observed action, not assumed random), position-aware, basic bet-sizing theory.
5. Full abstracted CFR blueprint strategy (card/bet abstraction + Discounted CFR solve), optionally refined in real time for the current subgame.

**ML engine levels** (behavioral, Python/PyTorch → ONNX at inference time): a **single LSTM architecture family** across all 5 levels — levels differ in model capacity (hidden size, sequence length considered, input feature richness) and in curriculum training stage, **not** in architecture family.

1. No opponent modeling — ignores all history.
2. Simple running statistics (fold-to-bet %, aggression factor) — counting, not a trained model.
3. Small LSTM trained on an early curriculum stage (weak opponents) → coarse behavioral read.
4. Medium LSTM, later curriculum stage (stronger opponents), sequence-aware range adjustment signal.
5. Full-capacity LSTM, final curriculum stage (near-equilibrium opponents), highest confidence-weighted influence on final decisions; may additionally incorporate board-texture features.

**Combination rule** (applies to every grid agent, all 25 combinations):

```
baseline_action = math_engine(level=m).recommend(state)
exploit_signal, confidence = ml_engine(level=k).read_opponent(state, opponent_history)
final_action = baseline_action + confidence_weight(k) × (exploit_signal − baseline_action)
```

`confidence_weight(k)` scales with ML level — level 1 has `confidence_weight ≡ 0` (pure baseline, no deviation ever), level 5 has the highest ceiling on how far it's allowed to deviate from baseline. Exact numeric scaling (e.g. linear vs. logistic in `k`) is an implementation-time tuning parameter, not a blocking decision — pick linear (`confidence_weight(k) = k/5`) as the default and revisit only if calibration results look wrong.

### 4.2 PD Archetype Agents (5 additional agents, hand-built, not part of the grid)

1. **Tit-for-Tat → Mirror aggression**: starts at a standard/GTO-baseline aggression level; shifts aggression up/down each hand in proportion to a rolling window of the opponent's recent aggression; decays old reads over time (forgiving).
2. **Always Defect → Max-aggression**: fixed near-maximum aggression range/sizing, no opponent-read module.
3. **Always Cooperate → GTO-pure**: static near-equilibrium strategy (reuses the math-engine level-5 solver standalone), never adapts.
4. **Pavlov → Win-stay/lose-shift**: maintains discrete strategy modes (tight-passive / loose-aggressive / balanced); switches mode after a loss above a stack-percentage threshold, stays in mode after a win.
5. **Grudger → Permanent punishment**: plays standard until it observes a clear bluff/overbet revealed at a showdown it won; then permanently switches to a maximally exploitative counter-strategy for the rest of the match (both legs — the grudge persists across the pairing's shared opponent memory, then flushes with everything else between pairings).

### 4.3 Agent Interface (language-agnostic contract; implemented as a Python abstract base class wrapping Rust/ONNX calls)

```
decide_action(state) -> Action

state includes:
  - own hole cards
  - public board cards so far
  - pot size, both stack sizes, current bet-to-call
  - own position (button / non-button)
  - full betting-action sequence for the CURRENT hand (street-tagged)
  - opponent match-history object (see 4.4)
  - current blind level, current hand number

Action returned:
  - action_type: FOLD | CHECK | CALL | RAISE | ALL_IN
  - size: required iff action_type == RAISE (pre-resolution pot-fraction choice; engine applies Section 3.3 resolution)
  - metadata: optional dict, logging-only (e.g. confidence scores, internal opponent read) — never consumed by engine logic
```

### 4.4 Opponent Memory Model

- Every agent starts each **pairing** (not each hand) fully anonymous to its opponent — no identity, archetype label, or prior read.
- Within a pairing, the opponent-history object accumulates across **both legs** (persists through the button swap).
- **Card visibility rule**: an agent sees its opponent's hole cards only in hands it **lost** at showdown (the winner must show to claim the pot). In hands it **won**, the opponent's cards are never revealed (modeling a real "muck when beaten" convention). Folded hands never reveal cards to anyone.
- The opponent-history object (and any Grudger-style internal state) is **flushed to zero** once a pairing's two legs are complete, before the next pairing begins.

---

## 5. ML Training Pipeline (Iterative Curriculum Self-Play)

1. **Stage 0**: Generate a large hand-history corpus by running the 5 **math-engine-only** agents (no ML component) against each other in self-play, at volume (target: hundreds of thousands of hands minimum — exact volume to be tuned per training-loss convergence during Sprint 11, not fixed in advance).
2. **ML level 1**: no training — rule-based counting agent, implemented directly, not a learned model.
3. **ML level 2**: no training — rule-based running-statistics agent, implemented directly.
4. **ML level 3**: train a small-capacity LSTM on the Stage-0 corpus restricted to hands involving math-engine levels 1–2 (weak opponents) only.
5. **ML level 4**: warm-start from ML level 3's weights; continue training on a corpus including math-engine levels 3–4 hands.
6. **ML level 5**: warm-start from ML level 4's weights; continue training on the full corpus including math-engine level-5 (CFR) self-play hands, and optionally hands generated by earlier grid-agent combinations once those exist (bootstrapping).
7. Each trained model is exported to **ONNX** for inference use inside the tournament loop (Section 2).
8. **Training/validation split**: standard 80/20 split by match (not by hand, to avoid leakage of a specific match's dynamics across train/val), tracked loss curves logged for each level's training run.

---

## 6. Scoring & Logging

### 6.1 Scoring

- **Per leg**: `score = (final_stack − starting_stack) / starting_stack`, bounded to **\[-1, +1\]** (table-stakes rules guarantee the lower bound; equal starting stacks guarantee the upper bound).
- **Per pairing**: `pairing_score = leg_1_score + leg_2_score`, bounded to **\[-2, +2\]**. Legs are **summed, not averaged** — each leg is treated as a fully independent result, not smoothed together.
- **Overall tournament ranking**: for each agent, aggregate its pairing scores across all 29 opponents (sum and/or mean — report both), plus:
- **Variance** of pairing scores (consistency metric — distinguishes a Tit-for-Tat-style "never does badly" agent from a high-variance boom/bust agent).
- **Win count / loss count** per agent (define a pairing "win" as `pairing_score > 0`, "loss" as `< 0`, "draw" as `== 0`).
- Per-agent and per-matchup breakdowns, not just a single global leaderboard number.

### 6.2 Logging

- One **Parquet file per match leg**, one row per hand, containing: hand number, blind level at time of hand, button position, both players' hole cards (recorded always, but only *revealed to the opposing agent* in-engine per the 4.4 visibility rule — the log itself is the omniscient ground truth for post-hoc analysis), full public board, complete action sequence (street-tagged: action type, size, acting agent), pot-size progression, whether the hand ended by fold or showdown, winner, amount won/lost, both agents' per-leg normalized score contribution after that hand.
- File naming/indexing: `logs/{agent_a_id}_vs_{agent_b_id}/leg_{1|2}.parquet`, plus a top-level `pairings_summary.parquet` aggregating final scores per pairing for fast analysis without reloading every hand log.
- **CSV export (explicit requirement, in addition to the Parquet logs above)**: every match leg must also write a **separate, independent CSV file** capturing the round-by-round progression of that specific game. Parquet remains the primary analysis format (Section 8 tests, Section 10 plotting); the CSV is a human-readable, spreadsheet-openable companion covering the same data, generated from the same in-memory `MatchResult` so the two are guaranteed consistent, never two independently-maintained sources of truth.
- **File naming**: `logs/{agent_a_id}_vs_{agent_b_id}/leg_{1|2}.csv`, written alongside the `.parquet` file for that leg (same directory, same base name, different extension) — one CSV per match leg, never combined across legs or pairings.
- **CSV row schema (one row per hand, matching the Parquet schema above)**: `hand_number, blind_level, small_blind, big_blind, button_position, hole_cards_agent_a, hole_cards_agent_b, board_flop, board_turn, board_river, action_sequence (street-tagged, semicolon-delimited, e.g. PREFLOP:RAISE:450;PREFLOP:CALL:450;FLOP:CHECK;...), pot_size_progression (semicolon-delimited per action), hand_ended_by (FOLD|SHOWDOWN), winner, amount_won, stack_agent_a_after, stack_agent_b_after, score_agent_a_this_hand, score_agent_b_this_hand`.
- This gives every game a fully self-contained, independently inspectable record of how that specific match progressed hand-by-hand — useful for manual spot-checking and debugging without needing a Parquet reader.

---

## 7. Tournament Orchestration

- **Calibration phase**: run the full round-robin (or a representative subset) under **fixed random seeds** first. Use this phase specifically to tune: the blind escalation rate (currently ×1.2/20 hands), denomination tier boundaries (currently fixed per Section 3.3 — confirm they produce sensible raise granularity across a typical match's pot-size range), and ML confidence-weighting scale. Document any parameter changes made after calibration as an addendum to this PRD, not a silent change.
- **Final phase**: re-run the entire round-robin with **fully random seeds** (no fixed seeding) using the calibrated parameters — this is the run whose results are reported as the final tournament outcome.
- **Pairing generation**: all C(30,2) = 435 unique pairings, each played as 2 legs → 870 total match legs per full round-robin run (calibration and final are each one full run).
- **Compute note**: 870 legs × up to 1000 hands each is a non-trivial but tractable simulation workload; ML inference cost (ONNX) per decision is the most likely bottleneck — profile this during Sprint 12 before committing to the final run's expected runtime.

---

## 8. Testing Strategy

Every sprint below has an explicit Definition of Done tied to passing tests. General rules:

- **Rust core**: `cargo test` must pass for all unit tests (hand evaluator correctness against known hand rankings, raise-resolution function against hand-picked edge cases, blind escalation schedule correctness, deck shuffle produces all 52 unique cards). Use `proptest` for invariants that must hold across random inputs (e.g. "raise resolution never produces a bet exceeding the agent's stack", "pot size after any sequence of legal actions equals the sum of all committed chips").
- **Python orchestrator/agents**: `pytest` for agent interface contract tests (every agent type returns a well-formed `Action` for any legal game state), scoring function tests (hand-picked stack deltas → expected normalized scores), logging schema tests (written Parquet files are readable and schema-conformant).
- **Integration tests**: full single-match-leg runs between two deterministic agents (e.g. two Always-Fold-style stub agents) under a fixed seed must produce byte-identical logs across repeated runs — this is the core reproducibility guarantee for the calibration phase.
- **No sprint is complete until its tests pass in CI (or local test runner) and a short note is added to the sprint's checklist confirming manual sanity-check of at least one full match's log file.**

---

## 9. Sprint Plan (Dependency-Ordered)

Each sprint lists: goal, key deliverables, tech involved, and Definition of Done (DoD). Sprints are strictly ordered — each one's output is required infrastructure for the next.

### Sprint 0 — Environment & Repo Setup

- **Goal**: working Rust + Python toolchain on Windows, repo skeleton in place.
- **Deliverables**: repo structure (Section 2.1), empty Rust crate building via `maturin develop`, empty Python package importable, CI config (optional but recommended).
- **DoD**: `cargo test` and `pytest` both run (even on empty test suites) with zero errors; a trivial Rust function is callable from Python via the PyO3 binding.

### Sprint 1 — Core Deck & Hand Evaluator (Rust)

- **Goal**: deck shuffling, dealing, and 7-card hand evaluation.
- **Deliverables**: `Deck` struct (fresh shuffle per hand), `evaluate_hand(7 cards) -> HandRank` function.
- **DoD**: unit tests cover every hand category (high card → royal flush) against known examples; property test confirms shuffled deck always contains all 52 unique cards; evaluator correctly ranks a curated set of tricky tie-break hands.

### Sprint 2 — Betting Engine: Legality, Raise Resolution, Denomination Tiers (Rust)

- **Goal**: implement Section 3.3 exactly.
- **Deliverables**: pure function `resolve_raise(pot_fraction_choice, pot_size, min_raise, stack_size) -> ResolvedAmount` implementing the 4-step precedence order; action-legality filter (check only when nothing to call, etc.).
- **DoD**: every branch of the 4-step resolution order has a dedicated unit test with hand-picked numbers; property test confirms output is never illegal (never exceeds stack, never below min-raise unless overridden by all-in).
- **Depends on**: Sprint 0.

### Sprint 3 — Single-Hand Game Loop (Rust)

- **Goal**: play one full hand end-to-end — blinds posted, hole cards dealt, 4 betting rounds, showdown or fold resolution, pot awarded.
- **Deliverables**: `play_hand(state, agent_a_action_fn, agent_b_action_fn) -> HandResult`.
- **DoD**: integration test runs a full hand between two scripted/stub decision functions and produces a correct, verifiable result (manually checked pot math); heads-up acting order (SB first preflop, BB first postflop) is explicitly unit-tested.
- **Depends on**: Sprints 1–2.

### Sprint 4 — Full Match Loop: Blinds, Button Rotation, End Conditions (Rust)

- **Goal**: wrap the single-hand loop into a full match (up to 1000 hands or bust), with blind escalation and button alternation.
- **Deliverables**: `play_match(...) -> MatchResult`, configurable blind schedule parameter.
- **DoD**: integration test with two stub agents under a fixed seed runs a full match to completion (cap or bust) and produces byte-identical results across repeated runs with the same seed (reproducibility test from Section 8).
- **Depends on**: Sprint 3.

### Sprint 5 — Python Bindings & Agent Interface Contract (Rust + Python)

- **Goal**: expose the match loop to Python; define and enforce the `decide_action()` contract from Section 4.3.
- **Deliverables**: PyO3 bindings for `play_match`; Python `Agent` abstract base class; a couple of trivial stub agents (always-fold, always-call) implementing it.
- **DoD**: a full match between two Python-defined stub agents runs via the Python orchestrator and produces a correctly formed log (next sprint formalizes logging, but raw result data must be inspectable here).
- **Depends on**: Sprint 4.

### Sprint 6 — Logging & Scoring (Python)

- **Goal**: implement Section 6 exactly.
- **Deliverables**: Parquet writer per the logging schema; per-leg and per-pairing scoring functions.
- **DoD**: hand-picked stack-delta test cases produce expected normalized scores; a full logged match leg is reloaded from Parquet and matches the in-memory match result exactly. The matching CSV file for the same leg (Section 6.2) is also generated and spot-checked by hand against the Parquet output for a sample match — both files must agree row-for-row.
- **Depends on**: Sprint 5.

### Sprint 7 — Math Engine Levels 1–5 (Rust, called from Python agents)

- **Goal**: implement the 5 deterministic math-engine levels from Section 4.1.
- **Deliverables**: heuristic thresholds (L1), pot-odds calculator (L2), Monte Carlo equity estimator (L3), range-vs-range equity + opponent-range estimation (L4), abstracted CFR solver with Discounted-CFR training (L5).
- **DoD**: each level has unit tests appropriate to it (e.g. L3's Monte Carlo equity converges to a known analytically-computable equity within tolerance over enough samples; L5's CFR solver's exploitability decreases monotonically over training iterations on a reduced test game). Each level is wrapped as a standalone `Agent` and can complete a full match via the Sprint-5 orchestrator.
- **Depends on**: Sprints 2, 5.

### Sprint 8 — PD Archetype Agents (Python)

- **Goal**: implement the 5 hand-built agents from Section 4.2.
- **Deliverables**: Tit-for-Tat/Mirror-Aggression, Always-Defect, Always-Cooperate (wraps math L5), Pavlov, Grudger.
- **DoD**: each archetype has a unit test confirming its core behavioral rule triggers correctly on a scripted opponent-history fixture (e.g. Grudger's permanent-punishment state correctly activates after a scripted bluff-catch and persists across a simulated leg boundary within the same pairing).
- **Depends on**: Sprint 7 (for Always-Cooperate's reuse of math L5).

### Sprint 9 — Round-Robin Orchestrator & Calibration Run (Python)

- **Goal**: pairing generation (all 435 pairings × 2 legs), opponent-memory persistence-then-flush logic (Section 4.4), full round-robin driver, fixed-seed calibration run using only math-engine + PD-archetype agents (ML grid doesn't exist yet — that's fine, calibration doesn't need it).
- **Deliverables**: orchestrator script; calibration-run results; a short addendum documenting any parameter changes to blind escalation rate or denomination tiers based on calibration findings (per Section 7).
- **DoD**: full calibration round-robin (10 agents: 5 math levels + 5 PD archetypes = 45 pairings × 2 legs = 90 match legs) completes without error, under fixed seeds, with reproducible results on re-run; bust-time distribution is reviewed against the Section-7 target (most matches retain strategic depth through \~60–70% of the hand cap).
- **Depends on**: Sprints 6, 8.

### Sprint 10 — Analysis & Visualization Harness (Python)

- **Goal**: implement the win/loss graphs, variance tracking, and per-agent/per-pairing breakdowns from Section 6.1.
- **Deliverables**: analysis scripts consuming `pairings_summary.parquet`; plots for win/loss counts per agent, score variance per agent, head-to-head matrices.
- **DoD**: full analysis suite runs against the Sprint-9 calibration data and produces a readable leaderboard + plots without manual data wrangling.
- **Depends on**: Sprint 9.

### Sprint 11 — ML Training Data Generation (Python)

- **Goal**: Stage 0 of Section 5 — generate the self-play corpus from math-engine agents at volume.
- **Deliverables**: large-scale self-play data generation script (reuses Sprint-9 orchestrator at higher volume, math-engine-only agents), corpus stored in a training-friendly format (e.g. Parquet, feature-extracted).
- **DoD**: corpus generated and validated (no malformed hands, correct label distribution across math-engine levels); train/val split implemented per Section 5.8.
- **Depends on**: Sprint 9.

### Sprint 12 — ML Model Architecture & Curriculum Training (Python/PyTorch)

- **Goal**: implement the LSTM architecture family and the levels 3–5 iterative curriculum training loop from Section 5.
- **Deliverables**: LSTM model definition (parameterized by capacity), training scripts for L3→L4→L5 warm-start curriculum, ONNX export for each trained level, rule-based L1/L2 implementations.
- **DoD**: each level's training run shows a sensible decreasing validation loss curve; exported ONNX models produce consistent outputs vs. the original PyTorch model on a held-out sample (numerical parity check); inference latency per decision is profiled and documented (per the Section-7 compute note).
- **Depends on**: Sprint 11.

### Sprint 13 — Hybrid Grid Agents & Confidence Blending (Python)

- **Goal**: implement the full 25-agent grid via the combination rule in Section 4.1.
- **Deliverables**: `GridAgent(math_level, ml_level)` wrapping both engines with the confidence-weighted blend formula.
- **DoD**: all 25 grid-agent combinations instantiate and complete a full match via the orchestrator without error; sanity-check matches at the grid's corners (5,1) vs (1,5) vs (5,5) vs (1,1) produce directionally sensible results (documented, not necessarily "correct" — this is exploratory validation, not a hard pass/fail test).
- **Depends on**: Sprints 8, 12.

### Sprint 14 — Full 30-Agent Integration Test

- **Goal**: confirm the complete system runs end-to-end at full scale before the final run.
- **Deliverables**: a full 30-agent round-robin executed under fixed seeds (reusing the calibration infrastructure), with full logging and analysis.
- **DoD**: all 435 pairings × 2 legs (870 match legs) complete without error; logs are complete and schema-valid for every leg; analysis harness (Sprint 10) produces a full leaderboard across all 30 agents; total runtime is recorded and reviewed against available time budget before committing to the final run.
- **Depends on**: Sprints 10, 13.

### Sprint 15 — Final Run

- **Goal**: the reported tournament result.
- **Deliverables**: full 30-agent round-robin under **fully random seeds** (no fixed seeding), using whatever parameters were finalized during calibration (Sprint 9) — explicitly re-confirm no fixed-seed artifacts remain in the configuration before launching.
- **DoD**: complete, logged, analyzed results; final leaderboard with win/loss counts, variance, and per-matchup breakdowns; this is the dataset referenced in any final writeup/report.
- **Depends on**: Sprint 14.

---

## 10. Open Items Explicitly Deferred (not blocking, revisit post-Sprint-15 if desired)

- Persistent opponent memory **across** pairings (not just within one pairing's two legs) — explicitly scoped out of this PRD; noted earlier as a possible future experiment, not part of this build.
- Exact numeric form of `confidence_weight(k)` beyond the linear default — tune only if Sprint 13's sanity checks look wrong.
- Training corpus volume for Sprint 11 — tune based on observed validation-loss convergence, not fixed here.
