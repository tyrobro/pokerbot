"""Sprint 6 DoD (Section 8): hand-picked stack-delta cases produce expected
normalized scores; a full logged match leg reloaded from Parquet matches
the in-memory match result exactly; the matching CSV agrees row-for-row.
"""

import csv

import pandas as pd

from agents.stub_agents import AlwaysCallAgent, AlwaysFoldAgent
from orchestrator.logging import build_hand_rows, write_leg_logs
from orchestrator.match_runner import run_match
from orchestrator.scoring import leg_score, pairing_score, pairing_outcome


# ---- hand-picked scoring cases (Section 8 DoD) ----

def test_leg_score_hand_picked_cases():
    assert leg_score(100_000, 100_000) == 0.0
    assert leg_score(150_000, 100_000) == 0.5
    assert leg_score(0, 100_000) == -1.0  # lower bound, table stakes
    assert leg_score(200_000, 100_000) == 1.0  # upper bound
    assert leg_score(250_000, 100_000) == 1.0  # clamped, never exceeds +1


def test_pairing_score_sums_not_averages():
    assert pairing_score(0.5, 0.3) == pytest_approx(0.8)
    assert pairing_score(1.0, 1.0) == 2.0  # upper bound
    assert pairing_score(-1.0, -1.0) == -2.0  # lower bound


def test_pairing_outcome_classification():
    assert pairing_outcome(0.4) == "WIN"
    assert pairing_outcome(-0.4) == "LOSS"
    assert pairing_outcome(0.0) == "DRAW"


def pytest_approx(x, tol=1e-9):
    # tiny local helper so this file has no pytest.approx import dependency
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol

    return _Approx()


# ---- full logged leg: Parquet reload matches in-memory, CSV agrees ----

def test_logged_leg_roundtrips_through_parquet_and_matches_csv(tmp_path):
    result = run_match(
        AlwaysCallAgent(),
        AlwaysFoldAgent(),
        seed=321,
        max_hands=25,
        blind_escalation_interval_hands=0,
    )
    starting_stack = 100_000
    rows = build_hand_rows(result, starting_stack)

    paths = write_leg_logs(rows, "agentA", "agentB", leg=1, logs_dir=str(tmp_path))

    # Parquet reload matches the in-memory rows exactly.
    reloaded = pd.read_parquet(paths["parquet"])
    assert len(reloaded) == len(rows)
    for i, row in enumerate(rows):
        for key, value in row.items():
            assert reloaded.iloc[i][key] == value, f"mismatch at row {i}, col {key}"

    # CSV agrees row-for-row with the Parquet-backing rows.
    with open(paths["csv"], newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    for i, row in enumerate(rows):
        for key, value in row.items():
            # CSV values come back as strings; compare stringified.
            assert csv_rows[i][key] == str(value), f"CSV mismatch at row {i}, col {key}"
            
def test_append_pairing_summary_overwrites_not_duplicates(tmp_path):
    from orchestrator.logging import append_pairing_summary
    import pandas as pd

    logs_dir = str(tmp_path)
    append_pairing_summary("x", "y", 0.1, -0.1, 0.1, -0.1, 0.2, -0.2, logs_dir)
    # Re-score the SAME pairing (simulating a re-run) with different numbers.
    append_pairing_summary("x", "y", 0.5, -0.5, 0.5, -0.5, 1.0, -1.0, logs_dir)

    df = pd.read_parquet(f"{logs_dir}/pairings_summary.parquet")
    assert len(df) == 1, "re-scoring the same pairing should replace, not duplicate, the row"
    assert df.iloc[0]["pairing_score_a"] == 1.0