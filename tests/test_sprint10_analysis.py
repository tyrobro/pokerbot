"""Sprint 10 DoD: full analysis suite runs against the Sprint-9
calibration data and produces a readable leaderboard + plots without
manual data wrangling."""

import pandas as pd
import pytest

from analysis.dashboard import build_dashboard_html
from analysis.metrics import build_head_to_head_matrix, build_leaderboard, to_agent_perspective_rows


@pytest.fixture
def sample_pairings_df():
    return pd.DataFrame(
        [
            {
                "agent_a_id": "x", "agent_b_id": "y",
                "leg_1_score_a": 0.5, "leg_1_score_b": -0.5,
                "leg_2_score_a": -0.2, "leg_2_score_b": 0.2,
                "pairing_score_a": 0.3, "pairing_score_b": -0.3,
            },
            {
                "agent_a_id": "x", "agent_b_id": "z",
                "leg_1_score_a": 1.0, "leg_1_score_b": -1.0,
                "leg_2_score_a": 1.0, "leg_2_score_b": -1.0,
                "pairing_score_a": 2.0, "pairing_score_b": -2.0,
            },
            {
                "agent_a_id": "y", "agent_b_id": "z",
                "leg_1_score_a": 0.0, "leg_1_score_b": 0.0,
                "leg_2_score_a": 0.0, "leg_2_score_b": 0.0,
                "pairing_score_a": 0.0, "pairing_score_b": 0.0,
            },
        ]
    )


def test_to_agent_perspective_rows_doubles_row_count(sample_pairings_df):
    rows = to_agent_perspective_rows(sample_pairings_df)
    assert len(rows) == len(sample_pairings_df) * 2


def test_leaderboard_has_one_row_per_agent_with_correct_opponent_count(sample_pairings_df):
    leaderboard = build_leaderboard(sample_pairings_df)
    assert set(leaderboard["agent_id"]) == {"x", "y", "z"}
    for _, row in leaderboard.iterrows():
        assert row["opponents_faced"] == 2  # each agent plays the other 2


def test_leaderboard_sum_and_win_loss_draw_counts_are_correct(sample_pairings_df):
    leaderboard = build_leaderboard(sample_pairings_df).set_index("agent_id")
    # x: +0.3 vs y, +2.0 vs z -> sum 2.3, 2 wins
    assert leaderboard.loc["x", "sum_pairing_score"] == pytest.approx(2.3)
    assert leaderboard.loc["x", "wins"] == 2
    assert leaderboard.loc["x", "losses"] == 0
    # z: -2.0 vs x, 0.0 vs y -> sum -2.0, 1 loss, 1 draw
    assert leaderboard.loc["z", "sum_pairing_score"] == pytest.approx(-2.0)
    assert leaderboard.loc["z", "losses"] == 1
    assert leaderboard.loc["z", "draws"] == 1


def test_head_to_head_matrix_is_antisymmetric(sample_pairings_df):
    matrix = build_head_to_head_matrix(sample_pairings_df, ["x", "y", "z"])
    assert matrix.loc["x", "y"] == pytest.approx(-matrix.loc["y", "x"])
    assert matrix.loc["x", "z"] == pytest.approx(-matrix.loc["z", "x"])
    assert pd.isna(matrix.loc["x", "x"])  # diagonal unused


def test_dashboard_builds_a_nonempty_self_contained_html_file(tmp_path, sample_pairings_df):
    parquet_path = tmp_path / "pairings_summary.parquet"
    sample_pairings_df.to_parquet(parquet_path, index=False)
    output_path = tmp_path / "dashboard.html"
    csv_path = tmp_path / "leaderboard.csv"

    build_dashboard_html(str(parquet_path), str(output_path), str(csv_path))

    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert len(html) > 10_000  # inline Plotly JS alone is large; catches an empty/broken write
    assert "plotly" in html.lower()
    # Narrative requirement (explicit DoD item, not optional): the
    # written sections must actually be present, not just the charts.
    assert "Methodology" in html
    assert "Leaderboard" in html
    assert "Head-to-Head" in html
    assert csv_path.exists()
    csv_content = csv_path.read_text()
    assert "agent_id" in csv_content