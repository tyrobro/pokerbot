"""Advanced final-report metrics (Elo, Sharpe, factor decomposition,
score-per-hand, opponent-class split, bust/survival rate)."""

import pandas as pd
import pytest

from analysis.advanced_metrics import (
    add_sharpe_ratio,
    compute_elo_ratings,
    factor_decomposition,
    grid_levels,
    is_grid_agent,
    opponent_class_split,
)
from analysis.metrics import build_leaderboard


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            {
                "agent_a_id": "grid_m1_k1", "agent_b_id": "grid_m5_k5",
                "leg_1_score_a": -1.0, "leg_1_score_b": 1.0,
                "leg_2_score_a": -1.0, "leg_2_score_b": 1.0,
                "pairing_score_a": -2.0, "pairing_score_b": 2.0,
            },
            {
                "agent_a_id": "grid_m1_k1", "agent_b_id": "pd_always_defect",
                "leg_1_score_a": 0.5, "leg_1_score_b": -0.5,
                "leg_2_score_a": 0.3, "leg_2_score_b": -0.3,
                "pairing_score_a": 0.8, "pairing_score_b": -0.8,
            },
            {
                "agent_a_id": "grid_m5_k5", "agent_b_id": "pd_always_defect",
                "leg_1_score_a": 0.9, "leg_1_score_b": -0.9,
                "leg_2_score_a": 0.7, "leg_2_score_b": -0.7,
                "pairing_score_a": 1.6, "pairing_score_b": -1.6,
            },
        ]
    )


def test_is_grid_agent_and_grid_levels():
    assert is_grid_agent("grid_m5_k3")
    assert not is_grid_agent("pd_grudger")
    assert grid_levels("grid_m5_k3") == (5, 3)
    assert grid_levels("pd_grudger") == (None, None)


def test_elo_ratings_rank_strong_agent_above_weak_one(sample_df):
    elo = compute_elo_ratings(sample_df).set_index("agent_id")
    assert elo.loc["grid_m5_k5", "elo_rating"] > elo.loc["grid_m1_k1", "elo_rating"]


def test_sharpe_ratio_is_mean_over_sqrt_variance():
    lb = pd.DataFrame({"agent_id": ["a"], "mean_pairing_score": [1.0], "variance_pairing_score": [4.0]})
    result = add_sharpe_ratio(lb)
    assert result.loc[0, "sharpe_ratio"] == pytest.approx(0.5)


def test_factor_decomposition_runs_on_grid_agents_only(sample_df):
    lb = build_leaderboard(sample_df)
    result = factor_decomposition(lb)
    assert result["n_observations"] == 2  # only grid_m1_k1 and grid_m5_k5, not pd_always_defect
    assert len(result["coefficients"]) == len(result["design_columns"])


def test_opponent_class_split_separates_grid_from_pd(sample_df):
    split = opponent_class_split(sample_df).set_index("agent_id")
    assert "grid" in split.columns
    assert "pd_archetype" in split.columns