"""Sprint 15 final-report additions: Elo, Sharpe-style risk adjustment,
factor decomposition (math vs ML level), score-per-hand, opponent-class
split, bust/survival rate. Built on top of analysis/metrics.py's
existing agent-perspective pivot; reads per-leg Parquet files directly
for anything pairings_summary.parquet doesn't carry (hands survived,
bust vs. cap).
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.metrics import to_agent_perspective_rows

GRID_ID_RE = re.compile(r"^grid_m(\d)_k(\d)$")


def is_grid_agent(agent_id: str) -> bool:
    return GRID_ID_RE.match(agent_id) is not None


def grid_levels(agent_id: str):
    m = GRID_ID_RE.match(agent_id)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


# ---- 1. Elo-style rating ----

def compute_elo_ratings(df: pd.DataFrame, k: float = 24.0, initial: float = 1500.0) -> pd.DataFrame:
    """Each LEG (not pairing) is one Elo game: win/loss/draw from its
    own leg score sign. Processes legs in the order they appear in the
    summary (round-robin has no inherent temporal order across
    pairings, so this is one reasonable, stable pass -- not claiming a
    single 'true' Elo, just a consistent strength-adjusted re-ranking
    to compare against the raw leaderboard)."""
    ratings = {}

    def get(agent_id):
        return ratings.setdefault(agent_id, initial)

    def expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for _, row in df.iterrows():
        for leg_a, leg_b in [
            (row["leg_1_score_a"], row["leg_1_score_b"]),
            (row["leg_2_score_a"], row["leg_2_score_b"]),
        ]:
            a_id, b_id = row["agent_a_id"], row["agent_b_id"]
            ra, rb = get(a_id), get(b_id)
            if leg_a > 0:
                score_a = 1.0
            elif leg_a < 0:
                score_a = 0.0
            else:
                score_a = 0.5
            exp_a = expected(ra, rb)
            ratings[a_id] = ra + k * (score_a - exp_a)
            ratings[b_id] = rb + k * ((1.0 - score_a) - (1.0 - exp_a))

    return pd.DataFrame({"agent_id": list(ratings.keys()), "elo_rating": list(ratings.values())}).sort_values(
        "elo_rating", ascending=False
    )


# ---- 2. Sharpe-style risk-adjusted score ----

def add_sharpe_ratio(leaderboard: pd.DataFrame) -> pd.DataFrame:
    lb = leaderboard.copy()
    std = np.sqrt(lb["variance_pairing_score"])
    # Avoid divide-by-zero for a (rare) perfectly consistent agent.
    lb["sharpe_ratio"] = np.where(std > 0, lb["mean_pairing_score"] / std, np.nan)
    return lb


# ---- 3. Factor decomposition (math level vs ML level), 25 grid agents only ----

def factor_decomposition(leaderboard: pd.DataFrame) -> dict:
    """OLS: sum_pairing_score ~ math_level (categorical) + ml_level
    (categorical), fit only on the 25 GridAgent rows. Isolates each
    factor's marginal effect independent of the other. Implemented with
    plain numpy least-squares (dummy coding) to avoid adding a
    statsmodels dependency for one analysis."""
    grid_rows = leaderboard[leaderboard["agent_id"].apply(is_grid_agent)].copy()
    grid_rows["math_level"], grid_rows["ml_level"] = zip(*grid_rows["agent_id"].apply(grid_levels))

    y = grid_rows["sum_pairing_score"].values
    n = len(grid_rows)

    # Dummy-code math_level (2-5, level 1 is baseline) and ml_level (2-5,
    # level 1 is baseline) + intercept.
    design_cols = ["intercept"] + [f"math_{m}" for m in range(2, 6)] + [f"ml_{k}" for k in range(2, 6)]
    X = np.zeros((n, len(design_cols)))
    X[:, 0] = 1.0
    for i, (_, row) in enumerate(grid_rows.iterrows()):
        if row["math_level"] > 1:
            X[i, 1 + (row["math_level"] - 2)] = 1.0
        if row["ml_level"] > 1:
            X[i, 1 + 4 + (row["ml_level"] - 2)] = 1.0

    coefs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    return {"design_columns": design_cols, "coefficients": coefs.tolist(), "n_observations": n}


# ---- 4. Score per hand ----

def score_per_hand(logs_dir: str) -> pd.DataFrame:
    """Reads per-leg Parquet files directly (hands_survived isn't in
    pairings_summary.parquet). Returns one row per (agent_id, leg) with
    hands survived and that leg's score, aggregated to one
    score-per-hand ratio per agent."""
    records = []
    for pair_dir in sorted(glob.glob(f"{logs_dir}/*_vs_*")):
        dir_name = Path(pair_dir).name
        if "__vs__" in dir_name:
            a_id, b_id = dir_name.split("__vs__")
        else:
            a_id, b_id = dir_name.split("_vs_")

        for leg_num in (1, 2):
            leg_path = Path(pair_dir) / f"leg_{leg_num}.parquet"
            if not leg_path.exists():
                continue
            leg_df = pd.read_parquet(leg_path)
            if leg_df.empty:
                continue
            hands_survived = len(leg_df)
            final_a = leg_df.iloc[-1]["stack_agent_a_after"]
            final_b = leg_df.iloc[-1]["stack_agent_b_after"]
            score_a = (final_a - 100_000) / 100_000
            score_b = (final_b - 100_000) / 100_000
            records.append({"agent_id": a_id, "hands_survived": hands_survived, "leg_score": score_a})
            records.append({"agent_id": b_id, "hands_survived": hands_survived, "leg_score": score_b})

    df = pd.DataFrame(records)
    agg = df.groupby("agent_id").agg(
        total_hands=("hands_survived", "sum"),
        total_score=("leg_score", "sum"),
        legs_played=("leg_score", "count"),
    )
    agg["score_per_hand"] = agg["total_score"] / agg["total_hands"]
    return agg.reset_index().sort_values("score_per_hand", ascending=False)


# ---- 5. Performance split by opponent class ----

def opponent_class_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = to_agent_perspective_rows(df)
    rows = rows.copy()
    rows["opponent_class"] = rows["opponent_id"].apply(lambda x: "grid" if is_grid_agent(x) else "pd_archetype")

    records = []
    for (agent_id, opp_class), group in rows.groupby(["agent_id", "opponent_class"]):
        records.append(
            {
                "agent_id": agent_id,
                "opponent_class": opp_class,
                "opponents_faced": len(group),
                "mean_pairing_score": group["pairing_score"].mean(),
            }
        )
    split = pd.DataFrame(records)
    return split.pivot(index="agent_id", columns="opponent_class", values="mean_pairing_score").reset_index()


# ---- 6. Bust rate / survival rate ----

def bust_survival_rates(logs_dir: str, max_hands: int = 1000) -> pd.DataFrame:
    """% of legs ending by elimination (hands_survived < max_hands)
    vs. reaching the hand cap, per agent."""
    records = []
    for pair_dir in sorted(glob.glob(f"{logs_dir}/*_vs_*")):
        dir_name = Path(pair_dir).name
        if "__vs__" in dir_name:
            a_id, b_id = dir_name.split("__vs__")
        else:
            a_id, b_id = dir_name.split("_vs_")

        for leg_num in (1, 2):
            leg_path = Path(pair_dir) / f"leg_{leg_num}.parquet"
            if not leg_path.exists():
                continue
            leg_df = pd.read_parquet(leg_path)
            if leg_df.empty:
                continue
            busted = len(leg_df) < max_hands
            records.append({"agent_id": a_id, "busted": busted})
            records.append({"agent_id": b_id, "busted": busted})

    df = pd.DataFrame(records)
    agg = df.groupby("agent_id").agg(legs_played=("busted", "count"), legs_busted=("busted", "sum"))
    agg["bust_rate"] = agg["legs_busted"] / agg["legs_played"]
    agg["survival_rate"] = 1.0 - agg["bust_rate"]
    return agg.reset_index().sort_values("survival_rate", ascending=False)