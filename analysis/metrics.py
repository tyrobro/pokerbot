"""Per-agent aggregation from pairings_summary.parquet (Section 6.1):
sum/mean pairing score, win/loss/draw counts, variance of pairing scores
-- one row per agent, pivoted from the stored per-pairing schema."""

import pandas as pd


def load_pairings_summary(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def to_agent_perspective_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pivots each pairing row into two agent-perspective rows (A's view,
    B's view) so every agent's full set of pairing_scores can be
    aggregated regardless of which seat (A/B) it occupied in storage."""
    a_view = df.rename(
        columns={"agent_a_id": "agent_id", "agent_b_id": "opponent_id", "pairing_score_a": "pairing_score"}
    )[["agent_id", "opponent_id", "pairing_score"]]
    b_view = df.rename(
        columns={"agent_b_id": "agent_id", "agent_a_id": "opponent_id", "pairing_score_b": "pairing_score"}
    )[["agent_id", "opponent_id", "pairing_score"]]
    return pd.concat([a_view, b_view], ignore_index=True)


def _outcome(score: float) -> str:
    if score > 0:
        return "WIN"
    if score < 0:
        return "LOSS"
    return "DRAW"


def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per agent: sum, mean, win/loss/draw counts, variance of
    pairing_score across all opponents faced (Section 6.1)."""
    rows = to_agent_perspective_rows(df)
    rows = rows.copy()
    rows["outcome"] = rows["pairing_score"].apply(_outcome)

    records = []
    for agent_id, group in rows.groupby("agent_id"):
        scores = group["pairing_score"]
        records.append(
            {
                "agent_id": agent_id,
                "opponents_faced": len(group),
                "sum_pairing_score": scores.sum(),
                "mean_pairing_score": scores.mean(),
                "variance_pairing_score": scores.var(ddof=0),  # population variance over the agent's own scores
                "wins": int((group["outcome"] == "WIN").sum()),
                "losses": int((group["outcome"] == "LOSS").sum()),
                "draws": int((group["outcome"] == "DRAW").sum()),
            }
        )
    leaderboard = pd.DataFrame(records).sort_values("sum_pairing_score", ascending=False).reset_index(drop=True)
    return leaderboard


def build_head_to_head_matrix(df: pd.DataFrame, agent_order: list) -> pd.DataFrame:
    """row[A][col=B] = pairing_score_A_vs_B (Section: asymmetric grid,
    row agent's own score). Diagonal is NaN (no agent plays itself)."""
    rows = to_agent_perspective_rows(df)
    matrix = pd.DataFrame(index=agent_order, columns=agent_order, dtype=float)
    for _, r in rows.iterrows():
        if r["agent_id"] in matrix.index and r["opponent_id"] in matrix.columns:
            matrix.loc[r["agent_id"], r["opponent_id"]] = r["pairing_score"]
    return matrix