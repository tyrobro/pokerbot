"""Sprint 10: single self-contained interactive HTML dashboard (Section
10 DoD) -- leaderboard table, win/loss bar chart, asymmetric 30x30 (or
NxN) head-to-head matrix, with written narrative sections interleaved
throughout. The narrative is a required, separate DoD item, not
decoration around the charts.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.metrics import build_head_to_head_matrix, build_leaderboard, load_pairings_summary, _outcome
from analysis.advanced_metrics import (
    add_sharpe_ratio,
    bust_survival_rates,
    compute_elo_ratings,
    factor_decomposition,
    opponent_class_split,
    score_per_hand,
)

PROJECT_GOAL_HTML = """
<h1>Hold'em Tournament Engine — Results Dashboard</h1>
<p>
This dashboard reports results from a heads-up No-Limit Hold'em round-robin tournament.
Every pair of agents in the field plays two button-swapped legs; each agent's performance
across the full round-robin is summarized below as a leaderboard, a win/loss breakdown,
and a full head-to-head matrix.
</p>
"""

METHODOLOGY_HTML = """
<h2>Methodology</h2>
<p>
Each agent is either a deterministic/Monte-Carlo math engine (levels 1&ndash;5, increasing
in sophistication from hand-strength heuristics up to an abstracted CFR solver), a hand-built
Prisoner's-Dilemma-style archetype (Tit-for-Tat, Always-Defect, Always-Cooperate, Pavlov,
Grudger), or, once the full grid exists, a hybrid of a math engine and an ML opponent-read
model. Every pairing plays two legs with the button flipped between them; a leg's score is
<code>(final_stack &minus; starting_stack) / starting_stack</code>, bounded to [&minus;1, +1].
A pairing's score sums its two leg scores (bounded [&minus;2, +2]) &mdash; legs are summed,
not averaged, so each leg counts as a fully independent result. An agent's overall standing
is the sum and mean of its pairing scores across every opponent it faced, with the variance
of those scores reported separately as a consistency signal (a low-variance agent performs
similarly against everyone; a high-variance agent swings between dominating and losing badly).
</p>
"""


def _leaderboard_table(leaderboard: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Agent", "Opponents", "Sum Score", "Mean Score", "Variance", "W", "L", "D"],
                    fill_color="#2c3e50",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=[
                        leaderboard["agent_id"],
                        leaderboard["opponents_faced"],
                        leaderboard["sum_pairing_score"].round(3),
                        leaderboard["mean_pairing_score"].round(3),
                        leaderboard["variance_pairing_score"].round(3),
                        leaderboard["wins"],
                        leaderboard["losses"],
                        leaderboard["draws"],
                    ],
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(title="Leaderboard (sorted by sum pairing score)", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _win_loss_bar(leaderboard: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Wins", x=leaderboard["agent_id"], y=leaderboard["wins"], marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="Losses", x=leaderboard["agent_id"], y=leaderboard["losses"], marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name="Draws", x=leaderboard["agent_id"], y=leaderboard["draws"], marker_color="#95a5a6"))
    fig.update_layout(
        title="Win / Loss / Draw count per agent",
        barmode="stack",
        xaxis_title="Agent",
        yaxis_title="Pairings",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _head_to_head_heatmap(matrix: pd.DataFrame, raw_df: pd.DataFrame) -> go.Figure:
    agents = list(matrix.index)
    z = matrix.values

    # Build hover text: exact score + win/loss/draw, blank on the diagonal.
    hover = []
    for row_agent in agents:
        row_text = []
        for col_agent in agents:
            if row_agent == col_agent:
                row_text.append("")
            else:
                val = matrix.loc[row_agent, col_agent]
                if pd.isna(val):
                    row_text.append(f"{row_agent} vs {col_agent}: no data")
                else:
                    row_text.append(f"{row_agent} vs {col_agent}<br>score: {val:+.2f} ({_outcome(val)})")
        hover.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=agents,
            y=agents,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-2,
            zmax=2,
            text=hover,
            hoverinfo="text",
            colorbar=dict(title="pairing_score"),
        )
    )
    fig.update_layout(
        title="Head-to-head matrix (row agent's pairing_score vs. column agent)",
        xaxis_title="Opponent",
        yaxis_title="Agent",
        margin=dict(l=10, r=10, t=40, b=10),
        height=700,
    )
    return fig


def build_dashboard_html(pairings_summary_path: str, output_path: str, leaderboard_csv_path: str = None, logs_dir: str = None,) -> None:
    """logs_dir: if provided, enables the per-leg-log-derived metrics
    (score-per-hand, bust/survival rate) -- these aren't in
    pairings_summary.parquet alone. Pass the same directory the summary
    file lives in (e.g. 'logs/final')."""
    
    df = load_pairings_summary(pairings_summary_path)
    leaderboard = build_leaderboard(df)
    leaderboard = add_sharpe_ratio(leaderboard)
    agent_order = leaderboard["agent_id"].tolist()
    matrix = build_head_to_head_matrix(df, agent_order)

    elo = compute_elo_ratings(df)
    class_split = opponent_class_split(df)
    factor_result = factor_decomposition(leaderboard)
    
    per_hand = score_per_hand(logs_dir) if logs_dir else None
    survival = bust_survival_rates(logs_dir) if logs_dir else None
    
    if leaderboard_csv_path:
        merged = leaderboard.merge(elo, on="agent_id", how="left")
        merged = merged.merge(class_split, on="agent_id", how="left")
        if per_hand is not None:
            merged = merged.merge(per_hand[["agent_id", "score_per_hand"]], on="agent_id", how="left")
        if survival is not None:
            merged = merged.merge(survival[["agent_id", "survival_rate"]], on="agent_id", how="left")
        merged.to_csv(leaderboard_csv_path, index=False)

    table_fig = _leaderboard_table(leaderboard)
    bar_fig = _win_loss_bar(leaderboard)
    matrix_fig = _head_to_head_heatmap(matrix, df)
    
    parts = [
        "<html><head><title>Tournament Dashboard</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;}"
        "h1{color:#2c3e50;} h2{color:#34495e;border-top:1px solid #ddd;padding-top:20px;}"
        ".howto{font-style:italic;color:#555;margin-bottom:6px;}</style></head><body>",
        PROJECT_GOAL_HTML,
        METHODOLOGY_HTML,
        "<h2>Leaderboard</h2>",
        '<p class="howto">Sorted by sum pairing score (best overall record first). '
        "Variance is computed over each agent's own pairing scores across every opponent faced "
        "-- a low number means consistent results, a high number means the agent swings between "
        "big wins and big losses depending on the matchup.</p>",
        table_fig.to_html(full_html=False, include_plotlyjs="inline"),
        "<h2>Win / Loss / Draw Breakdown</h2>",
        '<p class="howto">Each bar is one agent\'s record across all its pairings this round-robin; '
        "a pairing counts as a win/loss if its pairing_score is strictly positive/negative, a draw if exactly zero.</p>",
        bar_fig.to_html(full_html=False, include_plotlyjs=False),
                "<h2>Head-to-Head Matrix</h2>",
        '<p class="howto">Each cell is the ROW agent\'s pairing_score against the COLUMN agent '
        "(green = row agent won decisively, red = row agent lost decisively, the diagonal is "
        "unused since an agent never plays itself). Hover a cell for the exact score. "
        "Because starting stacks are equal, the matrix is exactly antisymmetric -- cell[A][B] = "
        "-cell[B][A] -- shown in full rather than as a triangle for easier reading.</p>",
        matrix_fig.to_html(full_html=False, include_plotlyjs=False),
    ]

    parts.append("<h2>Elo Rating (strength-adjusted re-ranking)</h2>")
    parts.append(
        '<p class="howto">Round-robin sum/mean rewards beating ANYONE equally -- it can\'t tell '
        "'beat weak agents a lot' apart from 'beat strong agents a little.' Elo (K=24, one pass "
        "over every leg as a win/loss/draw) re-ranks by strength-adjusted performance. Large "
        "divergences from the raw leaderboard below are informative -- they flag agents whose "
        "sum score was inflated or deflated by who they happened to play.</p>"
    )
    elo_table = go.Figure(
        data=[go.Table(
            header=dict(values=["Agent", "Elo Rating"], fill_color="#2c3e50", font=dict(color="white"), align="left"),
            cells=dict(values=[elo["agent_id"], elo["elo_rating"].round(1)], align="left"),
        )]
    )
    parts.append(elo_table.to_html(full_html=False, include_plotlyjs=False))

    parts.append("<h2>Risk-Adjusted Score (Sharpe-style)</h2>")
    parts.append(
        '<p class="howto">sharpe_ratio = mean_pairing_score / sqrt(variance_pairing_score). '
        "Separates 'consistently good' (high mean, low variance -> high Sharpe) from "
        "'occasionally spikes high' (high mean, high variance -> lower Sharpe). NaN means the "
        "agent's variance was exactly zero (perfectly consistent score every pairing).</p>"
    )
    sharpe_sorted = leaderboard.sort_values("sharpe_ratio", ascending=False)
    sharpe_table = go.Figure(
        data=[go.Table(
            header=dict(
                values=["Agent", "Mean Score", "Variance", "Sharpe Ratio"],
                fill_color="#2c3e50", font=dict(color="white"), align="left",
            ),
            cells=dict(
                values=[
                    sharpe_sorted["agent_id"],
                    sharpe_sorted["mean_pairing_score"].round(3),
                    sharpe_sorted["variance_pairing_score"].round(3),
                    sharpe_sorted["sharpe_ratio"].round(3),
                ],
                align="left",
            ),
        )]
    )
    parts.append(sharpe_table.to_html(full_html=False, include_plotlyjs=False))

    parts.append("<h2>Factor Decomposition: Math Level vs. ML Level</h2>")
    parts.append(
        '<p class="howto">OLS regression of sum_pairing_score on math_level and ml_level '
        "(dummy-coded, level 1 as baseline for each), fit on the 25 GridAgent rows only. "
        "Isolates each factor's marginal effect independent of the other -- the leaderboard "
        "alone entangles them. Coefficients are the estimated effect, in sum_pairing_score "
        "units, of moving from level 1 to that level on that factor, holding the other factor "
        "fixed.</p>"
    )
    coef_rows = "".join(
        f"<tr><td>{col}</td><td>{coef:.3f}</td></tr>"
        for col, coef in zip(factor_result["design_columns"], factor_result["coefficients"])
    )
    parts.append(
        f"<table border='1' cellpadding='6' style='border-collapse:collapse'>"
        f"<tr><th>Term</th><th>Coefficient</th></tr>{coef_rows}</table>"
        f"<p>(n={factor_result['n_observations']} grid agents)</p>"
    )

    if per_hand is not None:
        parts.append("<h2>Score per Hand</h2>")
        parts.append(
            '<p class="howto">score_per_hand = total pairing score / total hands survived across '
            "all legs. Controls for legs ending early on a bust -- an agent grinding out many "
            "hands at a mediocre per-hand rate can still out-SCORE an agent that busts fast but "
            "wins efficiently per hand while it lasts; this metric separates the two.</p>"
        )
        ph_table = go.Figure(
            data=[go.Table(
                header=dict(
                    values=["Agent", "Total Hands", "Score/Hand"],
                    fill_color="#2c3e50", font=dict(color="white"), align="left",
                ),
                cells=dict(
                    values=[per_hand["agent_id"], per_hand["total_hands"], per_hand["score_per_hand"].round(5)],
                    align="left",
                ),
            )]
        )
        parts.append(ph_table.to_html(full_html=False, include_plotlyjs=False))

    parts.append("<h2>Performance by Opponent Class</h2>")
    parts.append(
        '<p class="howto">Mean pairing_score split into "vs grid agents" and "vs PD archetypes" '
        "separately. Watch for a classic self-play-plateau signature: an agent elite against "
        "hand-built heuristics (PD archetypes) but only mediocre against other grid agents of "
        "similar sophistication.</p>"
    )
    class_split_display = class_split.round(3)
    class_cols = [c for c in class_split_display.columns if c != "agent_id"]
    class_table = go.Figure(
        data=[go.Table(
            header=dict(values=["Agent"] + class_cols, fill_color="#2c3e50", font=dict(color="white"), align="left"),
            cells=dict(values=[class_split_display["agent_id"]] + [class_split_display[c] for c in class_cols], align="left"),
        )]
    )
    parts.append(class_table.to_html(full_html=False, include_plotlyjs=False))

    if survival is not None:
        parts.append("<h2>Bust Rate / Survival Rate</h2>")
        parts.append(
            '<p class="howto">% of legs ending by elimination (busted before the hand cap) vs. '
            "reaching the cap. Independent of score -- tells you who plays a timid/durable game "
            "vs. a boom-or-bust one, regardless of whether that style wins.</p>"
        )
        surv_table = go.Figure(
            data=[go.Table(
                header=dict(
                    values=["Agent", "Legs Played", "Bust Rate", "Survival Rate"],
                    fill_color="#2c3e50", font=dict(color="white"), align="left",
                ),
                cells=dict(
                    values=[
                        survival["agent_id"], survival["legs_played"],
                        survival["bust_rate"].round(3), survival["survival_rate"].round(3),
                    ],
                    align="left",
                ),
            )]
        )
        parts.append(surv_table.to_html(full_html=False, include_plotlyjs=False))

    parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))