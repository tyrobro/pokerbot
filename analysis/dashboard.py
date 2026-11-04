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


def build_dashboard_html(pairings_summary_path: str, output_path: str, leaderboard_csv_path: str = None) -> None:
    df = load_pairings_summary(pairings_summary_path)
    leaderboard = build_leaderboard(df)
    agent_order = leaderboard["agent_id"].tolist()
    matrix = build_head_to_head_matrix(df, agent_order)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if leaderboard_csv_path:
        Path(leaderboard_csv_path).parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(leaderboard_csv_path, index=False)

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
        "</body></html>",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))