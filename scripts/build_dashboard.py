"""Run manually: python scripts/build_dashboard.py
Builds /analysis/output/dashboard.html + leaderboard.csv from the
existing calibration pairings_summary.parquet (Section 10 DoD)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.dashboard import build_dashboard_html

if __name__ == "__main__":
    build_dashboard_html(
        pairings_summary_path="logs/calibration/pairings_summary.parquet",
        output_path="analysis/output/dashboard.html",
        leaderboard_csv_path="analysis/output/leaderboard.csv",
    )
    print("Wrote analysis/output/dashboard.html and analysis/output/leaderboard.csv")