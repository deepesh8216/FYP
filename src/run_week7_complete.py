"""
Run full Week 7 pipeline: tables + ablation JSON + confusion figures.
Run from project root: python3 src/run_week7_complete.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "src" / "summarize_week7.py")],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        [sys.executable, str(root / "src" / "week7_export_figures.py")],
        cwd=str(root),
        check=True,
    )
    print(
        "\nWeek 7 complete: outputs/week7/master_comparison.csv, "
        "aggregate_by_variant.csv, ablation_summary.json, figures/"
    )


if __name__ == "__main__":
    main()
