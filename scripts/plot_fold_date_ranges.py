#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Gantt-style horizontal bar chart of test-set date ranges per fold for the two
primary temporal CV strategies: Temporal LOO Block CV and Temporal GW CV.

Usage:
    uv run scripts/plot_fold_date_ranges.py
    uv run scripts/plot_fold_date_ranges.py --output-dir data/eda
"""
from __future__ import annotations

import datetime
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

try:
    from scripts import EDA_DIR, INDEX_RESULTS_FILE
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import EDA_DIR, INDEX_RESULTS_FILE

LOO_TYPE = "temporal_loo_block_cross_validation"
GW_TYPE = "temporal_growing_window_cross_validation"

C_LOO = "#4477AA"
C_GW = "#EE6677"
C_HOLDOUT = "#228B22"

DPI = 600


def _parse_date(s: str) -> datetime.date:
    return datetime.date.fromisoformat(s[:10])


def load_data() -> pl.DataFrame:
    lf = (
        pl.scan_parquet(INDEX_RESULTS_FILE)
        .select(["fold_type", "fold_number", "phase", "publish_date"])
        .filter(pl.col("fold_type").is_in([LOO_TYPE, GW_TYPE]))
        .group_by(["fold_type", "fold_number", "phase"])
        .agg(
            pl.col("publish_date").min().alias("start"),
            pl.col("publish_date").max().alias("end"),
        )
        .sort(["phase", "fold_type", "fold_number"])
    )
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


def plot(df: pl.DataFrame) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="paper")

    test_df = df.filter(pl.col("phase") == "test")
    loo = test_df.filter(pl.col("fold_type") == LOO_TYPE).sort("fold_number")
    gw  = test_df.filter(pl.col("fold_type") == GW_TYPE).sort("fold_number")
    fold_nums = sorted(test_df["fold_number"].unique().to_list())

    # Holdout is the same for both strategies - take one row from either
    holdout_row = (
        df.filter((pl.col("phase") == "holdout") & (pl.col("fold_type") == LOO_TYPE))
        .sort("fold_number")
        .row(0, named=True)
    )
    h_start = _parse_date(holdout_row["start"])
    h_end   = _parse_date(holdout_row["end"])

    all_starts = [_parse_date(r) for r in df["start"].to_list()]
    all_ends   = [_parse_date(r) for r in df["end"].to_list()]
    x_min = min(all_starts)
    x_max = max(all_ends)

    fig, ax = plt.subplots(figsize=(12, 2.6))

    def _draw(rows: pl.DataFrame, y_offset: float, color: str, label: str) -> None:
        first = True
        for row in rows.iter_rows(named=True):
            s = _parse_date(row["start"])
            e = _parse_date(row["end"])
            fn = row["fold_number"]
            left = mdates.date2num(s)
            width = mdates.date2num(e) - left
            ax.barh(
                fn + y_offset, width, left=left, height=0.32,
                color=color, label=label if first else "_",
                alpha=0.85, zorder=3,
            )
            first = False

    _draw(loo, +0.18, C_LOO, "Temporal LOO")
    _draw(gw,  -0.18, C_GW,  "Temporal GW")

    # Holdout bar - one per fold row, spanning the full row height
    h_left  = mdates.date2num(h_start)
    h_width = mdates.date2num(h_end) - h_left
    for i, fn in enumerate(fold_nums):
        for j, offset in enumerate([+0.18, -0.18]):
            ax.barh(fn + offset, h_width, left=h_left, height=0.32,
                    color=C_HOLDOUT, label="Holdout" if i == 0 and j == 0 else "_",
                    alpha=0.85, zorder=3)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))

    ax.set_yticks(fold_nums)
    ax.set_yticklabels([f"Fold {i}" for i in fold_nums], fontsize=8)
    ax.set_ylim(-0.6, max(fold_nums) + 0.6)

    ax.set_xlabel("Date", fontsize=9)
    ax.set_title("Test-Set Periods by Fold - Temporal CV Strategies", fontsize=10, fontweight="bold")

    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    plt.tight_layout()
    return fig


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=EDA_DIR)
    args = p.parse_args()

    print(f"Loading {INDEX_RESULTS_FILE} …")
    df = load_data()
    print(f"Loaded {len(df)} rows")

    fig = plot(df)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fold_date_ranges_chart.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
