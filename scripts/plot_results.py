#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Unified results analysis and heatmap plotting for SCVD experiments.

Usage:
    uv run scripts/plot_results.py
    uv run scripts/plot_results.py --results data/results/results.parquet --output data/plots/
    uv run scripts/plot_results.py --tag my-tag --dataset megavul
"""
from __future__ import annotations

import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

try:
    from scripts import DATA_DIR, INDEX_RESULTS_FILE, PRIMARY_FOLD_TYPES, PROJECT_ROOT
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import DATA_DIR, INDEX_RESULTS_FILE, PRIMARY_FOLD_TYPES, PROJECT_ROOT

# ─── Constants ───────────────────────────────────────────────────────────────

GROUP_COLS = ["model", "fold_type", "fold_number", "run_mode", "datamodule", "dataset", "tag", "cw", "seed"]
METRICS = ["f1", "precision", "recall", "accuracy"]
NEEDED_COLS = ["phase", "prediction", "label"] + GROUP_COLS

FOLD_TYPE_PAIRS = [
    ("k_fold_cross_validation", "temporal_loo_block_cross_validation"),
    ("random_sliding_window_cross_validation", "temporal_sliding_window_cross_validation"),
    ("random_growing_window_cross_validation", "temporal_growing_window_cross_validation"),
]

# Omitted from every plot (heatmaps, CWE analysis, holdout scripts, etc.).
PLOT_EXCLUDED_MODELS: frozenset[str] = frozenset(
    {
        "gpt-oss-20b-gguf",
        "qwen3.5-27B-gguf",
        "llama3.2-1B-gguf",
    }
)


def apply_plot_model_exclusion(data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    """Remove rows for models that should not appear in any project plot."""
    if not PLOT_EXCLUDED_MODELS:
        return data
    excluded = list(PLOT_EXCLUDED_MODELS)
    if isinstance(data, pl.DataFrame):
        if "model" not in data.columns:
            return data
    elif "model" not in data.collect_schema().names():
        return data
    return data.filter(~pl.col("model").is_in(excluded))


def sort_models(models: list[str]) -> list[str]:
    """Sort model names: GGUF first (alphabetically), then non-GGUF (alphabetically)."""
    return sorted(models, key=lambda m: (not m.endswith("-gguf"), m))


def format_fold_type_for_plot(fold_type: str) -> str:
    """Axis/title label: drop the ``_cross_validation`` suffix; use spaces instead of underscores."""
    s = str(fold_type)
    if s.endswith("_cross_validation"):
        s = s[: -len("_cross_validation")]
    return s.replace("_", " ")


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_results(
    path: Path,
    tag: Optional[str] = None,
    dataset: Optional[str] = None,
    fold_types: Optional[list[str]] = None,
) -> pl.LazyFrame:
    """Return a LazyFrame with column pruning and pushed-down filters - no data loaded yet."""
    if fold_types is None:
        fold_types = PRIMARY_FOLD_TYPES
    available = pl.read_parquet_schema(path)
    select_cols = [c for c in NEEDED_COLS if c in available]

    lf = (
        pl.scan_parquet(path)
        .select(select_cols)
        .filter(pl.col("phase").is_in(["test", "holdout"]))
        .filter(pl.col("fold_type").is_in(fold_types))
    )
    if tag is not None:
        lf = lf.filter(pl.col("tag") == tag)
    if dataset is not None:
        lf = lf.filter(pl.col("dataset") == dataset)
    lf = apply_plot_model_exclusion(lf)
    if "run_mode" in available:
        lf = lf.filter(pl.col("run_mode").is_in(["fine_tuning", "opro"]))
    return lf


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame using streaming engine to minimise peak memory."""
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


# ─── Metrics computation ──────────────────────────────────────────────────────


def compute_metrics(df: pl.DataFrame, group_by: list[str]) -> pl.DataFrame:
    """Compute TP/TN/FP/FN and derived metrics per group."""
    agg = df.group_by(group_by).agg(
        [
            ((pl.col("prediction") == 1) & (pl.col("label") == 1)).sum().alias("tp"),
            ((pl.col("prediction") == 0) & (pl.col("label") == 0)).sum().alias("tn"),
            ((pl.col("prediction") == 1) & (pl.col("label") == 0)).sum().alias("fp"),
            ((pl.col("prediction") == 0) & (pl.col("label") == 1)).sum().alias("fn"),
        ]
    )
    agg = agg.with_columns(
        [
            (pl.col("tp") / (pl.col("tp") + pl.col("fp"))).alias("precision"),
            (pl.col("tp") / (pl.col("tp") + pl.col("fn"))).alias("recall"),
            ((pl.col("tp") + pl.col("tn")) / (pl.col("tp") + pl.col("tn") + pl.col("fp") + pl.col("fn"))).alias("accuracy"),
        ]
    )
    agg = agg.with_columns(
        (2 * pl.col("precision") * pl.col("recall") / (pl.col("precision") + pl.col("recall"))).alias("f1")
    )
    # Replace NaN with 0 (division by zero when no positives predicted)
    for col in ["precision", "recall", "f1"]:
        agg = agg.with_columns(pl.col(col).fill_nan(0.0))
    return agg.drop(["tp", "tn", "fp", "fn"])


def build_metric_frames(lf: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return (test_metrics, holdout_metrics, delta_metrics).

    Collects each phase separately to avoid holding the full dataset in memory.
    """
    existing_group_cols = [c for c in GROUP_COLS if c in lf.columns]

    test_metrics = compute_metrics(_collect(lf.filter(pl.col("phase") == "test")), existing_group_cols)
    holdout_metrics = compute_metrics(_collect(lf.filter(pl.col("phase") == "holdout")), existing_group_cols)

    # Delta = test - holdout (positive means test > holdout = potential overfit)
    joined = test_metrics.join(holdout_metrics, on=existing_group_cols, suffix="_h", how="inner")
    delta_metrics = joined.select(
        existing_group_cols
        + [
            (pl.col("f1") - pl.col("f1_h")).alias("f1"),
            (pl.col("precision") - pl.col("precision_h")).alias("precision"),
            (pl.col("recall") - pl.col("recall_h")).alias("recall"),
            (pl.col("accuracy") - pl.col("accuracy_h")).alias("accuracy"),
        ]
    )
    return test_metrics, holdout_metrics, delta_metrics


# ─── Pivot helpers ────────────────────────────────────────────────────────────


def make_pivot(df: pl.DataFrame, row: str, col: str, val: str) -> pd.DataFrame:
    """Aggregate mean over remaining dims and return a pandas pivot table."""
    agg = df.group_by([row, col]).agg(pl.col(val).mean())
    pdf = agg.to_pandas()
    pivot = pdf.pivot(index=row, columns=col, values=val)
    pivot.columns.name = None
    pivot.index.name = None
    return pivot


def _has_enough_data(pivot: pd.DataFrame, min_rows: int = 2, min_cols: int = 2) -> bool:
    return pivot.shape[0] >= min_rows and pivot.shape[1] >= min_cols


# ─── Plotting primitives ──────────────────────────────────────────────────────


def save_heatmap(
    pivot: pd.DataFrame,
    title: str,
    path: Path,
    cmap: str = "YlOrRd",
    fmt: str = ".3f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[tuple[float, float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if figsize is None:
        ncols = max(pivot.shape[1], 1)
        nrows = max(pivot.shape[0], 1)
        figsize = (max(6, ncols * 1.4), max(4, nrows * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _annotated_heatmap(
    pivot_mean: pd.DataFrame,
    pivot_std: pd.DataFrame,
    title: str,
    path: Path,
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Heatmap with mean ± std annotations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = max(pivot_mean.shape[1], 1)
    nrows = max(pivot_mean.shape[0], 1)
    figsize = (max(6, ncols * 1.6), max(4, nrows * 0.7))

    annot = pivot_mean.copy().astype(object)
    for r in pivot_mean.index:
        for c in pivot_mean.columns:
            m = pivot_mean.loc[r, c]
            s = pivot_std.loc[r, c] if r in pivot_std.index and c in pivot_std.columns else float("nan")
            if pd.isna(m):
                annot.loc[r, c] = ""
            elif pd.isna(s):
                annot.loc[r, c] = f"{m:.3f}"
            else:
                annot.loc[r, c] = f"{m:.3f}\n(±{s:.3f})"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_mean,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_pivot_stats(
    df: pl.DataFrame, row: str, col: str, val: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (mean_pivot, std_pivot, se_pivot) for val grouped by row × col."""
    agg = df.group_by([row, col]).agg([
        pl.col(val).mean().alias("mean"),
        pl.col(val).std().alias("std"),
        pl.col(val).count().alias("n"),
    ])
    pdf = agg.to_pandas()
    pdf["se"] = pdf["std"] / pdf["n"].pow(0.5)

    def _pivot(col_name: str) -> pd.DataFrame:
        p = pdf.pivot(index=row, columns=col, values=col_name)
        p.columns.name = None
        p.index.name = None
        return p

    return _pivot("mean"), _pivot("std"), _pivot("se")


# ─── Category 1 - Model × Fold Type ──────────────────────────────────────────


def plot_model_vs_fold_type(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    delta_metrics: pl.DataFrame,
    out_dir: Path,
) -> None:
    cat_dir = out_dir / "1_model_vs_fold_type"

    for metric in METRICS:
        for phase_label, data in [("test", test_metrics), ("holdout", holdout_metrics), ("delta", delta_metrics)]:
            if "model" not in data.columns or "fold_type" not in data.columns:
                continue
            mean_pivot, std_pivot, se_pivot = _make_pivot_stats(data, "model", "fold_type", metric)
            if not _has_enough_data(mean_pivot):
                warnings.warn(f"Category 1: skipping {metric}_{phase_label} (insufficient data)")
                continue
            for p in (mean_pivot, std_pivot, se_pivot):
                p.columns = [format_fold_type_for_plot(str(c)) for c in p.columns]
            cmap = "RdBu_r" if phase_label == "delta" else "YlOrRd"
            vmin, vmax = (-0.3, 0.3) if phase_label == "delta" else (0.0, 1.0)
            _annotated_heatmap(
                mean_pivot,
                std_pivot,
                title=f"{metric.upper()} - Model vs Fold Type ({phase_label})",
                path=cat_dir / f"{metric}_{phase_label}.png",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            save_heatmap(
                std_pivot,
                title=f"{metric.upper()} Std Dev - Model vs Fold Type ({phase_label})",
                path=cat_dir / "std" / f"{metric}_{phase_label}.png",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=None,
            )
            save_heatmap(
                se_pivot,
                title=f"{metric.upper()} Std Error - Model vs Fold Type ({phase_label})",
                path=cat_dir / "se" / f"{metric}_{phase_label}.png",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=None,
            )
    print(f"[1] Model × Fold Type plots written to {cat_dir}")


# ─── Category 2 - Per-Fold Analysis ──────────────────────────────────────────


def plot_per_fold_analysis(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    delta_metrics: pl.DataFrame,
    out_dir: Path,
) -> None:
    cat_dir = out_dir / "2_per_fold_analysis"

    if "fold_type" not in test_metrics.columns:
        print("[2] Skipping per-fold analysis (no fold_type column)")
        return

    fold_types = test_metrics["fold_type"].unique().to_list()

    for ft in fold_types:
        ft_dir = cat_dir / ft
        for metric in METRICS:
            for phase_label, data in [("test", test_metrics), ("holdout", holdout_metrics), ("delta", delta_metrics)]:
                subset = data.filter(pl.col("fold_type") == ft)
                if subset.is_empty() or "fold_number" not in subset.columns or "model" not in subset.columns:
                    continue
                pivot = make_pivot(subset, "fold_number", "model", metric)
                if not _has_enough_data(pivot):
                    warnings.warn(f"Category 2: skipping {ft}/{metric}_{phase_label} (insufficient data)")
                    continue
                cmap = "RdBu_r" if phase_label == "delta" else "YlOrRd"
                vmin, vmax = (-0.3, 0.3) if phase_label == "delta" else (0.0, 1.0)
                save_heatmap(
                    pivot,
                    title=f"{metric.upper()} - {format_fold_type_for_plot(ft)} per fold ({phase_label})",
                    path=ft_dir / f"{metric}_{phase_label}.png",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

    _plot_monolithic_f1_heatmap(test_metrics, holdout_metrics, delta_metrics, cat_dir, fold_types=PRIMARY_FOLD_TYPES, filename="monolithic_f1.png")
    _plot_monolithic_f1_heatmap(test_metrics, holdout_metrics, delta_metrics, cat_dir, fold_types=[ft for ft in test_metrics["fold_type"].unique().to_list() if ft not in PRIMARY_FOLD_TYPES], filename="monolithic_secondary_f1.png")
    _write_missing_monolithic(test_metrics, holdout_metrics, PROJECT_ROOT / "missing_monolithic.csv")
    _write_missing_secondary(test_metrics, holdout_metrics, PROJECT_ROOT / "missing_secondary.csv")
    print(f"[2] Per-Fold Analysis plots written to {cat_dir}")


def _format_fold_number(value: object) -> str:
    """Format fold_number for axis labels."""
    if value is None:
        return "unknown"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _build_monolithic_phase_pivot(
    data: pl.DataFrame,
    row_index: pd.MultiIndex,
    models: list[str],
) -> pd.DataFrame:
    """Build one phase block for the monolithic heatmap."""
    if not {"fold_type", "fold_number", "model", "f1"}.issubset(set(data.columns)):
        return pd.DataFrame(index=row_index, columns=models, dtype=float)

    agg = data.group_by(["fold_type", "fold_number", "model"]).agg(pl.col("f1").mean().alias("f1"))
    pdf = agg.to_pandas()

    if pdf.empty:
        pivot = pd.DataFrame(index=row_index, columns=models, dtype=float)
    else:
        pivot = pdf.pivot(index=["fold_type", "fold_number"], columns="model", values="f1")
        pivot = pivot.reindex(index=row_index, columns=models)
    return pivot


def _fold_type_groups(row_index: pd.MultiIndex) -> list[tuple[str, int, int]]:
    """Return contiguous fold_type groups as (fold_type, start_idx, end_idx_exclusive)."""
    groups: list[tuple[str, int, int]] = []
    if len(row_index) == 0:
        return groups

    fold_types = list(row_index.get_level_values("fold_type"))
    start = 0
    current = fold_types[0]
    for idx, fold_type in enumerate(fold_types[1:], start=1):
        if fold_type != current:
            groups.append((current, start, idx))
            current = fold_type
            start = idx
    groups.append((current, start, len(fold_types)))
    return groups


def _plot_monolithic_f1_heatmap(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    delta_metrics: pl.DataFrame,
    cat_dir: Path,
    fold_types: Optional[list[str]] = None,
    filename: str = "monolithic_f1.png",
) -> None:
    """Plot one monolithic figure with separate scales for test/holdout and delta.

    Args:
        fold_types: If provided, filter to only these fold types. If None, use all.
        filename: Output filename (default: monolithic_f1.png).
    """
    required_cols = {"fold_type", "fold_number", "model", "f1"}
    if not required_cols.issubset(set(test_metrics.columns)):
        warnings.warn("Category 2: skipping monolithic_f1 (missing required columns in test metrics)")
        return

    # Filter to requested fold types if specified
    if fold_types:
        test_metrics = test_metrics.filter(pl.col("fold_type").is_in(fold_types))
        holdout_metrics = holdout_metrics.filter(pl.col("fold_type").is_in(fold_types))
        delta_metrics = delta_metrics.filter(pl.col("fold_type").is_in(fold_types))

    row_source = pl.concat(
        [
            test_metrics.select("fold_type", "fold_number"),
            holdout_metrics.select("fold_type", "fold_number"),
            delta_metrics.select("fold_type", "fold_number"),
        ],
        how="vertical_relaxed",
    ).unique()
    if row_source.is_empty():
        warnings.warn("Category 2: skipping monolithic_f1 (no fold rows available)")
        return
    row_source = row_source.sort(["fold_type", "fold_number"])
    row_tuples = [(row["fold_type"], row["fold_number"]) for row in row_source.iter_rows(named=True)]
    row_index = pd.MultiIndex.from_tuples(row_tuples, names=["fold_type", "fold_number"])

    model_source = pl.concat(
        [
            test_metrics.select("model"),
            holdout_metrics.select("model"),
            delta_metrics.select("model"),
        ],
        how="vertical_relaxed",
    ).unique()
    models = sort_models(model_source["model"].to_list())
    if not models:
        warnings.warn("Category 2: skipping monolithic_f1 (no models found)")
        return

    test_pivot = _build_monolithic_phase_pivot(test_metrics, row_index, models)
    holdout_pivot = _build_monolithic_phase_pivot(holdout_metrics, row_index, models)
    delta_pivot = _build_monolithic_phase_pivot(delta_metrics, row_index, models)

    left_panel = pd.concat([test_pivot, holdout_pivot], axis=1)
    right_panel = delta_pivot.copy()

    if left_panel.shape[0] < 2 or left_panel.shape[1] < 2 or right_panel.shape[1] < 1:
        warnings.warn("Category 2: skipping monolithic_f1 (insufficient matrix shape)")
        return
    if left_panel.notna().sum().sum() == 0 and right_panel.notna().sum().sum() == 0:
        warnings.warn("Category 2: skipping monolithic_f1 (all cells are missing)")
        return

    fold_numbers = [
        _format_fold_number(fn) for fn in row_index.get_level_values("fold_number")
    ]
    fold_type_groups = _fold_type_groups(row_index)

    left_arr = left_panel.to_numpy(dtype=float)
    right_arr = right_panel.to_numpy(dtype=float)

    finite_delta = right_arr[np.isfinite(right_arr)]
    delta_abs_max = float(np.max(np.abs(finite_delta))) if finite_delta.size else 0.3
    if delta_abs_max < 1e-9:
        delta_abs_max = 0.3

    nrows = max(len(row_index), 1)
    n_left_cols = left_panel.shape[1]
    n_right_cols = right_panel.shape[1]
    n_models = len(models)

    # Size: 1 in/col, 0.65 in/row; +2.5 in for two colorbars; +1.8 in for margins
    fig_w = max(14.0, (n_left_cols + n_right_cols) * 1.0 + 2.5)
    fig_h = max(6.0, nrows * 0.65 + 1.8)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [n_left_cols, n_right_cols], "wspace": 0.08},
    )

    # ── Render cells with imshow (no seaborn layout side-effects) ────────────
    im_l = ax_left.imshow(
        left_arr, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0,
        interpolation="nearest", extent=(-0.5, n_left_cols - 0.5, nrows - 0.5, -0.5),
    )
    im_r = ax_right.imshow(
        right_arr, aspect="auto", cmap="RdBu_r", vmin=-delta_abs_max, vmax=delta_abs_max,
        interpolation="nearest", extent=(-0.5, n_right_cols - 0.5, nrows - 0.5, -0.5),
    )

    # ── Cell grid lines ───────────────────────────────────────────────────────
    for ax, ncols in [(ax_left, n_left_cols), (ax_right, n_right_cols)]:
        for i in range(nrows + 1):
            ax.axhline(i - 0.5, color="#d0d0d0", lw=0.5, zorder=3)
        for j in range(ncols + 1):
            ax.axvline(j - 0.5, color="#d0d0d0", lw=0.5, zorder=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    # ── Cell annotations ──────────────────────────────────────────────────────
    for arr, ax in [(left_arr, ax_left), (right_arr, ax_right)]:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, zorder=4)

    # ── Tick labels ───────────────────────────────────────────────────────────
    ax_left.set_xticks(range(n_left_cols))
    ax_left.set_xticklabels(models + models, rotation=90, ha="center", fontsize=8)
    ax_right.set_xticks(range(n_right_cols))
    ax_right.set_xticklabels(models, rotation=90, ha="center", fontsize=8)
    ax_left.set_yticks(range(nrows))
    ax_left.set_yticklabels(fold_numbers, fontsize=8)
    ax_right.tick_params(labelleft=False)

    # ── Phase labels above each block ─────────────────────────────────────────
    ax_left.text(n_models / 2 - 0.5, -0.65, "test",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_left.text(n_models * 1.5 - 0.5, -0.65, "holdout",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_right.text(n_right_cols / 2 - 0.5, -0.65, "delta",
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Vertical divider between test and holdout blocks
    ax_left.axvline(n_models - 0.5, color="#777777", lw=1.5, zorder=5)

    # ── Fold-type group labels (data coords, outside axes) ────────────────────
    for fold_type, start, end in fold_type_groups:
        center = (start + end - 1) / 2.0
        ax_left.text(
            -1.2, center,
            format_fold_type_for_plot(fold_type),
            ha="right", va="center", fontsize=8, fontweight="bold", clip_on=False,
        )
        if start > 0:
            ax_left.axhline(start - 0.5, color="#aaaaaa", lw=0.8, ls="--", clip_on=False)

    # ── Colorbars ─────────────────────────────────────────────────────────────
    # Lay out axes first so we can read their positions, then place identically-
    # sized colorbar axes manually - fraction= would scale with axis width and
    # produce mismatched bars across the two panels.
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()
    cbar_w = 0.012
    cbar_pad = 0.008
    cax_l = fig.add_axes([pos_l.x1 + cbar_pad, pos_l.y0, cbar_w, pos_l.height])
    cax_r = fig.add_axes([pos_r.x1 + cbar_pad, pos_r.y0, cbar_w, pos_r.height])
    cb_l = fig.colorbar(im_l, cax=cax_l)
    cb_l.set_label("F1", fontsize=8)
    cb_l.ax.tick_params(labelsize=7)
    cb_r = fig.colorbar(im_r, cax=cax_r)
    cb_r.set_label("Δ F1", fontsize=8)
    cb_r.ax.tick_params(labelsize=7)

    fig.savefig(cat_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_missing_monolithic(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    out_path: Path,
) -> None:
    required = {"model", "fold_type", "fold_number"}
    if not required.issubset(set(test_metrics.columns)) and not required.issubset(set(holdout_metrics.columns)):
        return

    id_cols = ["model", "fold_type", "fold_number"]

    # Filter to primary fold types only
    test_filtered = test_metrics.filter(pl.col("fold_type").is_in(PRIMARY_FOLD_TYPES))
    holdout_filtered = holdout_metrics.filter(pl.col("fold_type").is_in(PRIMARY_FOLD_TYPES))

    # Union of all (fold_type, fold_number) × models that appeared in either frame
    all_fold_rows = pl.concat([
        test_filtered.select(["fold_type", "fold_number"]),
        holdout_filtered.select(["fold_type", "fold_number"]),
    ]).unique()

    all_models = pl.concat([
        test_filtered.select("model"),
        holdout_filtered.select("model"),
    ]).unique()

    expected = all_fold_rows.join(all_models, how="cross")

    # Find all (model, fold_type, fold_number) that have BOTH phases completed
    test_present = test_filtered.select(id_cols).unique()
    holdout_present = holdout_filtered.select(id_cols).unique()
    both_present = test_present.join(holdout_present, on=id_cols, how="inner")

    # Find missing: expected but missing at least one phase (single entry per fold)
    missing = expected.join(both_present, on=id_cols, how="anti").with_columns(
        pl.lit("test").alias("missing_phase")  # Placeholder; relaunch will run both phases anyway
    )
    if missing.is_empty():
        print(f"[missing] No missing scenarios found - skipping {out_path}")
        return

    # Infer dataset and tag from available results (but NOT cw - infer cw from model)
    meta_cols = [c for c in ["dataset", "tag"] if c in test_filtered.columns]
    if meta_cols:
        meta = pl.concat([
            test_filtered.select(id_cols + meta_cols),
            holdout_filtered.select(id_cols + meta_cols),
        ]).unique()
        # Fallback: take first combo across all results
        fallback = meta.select(meta_cols).unique().head(1)
        for col in meta_cols:
            fallback_val = fallback[col][0]
            missing = missing.with_columns(pl.lit(fallback_val).alias(col))
    else:
        # Defaults matching launch_many.sh
        missing = missing.with_columns(
            pl.lit("megavul").alias("dataset"),
            pl.lit("test-holdout").alias("tag"),
        )

    # Always infer cw fresh from model type (GGUF → 16384, finetuned → 1024)
    # Never use values from results, as they're inconsistent across model types
    missing = missing.with_columns(
        pl.col("model").map_elements(
            lambda m: "16384" if m.endswith("-gguf") else "1024",
            return_dtype=pl.Utf8,
        ).alias("cw")
    )

    # Infer total_folds as max(fold_number) + 1 across primary fold types
    all_folds = pl.concat([
        test_filtered.select("fold_number"),
        holdout_filtered.select("fold_number"),
    ])["fold_number"]
    total_folds = int(all_folds.max()) + 1

    # Build flags string per row, mirroring launch_many.sh logic
    def _build_flags(row: dict) -> str:
        model = row["model"]
        parts = []
        if model.endswith("-gguf"):
            parts.append("--zero-shot --opro")
        parts += [
            f"--model={model}",
            f"--fold={int(row['fold_number'])}",
            f"--total_folds={total_folds}",
            f"--fold_type={row['fold_type']}",
            f"--dataset={row.get('dataset', 'megavul')}",
            f"--tag={row.get('tag', 'test-holdout')}",
        ]
        # Include cw if present and non-empty
        cw = row.get("cw", "")
        if cw:
            parts.append(f"--cw={cw}")
        parts += [
            "--batch_size=2",
            "--grad_acc=32",
            "--epochs=5",
        ]
        return " ".join(parts)

    flags_col = [_build_flags(r) for r in missing.iter_rows(named=True)]
    missing = missing.with_columns(pl.Series("flags", flags_col))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing.write_csv(out_path)
    print(f"[missing] {missing.height} missing scenario(s) written to {out_path}")


def _write_missing_secondary(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    out_path: Path,
) -> None:
    """Write missing scenarios for secondary (non-primary) fold types."""
    required = {"model", "fold_type", "fold_number"}
    if not required.issubset(set(test_metrics.columns)) and not required.issubset(set(holdout_metrics.columns)):
        return

    id_cols = ["model", "fold_type", "fold_number"]

    # Filter to secondary (non-primary) fold types only
    test_filtered = test_metrics.filter(~pl.col("fold_type").is_in(PRIMARY_FOLD_TYPES))
    holdout_filtered = holdout_metrics.filter(~pl.col("fold_type").is_in(PRIMARY_FOLD_TYPES))

    # Union of all (fold_type, fold_number) × models that appeared in either frame
    all_fold_rows = pl.concat([
        test_filtered.select(["fold_type", "fold_number"]),
        holdout_filtered.select(["fold_type", "fold_number"]),
    ]).unique()

    all_models = pl.concat([
        test_filtered.select("model"),
        holdout_filtered.select("model"),
    ]).unique()

    expected = all_fold_rows.join(all_models, how="cross")

    # Find all (model, fold_type, fold_number) that have BOTH phases completed
    test_present = test_filtered.select(id_cols).unique()
    holdout_present = holdout_filtered.select(id_cols).unique()
    both_present = test_present.join(holdout_present, on=id_cols, how="inner")

    # Find missing: expected but missing at least one phase (single entry per fold)
    missing = expected.join(both_present, on=id_cols, how="anti").with_columns(
        pl.lit("test").alias("missing_phase")
    )
    if missing.is_empty():
        print(f"[secondary] No missing secondary fold type scenarios found - skipping {out_path}")
        return

    # Infer dataset and tag from available results (but NOT cw - infer cw from model)
    meta_cols = [c for c in ["dataset", "tag"] if c in test_filtered.columns]
    if meta_cols:
        meta = pl.concat([
            test_filtered.select(id_cols + meta_cols),
            holdout_filtered.select(id_cols + meta_cols),
        ]).unique()
        # Fallback: take first combo across all results
        fallback = meta.select(meta_cols).unique().head(1)
        for col in meta_cols:
            fallback_val = fallback[col][0]
            missing = missing.with_columns(pl.lit(fallback_val).alias(col))
    else:
        # Defaults matching launch_many.sh
        missing = missing.with_columns(
            pl.lit("megavul").alias("dataset"),
            pl.lit("test-holdout").alias("tag"),
        )

    # Always infer cw fresh from model type (GGUF → 16384, finetuned → 1024)
    missing = missing.with_columns(
        pl.col("model").map_elements(
            lambda m: "16384" if m.endswith("-gguf") else "1024",
            return_dtype=pl.Utf8,
        ).alias("cw")
    )

    # Infer total_folds as max(fold_number) + 1 across secondary fold types
    all_folds = pl.concat([
        test_filtered.select("fold_number"),
        holdout_filtered.select("fold_number"),
    ])["fold_number"]
    total_folds = int(all_folds.max()) + 1

    # Build flags string per row
    def _build_flags(row: dict) -> str:
        model = row["model"]
        parts = []
        if model.endswith("-gguf"):
            parts.append("--zero-shot --opro")
        parts += [
            f"--model={model}",
            f"--fold={int(row['fold_number'])}",
            f"--total_folds={total_folds}",
            f"--fold_type={row['fold_type']}",
            f"--dataset={row.get('dataset', 'megavul')}",
            f"--tag={row.get('tag', 'test-holdout')}",
        ]
        # Include cw if present and non-empty
        cw = row.get("cw", "")
        if cw:
            parts.append(f"--cw={cw}")
        parts += [
            "--batch_size=2",
            "--grad_acc=32",
            "--epochs=5",
        ]
        return " ".join(parts)

    flags_col = [_build_flags(r) for r in missing.iter_rows(named=True)]
    missing = missing.with_columns(pl.Series("flags", flags_col))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing.write_csv(out_path)
    print(f"[secondary] {missing.height} missing secondary fold type scenario(s) written to {out_path}")


# ─── Category 4 - Random vs Temporal Pairs ───────────────────────────────────


def plot_random_vs_temporal(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    delta_metrics: pl.DataFrame,
    out_dir: Path,
) -> None:
    cat_dir = out_dir / "4_fold_type_pairs"
    cat_dir.mkdir(parents=True, exist_ok=True)

    if "fold_type" not in test_metrics.columns or "model" not in test_metrics.columns:
        print("[4] Skipping random vs temporal (missing columns)")
        return

    present_fold_types = set(test_metrics["fold_type"].unique().to_list())

    for phase_label, data in [("test", test_metrics), ("holdout", holdout_metrics), ("delta", delta_metrics)]:
        for rand_ft, temp_ft in FOLD_TYPE_PAIRS:
            if rand_ft not in present_fold_types or temp_ft not in present_fold_types:
                continue

            rand_agg = data.filter(pl.col("fold_type") == rand_ft).group_by("model").agg(
                [pl.col(m).mean().alias(f"{m}_rand") for m in METRICS]
            )
            temp_agg = data.filter(pl.col("fold_type") == temp_ft).group_by("model").agg(
                [pl.col(m).mean().alias(f"{m}_temp") for m in METRICS]
            )
            joined = rand_agg.join(temp_agg, on="model", how="inner")
            if joined.is_empty():
                continue

            # delta = temporal - random (positive = temporal harder)
            rows = []
            for m in METRICS:
                delta_col = joined.select((pl.col(f"{m}_temp") - pl.col(f"{m}_rand")).alias("delta"), "model")
                for row in delta_col.iter_rows(named=True):
                    rows.append({"model": row["model"], "metric": m, "delta": row["delta"]})

            pdf = pd.DataFrame(rows).pivot(index="model", columns="metric", values="delta")
            pdf.columns.name = None
            pdf.index.name = None

            if not _has_enough_data(pdf, min_rows=1, min_cols=1):
                continue

            rand_short = format_fold_type_for_plot(rand_ft)
            temp_short = format_fold_type_for_plot(temp_ft)
            pair_slug = f"{rand_ft}_vs_{temp_ft}"[:60]

            save_heatmap(
                pdf,
                title=f"Temporal − Random delta\n{temp_short} vs {rand_short} ({phase_label})",
                path=cat_dir / f"{pair_slug}_f1_{phase_label}.png",
                cmap="RdBu_r",
                vmin=-0.3,
                vmax=0.3,
            )

    print(f"[4] Random vs Temporal plots written to {cat_dir}")


# ─── Category 5 - Detailed Metrics Dashboard ─────────────────────────────────


def plot_detailed_metrics(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    delta_metrics: pl.DataFrame,
    out_dir: Path,
) -> None:
    cat_dir = out_dir / "5_detailed_metrics"
    cat_dir.mkdir(parents=True, exist_ok=True)

    if "model" not in test_metrics.columns:
        print("[5] Skipping detailed metrics (no model column)")
        return

    # Plot 5a - Full metrics grid (test + holdout in one figure)
    _plot_full_metrics_grid(test_metrics, holdout_metrics, cat_dir)

    # Plot 5b - Precision-Recall scatter
    _plot_precision_recall_scatter(test_metrics, holdout_metrics, cat_dir)

    print(f"[5] Detailed Metrics plots written to {cat_dir}")


def _plot_full_metrics_grid(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    cat_dir: Path,
) -> None:
    for phase_label, data in [("test", test_metrics), ("holdout", holdout_metrics)]:
        agg = data.group_by("model").agg([pl.col(m).mean() for m in METRICS])
        pdf = agg.to_pandas().set_index("model")
        if pdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(METRICS) * 2), max(4, pdf.shape[0] * 0.8)))
        sns.heatmap(
            pdf,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(f"Full Metrics Grid - {phase_label}", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        fig.savefig(cat_dir / f"full_metrics_grid_{phase_label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_precision_recall_scatter(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    cat_dir: Path,
) -> None:
    if "fold_type" not in test_metrics.columns:
        return

    agg = test_metrics.group_by(["model", "fold_type"]).agg(
        [pl.col("precision").mean(), pl.col("recall").mean(), pl.col("f1").mean()]
    ).to_pandas()

    if agg.empty:
        return

    models = sorted(agg["model"].unique())
    fold_types = sorted(agg["fold_type"].unique())

    model_palette = sns.color_palette("husl", n_colors=len(models))
    model_colors = dict(zip(models, model_palette))
    markers_list = ["o", "s", "^", "D", "v", "P", "X", "*"]
    ft_markers = {ft: markers_list[i % len(markers_list)] for i, ft in enumerate(fold_types)}

    fig, ax = plt.subplots(figsize=(8, 7))

    # Iso-F1 curves
    recall_range = np.linspace(0.01, 1.0, 200)
    for f1_target in [0.2, 0.4, 0.6, 0.8]:
        prec = f1_target * recall_range / (2 * recall_range - f1_target + 1e-9)
        mask = (prec >= 0) & (prec <= 1)
        ax.plot(recall_range[mask], prec[mask], "--", color="grey", alpha=0.4, linewidth=0.8)
        # Label at a mid-point
        mid_idx = mask.sum() // 2
        valid_idxs = np.where(mask)[0]
        if len(valid_idxs) > 0:
            mid = valid_idxs[len(valid_idxs) // 2]
            ax.text(recall_range[mid], prec[mid], f"F1={f1_target}", fontsize=7, color="grey",
                    ha="center", va="bottom", rotation=15)

    for _, row in agg.iterrows():
        ax.scatter(
            row["recall"],
            row["precision"],
            color=model_colors[row["model"]],
            marker=ft_markers[row["fold_type"]],
            s=80,
            alpha=0.8,
            zorder=3,
        )
        ax.annotate(
            row["model"][:12],
            xy=(row["recall"], row["precision"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[m], markersize=8, label=m)
        for m in models
    ] + [
        Line2D(
            [0],
            [0],
            marker=ft_markers[ft],
            color="grey",
            markersize=8,
            linestyle="None",
            label=format_fold_type_for_plot(ft),
        )
        for ft in fold_types
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="lower left", ncol=1, framealpha=0.7)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision–Recall Tradeoff (test, mean per model × fold type)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(cat_dir / "precision_recall_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── LaTeX table generation ───────────────────────────────────────────────────


def _f1_colour(val: float, vmin: float, vmax: float) -> str:
    """White (#FFFFFF) → green (#4CAF50) gradient mapped to [vmin, vmax]."""
    if np.isnan(val):
        return "FFFFFF"
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0))
    r = int(round(255 - t * (255 - 76)))
    g = int(round(255 - t * (255 - 175)))
    b = int(round(255 - t * (255 - 80)))
    return f"{r:02X}{g:02X}{b:02X}"


def _latex_cell(val: float, vmin: float, vmax: float) -> str:
    """Return a coloured cell string: \\cellcolor[HTML]{hex} XX.X"""
    colour = _f1_colour(val, vmin, vmax)
    if np.isnan(val):
        return f"\\cellcolor[HTML]{{{colour}}} ---"
    return f"\\cellcolor[HTML]{{{colour}}} {val * 100:.1f}"


def _canonical_fold_order(idx_a: "pd.Index", idx_b: "pd.Index") -> list[str]:
    """Return fold types ordered: FOLD_TYPE_PAIRS flattened (unique), then any others."""
    seen: dict[str, None] = {}
    for pair in FOLD_TYPE_PAIRS:
        for ft in pair:
            seen[ft] = None
    all_fts = list(idx_a) + [f for f in idx_b if f not in idx_a]
    remaining = [f for f in all_fts if f not in seen]
    ordered = [f for f in seen if f in set(all_fts)]
    return ordered + remaining


def _build_latex_summary_table(
    test_pivot: "pd.DataFrame",
    holdout_pivot: "pd.DataFrame",
    ft_models: list[str],
    zs_models: list[str],
    fold_order: list[str],
) -> str:
    all_models = ft_models + zs_models
    n_ft = len(ft_models)
    n_zs = len(zs_models)
    n_cols = 1 + n_ft + n_zs  # fold label + all model columns

    # global colour scale across both panels
    all_vals = np.array(
        [v for pivot in (test_pivot, holdout_pivot) for v in pivot.values.flatten() if not np.isnan(v)]
    )
    vmin = float(all_vals.min()) if len(all_vals) else 0.0
    vmax = float(all_vals.max()) if len(all_vals) else 1.0

    def _short(m: str) -> str:
        return m.replace("-gguf", "").replace("-", " ")

    def _data_rows(pivot: "pd.DataFrame") -> list[str]:
        rows = []
        for ft in fold_order:
            label = format_fold_type_for_plot(ft)
            cells = []
            for m in all_models:
                val = pivot.at[ft, m] if (ft in pivot.index and m in pivot.columns) else float("nan")
                cells.append(_latex_cell(val, vmin, vmax))
            rows.append(f"    {label} & {' & '.join(cells)} \\\\")
        return rows

    # column spec: l for label, grouped model cols with a | between ft and zs
    if n_ft and n_zs:
        col_spec = f"l *{{{n_ft}}}{{c}} | *{{{n_zs}}}{{c}}"
    elif n_ft:
        col_spec = f"l *{{{n_ft}}}{{c}}"
    else:
        col_spec = f"l *{{{n_zs}}}{{c}}"

    lines: list[str] = [
        "% Auto-generated by scripts/plot_results.py - do not edit manually.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{colortbl}, \\usepackage{xcolor}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\footnotesize",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
    ]

    # header row 1: group labels
    header1_parts = [""]
    if n_ft:
        header1_parts.append(f"\\multicolumn{{{n_ft}}}{{c}}{{\\textit{{Fine-tuning}}}}")
    if n_zs:
        header1_parts.append(f"\\multicolumn{{{n_zs}}}{{c}}{{\\textit{{Zero-shot}}}}")
    lines.append(f"    {' & '.join(header1_parts)} \\\\")

    # header row 2: model names
    header2 = ["Fold Type"] + [f"\\rotatebox{{60}}{{\\footnotesize {_short(m)}}}" for m in all_models]
    lines.append(f"    {' & '.join(header2)} \\\\")
    lines.append("    \\midrule")

    # test panel rows
    lines.extend(_data_rows(test_pivot))
    lines.append("    \\midrule\\midrule")

    # holdout panel label
    lines.append(f"    \\multicolumn{{{n_cols}}}{{c}}{{\\textit{{Holdout}}}} \\\\")
    lines.append("    \\midrule")

    # holdout panel rows
    lines.extend(_data_rows(holdout_pivot))
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(
        "  \\caption{Mean F1 (\\%) per model and cross-validation strategy."
        " Upper panel: cross-fold test results. Lower panel: held-out evaluation."
        " Colour scale: white (low) $\\to$ green (high), shared across both panels.}"
    )
    lines.append("  \\label{tab:summary_results}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def generate_summary_latex_table(
    test_metrics: pl.DataFrame,
    holdout_metrics: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Write a two-panel LaTeX table (test + holdout) to output_dir/summary_table.tex."""
    if test_metrics.is_empty() and holdout_metrics.is_empty():
        return

    test_pivot = make_pivot(test_metrics, row="fold_type", col="model", val="f1")
    holdout_pivot = make_pivot(holdout_metrics, row="fold_type", col="model", val="f1")

    all_model_names: list[str] = sorted(
        set(test_pivot.columns.tolist()) | set(holdout_pivot.columns.tolist())
    )
    ft_models = sorted(m for m in all_model_names if not m.endswith("-gguf"))
    zs_models = sorted(m for m in all_model_names if m.endswith("-gguf"))

    fold_order = _canonical_fold_order(test_pivot.index, holdout_pivot.index)
    test_pivot = test_pivot.reindex(index=fold_order, columns=ft_models + zs_models)
    holdout_pivot = holdout_pivot.reindex(index=fold_order, columns=ft_models + zs_models)

    latex = _build_latex_summary_table(test_pivot, holdout_pivot, ft_models, zs_models, fold_order)

    out_path = output_dir / "summary_table.tex"
    out_path.write_text(latex)
    print(f"  LaTeX summary table → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = ArgumentParser(description="Plot SCVD experiment results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=INDEX_RESULTS_FILE,
        help="Path to consolidated results parquet (default: scripts/eda/index_results.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "plots",
        help="Output directory for plots (default: data/plots/)",
    )
    parser.add_argument("--tag", type=str, default=None, help="Filter by experiment tag")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        print("Run `uv run scripts/collect_results.py` first to generate the consolidated results.")
        sys.exit(1)

    print(f"Loading results from {args.results} ...")
    lf = load_results(args.results, tag=args.tag, dataset=args.dataset)

    print("Computing metrics ...")
    test_metrics, holdout_metrics, delta_metrics = build_metric_frames(lf)

    if test_metrics.is_empty() and holdout_metrics.is_empty():
        print("No data after filtering. Exiting.")
        sys.exit(0)
    print(
        f"  test groups:    {test_metrics.height}\n"
        f"  holdout groups: {holdout_metrics.height}\n"
        f"  delta groups:   {delta_metrics.height}"
    )

    args.output.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white", context="paper")

    plot_model_vs_fold_type(test_metrics, holdout_metrics, delta_metrics, args.output)
    plot_per_fold_analysis(test_metrics, holdout_metrics, delta_metrics, args.output)
    plot_random_vs_temporal(test_metrics, holdout_metrics, delta_metrics, args.output)
    plot_detailed_metrics(test_metrics, holdout_metrics, delta_metrics, args.output)
    generate_summary_latex_table(test_metrics, holdout_metrics, args.output)

    print(f"\nAll plots written to {args.output}")


if __name__ == "__main__":
    main()
