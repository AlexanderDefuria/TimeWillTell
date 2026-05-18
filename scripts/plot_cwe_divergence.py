#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
CWE distribution divergence plots across cross-validation folds.

Three plot families:
  1. divergence_by_fold    - fold# vs divergence, lines: train→test and train→holdout
  2. divergence_to_holdout - fold# vs divergence, lines: train→holdout and test→holdout
  3. divergence_vs_f1      - scatter: div(train,*) vs F1, side-by-side test/holdout panels

Train CWE distributions come from processed split parquets (phase="train").
Test/holdout CWE distributions and F1 scores come from data/eda/index_results.parquet.

Usage:
    uv run scripts/plot_cwe_divergence.py
    uv run scripts/plot_cwe_divergence.py --metric kl
    uv run scripts/plot_cwe_divergence.py --fold-type k_fold_cross_validation
    uv run scripts/plot_cwe_divergence.py --model llama3.2-1B --model phi4
"""
from __future__ import annotations

import re
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

try:
    from scripts import DATA_DIR, EDA_DIR, INDEX_RESULTS_FILE, PROCESSED_DATA_DIR, PRIMARY_FOLD_TYPES
    from scripts.cwe_utils import explode_with_cwe_id
    from scripts.plot_results import (
        PLOT_EXCLUDED_MODELS,
        apply_plot_model_exclusion,
        compute_metrics,
        format_fold_type_for_plot,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import DATA_DIR, EDA_DIR, INDEX_RESULTS_FILE, PROCESSED_DATA_DIR, PRIMARY_FOLD_TYPES
    from scripts.cwe_utils import explode_with_cwe_id
    from scripts.plot_results import (
        PLOT_EXCLUDED_MODELS,
        apply_plot_model_exclusion,
        compute_metrics,
        format_fold_type_for_plot,
    )

DivMetric = Literal["js", "kl", "hellinger", "tv"]
OUTPUT_DIR = EDA_DIR / "cwe_divergence"

METRIC_LABELS: dict[str, str] = {
    "js": "JS Divergence",
    "kl": "KL Divergence (P‖Q)",
    "hellinger": "Hellinger Distance",
    "tv": "Total Variation Distance",
}


# Maps processed-split directory names → canonical fold_type names in index_results.
# Resampling fold types (mixed_resampling_*, undersampling_*, oversampling_*) keep
# their own names and are not aliased to baseline fold types.
_SPLIT_FOLD_TYPE_ALIASES: dict[str, str] = {
    f"{ft}__{strategy}": ft
    for ft in [
        "undersampling_random_loo_block",
        "undersampling_temporal_loo_block",
        "oversampling_random_loo_block",
        "oversampling_temporal_loo_block",
        "mixed_resampling_random_loo_block",
        "mixed_resampling_temporal_loo_block",
        "undersampling_random_single",
        "undersampling_temporal_single",
        "oversampling_random_single",
        "oversampling_temporal_single",
        "mixed_resampling_random_single",
        "mixed_resampling_temporal_single",
    ]
    for strategy in ["val_most_common", "primary", "explode"]
}

FOLD_TYPE_PALETTE: dict[str, str] = {}  # populated lazily per run


# ─── Divergence computation ───────────────────────────────────────────────────


def _align(p_counts: dict, q_counts: dict) -> tuple[np.ndarray, np.ndarray]:
    all_keys = sorted(set(p_counts) | set(q_counts))
    p = np.array([p_counts.get(k, 0) for k in all_keys], dtype=float)
    q = np.array([q_counts.get(k, 0) for k in all_keys], dtype=float)
    return p, q


def compute_divergence(p_counts: dict, q_counts: dict, metric: DivMetric = "js") -> float:
    """Compute divergence between two CWE count distributions."""
    if not p_counts or not q_counts:
        return float("nan")
    p, q = _align(p_counts, q_counts)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum

    if metric == "js":
        m = 0.5 * (p + q)
        # KL(P‖M) - terms where p=0 contribute 0
        def kl_pm(a: np.ndarray, b: np.ndarray) -> float:
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
        return 0.5 * kl_pm(p, m) + 0.5 * kl_pm(q, m)

    if metric == "kl":
        # KL(P‖Q); add eps to Q only where P > 0 to avoid log(0)
        eps = 1e-10
        mask = p > 0
        q_safe = np.where(mask, np.maximum(q, eps), 1.0)
        return float(np.sum(p[mask] * np.log(p[mask] / q_safe[mask])))

    if metric == "hellinger":
        return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))

    if metric == "tv":
        return 0.5 * float(np.sum(np.abs(p - q)))

    raise ValueError(f"Unknown divergence metric: {metric!r}")


# ─── CWE distribution helpers ─────────────────────────────────────────────────


def get_cwe_counts(df: pl.DataFrame) -> dict[int, int]:
    """Return {cwe_id: count} from a DataFrame with a cwe_ids list column."""
    if df.is_empty():
        return {}
    exploded = explode_with_cwe_id(df)
    if exploded.is_empty():
        return {}
    counts = exploded.group_by("cwe_id").agg(pl.len().alias("n"))
    return dict(zip(counts["cwe_id"].to_list(), counts["n"].to_list()))


NO_CWE = "no-cwe"


def _str_exploded_cwe_counts(df: pl.DataFrame) -> dict[str, int]:
    """String-form CWE counts (exploded) used to determine val_most_common weights."""
    if df.is_empty():
        return {}
    has_cwe = df.filter(pl.col("cwe_ids").list.len() > 0)
    if has_cwe.is_empty():
        return {}
    exploded = has_cwe.explode("cwe_ids").drop_nulls("cwe_ids")
    counts = exploded.group_by("cwe_ids").agg(pl.len().alias("n"))
    return dict(zip(counts["cwe_ids"].to_list(), counts["n"].to_list()))


def get_val_most_common_counts(split_df: pl.DataFrame, phase: str) -> dict[str, int]:
    """Count CWE assignments using val_most_common strategy (matches cwe_balance.py).

    Each row is assigned the CWE from its list that is most common in val.
    Rows with no CWEs (safe rows and unlabelled vulnerabilities) count as NO_CWE.
    """
    val_counts = _str_exploded_cwe_counts(split_df.filter(pl.col("phase") == "val"))
    target_df = split_df.filter(pl.col("phase") == phase)
    if target_df.is_empty():
        return {}
    result: dict[str, int] = {}
    for row in target_df.select(["cwe_ids"]).iter_rows(named=True):
        cwe_list = row["cwe_ids"] or []
        assigned = max(cwe_list, key=lambda c: val_counts.get(c, 0)) if cwe_list else NO_CWE
        result[assigned] = result.get(assigned, 0) + 1
    return result


# ─── Split parquet discovery ──────────────────────────────────────────────────


def _parse_split_path(path: Path) -> dict | None:
    """Extract fold metadata from a processed split parquet path."""
    parts = path.parts
    filename = parts[-1]
    fn_match = re.match(r"split_dataset_(\w+)_s(\d+)\.parquet$", filename)
    if not fn_match:
        return None
    fold_segment = parts[-3]
    fold_match = re.match(r"fold_(\d+)_of_(\d+)$", fold_segment)
    if not fold_match:
        return None
    try:
        fold_type_raw = parts[-5]
        return {
            "tag": parts[-8],
            "dataset": parts[-7],
            "datamodule": parts[-6],
            "fold_type": _SPLIT_FOLD_TYPE_ALIASES.get(fold_type_raw, fold_type_raw),
            "fold_number": int(fold_match.group(1)),
            "total_folds": int(fold_match.group(2)),
            "dataset_size": fn_match.group(1),
            "seed": int(fn_match.group(2)),
            "path": path,
        }
    except IndexError:
        return None


def discover_split_parquets(
    processed_dir: Path,
    *,
    tag: str | None,
    seed: int,
    total_folds: int,
    fold_types: list[str],
) -> list[dict]:
    """Return one metadata dict per (fold_type, fold_number) - full-dataset splits only."""
    seen: set[tuple[str, int]] = set()
    results: list[dict] = []
    for path in processed_dir.rglob("split_dataset_*.parquet"):
        meta = _parse_split_path(path)
        if meta is None:
            continue
        if meta["dataset_size"] != "None":
            continue  # skip dev/subset splits; only use full-dataset files
        if tag and meta["tag"] != tag:
            continue
        if meta["seed"] != seed:
            continue
        if meta["total_folds"] != total_folds:
            continue
        if meta["fold_type"] not in fold_types:
            continue
        key = (meta["fold_type"], meta["fold_number"])
        if key not in seen:
            seen.add(key)
            results.append(meta)
    return results


def _generate_missing_splits(
    existing_keys: set[tuple[str, int]],
    processed_dir: Path,
    *,
    dataset: str,
    tag: str,
    seed: int,
    total_folds: int,
    fold_types: list[str],
) -> bool:
    """Run main.py --processonly --splits-only for each missing (fold_type, fold_number). Return True if any splits were generated."""
    any_generated = False
    repo_root = Path(__file__).resolve().parents[1]
    for ft in fold_types:
        for fn in range(total_folds):
            if (ft, fn) in existing_keys:
                continue
            any_generated = True
            print(f"  [auto] Generating split: {ft!r} fold {fn} …")
            cmd = [
                "uv", "run", "main.py",
                "--processonly", "--splits-only",
                "--dataset", dataset,
                "--fold_type", ft,
                "--fold", str(fn),
                "--total_folds", str(total_folds),
                "--seed", str(seed),
                "--model", "llama3.2-1B",
                "--tag", tag,
            ]
            result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [auto] WARNING: failed for {ft!r} fold {fn}:\n{result.stderr[-500:]}")
    return any_generated


# ─── Divergence table ─────────────────────────────────────────────────────────

_RawData = tuple[
    dict[tuple[str, int, str], dict[int, int]],
    dict[tuple[str, int], dict[int, int]],
    pl.DataFrame,
]


def _load_raw_data(
    index_results_path: Path,
    processed_dir: Path,
    *,
    seed: int,
    cw: int,
    total_folds: int,
    fold_types: list[str],
    models: list[str] | None,
    tag: str | None,
    dataset: str | None = None,
) -> _RawData:
    """Load and process data once; reused across all divergence metrics."""
    print("[1/4] Loading index_results …")
    needed = ["fold_type", "fold_number", "model", "run_mode", "phase", "cwe_ids",
              "prediction", "label", "seed", "cw", "total_folds", "tag"]
    available = pl.read_parquet_schema(index_results_path)
    select_cols = [c for c in needed if c in available]

    lf = pl.scan_parquet(index_results_path).select(select_cols)
    lf = lf.filter(
        pl.col("seed") == seed,
        pl.col("cw") == cw,
        pl.col("total_folds") == total_folds,
        pl.col("fold_type").is_in(fold_types),
    )
    if tag:
        lf = lf.filter(pl.col("tag") == tag)
    lf = apply_plot_model_exclusion(lf)
    if "run_mode" in available:
        lf = lf.filter(pl.col("run_mode").is_in(["fine_tuning", "opro"]))
    if models:
        lf = lf.filter(pl.col("model").is_in(models))

    try:
        df = lf.collect(engine="streaming")
    except Exception:
        df = lf.collect()

    if df.is_empty():
        raise RuntimeError("No rows matched the filters in index_results.parquet.")

    print(f"    Loaded {df.height:,} rows, {df['model'].n_unique()} models, "
          f"{df['fold_type'].n_unique()} fold types")

    print("[2/3] Computing CWE distributions from split parquets (val_most_common) …")
    split_metas = discover_split_parquets(
        processed_dir,
        tag=tag or "default",
        seed=seed,
        total_folds=total_folds,
        fold_types=fold_types,
    )
    if dataset:
        existing_keys = {(m["fold_type"], m["fold_number"]) for m in split_metas}
        if _generate_missing_splits(
            existing_keys, processed_dir,
            dataset=dataset, tag=tag or "default",
            seed=seed, total_folds=total_folds, fold_types=fold_types,
        ):
            split_metas = discover_split_parquets(
                processed_dir,
                tag=tag or "default",
                seed=seed,
                total_folds=total_folds,
                fold_types=fold_types,
            )
    phase_dists: dict[tuple[str, int, str], dict[str, int]] = {}
    train_dists: dict[tuple[str, int], dict[str, int]] = {}
    for meta in split_metas:
        key = (meta["fold_type"], meta["fold_number"])
        split_df = pl.read_parquet(meta["path"], columns=["phase", "cwe_ids"])
        train_dists[key] = get_val_most_common_counts(split_df, "train")
        phase_dists[(key[0], key[1], "test")] = get_val_most_common_counts(split_df, "test")
        phase_dists[(key[0], key[1], "holdout")] = get_val_most_common_counts(split_df, "holdout")
        print(f"    {meta['fold_type']!r} fold {meta['fold_number']}: "
              f"{len(train_dists[key])} unique CWE buckets (train)")

    print("[3/3] Computing F1 scores …")
    metrics_df = compute_metrics(df, ["fold_type", "fold_number", "model", "run_mode", "phase"])

    return phase_dists, train_dists, metrics_df


def _assemble_table(
    phase_dists: dict[tuple[str, int, str], dict[int, int]],
    train_dists: dict[tuple[str, int], dict[int, int]],
    metrics_df: pl.DataFrame,
    metric: DivMetric,
) -> pl.DataFrame:
    """Compute divergences for one metric and return a tidy table."""
    rows: list[dict] = []
    for row in metrics_df.iter_rows(named=True):
        ft, fn, model, run_mode, phase = (
            row["fold_type"], row["fold_number"], row["model"], row["run_mode"], row["phase"]
        )
        key = (ft, fn)
        train_dist = train_dists.get(key, {})
        test_dist = phase_dists.get((ft, fn, "test"), {})
        holdout_dist = phase_dists.get((ft, fn, "holdout"), {})
        rows.append({
            "fold_type": ft,
            "fold_number": fn,
            "model": model,
            "run_mode": run_mode,
            "phase": phase,
            f"f1_{phase}": row["f1"],
            "div_train_test": compute_divergence(train_dist, test_dist, metric),
            "div_train_holdout": compute_divergence(train_dist, holdout_dist, metric),
            "div_test_holdout": compute_divergence(test_dist, holdout_dist, metric),
            "metric": metric,
        })

    raw = pl.DataFrame(rows)
    test_rows = raw.filter(pl.col("phase") == "test").select(
        ["fold_type", "fold_number", "model", "run_mode", "f1_test",
         "div_train_test", "div_train_holdout", "div_test_holdout", "metric"]
    )
    holdout_rows = raw.filter(pl.col("phase") == "holdout").select(
        ["fold_type", "fold_number", "model", "run_mode", "f1_holdout"]
    )
    tbl = test_rows.join(holdout_rows, on=["fold_type", "fold_number", "model", "run_mode"],
                         how="full", coalesce=True)
    return tbl.sort(["fold_type", "fold_number", "model", "run_mode"])


def build_divergence_tables(
    index_results_path: Path,
    processed_dir: Path,
    metrics: list[DivMetric],
    *,
    seed: int,
    cw: int,
    total_folds: int,
    fold_types: list[str],
    models: list[str] | None,
    tag: str | None,
    dataset: str | None = None,
) -> dict[str, pl.DataFrame]:
    """Load data once, then return one divergence table per requested metric."""
    phase_dists, train_dists, metrics_df = _load_raw_data(
        index_results_path, processed_dir,
        seed=seed, cw=cw, total_folds=total_folds,
        fold_types=fold_types, models=models, tag=tag, dataset=dataset,
    )
    return {m: _assemble_table(phase_dists, train_dists, metrics_df, m) for m in metrics}


# ─── Plot helpers ─────────────────────────────────────────────────────────────


def _fold_type_palette(fold_types: list[str]) -> dict[str, str]:
    colors = sns.color_palette("tab10", n_colors=len(fold_types))
    return {ft: c for ft, c in zip(sorted(fold_types), colors)}


def _save(fig: plt.Figure, path: Path, dpi: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {path}")


def _add_correlation(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    """Draw a pooled regression line and annotate with Pearson r and p-value."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    xm, ym = x[mask], y[mask]
    r, p = stats.pearsonr(xm, ym)
    coeffs = np.polyfit(xm, ym, 1)
    x_line = np.linspace(xm.min(), xm.max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), color="black",
            linewidth=1.5, linestyle="-", alpha=0.7, zorder=5)
    p_str = f"{p:.3f}" if p >= 0.001 else "< 0.001"
    ax.text(0.97, 0.97, f"r = {r:.3f}\np = {p_str}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


# ─── Plot 1: Divergence by fold (train-reference) ─────────────────────────────


def plot_divergence_by_fold(tbl: pl.DataFrame, metric: DivMetric, output_dir: Path) -> None:
    """
    Line plot: fold number vs divergence.
    Two lines per fold_type: train→test and train→holdout.
    One figure per fold_type; a combined figure shows all fold_types.
    """
    metric_label = METRIC_LABELS[metric]
    fold_types = sorted(tbl["fold_type"].unique().to_list())
    palette = _fold_type_palette(fold_types)

    # Combined figure (all fold_types as subplots)
    n = len(fold_types)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ft in zip(axes, fold_types):
        ft_df = tbl.filter(pl.col("fold_type") == ft)
        # Average over models
        agg = (
            ft_df.group_by("fold_number")
            .agg(
                pl.col("div_train_test").mean(),
                pl.col("div_train_holdout").mean(),
                pl.col("div_train_test").std().alias("div_train_test_std"),
                pl.col("div_train_holdout").std().alias("div_train_holdout_std"),
            )
            .sort("fold_number")
        )
        folds = agg["fold_number"].to_list()
        tt = agg["div_train_test"].to_list()
        th = agg["div_train_holdout"].to_list()
        tt_std = [v or 0 for v in agg["div_train_test_std"].to_list()]
        th_std = [v or 0 for v in agg["div_train_holdout_std"].to_list()]

        color = palette[ft]
        ax.plot(folds, tt, marker="o", label="train → test", color=color, linestyle="-")
        ax.fill_between(folds,
                        [a - s for a, s in zip(tt, tt_std)],
                        [a + s for a, s in zip(tt, tt_std)],
                        alpha=0.15, color=color)
        ax.plot(folds, th, marker="s", label="train → holdout", color=color, linestyle="--")
        ax.fill_between(folds,
                        [a - s for a, s in zip(th, th_std)],
                        [a + s for a, s in zip(th, th_std)],
                        alpha=0.10, color=color)
        ax.set_title(format_fold_type_for_plot(ft), fontsize=10)
        ax.set_xlabel("Fold Number")
        ax.set_xticks(folds)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    axes[0].set_ylabel(metric_label)
    fig.suptitle(f"CWE Distribution Shift vs Training Set  [{metric_label}]", fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir / f"divergence_by_fold_{metric}.png")

    # Per-fold-type individual figures
    for ft in fold_types:
        ft_df = tbl.filter(pl.col("fold_type") == ft)
        models = sorted(ft_df["model"].unique().to_list())
        agg = (
            ft_df.group_by("fold_number")
            .agg(
                pl.col("div_train_test").mean(),
                pl.col("div_train_holdout").mean(),
                pl.col("div_train_test").std().alias("div_train_test_std"),
                pl.col("div_train_holdout").std().alias("div_train_holdout_std"),
            )
            .sort("fold_number")
        )
        folds = agg["fold_number"].to_list()
        tt = agg["div_train_test"].to_list()
        th = agg["div_train_holdout"].to_list()
        tt_std = [v or 0 for v in agg["div_train_test_std"].to_list()]
        th_std = [v or 0 for v in agg["div_train_holdout_std"].to_list()]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(folds, tt, marker="o", label="train → test", color="steelblue", linewidth=2)
        ax.fill_between(folds,
                        [a - s for a, s in zip(tt, tt_std)],
                        [a + s for a, s in zip(tt, tt_std)],
                        alpha=0.15, color="steelblue")
        ax.plot(folds, th, marker="s", label="train → holdout", color="darkorange",
                linewidth=2, linestyle="--")
        ax.fill_between(folds,
                        [a - s for a, s in zip(th, th_std)],
                        [a + s for a, s in zip(th, th_std)],
                        alpha=0.15, color="darkorange")
        ax.set_xlabel("Fold Number", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_xticks(folds)
        ax.set_title(
            f"CWE Shift vs Training Set - {format_fold_type_for_plot(ft)}\n"
            f"(averaged over {len(models)} model(s))",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        sns.despine(ax=ax)
        plt.tight_layout()
        _save(fig, output_dir / "divergence_by_fold" / f"{ft}_{metric}.png")


# ─── Plot 2: Divergence to holdout ────────────────────────────────────────────


def plot_divergence_to_holdout(tbl: pl.DataFrame, metric: DivMetric, output_dir: Path) -> None:
    """
    Line plot: fold number vs divergence, anchored at holdout.
    Two lines per fold_type: train→holdout and test→holdout.
    Shows whether test is a good CWE proxy for holdout.
    """
    metric_label = METRIC_LABELS[metric]
    fold_types = sorted(tbl["fold_type"].unique().to_list())
    n = len(fold_types)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ft in zip(axes, fold_types):
        ft_df = tbl.filter(pl.col("fold_type") == ft)
        agg = (
            ft_df.group_by("fold_number")
            .agg(
                pl.col("div_train_holdout").mean(),
                pl.col("div_test_holdout").mean(),
                pl.col("div_train_holdout").std().alias("div_train_holdout_std"),
                pl.col("div_test_holdout").std().alias("div_test_holdout_std"),
            )
            .sort("fold_number")
        )
        folds = agg["fold_number"].to_list()
        th = agg["div_train_holdout"].to_list()
        tesh = agg["div_test_holdout"].to_list()
        th_std = [v or 0 for v in agg["div_train_holdout_std"].to_list()]
        tesh_std = [v or 0 for v in agg["div_test_holdout_std"].to_list()]

        ax.plot(folds, th, marker="o", label="train → holdout", color="darkorange", linewidth=2)
        ax.fill_between(folds,
                        [a - s for a, s in zip(th, th_std)],
                        [a + s for a, s in zip(th, th_std)],
                        alpha=0.15, color="darkorange")
        ax.plot(folds, tesh, marker="s", label="test → holdout", color="steelblue",
                linewidth=2, linestyle="--")
        ax.fill_between(folds,
                        [a - s for a, s in zip(tesh, tesh_std)],
                        [a + s for a, s in zip(tesh, tesh_std)],
                        alpha=0.15, color="steelblue")
        ax.set_title(format_fold_type_for_plot(ft), fontsize=10)
        ax.set_xlabel("Fold Number")
        ax.set_xticks(folds)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    axes[0].set_ylabel(metric_label)
    fig.suptitle(
        f"CWE Distribution Shift vs Holdout  [{metric_label}]\n"
        "Small gap between lines → test is a reliable CWE proxy for holdout",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    _save(fig, output_dir / f"divergence_to_holdout_{metric}.png")

    # Per-fold-type individual figures
    for ft in fold_types:
        ft_df = tbl.filter(pl.col("fold_type") == ft)
        models = sorted(ft_df["model"].unique().to_list())
        agg = (
            ft_df.group_by("fold_number")
            .agg(
                pl.col("div_train_holdout").mean(),
                pl.col("div_test_holdout").mean(),
                pl.col("div_train_holdout").std().alias("div_train_holdout_std"),
                pl.col("div_test_holdout").std().alias("div_test_holdout_std"),
            )
            .sort("fold_number")
        )
        folds = agg["fold_number"].to_list()
        th = agg["div_train_holdout"].to_list()
        tesh = agg["div_test_holdout"].to_list()
        th_std = [v or 0 for v in agg["div_train_holdout_std"].to_list()]
        tesh_std = [v or 0 for v in agg["div_test_holdout_std"].to_list()]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(folds, th, marker="o", label="train → holdout", color="darkorange", linewidth=2)
        ax.fill_between(folds,
                        [a - s for a, s in zip(th, th_std)],
                        [a + s for a, s in zip(th, th_std)],
                        alpha=0.15, color="darkorange")
        ax.plot(folds, tesh, marker="s", label="test → holdout", color="steelblue",
                linewidth=2, linestyle="--")
        ax.fill_between(folds,
                        [a - s for a, s in zip(tesh, tesh_std)],
                        [a + s for a, s in zip(tesh, tesh_std)],
                        alpha=0.15, color="steelblue")
        ax.set_xlabel("Fold Number", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_xticks(folds)
        ax.set_title(
            f"CWE Shift vs Holdout - {format_fold_type_for_plot(ft)}\n"
            f"(averaged over {len(models)} model(s))",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        sns.despine(ax=ax)
        plt.tight_layout()
        _save(fig, output_dir / "divergence_to_holdout" / f"{ft}_{metric}.png")


# ─── Plot 3: Divergence vs F1 scatter ─────────────────────────────────────────


def plot_divergence_vs_f1(tbl: pl.DataFrame, metric: DivMetric, output_dir: Path) -> None:
    """
    Side-by-side scatter: div(train, {test|holdout}) vs F1_{test|holdout}.
    Each point = one (model, fold_number) pair. Color encodes fold_type.
    Tests whether higher CWE distribution shift predicts lower model F1.
    """
    metric_label = METRIC_LABELS[metric]
    fold_types = sorted(tbl["fold_type"].unique().to_list())
    palette = _fold_type_palette(fold_types)

    pdf = tbl.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, (div_col, f1_col, split_label) in zip(
        axes,
        [
            ("div_train_test", "f1_test", "Test"),
            ("div_train_holdout", "f1_holdout", "Holdout"),
        ],
    ):
        sub = pdf[[div_col, f1_col, "fold_type", "fold_number", "model"]].dropna()
        for ft in fold_types:
            ft_sub = sub[sub["fold_type"] == ft]
            if ft_sub.empty:
                continue
            ax.scatter(
                ft_sub[div_col],
                ft_sub[f1_col],
                color=palette[ft],
                label=format_fold_type_for_plot(ft),
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
                s=60,
            )
        _add_correlation(ax, sub[div_col].values, sub[f1_col].values)
        ax.set_xlabel(f"CWE {metric_label} (train → {split_label.lower()})", fontsize=11)
        ax.set_ylabel(f"F1 Score ({split_label})", fontsize=11)
        ax.set_title(f"Train → {split_label}", fontsize=12)
        sns.despine(ax=ax)

    # Shared legend from the first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(fold_types)),
               fontsize=9, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(
        f"CWE Distribution Shift vs Model F1  [{metric_label}]\n"
        "Each point = one (model, fold); dashed line = linear trend per fold type",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    _save(fig, output_dir / f"divergence_vs_f1_{metric}.png")


def _scatter_panels(
    pdf,
    fold_types: list[str],
    palette: dict,
    metric_label: str,
    title: str,
    marker_size: int = 60,
    fig_size: tuple[int, int] = (10, 5),
    y=1.02,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create the standard 2-panel scatter figure and return (fig, axes) unfinalised."""
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=True)
    for ax, (div_col, f1_col, split_label) in zip(
        axes,
        [("div_train_test", "f1_test", "Test"),
         ("div_train_holdout", "f1_holdout", "Holdout")],
    ):
        sub = pdf[[div_col, f1_col, "fold_type"]].dropna()
        for ft in fold_types:
            ft_sub = sub[sub["fold_type"] == ft]
            if ft_sub.empty:
                continue
            ax.scatter(ft_sub[div_col], ft_sub[f1_col],
                       color=palette[ft], label=format_fold_type_for_plot(ft),
                       alpha=0.75, edgecolors="white", linewidths=0.4, s=marker_size)
        _add_correlation(ax, sub[div_col].values, sub[f1_col].values)
        ax.set_xlim(0, 0.5)
        ax.set_xlabel(f"CWE {metric_label} (train → {split_label.lower()})", fontsize=11)
        ax.set_ylabel(f"F1 Score ({split_label})", fontsize=11)
        ax.set_title(f"Train → {split_label}", fontsize=12)
        sns.despine(ax=ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(fold_types)),
               fontsize=9, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(title, fontsize=12, y=y)
    plt.tight_layout()
    return fig, axes


def plot_divergence_vs_f1_per_model(tbl: pl.DataFrame, metric: DivMetric, output_dir: Path) -> None:
    """One scatter figure per fine-tuning model."""
    metric_label = METRIC_LABELS[metric]
    ft_df = tbl.filter(pl.col("run_mode") == "fine_tuning")
    if ft_df.is_empty():
        print("    No fine-tuning rows - skipping per-model scatter.")
        return
    fold_types = sorted(tbl["fold_type"].unique().to_list())
    palette = _fold_type_palette(fold_types)
    for model in sorted(ft_df["model"].unique().to_list()):
        model_df = ft_df.filter(pl.col("model") == model).to_pandas()
        fig, _ = _scatter_panels(
            model_df, fold_types, palette, metric_label,
            f"{model} - CWE Divergence vs F1  [{metric_label}]",
        )
        _save(fig, output_dir / "divergence_vs_f1" / f"{model}_{metric}.png")


def plot_divergence_vs_f1_summary(tbl: pl.DataFrame, metric: DivMetric, output_dir: Path) -> None:
    """Summary scatter: mean over fine-tuning models per (fold_type, fold_number) - 50 points."""
    metric_label = METRIC_LABELS[metric]
    ft_df = tbl.filter(pl.col("run_mode") == "fine_tuning")
    if ft_df.is_empty():
        print("    No fine-tuning rows - skipping summary scatter.")
        return
    agg = (
        ft_df.group_by(["fold_type", "fold_number"])
        .agg(
            pl.col("div_train_test").mean(),
            pl.col("div_train_holdout").mean(),
            pl.col("f1_test").mean(),
            pl.col("f1_holdout").mean(),
        )
        .sort(["fold_type", "fold_number"])
    )
    n_points = agg.height
    fold_types = sorted(tbl["fold_type"].unique().to_list())
    palette = _fold_type_palette(fold_types)
    fig, _ = _scatter_panels(
        agg.to_pandas(), fold_types, palette, metric_label,
        f"Summary (mean over models, {n_points} points) - CWE Divergence vs F1  [{metric_label}]",
        marker_size=80,
    )
    _save(fig, output_dir / f"divergence_vs_f1_summary_{metric}.png")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--metric", dest="metrics", choices=list(METRIC_LABELS), action="append",
                   help="Divergence metric(s) to generate (may be repeated; default: js kl)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cw", type=int, default=1024)
    p.add_argument("--total-folds", type=int, default=10)
    p.add_argument("--fold-type", dest="fold_types", action="append",
                   help="Filter to specific fold type(s); may be repeated")
    p.add_argument("--model", dest="models", action="append",
                   help="Filter to specific model(s); may be repeated")
    p.add_argument("--tag", default=None,
                   help="Filter to specific experiment tag")
    p.add_argument("--dataset", default=None,
                   help="Dataset name (e.g. megavul); enables auto-generation of missing splits")
    p.add_argument("--save-table", action="store_true",
                   help="Write each divergence table as a parquet to data/eda/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fold_types = args.fold_types or PRIMARY_FOLD_TYPES
    metrics: list[str] = args.metrics or ["js", "kl"]

    print(f"Metrics: {[METRIC_LABELS[m] for m in metrics]}")
    print(f"Fold types: {fold_types}")
    print(f"Models filter: {args.models or '(all)'}")
    print()

    tables = build_divergence_tables(
        index_results_path=INDEX_RESULTS_FILE,
        processed_dir=PROCESSED_DATA_DIR,
        metrics=metrics,
        seed=args.seed,
        cw=args.cw,
        total_folds=args.total_folds,
        fold_types=fold_types,
        models=args.models,
        tag=args.tag,
        dataset=args.dataset,
    )

    for metric, tbl in tables.items():
        print(f"\n── {METRIC_LABELS[metric]} ──────────────────────────────")
        print(f"Divergence table: {tbl.height} rows")
        print(tbl.head(5))

        if args.save_table:
            out_path = EDA_DIR / f"cwe_divergence_{metric}.parquet"
            tbl.write_parquet(out_path)
            print(f"Table written → {out_path}")

        print("\nGenerating Plot 1: divergence by fold (train reference) …")
        plot_divergence_by_fold(tbl, metric, OUTPUT_DIR)

        print("\nGenerating Plot 2: divergence to holdout …")
        plot_divergence_to_holdout(tbl, metric, OUTPUT_DIR)

        print("\nGenerating Plot 3: divergence vs F1 scatter …")
        plot_divergence_vs_f1(tbl, metric, OUTPUT_DIR)

        print("\nGenerating Plot 3b: per-model scatter (fine-tuning) …")
        plot_divergence_vs_f1_per_model(tbl, metric, OUTPUT_DIR)

        print("\nGenerating Plot 3c: summary scatter (fine-tuning, 50 points) …")
        plot_divergence_vs_f1_summary(tbl, metric, OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
