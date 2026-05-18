#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Statistical comparison of fold types for SCVD experiments.

Runs pairwise paired t-tests and Wilcoxon signed-rank tests between all fold
types for each model, then produces CSV results and diagnostic plots
(p-value heatmaps, Cohen's d, box/violin plots, Q-Q plots, distributions,
and summary statistics).  Both test and holdout phases are evaluated separately.

Usage:
    uv run scripts/plot_statistical_tests.py
    uv run scripts/plot_statistical_tests.py --metric f1
    uv run scripts/plot_statistical_tests.py --tag fine-tuning --dataset megavul
"""
from __future__ import annotations

import sys
import warnings
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    from scripts import EDA_DIR, INDEX_RESULTS_FILE
    from scripts.plot_results import METRICS, compute_metrics, load_results
    from scripts.plot_holdout_results import normalize_fold_type_names
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import EDA_DIR, INDEX_RESULTS_FILE
    from scripts.plot_results import METRICS, compute_metrics, load_results
    from scripts.plot_holdout_results import normalize_fold_type_names

# ─── Constants ────────────────────────────────────────────────────────────────

OUTPUT_DIR = EDA_DIR / "statistical_tests"
PLOTS_DIR = OUTPUT_DIR / "plots"

GROUP_COLS = ["model", "fold_type", "fold_number", "run_mode", "datamodule", "dataset", "tag", "cw", "seed"]

PHASES = ["test", "holdout"]

MIN_PAIRED_FOLDS = 5


def _abbreviate_fold_type(ft: str) -> str:
    """Abbreviate fold type name for plot display."""
    if ft.startswith("K-Fold"):
        return "K-Fold"
    elif ft.startswith("Random Growing Window"):
        return "R-GW"
    elif ft.startswith("Random LOO Block"):
        return "R-LOO-B"
    elif ft.startswith("Temporal Growing Window"):
        return "T-GW"
    elif ft.startswith("Temporal LOO Block"):
        return "T-LOO-B"
    else:
        return ft


# ─── Data preparation ────────────────────────────────────────────────────────


def prepare_per_fold_metrics(
    df: pl.DataFrame,
    metric: str = "f1",
    phase: str = "test",
) -> pl.DataFrame:
    """Compute per-fold metric values from raw predictions."""
    phase_df = df.filter(pl.col("phase") == phase)
    existing_cols = [c for c in GROUP_COLS if c in phase_df.columns]
    metrics = compute_metrics(phase_df, existing_cols)
    metrics = normalize_fold_type_names(metrics, include_tag=True)
    return metrics


# ─── Statistical tests ───────────────────────────────────────────────────────


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d: mean(diff) / std(diff)."""
    diff = a - b
    sd = diff.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(diff.mean() / sd)


def run_pairwise_tests(
    metrics_df: pl.DataFrame,
    model: str,
    metric: str = "f1",
) -> pd.DataFrame:
    """Run paired t-test and Wilcoxon between all fold type pairs for one model."""
    model_df = metrics_df.filter(pl.col("model") == model)
    fold_types = sorted(model_df["fold_type"].unique().to_list())

    rows = []
    for ft1, ft2 in combinations(fold_types, 2):
        vals1 = (
            model_df.filter(pl.col("fold_type") == ft1)
            .select(["fold_number", metric])
            .sort("fold_number")
        )
        vals2 = (
            model_df.filter(pl.col("fold_type") == ft2)
            .select(["fold_number", metric])
            .sort("fold_number")
        )
        # Align by fold_number
        paired = vals1.join(vals2, on="fold_number", suffix="_b")
        if paired.height < MIN_PAIRED_FOLDS:
            continue

        a = paired[metric].to_numpy().astype(float)
        b = paired[f"{metric}_b"].to_numpy().astype(float)

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(a, b)

        # Wilcoxon signed-rank
        try:
            w_stat, w_pval = stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            w_stat, w_pval = float("nan"), float("nan")

        d = _cohens_d(a, b)

        rows.append(
            {
                "Model": model,
                "Fold Type 1": ft1,
                "Fold Type 2": ft2,
                "N": paired.height,
                "T-Statistic": t_stat,
                "P-Value": t_pval,
                "Wilcoxon Statistic": w_stat,
                "Wilcoxon P-Value": w_pval,
                "Cohens d": d,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("P-Value").reset_index(drop=True)
    return result


# ─── CSV output ───────────────────────────────────────────────────────────────


def save_test_results_csv(results_df: pd.DataFrame, model: str, phase: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{model}_fold_type_tests_{phase}.csv"
    results_df.to_csv(path, index=False)
    return path


# ─── Plot helpers ─────────────────────────────────────────────────────────────


def _build_symmetric_matrix(
    results_df: pd.DataFrame,
    fold_types: list[str],
    col: str,
    diag_val: float = 1.0,
) -> pd.DataFrame:
    """Build a symmetric fold_type x fold_type matrix from pairwise results."""
    n = len(fold_types)
    mat = np.full((n, n), diag_val)
    idx = {ft: i for i, ft in enumerate(fold_types)}
    for _, row in results_df.iterrows():
        i, j = idx.get(row["Fold Type 1"]), idx.get(row["Fold Type 2"])
        if i is not None and j is not None:
            mat[i, j] = row[col]
            mat[j, i] = row[col] if col != "Cohens d" else -row[col]
    return pd.DataFrame(mat, index=fold_types, columns=fold_types)


# ─── Plot functions ───────────────────────────────────────────────────────────


def plot_pvalue_heatmaps(model: str, results_df: pd.DataFrame, fold_types: list[str], phase: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ttest_mat = _build_symmetric_matrix(results_df, fold_types, "P-Value")
    wilcoxon_mat = _build_symmetric_matrix(results_df, fold_types, "Wilcoxon P-Value")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(fold_types) * 2.5), max(5, len(fold_types) * 0.9)))

    for ax, mat, title in [
        (ax1, ttest_mat, f"{model} - T-test P-values ({phase})"),
        (ax2, wilcoxon_mat, f"{model} - Wilcoxon P-values ({phase})"),
    ]:
        sns.heatmap(
            mat, ax=ax, annot=True, fmt=".4f",
            cmap="RdYlGn", vmin=0.0, vmax=0.10,
            linewidths=0.5, linecolor="white",
            xticklabels=True, yticklabels=True,
            annot_kws={"size": 12},
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        # Abbreviate tick labels for display
        abbrev_x_labels = [_abbreviate_fold_type(label.get_text()) for label in ax.get_xticklabels()]
        abbrev_y_labels = [_abbreviate_fold_type(label.get_text()) for label in ax.get_yticklabels()]
        ax.set_xticklabels(abbrev_x_labels, rotation=0)
        ax.set_yticklabels(abbrev_y_labels, rotation=0)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_pvalue_heatmaps_{phase}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_cohens_d(model: str, results_df: pd.DataFrame, fold_types: list[str], phase: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    mat = _build_symmetric_matrix(results_df, fold_types, "Cohens d", diag_val=0.0)

    vabs = max(abs(mat.values.min()), abs(mat.values.max()), 0.5)
    fig, ax = plt.subplots(figsize=(max(7, len(fold_types) * 1.5), max(5, len(fold_types) * 0.9)))
    sns.heatmap(
        mat, ax=ax, annot=True, fmt=".3f",
        cmap="RdBu_r", vmin=-vabs, vmax=vabs,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 12},
    )
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{model} - Cohen's d (paired, {phase})", fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    # Abbreviate tick labels for display
    abbrev_x_labels = [_abbreviate_fold_type(label.get_text()) for label in ax.get_xticklabels()]
    abbrev_y_labels = [_abbreviate_fold_type(label.get_text()) for label in ax.get_yticklabels()]
    ax.set_xticklabels(abbrev_x_labels, rotation=0)
    ax.set_yticklabels(abbrev_y_labels, rotation=0)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_cohens_d_{phase}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(model: str, per_fold_pdf: pd.DataFrame, metric: str, phase: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = per_fold_pdf.copy()
    pdf["fold_type"] = pdf["fold_type"].apply(_abbreviate_fold_type)
    fig, ax = plt.subplots(figsize=(max(8, len(pdf["fold_type"].unique()) * 1.2), 5))
    sns.boxplot(data=pdf, x="fold_type", y=metric, hue="fold_type", ax=ax, palette="pastel",
                showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
                legend=False)
    ax.set_title(f"{model} - {metric.upper()} by Fold Type ({phase})", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_boxplot_{phase}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_violinplot(model: str, per_fold_pdf: pd.DataFrame, metric: str, phase: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = per_fold_pdf.copy()
    pdf["fold_type"] = pdf["fold_type"].apply(_abbreviate_fold_type)
    fig, ax = plt.subplots(figsize=(max(8, len(pdf["fold_type"].unique()) * 1.2), 5))
    sns.violinplot(data=pdf, x="fold_type", y=metric, hue="fold_type", ax=ax, palette="pastel",
                   inner="quart", legend=False)
    ax.set_title(f"{model} - {metric.upper()} Distribution ({phase})", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=11)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_violinplot_{phase}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(model: str, per_fold_pdf: pd.DataFrame, metric: str, phase: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_types = sorted(per_fold_pdf["fold_type"].unique())
    ncols = min(3, len(fold_types))
    nrows = (len(fold_types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for i, ft in enumerate(fold_types):
        ax = axes[i // ncols][i % ncols]
        vals = per_fold_pdf[per_fold_pdf["fold_type"] == ft][metric].dropna()
        if len(vals) > 1:
            sns.histplot(vals, kde=True, ax=ax, color="steelblue", alpha=0.6)
        ax.axvline(vals.mean(), color="red", linestyle="--", label=f"Mean: {vals.mean():.4f}")
        ax.axvline(vals.median(), color="green", linestyle=":", label=f"Median: {vals.median():.4f}")
        ax.set_title(_abbreviate_fold_type(ft), fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=9)
        ax.set_xlabel(metric.upper(), fontsize=10)

    # Hide unused axes
    for i in range(len(fold_types), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(f"{model} - {metric.upper()} Distributions ({phase})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_distributions_{phase}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_qqplots(model: str, per_fold_pdf: pd.DataFrame, metric: str, phase: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_types = sorted(per_fold_pdf["fold_type"].unique())
    ncols = min(3, len(fold_types))
    nrows = (len(fold_types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for i, ft in enumerate(fold_types):
        ax = axes[i // ncols][i % ncols]
        vals = per_fold_pdf[per_fold_pdf["fold_type"] == ft][metric].dropna().values
        if len(vals) > 2:
            stats.probplot(vals, dist="norm", plot=ax)
        ax.set_title(_abbreviate_fold_type(ft), fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    for i in range(len(fold_types), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(f"{model} - Q-Q Plots ({metric.upper()}, {phase})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_qqplots_{phase}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_summary_stats(model: str, per_fold_pdf: pd.DataFrame, metric: str, phase: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_types = sorted(per_fold_pdf["fold_type"].unique())

    means, medians, ci95s, iqrs = [], [], [], []
    for ft in fold_types:
        vals = per_fold_pdf[per_fold_pdf["fold_type"] == ft][metric].dropna().values
        means.append(vals.mean())
        medians.append(np.median(vals))
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        ci95s.append(1.96 * se)
        q1, q3 = np.percentile(vals, [25, 75]) if len(vals) > 1 else (vals.mean(), vals.mean())
        iqrs.append((q3 - q1) / 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(fold_types) * 1.5), 5))

    y_pos = np.arange(len(fold_types))
    ax1.errorbar(means, y_pos, xerr=ci95s, fmt="o", color="steelblue", capsize=4)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([_abbreviate_fold_type(ft) for ft in fold_types], fontsize=11)
    ax1.set_xlabel(f"{metric.upper()} (mean +/- 95% CI)", fontsize=11)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.set_title("Mean with 95% CI", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()

    ax2.errorbar(medians, y_pos, xerr=iqrs, fmt="s", color="forestgreen", capsize=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([_abbreviate_fold_type(ft) for ft in fold_types], fontsize=11)
    ax2.set_xlabel(f"{metric.upper()} (median +/- IQR/2)", fontsize=11)
    ax2.tick_params(axis="x", labelsize=10)
    ax2.set_title("Median with IQR", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()

    fig.suptitle(f"{model} - Summary Statistics ({metric.upper()}, {phase})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"{model}_summary_stats_{phase}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ─── Model-level TOST RETIRED ───────────────────────────────────────────────
# The cross-model TOST that collapsed each model to one mean F1 has been
# replaced by per-(model, run_mode) paired TOST. See scripts/plot_tost_grouped.py
# and docs/tost_grouped.md.


# ─── Main ─────────────────────────────────────────────────────────────────────


def _run_phase(
    df: pl.DataFrame,
    phase: str,
    metric: str,
    alpha: float,
) -> None:
    """Run statistical tests and generate all plots for a single phase."""
    print(f"\n{'=' * 60}")
    print(f"  Phase: {phase}")
    print(f"{'=' * 60}")

    metrics = prepare_per_fold_metrics(df, metric=metric, phase=phase)
    if metrics.is_empty():
        print(f"  No {phase} data available. Skipping.")
        return

    models = sorted(metrics["model"].unique().to_list())
    print(f"Found {len(models)} model(s): {', '.join(models)}")

    for model in models:
        print(f"\n--- {model} ({phase}) ---")
        model_metrics = metrics.filter(pl.col("model") == model)
        fold_types = sorted(model_metrics["fold_type"].unique().to_list())
        print(f"  Fold types ({len(fold_types)}): {', '.join(fold_types)}")

        if len(fold_types) < 2:
            print("  Skipping (need at least 2 fold types)")
            continue

        # Run pairwise tests
        test_results = run_pairwise_tests(model_metrics, model, metric=metric)
        if test_results.empty:
            print("  No valid pairs found (insufficient matched folds)")
            continue

        # Save CSV
        csv_path = save_test_results_csv(test_results, model, phase)
        print(f"  CSV: {csv_path}")

        # Per-fold data as pandas for plotting
        per_fold_pdf = model_metrics.select(["fold_type", "fold_number", metric]).to_pandas()

        # Significant results
        sig = test_results[test_results["P-Value"] < alpha]
        if not sig.empty:
            print(f"  Significant pairs (alpha={alpha}):")
            for _, row in sig.iterrows():
                print(f"    {row['Fold Type 1']} vs {row['Fold Type 2']}: "
                      f"p={row['P-Value']:.4f}, d={row['Cohens d']:.3f}")
        else:
            print(f"  No significant differences at alpha={alpha}")

        # Generate all plots
        plot_pvalue_heatmaps(model, test_results, fold_types, phase)
        plot_cohens_d(model, test_results, fold_types, phase)
        plot_boxplot(model, per_fold_pdf, metric, phase)
        plot_violinplot(model, per_fold_pdf, metric, phase)
        plot_distributions(model, per_fold_pdf, metric, phase)
        plot_qqplots(model, per_fold_pdf, metric, phase)
        plot_summary_stats(model, per_fold_pdf, metric, phase)
        print(f"  Plots saved to {PLOTS_DIR}")


def main() -> None:
    parser = ArgumentParser(description="Statistical comparison of fold types for SCVD.")
    parser.add_argument("--results", type=Path, default=INDEX_RESULTS_FILE)
    parser.add_argument("--metric", type=str, default="f1", choices=METRICS)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        print("Run `uv run scripts/collect_results.py` first.")
        sys.exit(1)

    print(f"Loading results from {args.results} ...")
    df = load_results(args.results, tag=args.tag, dataset=args.dataset)
    if isinstance(df, pl.LazyFrame):
        try:
            df = df.collect(engine="streaming")
        except Exception:
            df = df.collect()
    if df.is_empty():
        print("No data after filtering. Exiting.")
        sys.exit(0)

    sns.set_theme(style="whitegrid", context="paper")

    for phase in PHASES:
        _run_phase(df, phase, metric=args.metric, alpha=args.alpha)

    # Cross-model TOST retired - see scripts/plot_tost_grouped.py for the
    # per-(model, run_mode) replacement.

    print(f"\nAll outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
