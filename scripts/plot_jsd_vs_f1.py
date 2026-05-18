#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
JSD vs F1 scatter plots split by model family (fine-tuned vs zero-shot).

Two per-family figures (test | holdout panels, fold-type colors) plus a
2×2 combined comparison figure.  Outputs to data/plots/7_jsd_vs_f1/.

Data loading and divergence computation reuse build_divergence_tables from
plot_cwe_divergence.py; no logic is duplicated here.

Usage:
    uv run scripts/plot_jsd_vs_f1.py
    uv run scripts/plot_jsd_vs_f1.py --metric js
    uv run scripts/plot_jsd_vs_f1.py --tag fine-tuning --dataset megavul
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

try:
    from scripts import INDEX_RESULTS_FILE, PROCESSED_DATA_DIR, PRIMARY_FOLD_TYPES, CWE_RESAMPLING_FOLD_TYPES
    from scripts.plot_cwe_divergence import (
        METRIC_LABELS,
        _add_correlation,
        _fold_type_palette,
        _save,
        _scatter_panels,
        build_divergence_tables,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import INDEX_RESULTS_FILE, PROCESSED_DATA_DIR, PRIMARY_FOLD_TYPES, CWE_RESAMPLING_FOLD_TYPES
    from scripts.plot_cwe_divergence import (
        METRIC_LABELS,
        _add_correlation,
        _fold_type_palette,
        _save,
        _scatter_panels,
        build_divergence_tables,
    )

PLOTS_DIR = Path(__file__).resolve().parents[1] / "data" / "plots" / "7_jsd_vs_f1"

_LLAMA_MODEL = "llama3.2-1B"
_ALL_FOLD_TYPES = list(dict.fromkeys(
    [ft for ft in PRIMARY_FOLD_TYPES if "growing_window" not in ft]
    + CWE_RESAMPLING_FOLD_TYPES
))

_RUN_MODES = {
    "fine_tuning": "Fine-Tuned (LoRA)",
    "opro": "OPRO (GGUF)",
}

_FILE_LABELS = {
    "fine_tuning": "fine_tuned",
    "opro": "opro",
}


# ─── Per-family scatter (2-panel) ─────────────────────────────────────────────


def plot_jsd_vs_f1_by_runmode(
    tbl: pl.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """One 2-panel figure per run_mode (test | holdout), fold-type colors."""
    metric_label = METRIC_LABELS[metric]
    all_fold_types = sorted(tbl["fold_type"].unique().to_list())
    palette = _fold_type_palette(all_fold_types)

    for run_mode, family_label in _RUN_MODES.items():
        sub = tbl.filter(pl.col("run_mode") == run_mode)
        if sub.is_empty():
            print(f"    No data for run_mode={run_mode!r} - skipping.")
            continue

        fold_types = sorted(sub["fold_type"].unique().to_list())
        n_models = sub["model"].n_unique()
        title = (
            f"{family_label} - CWE JS-Divergence vs F1  [{metric_label}]\n"
            f"Each point = one (model, fold)  |  {n_models} model(s)"
        )
        fig, _ = _scatter_panels(
            sub.to_pandas(),
            fold_types,
            {ft: palette[ft] for ft in fold_types},
            metric_label,
            title,
            fig_size=(8, 3),
            y=0.93,
        )
        file_label = _FILE_LABELS[run_mode]
        _save(fig, output_dir / f"jsd_vs_f1_{file_label}_{metric}.png")


# ─── Combined 2×2 comparison ──────────────────────────────────────────────────


def plot_jsd_vs_f1_per_model(
    tbl: pl.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """One 2-panel scatter figure per model (test | holdout panels, fold-type colors)."""
    metric_label = METRIC_LABELS[metric]
    all_fold_types = sorted(tbl["fold_type"].unique().to_list())
    palette = _fold_type_palette(all_fold_types)

    for model in sorted(tbl["model"].unique().to_list()):
        model_df = tbl.filter(pl.col("model") == model)
        fold_types = sorted(model_df["fold_type"].unique().to_list())
        if not fold_types:
            continue
        pdf = model_df.to_pandas()
        fig, _ = _scatter_panels(
            pdf,
            fold_types,
            {ft: palette[ft] for ft in fold_types},
            metric_label,
            f"{model} - CWE JS-Divergence vs F1  [{metric_label}]",
        )
        _save(fig, output_dir / "per_model" / f"{model}_{metric}.png")


def plot_jsd_vs_f1_combined(
    tbl: pl.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """2-row × 2-col figure: rows = run_mode, cols = test | holdout."""
    metric_label = METRIC_LABELS[metric]
    all_fold_types = sorted(tbl["fold_type"].unique().to_list())
    palette = _fold_type_palette(all_fold_types)

    run_modes = [rm for rm in _RUN_MODES if not tbl.filter(pl.col("run_mode") == rm).is_empty()]
    if not run_modes:
        print("    No data for any run_mode - skipping combined figure.")
        return

    panels = [
        ("div_train_test", "f1_test", "Test"),
        ("div_train_holdout", "f1_holdout", "Holdout"),
    ]
    nrows, ncols = len(run_modes), 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(10 * ncols / 2, 5 * nrows),
        sharey=False,
    )
    if nrows == 1:
        axes = [axes]

    for row_idx, run_mode in enumerate(run_modes):
        sub = tbl.filter(pl.col("run_mode") == run_mode)
        pdf = sub.to_pandas()
        fold_types = sorted(sub["fold_type"].unique().to_list())
        family_label = _RUN_MODES[run_mode]

        for col_idx, (div_col, f1_col, split_label) in enumerate(panels):
            ax = axes[row_idx][col_idx]
            sub_cols = pdf[[div_col, f1_col, "fold_type"]].dropna()
            for ft in fold_types:
                ft_sub = sub_cols[sub_cols["fold_type"] == ft]
                if ft_sub.empty:
                    continue
                ax.scatter(
                    ft_sub[div_col],
                    ft_sub[f1_col],
                    color=palette[ft],
                    label=ft,
                    alpha=0.75,
                    edgecolors="white",
                    linewidths=0.4,
                    s=55,
                )
            _add_correlation(
                ax,
                sub_cols[div_col].values,
                sub_cols[f1_col].values,
            )
            ax.set_xlabel(f"CWE {metric_label} (train → {split_label.lower()})", fontsize=10)
            ax.set_ylabel(f"F1 Score ({split_label})", fontsize=10)
            ax.set_title(f"{family_label}  |  Train → {split_label}", fontsize=11, fontweight="bold")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            sns.despine(ax=ax)

    # Shared legend from first row
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=min(4, len(all_fold_types)),
            fontsize=8,
            bbox_to_anchor=(0.5, -0.06),
        )

    fig.suptitle(
        f"CWE JS-Divergence vs F1 - Fine-Tuned vs OPRO  [{metric_label}]",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save(fig, output_dir / f"jsd_vs_f1_combined_{metric}.png")


# ─── Llama3.2-1B - all fold types (primary + CWE resampling) ─────────────────


def _extended_palette(fold_types: list[str]) -> dict[str, str]:
    """tab20-based palette for up to 20 fold types."""
    colors = sns.color_palette("tab20", n_colors=max(len(fold_types), 1))
    return {ft: colors[i] for i, ft in enumerate(sorted(fold_types))}


def plot_jsd_vs_f1_llama_all_folds(
    tbl: pl.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    """2-panel scatter (test | holdout) for llama3.2-1B across all fold types."""
    sub = tbl.filter(pl.col("model") == _LLAMA_MODEL)
    if sub.is_empty():
        print(f"    No data for model={_LLAMA_MODEL!r} - skipping llama all-folds figure.")
        return

    metric_label = METRIC_LABELS[metric]
    fold_types = sorted(sub["fold_type"].unique().to_list())
    palette = _extended_palette(fold_types)
    n_folds = len(fold_types)

    title = (
        f"{_LLAMA_MODEL} - CWE JS-Divergence vs F1  [{metric_label}]\n"
        f"Original + CWE-resampled fold types  |  {n_folds} fold type(s)"
    )
    fig, axes = _scatter_panels(
        sub.to_pandas(), fold_types, palette, metric_label, title,
        fig_size=(8, 3),
        y=0.93,
    )

    # _scatter_panels already places a figure legend; remove it and replace with
    # one sized for the larger fold-type set (ncol=6 → ~3 rows for 17 types).
    if fig.legends:
        fig.legends[0].remove()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(fold_types)),
            fontsize=7,
            bbox_to_anchor=(0.5, -0.12),
        )

    plt.tight_layout()
    _save(fig, output_dir / f"llama3.2_1B_{metric}.png")


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = ArgumentParser(description="JSD vs F1 scatter plots split by model family.")
    parser.add_argument(
        "--metric", dest="metrics", choices=list(METRIC_LABELS), action="append",
        help="Divergence metric(s); may be repeated. Default: js",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cw", type=int, default=1024)
    parser.add_argument("--total-folds", type=int, default=10)
    parser.add_argument(
        "--fold-type", dest="fold_types", action="append",
        help="Filter to specific fold type(s); may be repeated",
    )
    parser.add_argument(
        "--model", dest="models", action="append",
        help="Filter to specific model(s); may be repeated",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dataset", default=None,
                        help="Dataset name (e.g. megavul); enables auto-generation of missing splits")
    args = parser.parse_args()

    fold_types = args.fold_types or PRIMARY_FOLD_TYPES
    metrics: list[str] = args.metrics or ["js"]

    if not INDEX_RESULTS_FILE.exists():
        print(f"ERROR: Results file not found: {INDEX_RESULTS_FILE}")
        print("Run `uv run scripts/collect_results.py` first.")
        raise SystemExit(1)

    print(f"Metrics     : {[METRIC_LABELS[m] for m in metrics]}")
    print(f"Fold types  : {fold_types}")
    print(f"Models      : {args.models or '(all)'}")
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

    sns.set_theme(style="whitegrid", context="paper")

    for metric, tbl in tables.items():
        print(f"\n── {METRIC_LABELS[metric]} ──")
        print(f"  Rows: {tbl.height}, run_modes: {tbl['run_mode'].unique().to_list()}")

        print("  Generating per-family scatter …")
        plot_jsd_vs_f1_by_runmode(tbl, metric, PLOTS_DIR)

        print("  Generating per-model scatter …")
        plot_jsd_vs_f1_per_model(tbl, metric, PLOTS_DIR)

        print("  Generating combined 2×2 figure …")
        plot_jsd_vs_f1_combined(tbl, metric, PLOTS_DIR)

    print("\n── Llama3.2-1B - all fold types (primary + CWE resampling) ──")
    # Auto-detect the dataset so _generate_missing_splits can create train splits
    # for resampling fold types that have index results but no full-dataset split files.
    llama_dataset = args.dataset
    if llama_dataset is None:
        try:
            _hits = (
                pl.scan_parquet(INDEX_RESULTS_FILE)
                .filter(
                    (pl.col("model") == _LLAMA_MODEL) &
                    (pl.col("fold_type").is_in(CWE_RESAMPLING_FOLD_TYPES))
                )
                .select("dataset")
                .unique()
                .collect()
            )
            if _hits.height == 1:
                llama_dataset = _hits["dataset"][0]
                print(f"  Auto-detected dataset: {llama_dataset!r}")
        except Exception:
            pass

    llama_tables = build_divergence_tables(
        index_results_path=INDEX_RESULTS_FILE,
        processed_dir=PROCESSED_DATA_DIR,
        metrics=metrics,
        seed=args.seed,
        cw=args.cw,
        total_folds=args.total_folds,
        fold_types=_ALL_FOLD_TYPES,
        models=[_LLAMA_MODEL],
        tag=args.tag,
        dataset=llama_dataset,
    )
    for metric, tbl in llama_tables.items():
        print(f"  [{METRIC_LABELS[metric]}] rows: {tbl.height}")
        plot_jsd_vs_f1_llama_all_folds(tbl, metric, PLOTS_DIR)

    print(f"\nAll outputs written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
