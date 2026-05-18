#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
CWE resampling results: per-fold heatmaps and test vs holdout bar chart.

Model: llama3.2-1B only.
Fold types: loo_block resampling variants (under/over/mixed × random/temporal).

Usage:
    uv run scripts/plot_cwe_resampling.py
    uv run scripts/plot_cwe_resampling.py --dataset megavul
    uv run scripts/plot_cwe_resampling.py --tag my-tag
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
    from scripts import CWE_RESAMPLING_BASELINES, CWE_RESAMPLING_FOLD_TYPES, INDEX_RESULTS_FILE, PROCESSED_DATA_DIR
    from scripts.plot_cwe_divergence import compute_divergence, discover_split_parquets, get_cwe_counts, get_val_most_common_counts
    from scripts.plot_results import build_metric_frames, load_results, make_pivot
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import CWE_RESAMPLING_BASELINES, CWE_RESAMPLING_FOLD_TYPES, INDEX_RESULTS_FILE, PROCESSED_DATA_DIR
    from scripts.plot_cwe_divergence import compute_divergence, discover_split_parquets, get_cwe_counts, get_val_most_common_counts
    from scripts.plot_results import build_metric_frames, load_results, make_pivot

PLOTS_DIR = Path(__file__).resolve().parents[1] / "data" / "plots" / "8_cwe_resampling"

CMAP = "YlOrRd"
DPI = 600
MODEL = "llama3.2-1B"

LOO_BLOCK_FOLD_TYPES = [ft for ft in CWE_RESAMPLING_FOLD_TYPES if "loo_block" in ft]
TEMPORAL_LOO_BLOCK_FOLD_TYPES = [ft for ft in LOO_BLOCK_FOLD_TYPES if "temporal" in ft]
TEMPORAL_LOO_BLOCK_BASELINES = ["temporal_loo_block_cross_validation"]

_STRATEGIES = ["val_most_common", "explode", "primary"]
_STRATEGY_LABELS = {
    "val_most_common": "Most Common",
    "explode":         "Explode",
    "primary":         "Primary",
}

_LABELS: dict[str, str] = {
    "undersampling_random_loo_block":      "Under Random LOO",
    "undersampling_temporal_loo_block":    "Under Temporal LOO",
    "oversampling_random_loo_block":       "Over Random LOO",
    "oversampling_temporal_loo_block":     "Over Temporal LOO",
    "mixed_resampling_random_loo_block":   "Mixed Random LOO",
    "mixed_resampling_temporal_loo_block": "Mixed Temporal LOO",
    "random_loo_block_cross_validation":   "Random LOO (Baseline)",
    "temporal_loo_block_cross_validation": "Temporal LOO (Baseline)",
}

_SHORT_LABELS: dict[str, str] = {
    "undersampling_random_loo_block":      "Under R-LOO",
    "undersampling_temporal_loo_block":    "Under T-LOO",
    "oversampling_random_loo_block":       "Over R-LOO",
    "oversampling_temporal_loo_block":     "Over T-LOO",
    "mixed_resampling_random_loo_block":   "Mixed R-LOO",
    "mixed_resampling_temporal_loo_block": "Mixed T-LOO",
    "random_loo_block_cross_validation":   "R-LOO Base",
    "temporal_loo_block_cross_validation": "T-LOO Base",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_heatmap(
    df: pl.DataFrame,
    phase: str,
    fold_types: list[str],
    baseline_types: list[str],
) -> plt.Figure:
    """Heatmap of F1 (%) per (resampling method × fold number).

    Resampling rows appear first; baselines are separated by a horizontal line.
    """
    pivot = make_pivot(df, row="fold_type", col="fold_number", val="f1")
    ordered = [ft for ft in fold_types if ft in pivot.index] + [
        ft for ft in baseline_types if ft in pivot.index
    ]
    pivot = pivot.loc[ordered]
    n_resample = sum(ft in pivot.index for ft in fold_types)
    pivot.index = [_LABELS.get(ft, ft) for ft in pivot.index]

    n_rows, n_cols = pivot.shape
    fig_w = max(6, n_cols * 0.9 + 2)
    fig_h = max(2, n_rows * 0.8 + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=CMAP,
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "F1"},
        linewidths=0.5,
    )
    # Separator line between resampling rows and baseline rows
    if n_resample > 0 and n_resample < n_rows:
        ax.axhline(n_resample, color="black", linewidth=2.0)
    ax.set_title(
        f"F1 per Fold - {phase.capitalize()} ({MODEL})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Fold Number")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    return fig


def plot_jsd_bar(
    fold_types: list[str],
    baseline_types: list[str],
    processed_dir: Path,
    tag: str | None,
    seed: int,
    total_folds: int,
) -> plt.Figure:
    """Bar chart: JSD(train CWE dist, test CWE dist) per resampling method.

    Each bar = mean ± std across folds. Resampling bars are solid; baseline bars are hatched.
    """
    all_types = fold_types + baseline_types
    split_metas = discover_split_parquets(
        processed_dir,
        tag=tag or "default",
        seed=seed,
        total_folds=total_folds,
        fold_types=all_types,
    )

    jsd_by_type: dict[str, list[float]] = {ft: [] for ft in all_types}
    for meta in split_metas:
        ft = meta["fold_type"]
        if ft not in jsd_by_type:
            continue
        split_df = pl.read_parquet(meta["path"], columns=["phase", "cwe_ids"])
        train_dist = get_val_most_common_counts(split_df, "train")
        test_dist = get_val_most_common_counts(split_df, "test")
        jsd = compute_divergence(train_dist, test_dist, "js")
        if not np.isnan(jsd):
            jsd_by_type[ft].append(jsd)

    colors = sns.color_palette("tab10", n_colors=len(all_types))
    x = np.arange(len(all_types), dtype=float)
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(10, len(all_types) * 1.4 + 2), 3.5))
    for i, ft in enumerate(all_types):
        vals = jsd_by_type[ft]
        mean = float(np.nanmean(vals)) if vals else float("nan")
        std = float(np.nanstd(vals)) if vals else 0.0
        is_baseline = ft in baseline_types
        ax.bar(
            x[i], mean, width,
            label=_LABELS.get(ft, ft),
            color=colors[i],
            alpha=0.85,
            hatch="//" if is_baseline else None,
            zorder=3,
        )
        ax.errorbar(
            x[i], mean, yerr=std,
            fmt="none", color="black", capsize=4, linewidth=1.0, zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_LABELS.get(ft, ft) for ft in all_types],
        rotation=20, ha="right", fontsize=8,
    )
    ax.set_ylabel("JSD (mean ± std over folds)", fontsize=10)
    ax.set_title(
        f"CWE Distribution JSD - Train vs Test | Temporal LOO Block CV ({MODEL})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 0.5), loc="center left",
              framealpha=0.9, borderaxespad=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def _get_cwe_dist_for_strategy(
    split_df: pl.DataFrame, phase: str, strategy: str
) -> dict[str, int]:
    """CWE count distribution for `phase` rows using the given assignment strategy."""
    phase_df = split_df.filter(pl.col("phase") == phase)
    if strategy == "val_most_common":
        return get_val_most_common_counts(split_df, phase)
    if strategy == "explode":
        return get_cwe_counts(phase_df)
    if strategy == "primary":
        result: dict[str, int] = {}
        for row in phase_df.select(["cwe_ids"]).iter_rows(named=True):
            cwe_list = row["cwe_ids"] or []
            key = cwe_list[0] if cwe_list else "no-cwe"
            result[key] = result.get(key, 0) + 1
        return result
    raise ValueError(f"Unknown strategy: {strategy!r}")


def plot_summary_jsd_by_strategy(
    fold_types: list[str],
    baseline_types: list[str],
    processed_dir: Path,
    tag: str | None,
    seed: int,
    total_folds: int,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """JSD(train, test) per CWE strategy × resampling approach.

    x-axis = CWE assignment strategy; grouped bars = resampling approach.
    Averages JSD over all available folds.
    """
    all_types = fold_types + baseline_types
    split_metas = discover_split_parquets(
        processed_dir, tag=tag or "default",
        seed=seed, total_folds=total_folds, fold_types=all_types,
    )

    jsd_data: dict[str, dict[str, list[float]]] = {
        strat: {ft: [] for ft in all_types} for strat in _STRATEGIES
    }
    for meta in split_metas:
        ft = meta["fold_type"]
        if ft not in all_types:
            continue
        split_df = pl.read_parquet(meta["path"], columns=["phase", "cwe_ids"])
        for strat in _STRATEGIES:
            train_dist = _get_cwe_dist_for_strategy(split_df, "train", strat)
            test_dist  = _get_cwe_dist_for_strategy(split_df, "test",  strat)
            jsd = compute_divergence(train_dist, test_dist, "js")
            if not np.isnan(jsd):
                jsd_data[strat][ft].append(jsd)

    n_strat = len(_STRATEGIES)
    n_types = len(all_types)
    x = np.arange(n_strat, dtype=float)
    width = 0.7 / n_types
    offsets = np.linspace(-(n_types - 1) / 2, (n_types - 1) / 2, n_types) * width
    colors = sns.color_palette("tab10", n_colors=n_types)

    fig, ax = plt.subplots(figsize=figsize or (max(10, n_strat * 3 + 2), 3))
    for i, ft in enumerate(all_types):
        is_baseline = ft in baseline_types
        means = [
            float(np.nanmean(jsd_data[strat][ft])) if jsd_data[strat][ft] else float("nan")
            for strat in _STRATEGIES
        ]
        stds = [
            float(np.nanstd(jsd_data[strat][ft])) if jsd_data[strat][ft] else 0.0
            for strat in _STRATEGIES
        ]
        ax.bar(
            x + offsets[i], means, width,
            label=_LABELS.get(ft, ft), color=colors[i],
            alpha=0.85, hatch="//" if is_baseline else None, zorder=3,
        )
        ax.errorbar(
            x + offsets[i], means, yerr=stds,
            fmt="none", color="black", capsize=3, linewidth=1.0, zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_STRATEGY_LABELS[s] for s in _STRATEGIES], fontsize=10)
    ax.set_ylabel("JSD (mean ± std over folds)", fontsize=10)
    ax.set_title(
        f"CWE Distribution JSD - Train vs Test by CWE Strategy | Temporal LOO Block CV ({MODEL})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 0.5), loc="center left",
              framealpha=0.9, borderaxespad=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def plot_summary_jsd_paired(
    fold_types: list[str],
    baseline_types: list[str],
    processed_dir: Path,
    tag: str | None,
    seed: int,
    total_folds: int,
    title_suffix: str = "",
) -> plt.Figure:
    """Side-by-side JSD(train, test) and JSD(train, holdout) per CWE strategy × resampling approach."""
    all_types = fold_types + baseline_types
    split_metas = discover_split_parquets(
        processed_dir, tag=tag or "default",
        seed=seed, total_folds=total_folds, fold_types=all_types,
    )

    jsd_test: dict[str, dict[str, list[float]]] = {
        strat: {ft: [] for ft in all_types} for strat in _STRATEGIES
    }
    jsd_holdout: dict[str, dict[str, list[float]]] = {
        strat: {ft: [] for ft in all_types} for strat in _STRATEGIES
    }
    for meta in split_metas:
        ft = meta["fold_type"]
        if ft not in all_types:
            continue
        split_df = pl.read_parquet(meta["path"], columns=["phase", "cwe_ids"])
        for strat in _STRATEGIES:
            train_dist = _get_cwe_dist_for_strategy(split_df, "train", strat)
            test_dist = _get_cwe_dist_for_strategy(split_df, "test", strat)
            holdout_dist = _get_cwe_dist_for_strategy(split_df, "holdout", strat)
            jsd_t = compute_divergence(train_dist, test_dist, "js")
            jsd_h = compute_divergence(train_dist, holdout_dist, "js")
            if not np.isnan(jsd_t):
                jsd_test[strat][ft].append(jsd_t)
            if not np.isnan(jsd_h):
                jsd_holdout[strat][ft].append(jsd_h)

    n_strat = len(_STRATEGIES)
    n_types = len(all_types)
    x = np.arange(n_strat, dtype=float)
    width = 0.7 / n_types
    offsets = np.linspace(-(n_types - 1) / 2, (n_types - 1) / 2, n_types) * width
    colors = sns.color_palette("tab10", n_colors=n_types)

    fig_w = max(16, n_strat * 5 + 4)
    fig, (ax_test, ax_hol) = plt.subplots(1, 2, figsize=(fig_w, 4), sharey=True)

    for ax, jsd_data, phase_label in [
        (ax_test, jsd_test, "Train vs Test"),
        (ax_hol, jsd_holdout, "Train vs Holdout"),
    ]:
        for i, ft in enumerate(all_types):
            is_baseline = ft in baseline_types
            means = [
                float(np.nanmean(jsd_data[strat][ft])) if jsd_data[strat][ft] else float("nan")
                for strat in _STRATEGIES
            ]
            stds = [
                float(np.nanstd(jsd_data[strat][ft])) if jsd_data[strat][ft] else 0.0
                for strat in _STRATEGIES
            ]
            ax.bar(
                x + offsets[i], means, width,
                label=_LABELS.get(ft, ft), color=colors[i],
                alpha=0.85, hatch="//" if is_baseline else None, zorder=3,
            )
            ax.errorbar(
                x + offsets[i], means, yerr=stds,
                fmt="none", color="black", capsize=3, linewidth=1.0, zorder=4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_STRATEGY_LABELS[s] for s in _STRATEGIES], fontsize=10)
        ax.set_title(phase_label, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    ax_test.set_ylabel("JSD (mean ± std over folds)", fontsize=10)
    ax_hol.legend(
        fontsize=8, bbox_to_anchor=(1.02, 0.5), loc="center left",
        framealpha=0.9, borderaxespad=0,
    )
    suffix = f" | {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"CWE Distribution JSD by CWE Strategy{suffix} ({MODEL})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return fig


def build_jsd_table(
    fold_types: list[str],
    baseline_types: list[str],
    processed_dir: Path,
    tag: str | None,
    seed: int,
    total_folds: int,
) -> str:
    """LaTeX booktabs table: JSD(train,test) and JSD(train,holdout) per CWE strategy × fold type.

    Rows: strategy × phase (6 rows: 3 strategies × {Test, Holdout}).
    Columns: fold types (resampling then baselines), with rotated headers.
    Cells: mean ± std over folds.
    Requires: booktabs, multirow.
    """
    all_types = fold_types + baseline_types
    split_metas = discover_split_parquets(
        processed_dir, tag=tag or "default",
        seed=seed, total_folds=total_folds, fold_types=all_types,
    )

    # jsd_data[phase][strat][ft] = list of JSD values
    jsd_data: dict[str, dict[str, dict[str, list[float]]]] = {
        phase: {strat: {ft: [] for ft in all_types} for strat in _STRATEGIES}
        for phase in ("test", "holdout")
    }
    for meta in split_metas:
        ft = meta["fold_type"]
        if ft not in all_types:
            continue
        split_df = pl.read_parquet(meta["path"], columns=["phase", "cwe_ids"])
        for strat in _STRATEGIES:
            train_dist = _get_cwe_dist_for_strategy(split_df, "train", strat)
            for phase in ("test", "holdout"):
                other_dist = _get_cwe_dist_for_strategy(split_df, phase, strat)
                jsd = compute_divergence(train_dist, other_dist, "js")
                if not np.isnan(jsd):
                    jsd_data[phase][strat][ft].append(jsd)

    # Strategy abbreviations for column headers
    _STRAT_ABBR = {"val_most_common": "MC", "explode": "E", "primary": "P"}
    _VMAX = 0.40  # JSD above this maps to full red

    def _cell(vals: list[float]) -> str:
        if not vals:
            return "---"
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        t = min(1.0, mean / _VMAX)
        g = b = int(round(255 - t * (255 - 50)))
        colour = f"{255:02X}{g:02X}{b:02X}"
        return f"\\cellcolor[HTML]{{{colour}}} {mean:.3f}$\\pm${std:.3f}"

    # Layout: fold types as rows, strategy×phase as columns
    # col spec: Fold Type | (MC-T MC-H) (E-T E-H) (P-T P-H)
    col_spec = "l " + " ".join(["cc"] * len(_STRATEGIES))

    # Header row 1: strategy names spanning 2 columns each (col 2 onward)
    strat_spans = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{_STRATEGY_LABELS[s]}}}" for s in _STRATEGIES
    )
    # Cmidrules under each strategy pair
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}" for i in range(len(_STRATEGIES))
    )
    # Header row 2: plain phase labels under each strategy group
    phase_heads = " & ".join("T & H" for _ in _STRATEGIES)

    def _data_rows(types: list[str]) -> list[str]:
        rows = []
        for ft in types:
            label = _SHORT_LABELS.get(ft, ft)
            cells = " & ".join(
                f"{_cell(jsd_data['test'][s][ft])} & {_cell(jsd_data['holdout'][s][ft])}"
                for s in _STRATEGIES
            )
            rows.append(f"    {label} & {cells} \\\\")
        return rows

    lines: list[str] = [
        "% Auto-generated by scripts/plot_cwe_resampling.py - do not edit manually.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{colortbl}, \\usepackage{graphicx}",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\scriptsize",
        "  \\resizebox{\\linewidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    & {strat_spans} \\\\",
        f"    {cmidrules}",
        f"    Fold Type & {phase_heads} \\\\",
        "    \\midrule",
        *_data_rows(fold_types),
        "    \\midrule",
        *_data_rows(baseline_types),
        "    \\bottomrule",
        "  \\end{tabular}}%",
        "  \\caption{CWE distribution JSD (mean~$\\pm$~std over folds) between train and"
        " test (T) or holdout (H), per fold type and CWE assignment strategy."
        " MC = Most Common, E = Explode, P = Primary."
        " Baselines (bottom section) are non-resampling LOO block CV."
        " Colour: white (JSD=0) $\\to$ red (JSD$\\geq$0.4).}",
        "  \\label{tab:cwe_jsd_by_strategy}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = ArgumentParser(description="CWE resampling heatmaps and JSD bar chart.")
    parser.add_argument("--results", type=Path, default=INDEX_RESULTS_FILE)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-folds", type=int, default=10)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        print("Run `uv run scripts/collect_results.py` first.")
        sys.exit(1)

    print(f"Loading results from {args.results} …")
    all_fold_types = LOO_BLOCK_FOLD_TYPES + CWE_RESAMPLING_BASELINES
    lf = load_results(
        args.results,
        fold_types=all_fold_types,
        tag=args.tag,
        dataset=args.dataset,
    )
    lf = lf.filter(pl.col("model") == MODEL)

    print("Computing per-fold metrics …")
    test_df, holdout_df, _ = build_metric_frames(lf)

    present = set(test_df["fold_type"].unique().to_list()) | set(
        holdout_df["fold_type"].unique().to_list()
    )
    fold_types = [ft for ft in LOO_BLOCK_FOLD_TYPES if ft in present]
    baseline_types = [ft for ft in CWE_RESAMPLING_BASELINES if ft in present]
    if not fold_types and not baseline_types:
        print("No CWE resampling fold types found for llama3.2-1B. Exiting.")
        sys.exit(0)

    print(f"Fold types : {fold_types}")
    print(f"Baselines  : {baseline_types}")

    all_split_fold_types = LOO_BLOCK_FOLD_TYPES + CWE_RESAMPLING_BASELINES
    split_present = {
        m["fold_type"]
        for m in discover_split_parquets(
            PROCESSED_DATA_DIR,
            tag=args.tag or "default",
            seed=args.seed,
            total_folds=args.total_folds,
            fold_types=all_split_fold_types,
        )
    }
    print(f"Split parquets found for: {sorted(split_present)}")

    jsd_fold_types     = [ft for ft in LOO_BLOCK_FOLD_TYPES    if ft in split_present]
    jsd_baseline_types = [ft for ft in CWE_RESAMPLING_BASELINES if ft in split_present]

    sns.set_theme(style="whitegrid", context="paper")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _save(plot_heatmap(test_df, "test", fold_types, baseline_types), PLOTS_DIR / "heatmap_test.png")
    _save(plot_heatmap(holdout_df, "holdout", fold_types, baseline_types), PLOTS_DIR / "heatmap_holdout.png")
    _save(
        plot_jsd_bar(jsd_fold_types, jsd_baseline_types, PROCESSED_DATA_DIR, args.tag, args.seed, args.total_folds),
        PLOTS_DIR / "bar_jsd_train_test.png",
    )
    _save(
        plot_summary_jsd_by_strategy(
            jsd_fold_types, jsd_baseline_types, PROCESSED_DATA_DIR,
            args.tag, args.seed, args.total_folds,
        ),
        PLOTS_DIR / "summary_jsd_by_strategy.png",
    )
    _save(
        plot_summary_jsd_paired(
            jsd_fold_types, jsd_baseline_types, PROCESSED_DATA_DIR,
            args.tag, args.seed, args.total_folds,
            title_suffix="LOO Block CV",
        ),
        PLOTS_DIR / "summary_jsd_by_strategy_paired.png",
    )
    temporal_fold_types     = [ft for ft in TEMPORAL_LOO_BLOCK_FOLD_TYPES if ft in split_present]
    temporal_baseline_types = [ft for ft in TEMPORAL_LOO_BLOCK_BASELINES  if ft in split_present]
    if temporal_fold_types or temporal_baseline_types:
        _save(
            plot_summary_jsd_by_strategy(
                temporal_fold_types, temporal_baseline_types, PROCESSED_DATA_DIR,
                args.tag, args.seed, args.total_folds,
                figsize=(9.9, 3.11),
            ),
            PLOTS_DIR / "summary_jsd_by_strategy_temporal.png",
        )
        _save(
            plot_summary_jsd_paired(
                temporal_fold_types, temporal_baseline_types, PROCESSED_DATA_DIR,
                args.tag, args.seed, args.total_folds,
                title_suffix="Temporal LOO Block CV",
            ),
            PLOTS_DIR / "summary_jsd_by_strategy_paired_temporal.png",
        )

    tex = build_jsd_table(
        jsd_fold_types, jsd_baseline_types, PROCESSED_DATA_DIR,
        args.tag, args.seed, args.total_folds,
    )
    tex_path = PLOTS_DIR / "table_jsd_by_strategy.tex"
    tex_path.write_text(tex)
    print(f"  Saved: {tex_path}")

    print(f"\nAll outputs written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
