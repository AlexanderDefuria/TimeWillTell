#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
CWE distribution stacked bar chart: Random CFV Fold N | Temporal LOO all folds | Holdout.

Each bar shows the proportion of CWE tags in that fold's test (or holdout) split.
Samples are deduplicated by sample_index to avoid double-counting across models.

Usage:
    uv run scripts/plot_cwe_fold_distribution.py
    uv run scripts/plot_cwe_fold_distribution.py --top-k 20 --total-folds 10
    uv run scripts/plot_cwe_fold_distribution.py --random-fold 2 --tag my-tag
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

try:
    from scripts import EDA_DIR, INDEX_RESULTS_FILE
    from scripts.cwe_utils import explode_with_cwe_id
    from scripts.plot_results import format_fold_type_for_plot
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import EDA_DIR, INDEX_RESULTS_FILE
    from scripts.cwe_utils import explode_with_cwe_id
    from scripts.plot_results import format_fold_type_for_plot

OUTPUT_DIR = EDA_DIR / "cwe_distribution"

DEFAULT_RANDOM_FOLD_TYPE = "k_fold_cross_validation"
DEFAULT_TEMPORAL_FOLD_TYPE = "temporal_loo_block_cross_validation"


# ─── Data loading ─────────────────────────────────────────────────────────────


def _load_data(
    *,
    random_fold_type: str,
    temporal_fold_type: str,
    total_folds: int,
    seed: int,
    cw: int,
    tag: str | None,
) -> pl.DataFrame:
    needed = ["fold_type", "fold_number", "phase", "cwe_ids", "sample_index",
              "seed", "cw", "total_folds", "tag"]
    schema = pl.read_parquet_schema(INDEX_RESULTS_FILE)
    select_cols = [c for c in needed if c in schema]

    lf = pl.scan_parquet(INDEX_RESULTS_FILE).select(select_cols)
    lf = lf.filter(
        pl.col("fold_type").is_in([random_fold_type, temporal_fold_type]),
        pl.col("total_folds") == total_folds,
        pl.col("seed") == seed,
        pl.col("cw") == cw,
    )
    if tag:
        lf = lf.filter(pl.col("tag") == tag)

    try:
        df = lf.collect(engine="streaming")
    except Exception:
        df = lf.collect()

    if df.is_empty():
        raise RuntimeError(
            "No rows matched the filters. "
            f"fold_types={[random_fold_type, temporal_fold_type]}, "
            f"total_folds={total_folds}, seed={seed}, cw={cw}, tag={tag!r}"
        )
    return df


# ─── CWE proportion computation ───────────────────────────────────────────────


def _cwe_counts(sub: pl.DataFrame) -> dict[int, int]:
    """Count CWE occurrences in a deduplicated sample set."""
    if sub.is_empty():
        return {}
    deduped = sub.unique(subset=["sample_index"])
    exploded = explode_with_cwe_id(deduped)
    if exploded.is_empty():
        return {}
    counts = exploded.group_by("cwe_id").agg(pl.len().alias("n"))
    return dict(zip(counts["cwe_id"].to_list(), counts["n"].to_list()))


def _top_k_global(all_counts: list[dict[int, int]], top_k: int) -> list[int]:
    """Return the top-K CWE IDs by total count across all bars."""
    totals: dict[int, int] = {}
    for counts in all_counts:
        for cwe_id, n in counts.items():
            totals[cwe_id] = totals.get(cwe_id, 0) + n
    sorted_ids = sorted(totals, key=lambda k: totals[k], reverse=True)
    return sorted_ids[:top_k]


def _proportions(counts: dict[int, int], top_ids: list[int]) -> dict[str, float]:
    """Convert counts → proportions, bucketing non-top CWEs into 'Other'."""
    total = sum(counts.values())
    if total == 0:
        return {f"CWE-{i}": 0.0 for i in top_ids} | {"Other": 0.0}
    props: dict[str, float] = {}
    other = 0.0
    for cwe_id, n in counts.items():
        if cwe_id in top_ids:
            props[f"CWE-{cwe_id}"] = n / total
        else:
            other += n / total
    for cwe_id in top_ids:
        props.setdefault(f"CWE-{cwe_id}", 0.0)
    props["Other"] = other
    return props


# ─── Bar assembly ─────────────────────────────────────────────────────────────


def _build_bars(
    df: pl.DataFrame,
    *,
    random_fold_type: str,
    temporal_fold_type: str,
    random_fold: int,
    total_folds: int,
) -> tuple[list[str], list[dict[int, int]], list[int]]:
    """
    Returns (labels, counts_per_bar, group_starts).

    group_starts[i] is the bar index where group i starts.
    """
    labels: list[str] = []
    counts_list: list[dict[int, int]] = []
    group_starts: list[int] = []

    # Group 0: single random CFV fold
    group_starts.append(0)
    rand_sub = df.filter(
        pl.col("fold_type") == random_fold_type,
        pl.col("fold_number") == random_fold,
        pl.col("phase") == "test",
    )
    labels.append(f"Random\nFold {random_fold}")
    counts_list.append(_cwe_counts(rand_sub))

    # Group 1: all temporal LOO folds
    group_starts.append(len(labels))
    temporal_df = df.filter(
        pl.col("fold_type") == temporal_fold_type,
        pl.col("phase") == "test",
    )
    available_folds = sorted(temporal_df["fold_number"].unique().to_list())
    if not available_folds:
        print(f"WARNING: no temporal folds found for {temporal_fold_type!r}. "
              "Check --temporal-fold-type and --total-folds.")
    for fn in available_folds:
        fold_sub = temporal_df.filter(pl.col("fold_number") == fn)
        labels.append(f"Temporal\nFold {fn}")
        counts_list.append(_cwe_counts(fold_sub))

    # Group 2: holdout (same set regardless of fold - take temporal fold 0)
    group_starts.append(len(labels))
    holdout_sub = df.filter(
        pl.col("fold_type") == temporal_fold_type,
        pl.col("fold_number") == 0,
        pl.col("phase") == "holdout",
    )
    if holdout_sub.is_empty():
        holdout_sub = df.filter(pl.col("phase") == "holdout")
    labels.append("Holdout")
    counts_list.append(_cwe_counts(holdout_sub))

    return labels, counts_list, group_starts


# ─── Plotting ─────────────────────────────────────────────────────────────────


def plot_cwe_fold_distribution(
    labels: list[str],
    counts_list: list[dict[int, int]],
    group_starts: list[int],
    top_ids: list[int],
    *,
    random_fold_type: str,
    temporal_fold_type: str,
    output_path: Path,
) -> None:
    n_bars = len(labels)
    cwe_labels = [f"CWE-{i}" for i in top_ids]
    all_keys = cwe_labels + ["Other"]

    prop_matrix = [_proportions(c, top_ids) for c in counts_list]

    cmap = plt.get_cmap("tab20")
    colors = {f"CWE-{i}": cmap(idx / max(1, len(top_ids) - 1)) for idx, i in enumerate(top_ids)}
    colors["Other"] = "#d3d3d3"

    fig_w = max(10, n_bars * 1.0 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))

    x = np.arange(n_bars)
    bottoms = np.zeros(n_bars)

    for key in all_keys:
        vals = np.array([p.get(key, 0.0) for p in prop_matrix])
        ax.bar(x, vals, bottom=bottoms, color=colors[key], label=key,
               width=0.75, edgecolor="white", linewidth=0.4)
        bottoms += vals

    # Group separators
    group_label_texts = [
        format_fold_type_for_plot(random_fold_type),
        format_fold_type_for_plot(temporal_fold_type),
        "Holdout",
    ]
    for sep_bar in group_starts[1:]:
        ax.axvline(sep_bar - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    for group_idx, (gs, text) in enumerate(zip(group_starts, group_label_texts)):
        end = group_starts[group_idx + 1] if group_idx + 1 < len(group_starts) else n_bars
        mid = (gs + end - 1) / 2
        ax.text(mid, 1.01, text, ha="center", va="bottom", fontsize=9,
                fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center")
    ax.set_ylabel("CWE Proportion", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, n_bars - 0.5)
    ax.set_title("CWE Distribution by Fold and Holdout Set", fontsize=12, pad=32)
    sns.despine(ax=ax)

    handles = [mpatches.Patch(color=colors[k], label=k) for k in all_keys]
    ax.legend(
        handles=handles,
        fontsize=7,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        ncol=2,
        title="CWE",
        title_fontsize=8,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--random-fold-type", default=DEFAULT_RANDOM_FOLD_TYPE)
    p.add_argument("--temporal-fold-type", default=DEFAULT_TEMPORAL_FOLD_TYPE)
    p.add_argument("--random-fold", type=int, default=0,
                   help="Fold number to show for random CFV (default: 0)")
    p.add_argument("--total-folds", type=int, default=10)
    p.add_argument("--top-k", type=int, default=15,
                   help="Number of named CWEs; rest → 'Other' (default: 15)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cw", type=int, default=1024)
    p.add_argument("--tag", default=None)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Random fold type : {args.random_fold_type}  (fold {args.random_fold})")
    print(f"Temporal fold type: {args.temporal_fold_type}  (all folds)")
    print(f"total_folds={args.total_folds}, seed={args.seed}, cw={args.cw}, tag={args.tag!r}")
    print()

    print("[1/4] Loading data …")
    df = _load_data(
        random_fold_type=args.random_fold_type,
        temporal_fold_type=args.temporal_fold_type,
        total_folds=args.total_folds,
        seed=args.seed,
        cw=args.cw,
        tag=args.tag,
    )
    print(f"    {df.height:,} rows loaded")

    print("[2/4] Building bars …")
    labels, counts_list, group_starts = _build_bars(
        df,
        random_fold_type=args.random_fold_type,
        temporal_fold_type=args.temporal_fold_type,
        random_fold=args.random_fold,
        total_folds=args.total_folds,
    )
    for label, counts in zip(labels, counts_list):
        print(f"    {label.replace(chr(10), ' '):25s}  total CWE tags: {sum(counts.values()):,}")

    print(f"[3/4] Computing top-{args.top_k} CWEs …")
    top_ids = _top_k_global(counts_list, args.top_k)
    print(f"    Top CWEs: {[f'CWE-{i}' for i in top_ids]}")

    print("[4/4] Plotting …")
    out_path = args.output_dir / "fold_distribution_stacked.png"
    plot_cwe_fold_distribution(
        labels,
        counts_list,
        group_starts,
        top_ids,
        random_fold_type=args.random_fold_type,
        temporal_fold_type=args.temporal_fold_type,
        output_path=out_path,
    )


if __name__ == "__main__":
    main()
