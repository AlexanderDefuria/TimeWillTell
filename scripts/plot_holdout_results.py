from __future__ import annotations

from pathlib import Path
import re
import shutil
import os
import sys
from typing import List, Optional
import polars as pl
try:
    from scripts import INDEX_RESULTS_FILE, PRIMARY_FOLD_TYPES
    from scripts.cwe_mapping import get_research_concepts_category
    from scripts.plot_results import apply_plot_model_exclusion, format_fold_type_for_plot, sort_models
except ModuleNotFoundError:
    # Support running as a file path: `uv run scripts/plot_holdout_results.py`.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import INDEX_RESULTS_FILE, PRIMARY_FOLD_TYPES
    from scripts.cwe_mapping import get_research_concepts_category
    from scripts.plot_results import apply_plot_model_exclusion, format_fold_type_for_plot, sort_models
import seaborn as sns
from matplotlib import pyplot as plt

pl.Config.set_tbl_rows(500)
pl.Config.set_tbl_cols(10)


def load_and_filter_results(
    df: pl.DataFrame,
    fold_count: int = 10,
    seed: int = 42,
    cw: int | List[int] = 1024,
    fold_types: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Load results from CSV and apply standard filters.

    Parameters:
        csv_path: Path to TEST_RESULTS_CSV
        fold_count: Number of folds (default: 10)
        seed: Random seed (default: 42)
        cw: Context window size(s) (default: 1024). Pass a list to allow multiple.
        fold_types: List of fold types to include (default: all standard types)
        models: List of models to include (default: all standard models)
        tags: List of tags to include (default: ['default'])

    Returns:
        Filtered Polars DataFrame
    """
    # Default lists
    if fold_types is None:
        fold_types = [
            "k_fold_cross_validation",
            "random_growing_window_cross_validation",
            "random_loo_block_cross_validation",
            "temporal_growing_window_cross_validation",
            "temporal_loo_block_cross_validation",
        ]

    if models is None:
        models = [
            "llama3.2-1B",
            "deepseekcoder",
            "llama3.1",
            "phi4",
            "mistral3",
        ]

    if tags is None:
        tags = ["default"]

    cw_list = [cw] if isinstance(cw, int) else cw

    df = df.filter(
        pl.col("total_folds") == fold_count,
        pl.col("seed") == seed,
        pl.col("cw").is_in(cw_list),
        pl.col("fold_type").is_in(fold_types),
        pl.col("model").is_in(models),
        pl.col("tag").is_in(tags),
    )
    if "run_mode" in df.columns:
        df = df.filter(pl.col("run_mode").is_in(["fine_tuning", "opro"]))

    return df


def normalize_fold_type_names(df: pl.DataFrame, include_tag: bool = False) -> pl.DataFrame:
    """
    Standardize fold type column names.

    Parameters:
        df: Input DataFrame
        include_tag: Whether to append tag info (Commit Splitting vs Default)

    Returns:
        DataFrame with normalized fold_type column
    """
    df = df.with_columns(
        pl.col("fold_type")
        .str.replace("k_fold_cross_validation", "K-Fold CV")
        .str.replace("random_growing_window_cross_validation", "Random Growing Window CV")
        .str.replace("random_loo_block_cross_validation", "Random LOO Block CV")
        .str.replace("temporal_growing_window_cross_validation", "Temporal Growing Window CV")
        .str.replace("temporal_loo_block_cross_validation", "Temporal LOO Block CV")
    )

    if include_tag:
        df = df.with_columns(
            pl.when(pl.col("tag") == "commit-splitting")
            .then(pl.col("fold_type") + " (Commit Splitting)")
            .otherwise(pl.col("fold_type") + " (Default)")
            .alias("fold_type")
        )

    return df


def summarize_df_statistics(index_df: pl.DataFrame) -> pl.DataFrame:
    """
    Print summary statistics of the DataFrame.
    Parameters:
        df: Input DataFrame
    """
    summarized_list = []
    for group in index_df.group_by(
        [
            "fold_type",
            "fold_number",
            "model",
            "dataset",
            "datamodule",
            "tag",
            "batch_size",
            "seed",
            "cw",
            "phase",
            "total_folds",
            "accumulation_steps",
        ]
    ):
        df = group[1]
        df = df.select(["prediction", "label"])
        tp = ((df["prediction"] == 1) & (df["label"] == 1)).sum()
        tn = ((df["prediction"] == 0) & (df["label"] == 0)).sum()
        fp = ((df["prediction"] == 1) & (df["label"] == 0)).sum()
        fn = ((df["prediction"] == 0) & (df["label"] == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        summarized_list.append(
            {
                "fold_type": group[0][0],
                "fold_number": f"fold_{group[0][1]}_of_{group[0][10]}",
                "model": group[0][2],
                "dataset": group[0][3],
                "datamodule": group[0][4],
                "tag": group[0][5],
                "batch_size": group[0][6],
                "seed": group[0][7],
                "cw": group[0][8],
                "phase": group[0][9],
                "total_folds": group[0][10],
                "accumulation_steps": group[0][11],
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    summary_df = pl.DataFrame(summarized_list)
    return summary_df


def pivot_and_aggregate(df: pl.DataFrame, fold_count: int, metric: str) -> pl.DataFrame:
    """
    Pivot folds into columns and compute aggregate statistics.

    Parameters:
        df: Filtered and normalized DataFrame
        fold_count: Number of folds

    Returns:
        Aggregated DataFrame with statistics
    """
    # Clean and deduplicate
    df = df.unique(subset=["seed", "model", "fold_number", "fold_type", metric])
    df = df.select(["model", metric, "fold_type", "fold_number", "tag"])
    df = df.sort(by=["model", "fold_type", "fold_number"])

    # Pivot folds into columns
    df_pivoted = df.pivot(
        values=metric,
        index=["model", "fold_type"],
        on="fold_number",
        aggregate_function="mean",
    )

    # Create folds list and compute statistics
    fold_cols = [f"fold_{i}_of_{fold_count}" for i in range(fold_count)]

    df_aggregated = df_pivoted.with_columns(pl.concat_list(fold_cols).alias("folds_list"))

    df_aggregated = df_aggregated.with_columns(
        pl.col("folds_list").list.mean().alias(f"Mean {metric}"),
        pl.col("folds_list").list.std().alias(f"STD {metric}"),
        pl.col("folds_list").list.min().alias(f"Min {metric}"),
        pl.col("folds_list").list.max().alias(f"Max {metric}"),
        pl.col("folds_list").list.drop_nulls().list.len().cast(pl.Float64).mul(0.01).alias("Num Folds"),
    )

    return df_aggregated


def plot_high_level_summary_heatmaps(
    df: pl.DataFrame,
    fold_count: int,
    output_dir: Path,
    metrics: Optional[List[str]] = ["TP", "TN", "FP", "FN", "Precision", "Recall", "F1"],
) -> None:
    """
    Generate heatmaps for each metric.

    Parameters:
        df: Summarized DataFrame
        fold_count: Fold count for title
        output_dir: Path to save PNG files
        metrics: List of metrics to plot
    """
    aggreation = ["Mean", "STD", "Min", "Max", "Num Folds"]

    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = df.group_by(
        [
            "dataset",
            "datamodule",
            "tag",
            "seed",
            "cw",
            "phase",
            "total_folds",
        ]
    )
    for group in groups:
        dataset, datamodule, tag, seed, cw, phase, total_folds = group[0]
        group_df = group[1]

        for metric in ["Precision", "Recall", "F1"]:
            plt.figure(figsize=(10, 6))

            pivot_data = (
                group_df.filter(pl.col("phase") == phase)
                .select(["model", "fold_type", metric])
                .to_pandas()
                .pivot_table(
                    columns="model",
                    index="fold_type",
                    values=metric,
                    aggfunc="mean",
                )
            )
            pivot_data = pivot_data.rename(index=lambda x: format_fold_type_for_plot(str(x)))

            sns.heatmap(
                pivot_data * 100,
                cbar_kws={"label": f"{metric} (%)"},
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                vmin=10 if metric == "F1" else 0,
                vmax=50 if metric == "F1" else 100,
            )

            os.makedirs(output_dir / f"{phase}", exist_ok=True)
            plt.title(f"Cross-Validation {phase} {metric} by Model and Fold Type ({fold_count} Folds)")
            plt.ylabel("Fold Type")
            plt.xlabel("Model")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / f"{phase}" / f"{tag}_{metric.replace(' ', '_')}_heatmap_overall.png", dpi=500)
            plt.close()

            # STD heatmap
            plt.figure(figsize=(10, 6))

            std_pivot_data = (
                group_df.filter(pl.col("phase") == phase)
                .select(["model", "fold_type", metric])
                .to_pandas()
                .pivot_table(
                    columns="model",
                    index="fold_type",
                    values=metric,
                    aggfunc="std",
                )
            )
            std_pivot_data = std_pivot_data.rename(index=lambda x: format_fold_type_for_plot(str(x)))

            sns.heatmap(
                std_pivot_data * 100,
                cbar_kws={"label": f"{metric} STD (%)"},
                annot=True,
                fmt=".1f",
                cmap="Blues",
                vmin=0,
                vmax=20 if metric == "F1" else 30,
            )

            os.makedirs(output_dir / f"{phase}" / "std", exist_ok=True)
            plt.title(f"Cross-Validation {phase} {metric} STD by Model and Fold Type ({fold_count} Folds)")
            plt.ylabel("Fold Type")
            plt.xlabel("Model")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / f"{phase}" / "std" / f"{tag}_{metric.replace(' ', '_')}_heatmap_overall.png", dpi=500)
            plt.close()

            # Plot folds individually
            for fold_type in group_df.select("fold_type").unique().to_series().to_list():
                fold_type_df = group_df.filter(pl.col("fold_type") == fold_type)
                fold_type_df = fold_type_df.filter(pl.col("phase") == phase)
                fold_type_df = fold_type_df.sort(by=["model", "fold_number"])
                fold_type_df = fold_type_df.select(["model", "fold_number", metric])
                fold_type_df = (
                    fold_type_df.pivot(
                        values=metric,
                        on="fold_number",
                        index="model",
                    )
                    .to_pandas()
                    .set_index("model")
                )
                fold_type_df = fold_type_df[[f"fold_{i}_of_{fold_count}" for i in range(fold_count) if f"fold_{i}_of_{fold_count}" in fold_type_df.columns]]
                fold_type_df = fold_type_df.reindex(sort_models(fold_type_df.index.tolist()))

                plt.figure(figsize=(10, 6))
                sns.heatmap(
                    fold_type_df * 100,
                    cbar_kws={"label": f"{metric} (%)"},
                    annot=True,
                    fmt=".1f",
                    cmap="YlGnBu",
                    vmin=10 if metric == "F1" else 0,
                    vmax=50 if metric == "F1" else 100,
                )
                plt.title(f"{format_fold_type_for_plot(fold_type)} {phase} {metric} by Model and Fold ({fold_count} Folds)")
                plt.ylabel("Fold")
                plt.xlabel("Model")
                plt.tight_layout()
                plt.savefig(
                    output_dir / f"{phase}" / f"{tag}_{fold_type}_{metric.replace(' ', '_')}_folds_heatmap.png",
                    dpi=500,
                )
                plt.close()


def plot_random_temporal_delta(df: pl.DataFrame, fold_count: int, output_dir: Path) -> None:
    """
    Plot fold-level F1 delta heatmaps between temporal methods and k-fold baseline.
    """
    output_dir = Path(output_dir)
    baseline_fold_type = "k_fold_cross_validation"
    temporal_fold_types = [
        "temporal_growing_window_cross_validation",
        "temporal_loo_block_cross_validation",
    ]
    fold_cols = [f"fold_{i}_of_{fold_count}" for i in range(fold_count)]

    groups = df.group_by(
        [
            "dataset",
            "datamodule",
            "tag",
            "seed",
            "cw",
            "phase",
            "total_folds",
        ]
    )

    for group in groups:
        _, _, tag, _, _, phase, _ = group[0]
        group_df = group[1].filter(pl.col("phase") == phase)
        os.makedirs(output_dir / f"{phase}", exist_ok=True)

        baseline_df = (
            group_df.filter(pl.col("fold_type") == baseline_fold_type)
            .sort(by=["model", "fold_number"])
            .select(["model", "fold_number", "F1"])
        )
        if baseline_df.height == 0:
            continue

        baseline_pivot = (
            baseline_df.pivot(
                values="F1",
                on="fold_number",
                index="model",
            )
            .to_pandas()
            .set_index("model")
        )
        baseline_pivot = baseline_pivot[[col for col in fold_cols if col in baseline_pivot.columns]]
        if baseline_pivot.empty:
            continue

        for temporal_fold_type in temporal_fold_types:
            temporal_df = (
                group_df.filter(pl.col("fold_type") == temporal_fold_type)
                .sort(by=["model", "fold_number"])
                .select(["model", "fold_number", "F1"])
            )
            if temporal_df.height == 0:
                continue

            temporal_pivot = (
                temporal_df.pivot(
                    values="F1",
                    on="fold_number",
                    index="model",
                )
                .to_pandas()
                .set_index("model")
            )
            temporal_pivot = temporal_pivot[[col for col in fold_cols if col in temporal_pivot.columns]]
            if temporal_pivot.empty:
                continue

            common_models = sort_models(list(set(temporal_pivot.index).intersection(set(baseline_pivot.index))))
            common_cols = [col for col in fold_cols if col in temporal_pivot.columns and col in baseline_pivot.columns]
            if len(common_models) == 0 or len(common_cols) == 0:
                continue

            delta_df = temporal_pivot.loc[common_models, common_cols] - baseline_pivot.loc[common_models, common_cols]
            if delta_df.empty:
                continue

            delta_percent_points = delta_df * 100
            min_delta = float(delta_percent_points.min().min())
            max_delta = float(delta_percent_points.max().max())
            limit = max(abs(min_delta), abs(max_delta))
            if limit == 0:
                limit = 1.0

            plt.figure(figsize=(10, 6))
            sns.heatmap(
                delta_percent_points,
                cbar_kws={"label": "F1 Delta (pp)"},
                annot=True,
                fmt=".1f",
                cmap="RdBu_r",
                center=0,
                vmin=-limit,
                vmax=limit,
            )
            plt.title(
                f"{format_fold_type_for_plot(temporal_fold_type)} - {format_fold_type_for_plot(baseline_fold_type)} "
                f"{phase} F1 Delta by Model and Fold ({fold_count} Folds)"
            )
            plt.ylabel("Model")
            plt.xlabel("Fold Number")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{phase}" / f"{tag}_{temporal_fold_type}_minus_k_fold_F1_folds_delta_heatmap.png",
                dpi=500,
            )
            plt.close()


def _sanitize_for_path(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _with_single_cwe_id(df: pl.DataFrame) -> pl.DataFrame:
    """Assign one CWE id per sample (the highest numeric CWE id)."""
    return df.with_columns(
        pl.col("cwe_ids")
        .list.eval(
            pl.element()
            .cast(pl.Utf8)
            .str.extract(r"(\d+)")
            .cast(pl.Int64, strict=False)
        )
        .list.max()
        .alias("cwe_id")
    )


def _with_cwe_rollup(df: pl.DataFrame, source_col: str, target_col: str, depth: int = 1) -> pl.DataFrame:
    """Map CWE IDs in source_col to View 1000 rollup categories."""

    def _to_rollup(value: int | None) -> int | None:
        if value is None:
            return None
        try:
            return get_research_concepts_category(int(value), depth=depth)
        except Exception:
            return None

    return df.with_columns(
        pl.col(source_col).map_elements(_to_rollup, return_dtype=pl.Int64).alias(target_col)
    )


def _explode_with_cwe_id(df: pl.DataFrame) -> pl.DataFrame:
    """Explode cwe_ids and parse numeric CWE id into cwe_id."""
    return (
        df.explode("cwe_ids")
        .drop_nulls("cwe_ids")
        .with_columns(
            pl.col("cwe_ids")
            .cast(pl.Utf8)
            .str.extract(r"(\d+)")
            .cast(pl.Int64, strict=False)
            .alias("cwe_id")
        )
        .drop_nulls("cwe_id")
    )


def plot_cwe_contents_of_each_fold(index_df: pl.DataFrame, output_dir: Path, top_n_cwes: int = 20) -> None:
    """
    Plot the CWE contents of each fold.
    Parameters:
        summary_df: Summarized DataFrame
        output_dir: Path to save PNG files
    """
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_df = index_df.filter(
        pl.col("tag") == "test-holdout",
        pl.col("total_folds").cast(pl.Int64) == 10,
        pl.col("dataset") == "megavul",
        pl.col("seed").cast(pl.Int64) == 42,
        pl.col("cw").cast(pl.Int64) == 1024,
    )
    combinations = (
        index_df.select(["fold_type", "model"])
        .unique()
        .sort(by=["fold_type", "model"])
        .iter_rows(named=True)
    )

    tp_expr = (pl.when((pl.col("label") == 1) & (pl.col("prediction") == 1)).then(1).otherwise(0)).sum().alias("tp")
    fp_expr = (pl.when((pl.col("label") == 0) & (pl.col("prediction") == 1)).then(1).otherwise(0)).sum().alias("fp")
    fn_expr = (pl.when((pl.col("label") == 1) & (pl.col("prediction") == 0)).then(1).otherwise(0)).sum().alias("fn")

    rollup_depths = [1, 2, 3, 4]

    for combo in combinations:
        fold_type = combo["fold_type"]
        model = combo["model"]
        combo_df = index_df.filter(
            pl.col("fold_type") == fold_type,
            pl.col("model") == model,
        )

        cwe_overall_counts = (
            combo_df.filter(pl.col("phase") == "test")
            .explode("cwe_ids")
            .drop_nulls("cwe_ids")
            .group_by("cwe_ids")
            .agg(
                pl.len().alias("count"),
            )
            .sort("count")
        )

        if cwe_overall_counts.height == 0:
            cwe_overall_counts = (
                combo_df.explode("cwe_ids")
                .drop_nulls("cwe_ids")
                .group_by("cwe_ids")
                .agg(pl.len().alias("count"))
                .sort("count")
            )
        if cwe_overall_counts.height == 0:
            continue

        top_cwes = cwe_overall_counts.tail(top_n_cwes).select("cwe_ids").to_series().reverse().to_list()
        combo_exploded_test = _explode_with_cwe_id(combo_df.filter(pl.col("phase") == "test"))
        combo_exploded_all = _explode_with_cwe_id(combo_df)
        top_rollup_categories: dict[int, list[int]] = {}
        for depth in rollup_depths:
            rollup_col = f"cwe_rollup_depth{depth}"
            rollup_top = (
                _with_cwe_rollup(
                    combo_exploded_test,
                    source_col="cwe_id",
                    target_col=rollup_col,
                    depth=depth,
                )
                .drop_nulls(rollup_col)
                .group_by(rollup_col)
                .agg(pl.len().alias("count"))
                .sort("count")
                .tail(top_n_cwes)
                .select(rollup_col)
                .to_series()
                .reverse()
                .to_list()
            )
            if len(rollup_top) == 0:
                rollup_top = (
                    _with_cwe_rollup(
                        combo_exploded_all,
                        source_col="cwe_id",
                        target_col=rollup_col,
                        depth=depth,
                    )
                    .drop_nulls(rollup_col)
                    .group_by(rollup_col)
                    .agg(pl.len().alias("count"))
                    .sort("count")
                    .tail(top_n_cwes)
                    .select(rollup_col)
                    .to_series()
                    .reverse()
                    .to_list()
                )
            top_rollup_categories[depth] = rollup_top

        for phase in ["holdout", "test"]:
            cwe_counts_df = (
                combo_df.filter(
                    pl.col("phase") == phase,
                )
                .explode("cwe_ids")
                .drop_nulls("cwe_ids")
                .group_by(["fold_number", "cwe_ids"])
                .agg(
                    pl.len().alias("count"),
                    fp_expr,
                    tp_expr,
                    fn_expr,
                )
                .with_columns(
                    pl.when((pl.col("tp") + pl.col("fn")) > 0).then(pl.col("tp") / (pl.col("tp") + pl.col("fn"))).otherwise(0.0).alias("recall"),
                    pl.when((pl.col("tp") + pl.col("fp")) > 0).then(pl.col("tp") / (pl.col("tp") + pl.col("fp"))).otherwise(0.0).alias("precision"),
                )
                .with_columns(
                    pl.when((pl.col("precision") + pl.col("recall")) > 0)
                    .then(2 * pl.col("precision") * pl.col("recall") / (pl.col("precision") + pl.col("recall")))
                    .otherwise(0.0)
                    .alias("f1"),
                )
                .filter(pl.col("cwe_ids").is_in(top_cwes))
            )
            if cwe_counts_df.height == 0:
                continue

            cwe_counts = cwe_counts_df.to_pandas()
            model_slug = _sanitize_for_path(model)
            fold_type_slug = _sanitize_for_path(fold_type)
            phase_dir = output_dir / phase / model_slug / fold_type_slug
            os.makedirs(phase_dir, exist_ok=True)

            for metric in ["count", "f1", "precision", "recall"]:
                pivot_data = cwe_counts.pivot_table(
                    index="cwe_ids",
                    columns=["fold_number"],
                    values=metric,
                    aggfunc="mean" if metric != "count" else "max",
                    fill_value=0,
                )
                pivot_data = pivot_data.reindex(index=top_cwes, fill_value=0)
                pivot_data = pivot_data.reindex(columns=sorted(pivot_data.columns), fill_value=0)

                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    pivot_data,
                    cmap="YlGnBu",
                    cbar_kws={"label": f"{metric} of Examples in Set"},
                    annot=True,
                    fmt=".2f" if metric != "count" else ".0f",
                    vmin=0,
                    vmax=1.0 if metric in ["f1", "precision", "recall"] else 10_000,
                )
                metric_title = metric.upper() if metric == "f1" else metric.capitalize()
                plt.title(
                    f"{metric_title} for Each Fold ({phase.capitalize()} Phase)\n"
                    f"Model: {model} | Fold Type: {format_fold_type_for_plot(fold_type)}"
                )
                plt.ylabel("CWE IDs (increasing prevalence in test set →)")
                plt.xlabel("Fold Number")
                plt.tight_layout()
                plt.savefig(phase_dir / f"cwe_{metric}_heatmap.png", dpi=500)
                plt.close()

            phase_exploded_cwe = _explode_with_cwe_id(combo_df.filter(pl.col("phase") == phase))
            for depth in rollup_depths:
                top_rollups = top_rollup_categories.get(depth, [])
                if len(top_rollups) == 0:
                    continue

                rollup_col = f"cwe_rollup_depth{depth}"
                cwe_rollup_counts_df = (
                    _with_cwe_rollup(
                        phase_exploded_cwe,
                        source_col="cwe_id",
                        target_col=rollup_col,
                        depth=depth,
                    )
                    .drop_nulls(rollup_col)
                    .group_by(["fold_number", rollup_col])
                    .agg(
                        pl.len().alias("count"),
                        fp_expr,
                        tp_expr,
                        fn_expr,
                    )
                    .with_columns(
                        pl.when((pl.col("tp") + pl.col("fn")) > 0).then(pl.col("tp") / (pl.col("tp") + pl.col("fn"))).otherwise(0.0).alias("recall"),
                        pl.when((pl.col("tp") + pl.col("fp")) > 0).then(pl.col("tp") / (pl.col("tp") + pl.col("fp"))).otherwise(0.0).alias("precision"),
                    )
                    .with_columns(
                        pl.when((pl.col("precision") + pl.col("recall")) > 0)
                        .then(2 * pl.col("precision") * pl.col("recall") / (pl.col("precision") + pl.col("recall")))
                        .otherwise(0.0)
                        .alias("f1"),
                    )
                    .filter(pl.col(rollup_col).is_in(top_rollups))
                )
                if cwe_rollup_counts_df.height == 0:
                    continue

                cwe_rollup_counts = cwe_rollup_counts_df.to_pandas()
                depth_dir = phase_dir / "rollup" / f"depth_{depth}"
                os.makedirs(depth_dir, exist_ok=True)
                for metric in ["count", "f1", "precision", "recall"]:
                    pivot_data = cwe_rollup_counts.pivot_table(
                        index=rollup_col,
                        columns=["fold_number"],
                        values=metric,
                        aggfunc="mean" if metric != "count" else "max",
                        fill_value=0,
                    )
                    pivot_data = pivot_data.reindex(index=top_rollups, fill_value=0)
                    pivot_data = pivot_data.reindex(columns=sorted(pivot_data.columns), fill_value=0)

                    plt.figure(figsize=(12, 8))
                    sns.heatmap(
                        pivot_data,
                        cmap="YlGnBu",
                        cbar_kws={"label": f"{metric} of Examples in Set"},
                        annot=True,
                        fmt=".2f" if metric != "count" else ".0f",
                        vmin=0,
                        vmax=1.0 if metric in ["f1", "precision", "recall"] else 10_000,
                    )
                    metric_title = metric.upper() if metric == "f1" else metric.capitalize()
                    plt.title(
                        f"{metric_title} for Each Fold ({phase.capitalize()} Phase) [Depth-{depth} Rollup]\n"
                        f"Model: {model} | Fold Type: {format_fold_type_for_plot(fold_type)}"
                    )
                    plt.ylabel(f"CWE Research Concepts Depth-{depth}")
                    plt.xlabel("Fold Number")
                    plt.tight_layout()
                    plt.savefig(depth_dir / f"cwe_{metric}_heatmap.png", dpi=500)
                    plt.close()


def plot_cwe_trend_lines(index_df: pl.DataFrame, output_dir: Path, top_n_cwes: int = 10, min_examples: int = 10) -> None:
    """
    Plot fold-wise CWE trends for prevalence and performance metrics.
    """
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = index_df.filter(
        pl.col("tag") == "test-holdout",
        pl.col("total_folds").cast(pl.Int64) == 10,
        pl.col("dataset") == "megavul",
        pl.col("seed").cast(pl.Int64) == 42,
        pl.col("cw").cast(pl.Int64) == 1024,
    )
    base_df = _with_single_cwe_id(base_df).drop_nulls("cwe_id")
    if base_df.height == 0:
        return

    tp_expr = (pl.when((pl.col("label") == 1) & (pl.col("prediction") == 1)).then(1).otherwise(0)).sum().alias("tp")
    fp_expr = (pl.when((pl.col("label") == 0) & (pl.col("prediction") == 1)).then(1).otherwise(0)).sum().alias("fp")
    fn_expr = (pl.when((pl.col("label") == 1) & (pl.col("prediction") == 0)).then(1).otherwise(0)).sum().alias("fn")

    combinations = (
        base_df.select(["fold_type", "model"])
        .unique()
        .sort(by=["fold_type", "model"])
        .iter_rows(named=True)
    )
    metric_specs = [
        ("prevalence", "count", "Prevalence (Raw Count)", 0.0, None),
        ("count", "count", "Count", 0.0, None),
        ("f1", "f1", "F1", 0.0, 1.0),
        ("precision", "precision", "Precision", 0.0, 1.0),
        ("recall", "recall", "Recall", 0.0, 1.0),
    ]

    for combo in combinations:
        fold_type = combo["fold_type"]
        model = combo["model"]
        combo_base_df = base_df.filter(
            pl.col("fold_type") == fold_type,
            pl.col("model") == model,
        )
        if combo_base_df.height == 0:
            continue

        for phase in ["holdout", "test", ]:
            combo_df = combo_base_df.filter(pl.col("phase") == phase)
            if combo_df.height == 0:
                continue

            fold_numbers = sorted(combo_df.select("fold_number").unique().to_series().to_list())
            required_fold_coverage = len(fold_numbers)
            if required_fold_coverage == 0:
                continue

            model_slug = _sanitize_for_path(model)
            fold_type_slug = _sanitize_for_path(fold_type)
            combo_output_dir = output_dir / phase / model_slug / fold_type_slug
            os.makedirs(combo_output_dir, exist_ok=True)

            grouping_specs = [("cwe_id", combo_df, "cwe", "CWE", "CWE", combo_output_dir)]
            for depth in [1, 2, 3, 4]:
                rollup_col = f"cwe_rollup_depth{depth}"
                rollup_df = _with_cwe_rollup(
                    combo_df, source_col="cwe_id", target_col=rollup_col, depth=depth
                ).drop_nulls(rollup_col)
                grouping_specs.append(
                    (
                        rollup_col,
                        rollup_df,
                        "cwe",
                        f"CWE Depth-{depth} Rollup",
                        f"Depth-{depth}",
                        combo_output_dir / "rollup" / f"depth_{depth}",
                    )
                )

            for group_col, group_df, filename_prefix, plot_label_prefix, legend_title, group_output_dir in grouping_specs:
                if group_df.height == 0:
                    continue

                cwe_fold_counts = group_df.group_by([group_col, "fold_number"]).agg(
                    pl.len().alias("fold_count"),
                )
                eligible_top_groups = (
                    cwe_fold_counts.group_by(group_col)
                    .agg(
                        pl.col("fold_count").sum().alias("total_count"),
                        pl.col("fold_number").n_unique().alias("num_folds_present"),
                    )
                    .filter(
                        pl.col("num_folds_present") == required_fold_coverage,
                        pl.col("total_count") >= min_examples,
                    )
                    .sort("total_count")
                    .tail(top_n_cwes)
                    .select(group_col)
                    .to_series()
                    .reverse()
                    .to_list()
                )
                if len(eligible_top_groups) == 0:
                    continue

                cwe_trends_df = (
                    group_df.group_by(["fold_number", group_col])
                    .agg(
                        pl.len().alias("count"),
                        fp_expr,
                        tp_expr,
                        fn_expr,
                    )
                    .with_columns(
                        pl.when((pl.col("tp") + pl.col("fn")) > 0).then(pl.col("tp") / (pl.col("tp") + pl.col("fn"))).otherwise(0.0).alias("recall"),
                        pl.when((pl.col("tp") + pl.col("fp")) > 0).then(pl.col("tp") / (pl.col("tp") + pl.col("fp"))).otherwise(0.0).alias("precision"),
                    )
                    .with_columns(
                        pl.when((pl.col("precision") + pl.col("recall")) > 0)
                        .then(2 * pl.col("precision") * pl.col("recall") / (pl.col("precision") + pl.col("recall")))
                        .otherwise(0.0)
                        .alias("f1"),
                    )
                    .filter(pl.col(group_col).is_in(eligible_top_groups))
                )
                if cwe_trends_df.height == 0:
                    continue

                trend_df = cwe_trends_df.to_pandas()
                palette = sns.color_palette("tab10", n_colors=max(len(eligible_top_groups), 1))
                os.makedirs(group_output_dir, exist_ok=True)

                for metric_name, metric_col, y_label, y_min, y_max in metric_specs:
                    plt.figure(figsize=(12, 7))
                    for idx, group_id in enumerate(eligible_top_groups):
                        group_data = (
                            trend_df[trend_df[group_col] == group_id][["fold_number", metric_col]]
                            .sort_values("fold_number")
                        )
                        plt.plot(
                            group_data["fold_number"],
                            group_data[metric_col],
                            marker="o",
                            linewidth=2,
                            color=palette[idx % len(palette)],
                            label=f"{plot_label_prefix}-{int(group_id)}",
                        )

                    plt.title(
                        f"{plot_label_prefix} {y_label} Across {phase.capitalize()} Folds\n"
                        f"Model: {model} | Fold Type: {format_fold_type_for_plot(fold_type)} | Min Samples: {min_examples}"
                    )
                    plt.xlabel("Fold Number")
                    plt.ylabel(y_label)
                    plt.xticks(fold_numbers)
                    plt.ylim(bottom=y_min)
                    if y_max is not None:
                        plt.ylim(top=y_max)
                    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.tight_layout()
                    plt.savefig(group_output_dir / f"{filename_prefix}_{metric_name}_trend_line.png", dpi=500)
                    plt.close()


def plot_holdout_results(df: pl.DataFrame, eda_dir: Path) -> None:
    df = df.filter(
        pl.col("tag") == "test-holdout",
        pl.col("total_folds").cast(pl.Int64) == 10,
        pl.col("model") == "llama3.2-1B",
        pl.col("dataset") == "megavul",
        pl.col("seed").cast(pl.Int64) == 42,
        pl.col("cw").cast(pl.Int64) == 1024,
        pl.col("fold_type") == "k_fold_cross_validation",
        pl.col("fold_number") == 0,
        pl.col("phase") == "holdout",
    )
    df = df.sort(by="date")
    df = df.unique(subset=["index"], keep="last")
    df = df.sort(by="publish_date")


def plot_raw_vs_alpaca_results(df: pl.DataFrame, eda_dir: Path) -> None:
    os.makedirs(eda_dir, exist_ok=True)
    df = df.filter(
        pl.col("total_folds").cast(pl.Int64) == 10,
        pl.col("model") == "llama3.2-1B",
        pl.col("tag") == "test-holdout",
        pl.col("seed").cast(pl.Int64) == 42,
        pl.col("cw").cast(pl.Int64) == 1024,
    )
    if {"epoch", "step"}.issubset(set(df.columns)):
        df = df.sort(by=["epoch", "step"])
    df = df.unique(
        subset=[
            "index",
            "datamodule",
            "fold_number",
            "fold_type",
            "phase",
        ],
        keep="last",
    )

    for phase in ["holdout", "test"]:
        summary_df = summarize_df_statistics(df.filter(pl.col("phase") == phase))
        summary_df = summary_df.with_columns(
            pl.concat_str(
                [
                    pl.col("fold_type"),
                    pl.lit(" | "),
                    pl.col("datamodule"),
                ]
            ).alias("label")
        ).select(["label", "Precision", "Recall", "F1", "fold_number"])

        for metric in ["Precision", "Recall", "F1"]:
            plt.figure(figsize=(12, 8))
            data = (
                summary_df.select(["label", "fold_number", metric])
                .to_pandas()
                .pivot(
                    index="label",
                    columns="fold_number",
                    values=metric,
                )
            )
            # add a mean column on the right
            data["Mean"] = data.mean(axis=1)
            data["STD"] = data.iloc[:, :-1].std(axis=1)
            sns.heatmap(
                data=data,
                cmap="YlGnBu",
                cbar_kws={"label": f"{metric} (%)"},
                annot=True,
                fmt=".3f",
                vmin=0,
                vmax=1,
            )
            plt.title(f"{metric} Comparison of Raw vs Alpaca Datasets ({phase.capitalize()} Phase)")
            plt.ylabel("Dataset and Fold Type")
            plt.xlabel("Fold Number")
            plt.tight_layout()
            plt.savefig(eda_dir / f"{metric.lower()}_{phase}_heatmap.png", dpi=500)
            plt.close()


if __name__ == "__main__":
    eda_dir: Path = INDEX_RESULTS_FILE.parent
    df = apply_plot_model_exclusion(pl.read_parquet(INDEX_RESULTS_FILE))
    df = df.filter(pl.col("datamodule").is_in(["AlpacaDataset"]))
    df = df.filter(pl.col("fold_type").is_in(PRIMARY_FOLD_TYPES))
    if "run_mode" in df.columns:
        df = df.filter(pl.col("run_mode").is_in(["fine_tuning", "opro"]))
    summary_df = summarize_df_statistics(df)

    plot_high_level_summary_heatmaps(summary_df, fold_count=10, output_dir=eda_dir / "heatmaps")
    plot_random_temporal_delta(summary_df, fold_count=10, output_dir=eda_dir / "delta")
    plot_cwe_contents_of_each_fold(df, eda_dir / "cwe_contents")
    plot_cwe_trend_lines(df, eda_dir / "cwe_trends")
    plot_holdout_results(df, eda_dir / "inidivudal_holdout_plots")
    plot_raw_vs_alpaca_results(df, eda_dir / "raw_vs_alpaca_plots")
