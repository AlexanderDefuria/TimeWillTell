#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Test vs holdout performance contrast across the 5 primary fold types.

Three figure families (one figure per metric - f1, precision, recall):
  bar     - grouped bar chart: mean ± std per fold type, test vs holdout side-by-side
  scatter - test metric vs holdout metric; one point = (model, fold_number); y=x line
  delta   - violin of (holdout − test) per fold type, centred at zero

Outputs to data/plots/9_test_holdout_contrast/.

Usage:
    uv run scripts/plot_test_holdout_contrast.py
    uv run scripts/plot_test_holdout_contrast.py --metric f1 --dataset megavul
    uv run scripts/plot_test_holdout_contrast.py --tag fine-tuning
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
    from scripts import INDEX_RESULTS_FILE, PRIMARY_FOLD_TYPES
    from scripts.plot_results import (
        GROUP_COLS, METRICS, build_metric_frames, format_fold_type_for_plot, load_results,
        _f1_colour, _latex_cell,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import INDEX_RESULTS_FILE, PRIMARY_FOLD_TYPES
    from scripts.plot_results import (
        GROUP_COLS, METRICS, build_metric_frames, format_fold_type_for_plot, load_results,
        _f1_colour, _latex_cell,
    )

PLOTS_DIR = Path(__file__).resolve().parents[1] / "data" / "plots" / "9_test_holdout_contrast"

_SHORT_LABELS = {
    "k fold": "K-Fold",
    "random growing window": "R-GW",
    "random loo block": "R-LOO",
    "temporal growing window": "T-GW",
    "temporal loo block": "T-LOO",
}

_FT_PALETTE = {ft: c for ft, c in zip(
    PRIMARY_FOLD_TYPES,
    sns.color_palette("tab10", n_colors=5),
)}


def _short(fold_type: str) -> str:
    return _SHORT_LABELS.get(format_fold_type_for_plot(fold_type), format_fold_type_for_plot(fold_type))


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _delta_colour(delta: float, vmin: float = -0.1, vmax: float = 0.1) -> str:
    """Interpolate red (delta<0) to white (0) to blue (delta>0)."""
    if np.isnan(delta):
        return "FFFFFF"
    # Clamp to [-vmin, vmax]
    clamped = np.clip(delta, -abs(vmin), vmax)
    if clamped < 0:
        # Negative: interpolate white → red
        t = -clamped / abs(vmin)  # 0 to 1
        r = int(round(255 - t * (255 - 229)))
        g = int(round(255 - t * (255 - 87)))
        b = int(round(255 - t * (255 - 87)))
    else:
        # Positive: interpolate white → blue
        t = clamped / vmax  # 0 to 1
        r = int(round(255 - t * (255 - 76)))
        g = int(round(255 - t * (255 - 175)))
        b = int(round(255 - t * (255 - 80)))
    return f"{r:02X}{g:02X}{b:02X}"


def _delta_cell(val: float) -> str:
    """Return a coloured cell: \\cellcolor[HTML]{hex} ±XXX"""
    if np.isnan(val):
        return "\\cellcolor[HTML]{FFFFFF} ---"
    colour = _delta_colour(val)
    sign = "+" if val > 0 else ""
    return f"\\cellcolor[HTML]{{{colour}}} {sign}{val * 100:.1f}"


def _relative_cell(val: float, vmax: float = 0.5) -> str:
    """Colour cell for relative test inflation: white (0%) → orange-red (high %)."""
    if np.isnan(val):
        return "\\cellcolor[HTML]{FFFFFF} ---"
    t = min(abs(val) / vmax, 1.0)
    # White → orange-red: (255,255,255) → (229,87,87)
    r = int(round(255 - t * (255 - 229)))
    g = int(round(255 - t * (255 - 87)))
    b = int(round(255 - t * (255 - 87)))
    colour = f"{r:02X}{g:02X}{b:02X}"
    sign = "+" if val > 0 else ""
    return f"\\cellcolor[HTML]{{{colour}}} {sign}{val * 100:.1f}"


def generate_contrast_latex_table(
    test_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    fold_types: list[str],
    family_key: str,
    output_dir: Path,
) -> None:
    """Generate a LaTeX table with Test F1, Holdout F1, and delta."""
    import pandas as pd

    rows = []
    for ft in fold_types:
        t = test_df.filter(pl.col("fold_type") == ft)["f1"].to_numpy()
        h = holdout_df.filter(pl.col("fold_type") == ft)["f1"].to_numpy()
        if len(t) == 0 or len(h) == 0:
            continue
        t_mean = float(np.mean(t))
        h_mean = float(np.mean(h))
        relative = (t_mean - h_mean) / h_mean if h_mean > 0 else float("nan")
        rows.append({
            "fold_type": ft,
            "test_mean": t_mean,
            "test_std": float(np.std(t)),
            "holdout_mean": h_mean,
            "holdout_std": float(np.std(h)),
            "delta_mean": float(h_mean - t_mean),
            "relative_mean": relative,
        })

    if not rows:
        return

    pdf = pd.DataFrame(rows)
    col_spec = "l c c c c"

    # Global colour scale for F1 columns
    all_f1 = np.concatenate([pdf["test_mean"].values, pdf["holdout_mean"].values])
    all_f1 = all_f1[np.isfinite(all_f1)]
    vmin_f1 = float(all_f1.min()) if len(all_f1) else 0.0
    vmax_f1 = float(all_f1.max()) if len(all_f1) else 1.0

    def _short(fold_type: str) -> str:
        return _SHORT_LABELS.get(format_fold_type_for_plot(fold_type), format_fold_type_for_plot(fold_type))

    lines: list[str] = [
        "% Auto-generated by scripts/plot_test_holdout_contrast.py -- do not edit manually.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{colortbl}, \\usepackage{xcolor}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\footnotesize",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    Fold Type & T-F1 $\\pm$ std (\\%) & H-F1 $\\pm$ std (\\%) & $\\Delta$ F1 (pp) & RI-F1 (\\%) \\\\",
        "    \\midrule",
    ]

    for _, row in pdf.iterrows():
        ft = row["fold_type"]
        label = _short(ft)
        test_val = row["test_mean"]
        holdout_val = row["holdout_mean"]
        delta_val = row["delta_mean"]
        relative_val = row["relative_mean"]

        test_cell = _latex_cell(test_val, vmin_f1, vmax_f1)
        holdout_cell = _latex_cell(holdout_val, vmin_f1, vmax_f1)
        delta_cell = _delta_cell(delta_val)
        relative_cell = _relative_cell(relative_val)

        lines.append(f"    {label} & {test_cell} & {holdout_cell} & {delta_cell} & {relative_cell} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    family_label = "Fine-Tuned (LoRA)" if family_key == "fine_tuned" else "OPRO (GGUF)"
    lines.append(
        f"  \\caption{{Test vs holdout F1 performance by fold type ({family_label}). "
        f"T-F1 ± std: mean cross-validation test-set F1 with std dev. "
        f"H-F1 ± std: mean holdout-set F1 with std dev. "
        f"$\\Delta$ F1: absolute gap (holdout - test) in percentage points. "
        f"RI-F1: relative inflation ((test − holdout) / holdout) as a percentage.}}"
    )
    lines.append(f"  \\label{{tab:contrast_{family_key}}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"contrast_table_{family_key}.tex"
    out_path.write_text(latex)
    print(f"  LaTeX contrast table → {out_path}")


# Combined contrast table (fine-tuning + zero-shot side-by-side)


def generate_combined_contrast_latex_table(
    test_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    fold_types: list[str],
    output_dir: Path,
) -> None:
    """Generate a combined LaTeX table with fine-tuning and zero-shot side-by-side."""
    import pandas as pd

    def _build_family_rows(t_df: pl.DataFrame, h_df: pl.DataFrame) -> pd.DataFrame:
        """Build rows dict for a single family."""
        rows = []
        for ft in fold_types:
            t = t_df.filter(pl.col("fold_type") == ft)["f1"].to_numpy()
            h = h_df.filter(pl.col("fold_type") == ft)["f1"].to_numpy()
            if len(t) == 0 or len(h) == 0:
                continue
            t_mean = float(np.mean(t))
            h_mean = float(np.mean(h))
            relative = (t_mean - h_mean) / h_mean if h_mean > 0 else float("nan")
            rows.append({
                "fold_type": ft,
                "test_mean": t_mean,
                "holdout_mean": h_mean,
                "delta_mean": float(h_mean - t_mean),
                "relative_mean": relative,
            })
        return pd.DataFrame(rows)

    # Get fine-tuning and zero-shot data
    ft_test = test_df.filter(pl.col("run_mode") == "fine_tuning")
    ft_holdout = holdout_df.filter(pl.col("run_mode") == "fine_tuning")
    zs_test = test_df.filter(pl.col("run_mode") == "opro")
    zs_holdout = holdout_df.filter(pl.col("run_mode") == "opro")

    ft_rows = _build_family_rows(ft_test, ft_holdout)
    zs_rows = _build_family_rows(zs_test, zs_holdout)

    if ft_rows.empty or zs_rows.empty:
        return

    # Merge on fold_type and align rows
    combined = ft_rows.merge(zs_rows, on="fold_type", suffixes=("_ft", "_zs"))

    # Global colour scales for each family's F1 columns
    ft_f1 = np.concatenate([ft_rows["test_mean"].values, ft_rows["holdout_mean"].values])
    ft_f1 = ft_f1[np.isfinite(ft_f1)]
    vmin_ft = float(ft_f1.min()) if len(ft_f1) else 0.0
    vmax_ft = float(ft_f1.max()) if len(ft_f1) else 1.0

    zs_f1 = np.concatenate([zs_rows["test_mean"].values, zs_rows["holdout_mean"].values])
    zs_f1 = zs_f1[np.isfinite(zs_f1)]
    vmin_zs = float(zs_f1.min()) if len(zs_f1) else 0.0
    vmax_zs = float(zs_f1.max()) if len(zs_f1) else 1.0

    def _short(fold_type: str) -> str:
        return _SHORT_LABELS.get(format_fold_type_for_plot(fold_type), format_fold_type_for_plot(fold_type))

    lines: list[str] = [
        "% Auto-generated by scripts/plot_test_holdout_contrast.py -- do not edit manually.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{colortbl}, \\usepackage{xcolor}, \\usepackage{tabularx}, \\usepackage{array}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\footnotesize",
        "  \\begin{tabularx}{\\textwidth}{l|*{4}{>{\\centering\\arraybackslash}X}|*{4}{>{\\centering\\arraybackslash}X}}",
        "    \\toprule",
        "    & \\multicolumn{4}{c}{\\textit{Fine-Tuned (LoRA)}} & \\multicolumn{4}{c}{\\textit{OPRO (GGUF)}} \\\\",
        "    Fold Type & T-F1 (\\%) & H-F1 (\\%) & $\\Delta$ F1 (pp) & RI-F1 (\\%) & T-F1 (\\%) & H-F1 (\\%) & $\\Delta$ F1 (pp) & RI-F1 (\\%) \\\\",
        "    \\midrule",
    ]

    for _, row in combined.iterrows():
        ft = row["fold_type"]
        label = _short(ft)

        # Fine-tuning cells
        ft_test_cell = _latex_cell(row["test_mean_ft"], vmin_ft, vmax_ft)
        ft_holdout_cell = _latex_cell(row["holdout_mean_ft"], vmin_ft, vmax_ft)
        ft_delta_cell = _delta_cell(row["delta_mean_ft"])
        ft_relative_cell = _relative_cell(row["relative_mean_ft"])

        # Zero-shot cells
        zs_test_cell = _latex_cell(row["test_mean_zs"], vmin_zs, vmax_zs)
        zs_holdout_cell = _latex_cell(row["holdout_mean_zs"], vmin_zs, vmax_zs)
        zs_delta_cell = _delta_cell(row["delta_mean_zs"])
        zs_relative_cell = _relative_cell(row["relative_mean_zs"])

        lines.append(
            f"    {label} & {ft_test_cell} & {ft_holdout_cell} & {ft_delta_cell} & {ft_relative_cell} & "
            f"{zs_test_cell} & {zs_holdout_cell} & {zs_delta_cell} & {zs_relative_cell} \\\\"
        )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabularx}")
    lines.append(
        "  \\caption{Test vs holdout F1 performance by model family. "
        "T-F1: mean cross-validation test-set F1 (\\%). "
        "H-F1: mean holdout-set F1 (\\%). "
        "$\\Delta$ F1: absolute gap (holdout - test) in percentage points. "
        "RI-F1: relative inflation ((test - holdout) / holdout) as a percentage.}"
    )
    lines.append("  \\label{tab:contrast_combined}")
    lines.append("\\end{table}")

    latex = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "contrast_table_combined.tex"
    out_path.write_text(latex)
    print(f"  LaTeX combined contrast table → {out_path}")


# Combined contrast table + last-fold rows


_LAST_FOLD_FOLD_TYPES = [
    "temporal_growing_window_cross_validation",
    "temporal_loo_block_cross_validation",
]


def generate_combined_contrast_latex_table_with_last_fold(
    test_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    fold_types: list[str],
    output_dir: Path,
) -> None:
    """Same as the combined contrast table, plus two extra rows for the most
    recent fold of the two temporal strategies."""
    import pandas as pd

    def _build_family_rows(
        t_df: pl.DataFrame,
        h_df: pl.DataFrame,
        fts: list[str],
        last_fold_only: bool,
    ) -> pd.DataFrame:
        rows = []
        for ft in fts:
            t_sub = t_df.filter(pl.col("fold_type") == ft)
            h_sub = h_df.filter(pl.col("fold_type") == ft)
            if last_fold_only:
                if t_sub.is_empty() or h_sub.is_empty():
                    continue
                # For both temporal strategies the dataset is sorted by date
                # ascending before slicing, so fold index is a direct position
                # offset into the sorted array: fold 0 = earliest test window,
                # fold N-1 = most recent test window.  max(fold_number) therefore
                # gives the most chronologically recent fold that completed both
                # test and holdout phases.
                t_folds = set(t_sub["fold_number"].to_list())
                h_folds = set(h_sub["fold_number"].to_list())
                common_folds = t_folds & h_folds
                if not common_folds:
                    continue
                last_fold = max(common_folds)  # most recent completed fold
                t_sub = t_sub.filter(pl.col("fold_number") == last_fold)
                h_sub = h_sub.filter(pl.col("fold_number") == last_fold)
            t = t_sub["f1"].to_numpy()
            h = h_sub["f1"].to_numpy()
            if len(t) == 0 or len(h) == 0:
                continue
            t_mean = float(np.mean(t))
            h_mean = float(np.mean(h))
            t_std = float(np.std(t))
            h_std = float(np.std(h))
            relative = (t_mean - h_mean) / h_mean if h_mean > 0 else float("nan")
            rows.append({
                "fold_type": ft,
                "starred": last_fold_only,
                "test_mean": t_mean,
                "test_std": t_std,
                "holdout_mean": h_mean,
                "holdout_std": h_std,
                "delta_mean": float(h_mean - t_mean),
                "relative_mean": relative,
            })
        return pd.DataFrame(rows)

    def _mbox_cell(val: float, vmin: float, vmax: float, std: float) -> str:
        colour = _f1_colour(val, vmin, vmax)
        if np.isnan(val):
            return f"\\cellcolor[HTML]{{{colour}}} ---"
        return (
            f"\\cellcolor[HTML]{{{colour}}} "
            f"\\mbox{{{val * 100:.1f} {{\\tiny {{±{std * 100:.1f}}}}}}}"
        )

    ft_test = test_df.filter(pl.col("run_mode") == "fine_tuning")
    ft_holdout = holdout_df.filter(pl.col("run_mode") == "fine_tuning")
    zs_test = test_df.filter(pl.col("run_mode") == "opro")
    zs_holdout = holdout_df.filter(pl.col("run_mode") == "opro")

    last_fold_fts = [ft for ft in _LAST_FOLD_FOLD_TYPES if ft in fold_types]

    ft_std = _build_family_rows(ft_test, ft_holdout, fold_types, last_fold_only=False)
    ft_last = _build_family_rows(ft_test, ft_holdout, last_fold_fts, last_fold_only=True)
    zs_std = _build_family_rows(zs_test, zs_holdout, fold_types, last_fold_only=False)
    zs_last = _build_family_rows(zs_test, zs_holdout, last_fold_fts, last_fold_only=True)

    ft_rows = pd.concat([ft_std, ft_last], ignore_index=True)
    zs_rows = pd.concat([zs_std, zs_last], ignore_index=True)

    if ft_rows.empty or zs_rows.empty:
        return

    combined = ft_rows.merge(zs_rows, on=["fold_type", "starred"], suffixes=("_ft", "_zs"))
    if combined.empty:
        return

    fold_order = {ft: i for i, ft in enumerate(fold_types)}
    combined["_sort_key"] = combined.apply(
        lambda r: (1 if r["starred"] else 0, fold_order.get(r["fold_type"], len(fold_order))),
        axis=1,
    )
    combined = combined.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

    ft_f1 = np.concatenate([combined["test_mean_ft"].values, combined["holdout_mean_ft"].values])
    ft_f1 = ft_f1[np.isfinite(ft_f1)]
    vmin_ft = float(ft_f1.min()) if len(ft_f1) else 0.0
    vmax_ft = float(ft_f1.max()) if len(ft_f1) else 1.0

    zs_f1 = np.concatenate([combined["test_mean_zs"].values, combined["holdout_mean_zs"].values])
    zs_f1 = zs_f1[np.isfinite(zs_f1)]
    vmin_zs = float(zs_f1.min()) if len(zs_f1) else 0.0
    vmax_zs = float(zs_f1.max()) if len(zs_f1) else 1.0

    def _short(fold_type: str) -> str:
        return _SHORT_LABELS.get(format_fold_type_for_plot(fold_type), format_fold_type_for_plot(fold_type))

    lines: list[str] = [
        "% Auto-generated by scripts/plot_test_holdout_contrast.py -- do not edit manually.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{colortbl}, \\usepackage{xcolor}, \\usepackage{tabularx}, \\usepackage{array}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\footnotesize",
        "  \\begin{tabularx}{\\textwidth}{l"
        "|>{\\centering\\arraybackslash\\hsize=1.2\\hsize}X"
        " >{\\centering\\arraybackslash\\hsize=1.2\\hsize}X"
        " >{\\centering\\arraybackslash\\hsize=0.8\\hsize}X"
        " >{\\centering\\arraybackslash\\hsize=0.8\\hsize}X"
        "|>{\\centering\\arraybackslash\\hsize=1.2\\hsize}X"
        " >{\\centering\\arraybackslash\\hsize=1.2\\hsize}X"
        " >{\\centering\\arraybackslash\\hsize=0.8\\hsize}X"
        " >{\\centering\\arraybackslash\\hsize=0.8\\hsize}X}",
        "    \\toprule",
        "    & \\multicolumn{4}{c}{\\textit{Fine-Tuned (LoRA)}} & \\multicolumn{4}{c}{\\textit{OPRO (GGUF)}} \\\\",
        "    Fold Type & \\shortstack{T-F1\\\\{\\tiny$\\pm$std}} & \\shortstack{H-F1\\\\{\\tiny$\\pm$std}} & \\shortstack{$\\Delta$F1\\\\{\\tiny(pp)}} & \\shortstack{RI-F1\\\\{\\tiny(\\%)}} & \\shortstack{T-F1\\\\{\\tiny$\\pm$std}} & \\shortstack{H-F1\\\\{\\tiny$\\pm$std}} & \\shortstack{$\\Delta$F1\\\\{\\tiny(pp)}} & \\shortstack{RI-F1\\\\{\\tiny(\\%)}} \\\\",
        "    \\midrule",
    ]

    prev_starred = False
    for _, row in combined.iterrows():
        if row["starred"] and not prev_starred:
            lines.append("    \\midrule")
        prev_starred = row["starred"]

        label = _short(row["fold_type"]) + ("*" if row["starred"] else "")

        ft_test_cell = _mbox_cell(row["test_mean_ft"], vmin_ft, vmax_ft, row["test_std_ft"])
        ft_holdout_cell = _mbox_cell(row["holdout_mean_ft"], vmin_ft, vmax_ft, row["holdout_std_ft"])
        ft_delta_cell = _delta_cell(row["delta_mean_ft"])
        ft_relative_cell = _relative_cell(row["relative_mean_ft"])

        zs_test_cell = _mbox_cell(row["test_mean_zs"], vmin_zs, vmax_zs, row["test_std_zs"])
        zs_holdout_cell = _mbox_cell(row["holdout_mean_zs"], vmin_zs, vmax_zs, row["holdout_std_zs"])
        zs_delta_cell = _delta_cell(row["delta_mean_zs"])
        zs_relative_cell = _relative_cell(row["relative_mean_zs"])

        lines.append(
            f"    {label} & {ft_test_cell} & {ft_holdout_cell} & {ft_delta_cell} & {ft_relative_cell} & "
            f"{zs_test_cell} & {zs_holdout_cell} & {zs_delta_cell} & {zs_relative_cell} \\\\"
        )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabularx}")
    lines.append(
        "  \\caption{Test vs holdout F1 performance by model family. "
        "T-F1 ± std: mean cross-validation test-set F1 with standard deviation. "
        "H-F1 ± std: mean holdout-set F1 with standard deviation. "
        "$\\Delta$ F1: absolute gap (holdout - test) in percentage points. "
        "RI-F1: relative inflation ((test - holdout) / holdout) as a percentage. "
        "* values computed from the most recent temporal fold only.}"
    )
    lines.append("  \\label{tab:contrast_combined_last_fold}")
    lines.append("\\end{table}")

    latex = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "contrast_table_combined_last_fold.tex"
    out_path.write_text(latex)
    print(f"  LaTeX combined contrast table with last fold → {out_path}")


# Figure 1: Grouped bar chart


def plot_bar(
    test_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    metric: str,
    fold_types: list[str],
    family_label: str = "",
) -> plt.Figure:
    """Mean ± std grouped bar chart: test vs holdout per fold type."""
    rows = []
    for ft in fold_types:
        t = test_df.filter(pl.col("fold_type") == ft)[metric].to_numpy()
        h = holdout_df.filter(pl.col("fold_type") == ft)[metric].to_numpy()
        rows.append({"fold_type": ft, "phase": "Test",    "mean": float(np.mean(t)), "std": float(np.std(t))})
        rows.append({"fold_type": ft, "phase": "Holdout", "mean": float(np.mean(h)), "std": float(np.std(h))})

    import pandas as pd
    pdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(fold_types))
    width = 0.35
    colors = {"Test": "#4C72B0", "Holdout": "#DD8452"}

    for i, phase in enumerate(["Test", "Holdout"]):
        sub = pdf[pdf["phase"] == phase]
        offset = (i - 0.5) * width
        ax.bar(x + offset, sub["mean"].values, width,
               label=phase, color=colors[phase], alpha=0.85, zorder=3)
        ax.errorbar(x + offset, sub["mean"].values, yerr=sub["std"].values,
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_short(ft) for ft in fold_types], fontsize=10)
    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.set_ylim(bottom=0)
    suffix = f" - {family_label}" if family_label else ""
    ax.set_title(f"Test vs Holdout - {metric.upper()} (mean ± std){suffix}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


# Figure 2: Scatter test vs holdout


def plot_scatter(
    test_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    metric: str,
    fold_types: list[str],
    family_label: str = "",
) -> plt.Figure:
    """Scatter: test metric (x) vs holdout metric (y); one point per (model, fold_number)."""
    join_cols = [c for c in GROUP_COLS if c in test_df.columns and c in holdout_df.columns]
    joined = test_df.join(holdout_df, on=join_cols, suffix="_h")

    fig, ax = plt.subplots(figsize=(6, 6))

    all_vals: list[float] = []
    for ft in fold_types:
        sub = joined.filter(pl.col("fold_type") == ft)
        if sub.is_empty():
            continue
        x = sub[metric].to_numpy()
        y = sub[f"{metric}_h"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask],
                   color=_FT_PALETTE[ft], label=_short(ft),
                   alpha=0.7, edgecolors="white", linewidths=0.3, s=50, zorder=3)
        all_vals.extend(x[mask].tolist())
        all_vals.extend(y[mask].tolist())

    if all_vals:
        lo = max(0.0, min(all_vals) - 0.02)
        hi = min(1.0, max(all_vals) + 0.02)
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--",
                linewidth=1.2, alpha=0.6, zorder=2, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel(f"Test {metric.upper()}", fontsize=11)
    ax.set_ylabel(f"Holdout {metric.upper()}", fontsize=11)
    suffix = f" - {family_label}" if family_label else ""
    ax.set_title(f"Test vs Holdout {metric.upper()}{suffix}\nPoints above y=x: holdout > test",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


# Figure 3: Delta violin


def plot_delta(
    delta_df: pl.DataFrame,
    metric: str,
    fold_types: list[str],
    family_label: str = "",
) -> plt.Figure:
    """Violin of (holdout − test) per fold type. delta_df contains test − holdout, so negate."""
    import pandas as pd
    rows = []
    for ft in fold_types:
        sub = delta_df.filter(pl.col("fold_type") == ft)
        if sub.is_empty():
            continue
        vals = (-sub[metric]).to_numpy()  # negate: holdout − test
        vals = vals[np.isfinite(vals)]
        for v in vals:
            rows.append({"fold_type": _short(ft), "delta": v})

    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    pdf = pd.DataFrame(rows)
    short_order = [_short(ft) for ft in fold_types if _short(ft) in pdf["fold_type"].values]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = {_short(ft): _FT_PALETTE[ft] for ft in fold_types}
    sns.violinplot(data=pdf, x="fold_type", y="delta", order=short_order,
                   hue="fold_type", palette=palette, legend=False,
                   inner="box", ax=ax, linewidth=0.8)
    sns.stripplot(data=pdf, x="fold_type", y="delta", order=short_order,
                  hue="fold_type", palette=palette, legend=False,
                  alpha=0.4, size=3, jitter=True, ax=ax, zorder=3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("", fontsize=10)
    ax.set_ylabel(f"Holdout − Test {metric.upper()}", fontsize=11)
    suffix = f" - {family_label}" if family_label else ""
    ax.set_title(f"Holdout − Test {metric.upper()} per Fold Type{suffix}\nNegative = degradation on holdout",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


# Figure 4: Temporal trend line chart

_TEMPORAL_FOLD_TYPES = [
    "temporal_growing_window_cross_validation",
    "temporal_loo_block_cross_validation",
]

_RUN_MODE_LABELS = {
    "fine_tuning": "Fine-Tuned (LoRA)",
    "opro":        "OPRO (GGUF)",
}


def plot_temporal_trend(
    test_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    metric: str = "f1",
) -> dict[str, plt.Figure]:
    """Line charts of test vs holdout F1 over fold number for each temporal
    fold type.

    Returns a dict mapping fold_type slug → Figure.  Each figure contains one
    subplot per run_mode (fine_tuning / opro) that has data.  Within each
    subplot:
      • one solid line  = mean across models for the *test* phase
      • one dashed line = mean across models for the *holdout* phase
      • shaded band     = ±1 std across models
      • thin lines (low alpha) = individual model trajectories
    """
    temporal_fts = [
        ft for ft in _TEMPORAL_FOLD_TYPES
        if ft in test_df["fold_type"].unique().to_list()
    ]

    run_modes = [rm for rm in _RUN_MODE_LABELS if rm in test_df["run_mode"].unique().to_list()]

    figs: dict[str, plt.Figure] = {}

    for ft in temporal_fts:
        ft_test = test_df.filter(pl.col("fold_type") == ft)
        ft_hold = holdout_df.filter(pl.col("fold_type") == ft)

        active_modes = [rm for rm in run_modes if not ft_test.filter(pl.col("run_mode") == rm).is_empty()]
        if not active_modes:
            continue

        n_cols = len(active_modes)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5), sharey=True, squeeze=False)

        for col_idx, run_mode in enumerate(active_modes):
            ax = axes[0][col_idx]
            t_sub = ft_test.filter(pl.col("run_mode") == run_mode)
            h_sub = ft_hold.filter(pl.col("run_mode") == run_mode)

            fold_nums = sorted(
                set(t_sub["fold_number"].to_list()) | set(h_sub["fold_number"].to_list())
            )
            models = sorted(
                set(t_sub["model"].to_list()) | set(h_sub["model"].to_list())
            )

            # Per-model series (thin background lines)
            for model in models:
                t_m = t_sub.filter(pl.col("model") == model)
                h_m = h_sub.filter(pl.col("model") == model)
                t_pts = {r["fold_number"]: r[metric] for r in t_m.iter_rows(named=True)}
                h_pts = {r["fold_number"]: r[metric] for r in h_m.iter_rows(named=True)}
                t_vals = [t_pts.get(fn) for fn in fold_nums]
                h_vals = [h_pts.get(fn) for fn in fold_nums]
                ax.plot(fold_nums, t_vals, color="#4C72B0", alpha=0.18, linewidth=0.9, zorder=2)
                ax.plot(fold_nums, h_vals, color="#DD8452", alpha=0.18, linewidth=0.9, zorder=2)

            # Mean ± std across models
            t_matrix = np.array([
                [
                    float(t_sub.filter(pl.col("fold_number") == fn).filter(pl.col("model") == m)[metric].mean() or float("nan"))
                    for m in models
                ]
                for fn in fold_nums
            ])  # shape: (n_folds, n_models)
            h_matrix = np.array([
                [
                    float(h_sub.filter(pl.col("fold_number") == fn).filter(pl.col("model") == m)[metric].mean() or float("nan"))
                    for m in models
                ]
                for fn in fold_nums
            ])

            t_mean = np.nanmean(t_matrix, axis=1)
            t_std  = np.nanstd(t_matrix, axis=1)
            h_mean = np.nanmean(h_matrix, axis=1)
            h_std  = np.nanstd(h_matrix, axis=1)

            ax.plot(fold_nums, t_mean, color="#4C72B0", linewidth=2.2, marker="o",
                    markersize=5, label="Test", zorder=4)
            ax.fill_between(fold_nums, t_mean - t_std, t_mean + t_std,
                            color="#4C72B0", alpha=0.18, zorder=3)

            ax.plot(fold_nums, h_mean, color="#DD8452", linewidth=2.2, marker="s",
                    markersize=5, linestyle="--", label="Holdout", zorder=4)
            ax.fill_between(fold_nums, h_mean - h_std, h_mean + h_std,
                            color="#DD8452", alpha=0.18, zorder=3)

            ax.set_xlabel("Fold Number (increasing = more recent)", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{metric.upper()}", fontsize=11)
            ax.set_title(_RUN_MODE_LABELS.get(run_mode, run_mode), fontsize=10, fontweight="bold")
            ax.set_xticks(fold_nums)
            ax.set_ylim(bottom=0.0)
            ax.legend(fontsize=9)
            sns.despine(ax=ax)

        ft_label = _short(ft)
        fig.suptitle(
            f"{metric.upper()} over Time - {ft_label}\n"
            f"Mean ± std across {len(models)} model(s); thin lines = individual models",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        figs[ft] = fig

    return figs


# Entry point


def main() -> None:
    parser = ArgumentParser(description="Test vs holdout performance contrast plots.")
    parser.add_argument("--results", type=Path, default=INDEX_RESULTS_FILE)
    parser.add_argument("--metric", dest="metrics", action="append",
                        choices=["f1", "precision", "recall", "accuracy"],
                        help="Metric(s) to plot; may be repeated. Default: f1 precision recall")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        print("Run `uv run scripts/collect_results.py` first.")
        sys.exit(1)

    metrics: list[str] = args.metrics or ["f1", "precision", "recall"]

    print(f"Loading results from {args.results} …")
    lf = load_results(args.results, tag=args.tag, dataset=args.dataset)

    print("Computing per-fold metrics …")
    test_df, holdout_df, delta_df = build_metric_frames(lf)

    # Filter to primary fold types present in the data
    present = set(test_df["fold_type"].unique().to_list())
    fold_types = [ft for ft in PRIMARY_FOLD_TYPES if ft in present]
    if not fold_types:
        print("No primary fold types found in data. Exiting.")
        sys.exit(0)

    test_df = test_df.filter(pl.col("fold_type").is_in(fold_types))
    holdout_df = holdout_df.filter(pl.col("fold_type").is_in(fold_types))
    delta_df = delta_df.filter(pl.col("fold_type").is_in(fold_types))

    print(f"Fold types : {[_short(ft) for ft in fold_types]}")
    print(f"Models     : {sorted(test_df['model'].unique().to_list())}")
    print(f"Metrics    : {metrics}")
    print()

    sns.set_theme(style="whitegrid", context="paper")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    families = {
        "fine_tuned": ("fine_tuning", "Fine-Tuned (LoRA)"),
        "zero_shot":  ("opro",        "OPRO (GGUF)"),
    }

    for family_key, (run_mode_val, family_label) in families.items():
        t = test_df.filter(pl.col("run_mode") == run_mode_val)
        h = holdout_df.filter(pl.col("run_mode") == run_mode_val)
        d = delta_df.filter(pl.col("run_mode") == run_mode_val)
        if t.is_empty():
            print(f"  No data for {family_label} - skipping.")
            continue

        ft_present = [ft for ft in fold_types if ft in t["fold_type"].unique().to_list()]
        print(f"\n  [{family_label}]  models: {sorted(t['model'].unique().to_list())}")

        for metric in metrics:
            print(f"    [{metric}]")
            _save(plot_bar(t, h, metric, ft_present, family_label),
                  PLOTS_DIR / f"bar_{metric}_{family_key}.png")
            _save(plot_scatter(t, h, metric, ft_present, family_label),
                  PLOTS_DIR / f"scatter_{metric}_{family_key}.png")
            _save(plot_delta(d, metric, ft_present, family_label),
                  PLOTS_DIR / f"delta_{metric}_{family_key}.png")

        # Generate LaTeX table for this family
        generate_contrast_latex_table(t, h, ft_present, family_key, PLOTS_DIR)

    # Generate combined contrast table (fine-tuning + zero-shot side-by-side)
    generate_combined_contrast_latex_table(test_df, holdout_df, fold_types, PLOTS_DIR)

    # Generate combined contrast table + last-fold rows for temporal strategies
    generate_combined_contrast_latex_table_with_last_fold(test_df, holdout_df, fold_types, PLOTS_DIR)

    # Temporal trend line charts (test vs holdout F1 over fold number / time)
    for metric in metrics:
        trend_figs = plot_temporal_trend(test_df, holdout_df, metric=metric)
        for ft, fig in trend_figs.items():
            slug = ft.replace("_cross_validation", "").replace("_", "-")
            _save(fig, PLOTS_DIR / f"temporal_trend_{metric}_{slug}.png")

    print(f"\nAll outputs written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
