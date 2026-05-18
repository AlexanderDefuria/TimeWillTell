#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Direct comparison of LoRA fine-tuning vs OPRO zero-shot for overlapping model families.

Generates a single figure with three stacked panels:
  Panel 1 - Holdout F1: FT vs OPRO (absolute)
  Panel 2 - Delta bars: approach advantage (FT−OPRO) + per-approach overfitting (test−holdout)
  Panel 3 - Relative inflation %: same two quantities expressed as percentages

Usage:
    uv run scripts/plot_approach_comparison.py
    uv run scripts/plot_approach_comparison.py --tag my-tag --dataset megavul
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
import numpy as np
import polars as pl

try:
    from scripts import EDA_DIR, INDEX_RESULTS_FILE, PRIMARY_FOLD_TYPES
    from scripts.plot_results import (
        apply_plot_model_exclusion,
        compute_metrics,
        format_fold_type_for_plot,
        load_results,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import EDA_DIR, INDEX_RESULTS_FILE, PRIMARY_FOLD_TYPES
    from scripts.plot_results import (
        apply_plot_model_exclusion,
        compute_metrics,
        format_fold_type_for_plot,
        load_results,
    )

# ─── Model pair registry ──────────────────────────────────────────────────────

# (display_label, ft_model_name, opro_model_name)
MODEL_PAIRS: list[tuple[str, str, str]] = [
    ("DeepSeek Coder", "deepseekcoder", "deepseekcoder-7B-gguf"),
    ("Mistral 3", "mistral3", "mistral3-7B-gguf"),
    ("Llama 3.2 3B", "llama3.2-3B", "llama3.2-3B-gguf"),
]

_ALL_PAIR_MODELS: list[str] = [m for _, ft, op in MODEL_PAIRS for m in (ft, op)]

# ─── Colour / style constants ─────────────────────────────────────────────────

C_FT = "#4477AA"      # blue - fine-tuning bars (panel 1)
C_OPRO = "#EE6677"    # red/salmon - OPRO bars (panel 1)

C_APPROACH = "#228B22"  # teal-green - FT−OPRO advantage
C_FT_OVER = "#CC3311"   # dark red - FT test−holdout
C_OP_OVER = "#EE7733"   # orange - OPRO test−holdout

BAR_WIDTH = 0.25
GROUP_GAP = 0.35   # extra gap between model-pair groups

_FOLD_LABELS: dict[str, str] = {
    "k_fold_cross_validation":                  "K-Fold",
    "random_loo_block_cross_validation":         "R-LOO",
    "random_growing_window_cross_validation":    "R-GW",
    "temporal_loo_block_cross_validation":       "T-LOO",
    "temporal_growing_window_cross_validation":  "T-GW",
}


# ─── Data helpers ─────────────────────────────────────────────────────────────


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except Exception:
        return lf.collect()


def load_and_compute(
    results_path: Path,
    tag: Optional[str],
    dataset: Optional[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (test_f1, holdout_f1) DataFrames aggregated per (model, fold_type)."""
    group_cols = ["model", "run_mode", "fold_type", "fold_number"]

    # Load test phase
    lf_test = (
        load_results(results_path, tag=tag, dataset=dataset)
        .filter(pl.col("phase") == "test")
        .filter(pl.col("model").is_in(_ALL_PAIR_MODELS))
    )
    test_raw = compute_metrics(_collect(lf_test), group_cols)
    test_agg = (
        test_raw
        .group_by(["model", "run_mode", "fold_type"])
        .agg(pl.col("f1").mean().alias("f1"), pl.col("f1").std().alias("f1_std"))
    )

    # Load holdout phase
    lf_holdout = (
        load_results(results_path, tag=tag, dataset=dataset)
        .filter(pl.col("phase") == "holdout")
        .filter(pl.col("model").is_in(_ALL_PAIR_MODELS))
    )
    holdout_raw = compute_metrics(_collect(lf_holdout), group_cols)
    holdout_agg = (
        holdout_raw
        .group_by(["model", "run_mode", "fold_type"])
        .agg(pl.col("f1").mean().alias("f1"), pl.col("f1").std().alias("f1_std"))
    )

    return test_agg, holdout_agg


def _get_f1(df: pl.DataFrame, model: str, fold_type: str) -> Optional[float]:
    rows = df.filter((pl.col("model") == model) & (pl.col("fold_type") == fold_type))
    if rows.is_empty():
        return None
    return float(rows["f1"][0])


def _get_f1_std(df: pl.DataFrame, model: str, fold_type: str) -> float:
    rows = df.filter((pl.col("model") == model) & (pl.col("fold_type") == fold_type))
    if rows.is_empty() or "f1_std" not in rows.columns:
        return 0.0
    val = rows["f1_std"][0]
    return float(val) if val is not None else 0.0


# ─── X-axis layout ────────────────────────────────────────────────────────────


def build_x_positions(
    pairs: list[tuple[str, str, str]],
    fold_types: list[str],
) -> tuple[list[float], list[str], list[float], list[str]]:
    """Return (tick_centers, tick_labels, group_centers, group_labels).

    Each (pair, fold_type) gets a centre position. A larger gap separates pairs.
    """
    n_folds = len(fold_types)
    tick_centers: list[float] = []
    tick_labels: list[str] = []
    group_centers: list[float] = []
    group_labels: list[str] = []

    cursor = 0.0
    slot_width = 3 * BAR_WIDTH + 0.1  # room for 3 bars + small intra-slot gap

    for pair_label, _, _ in pairs:
        group_start = cursor
        for ft in fold_types:
            tick_centers.append(cursor + slot_width / 2)
            tick_labels.append(_FOLD_LABELS.get(ft, format_fold_type_for_plot(ft)))
            cursor += slot_width
        group_end = cursor
        group_centers.append((group_start + group_end) / 2)
        group_labels.append(pair_label)
        cursor += GROUP_GAP  # extra gap between model-pair groups

    return tick_centers, tick_labels, group_centers, group_labels


# ─── Panel drawing ────────────────────────────────────────────────────────────


def _draw_panel1(
    ax: plt.Axes,
    pairs: list[tuple[str, str, str]],
    fold_types: list[str],
    holdout: pl.DataFrame,
    tick_centers: list[float],
    fs: int = 10,
) -> None:
    """Holdout F1: FT (blue) vs OPRO (orange), 2 bars per x-tick."""
    ft_vals: list[float] = []
    op_vals: list[float] = []
    ft_stds: list[float] = []
    op_stds: list[float] = []

    for _, ft_model, op_model in pairs:
        for ft in fold_types:
            ft_vals.append(_get_f1(holdout, ft_model, ft) or 0.0)
            op_vals.append(_get_f1(holdout, op_model, ft) or 0.0)
            ft_stds.append(_get_f1_std(holdout, ft_model, ft))
            op_stds.append(_get_f1_std(holdout, op_model, ft))

    offset = BAR_WIDTH / 2
    xs = np.array(tick_centers)
    ax.bar(xs - offset, ft_vals, BAR_WIDTH, color=C_FT, label="Fine-tuning (LoRA)", zorder=3)
    ax.bar(xs + offset, op_vals, BAR_WIDTH, color=C_OPRO, label="OPRO (zero-shot)", zorder=3)
    ax.errorbar(xs - offset, ft_vals, yerr=ft_stds, fmt="none", color="black", capsize=3, linewidth=1.0, zorder=4)
    ax.errorbar(xs + offset, op_vals, yerr=op_stds, fmt="none", color="black", capsize=3, linewidth=1.0, zorder=4)

    ax.axhline(0.5, color="grey", lw=0.8, ls="--", zorder=2)
    ax.set_ylabel("F1 (holdout)", fontsize=fs)
    ax.set_ylim(0, 0.5)
    ax.set_title("Holdout F1 - Fine-tuning vs OPRO", fontsize=fs + 1, fontweight="bold")
    ax.legend(fontsize=fs - 1, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
              title="error bars = ±1 std\nacross folds", title_fontsize=fs - 2)
    ax.grid(axis="y", alpha=0.3, zorder=1)


def _draw_panel2(
    ax: plt.Axes,
    pairs: list[tuple[str, str, str]],
    fold_types: list[str],
    test: pl.DataFrame,
    holdout: pl.DataFrame,
    tick_centers: list[float],
    fs: int = 10,
) -> None:
    """Delta bars: FT−OPRO advantage, FT overfitting, OPRO overfitting."""
    approach_delta: list[float] = []
    ft_overfit: list[float] = []
    op_overfit: list[float] = []
    approach_delta_stds: list[float] = []
    ft_overfit_stds: list[float] = []
    op_overfit_stds: list[float] = []

    for _, ft_model, op_model in pairs:
        for ft in fold_types:
            ft_h = _get_f1(holdout, ft_model, ft) or 0.0
            op_h = _get_f1(holdout, op_model, ft) or 0.0
            ft_t = _get_f1(test, ft_model, ft) or 0.0
            op_t = _get_f1(test, op_model, ft) or 0.0
            s_ft_h = _get_f1_std(holdout, ft_model, ft)
            s_op_h = _get_f1_std(holdout, op_model, ft)
            s_ft_t = _get_f1_std(test, ft_model, ft)
            s_op_t = _get_f1_std(test, op_model, ft)

            approach_delta.append(ft_h - op_h)
            ft_overfit.append(ft_t - ft_h)
            op_overfit.append(op_t - op_h)
            approach_delta_stds.append(float(np.sqrt(s_ft_h ** 2 + s_op_h ** 2)))
            ft_overfit_stds.append(float(np.sqrt(s_ft_t ** 2 + s_ft_h ** 2)))
            op_overfit_stds.append(float(np.sqrt(s_op_t ** 2 + s_op_h ** 2)))

    xs = np.array(tick_centers)
    ax.bar(xs - BAR_WIDTH, approach_delta, BAR_WIDTH, color=C_APPROACH,
           label="FT − OPRO (approach advantage)", zorder=3)
    ax.bar(xs, ft_overfit, BAR_WIDTH, color=C_FT_OVER, hatch="//",
           label="FT: test − holdout (overfit)", zorder=3)
    ax.bar(xs + BAR_WIDTH, op_overfit, BAR_WIDTH, color=C_OP_OVER, hatch="\\\\",
           label="OPRO: test − holdout (overfit)", zorder=3)

    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=2)
    ax.set_ylabel("ΔF1", fontsize=fs)
    ax.set_title("F1 Deltas - Approach Advantage & Overfitting", fontsize=fs + 1, fontweight="bold")
    ax.legend(fontsize=fs - 1, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(axis="y", alpha=0.3, zorder=1)


def _draw_panel3(
    ax: plt.Axes,
    pairs: list[tuple[str, str, str]],
    fold_types: list[str],
    test: pl.DataFrame,
    holdout: pl.DataFrame,
    tick_centers: list[float],
    fs: int = 10,
) -> None:
    """Relative inflation %: FT gain over OPRO, FT inflation, OPRO inflation."""
    approach_pct: list[float] = []
    ft_infl: list[float] = []
    op_infl: list[float] = []
    approach_pct_stds: list[float] = []
    ft_infl_stds: list[float] = []
    op_infl_stds: list[float] = []

    for _, ft_model, op_model in pairs:
        for ft in fold_types:
            ft_h = _get_f1(holdout, ft_model, ft) or 0.0
            op_h = _get_f1(holdout, op_model, ft) or 0.0
            ft_t = _get_f1(test, ft_model, ft) or 0.0
            op_t = _get_f1(test, op_model, ft) or 0.0
            s_ft_h = _get_f1_std(holdout, ft_model, ft)
            s_op_h = _get_f1_std(holdout, op_model, ft)
            s_ft_t = _get_f1_std(test, ft_model, ft)
            s_op_t = _get_f1_std(test, op_model, ft)

            approach_pct.append((ft_h / op_h - 1) * 100 if op_h > 0 else 0.0)
            ft_infl.append((ft_t / ft_h - 1) * 100 if ft_h > 0 else 0.0)
            op_infl.append((op_t / op_h - 1) * 100 if op_h > 0 else 0.0)

            if op_h > 0:
                approach_pct_stds.append(100.0 * float(np.sqrt((s_ft_h / op_h) ** 2 + (ft_h * s_op_h / op_h ** 2) ** 2)))
            else:
                approach_pct_stds.append(0.0)
            if ft_h > 0:
                ft_infl_stds.append(100.0 * float(np.sqrt((s_ft_t / ft_h) ** 2 + (ft_t * s_ft_h / ft_h ** 2) ** 2)))
            else:
                ft_infl_stds.append(0.0)
            if op_h > 0:
                op_infl_stds.append(100.0 * float(np.sqrt((s_op_t / op_h) ** 2 + (op_t * s_op_h / op_h ** 2) ** 2)))
            else:
                op_infl_stds.append(0.0)

    xs = np.array(tick_centers)
    ax.bar(xs - BAR_WIDTH, approach_pct, BAR_WIDTH, color=C_APPROACH,
           label="(FT / OPRO − 1) × 100%", zorder=3)
    ax.bar(xs, ft_infl, BAR_WIDTH, color=C_FT_OVER, hatch="//",
           label="FT: (test / holdout − 1) × 100%", zorder=3)
    ax.bar(xs + BAR_WIDTH, op_infl, BAR_WIDTH, color=C_OP_OVER, hatch="\\\\",
           label="OPRO: (test / holdout − 1) × 100%", zorder=3)

    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=2)
    ax.set_ylabel("Relative change (%)", fontsize=fs)
    ax.set_ylim(0, 140)
    ax.set_title("Relative Inflation - Approach Gain & Overfitting %", fontsize=fs + 1, fontweight="bold")
    ax.legend(fontsize=fs - 1, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(axis="y", alpha=0.3, zorder=1)


def _apply_x_axis(
    ax: plt.Axes,
    tick_centers: list[float],
    tick_labels: list[str],
    group_centers: list[float],
    group_labels: list[str],
    is_bottom: bool,
    fs: int = 7,
) -> None:
    """Set x-ticks for fold types; add model-pair group labels as a secondary axis."""
    ax.set_xticks(tick_centers)
    if is_bottom:
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=fs)
    else:
        ax.set_xticklabels([], fontsize=0)

    # Draw vertical separators at pair group boundaries
    n_folds = len(tick_labels) // len(group_labels)
    slot_width = 3 * BAR_WIDTH + 0.1
    for i in range(1, len(group_labels)):
        sep_x = group_centers[i] - (n_folds * slot_width + GROUP_GAP) / 2
        ax.axvline(sep_x, color="black", lw=0.6, ls=":", zorder=4)

    # Group label annotations inside top of each panel (white box avoids title overlap)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for cx, lbl in zip(group_centers, group_labels):
        ax.text(cx, 0.97, lbl, transform=trans,
                ha="center", va="top", fontsize=fs + 2, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"))


# ─── Individual panel export ──────────────────────────────────────────────────


def _save_individual_panels(
    out_dir: Path,
    pairs: list[tuple[str, str, str]],
    fold_types: list[str],
    test_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    tick_centers: list[float],
    tick_labels: list[str],
    group_centers: list[float],
    group_labels: list[str],
) -> None:
    """Save each panel as a standalone figure with full x-axis labels."""
    fs = 11
    panels = [
        (
            "panel1_holdout_f1",
            lambda ax: _draw_panel1(ax, pairs, fold_types, holdout_df, tick_centers, fs=fs),
        ),
        (
            "panel2_deltas",
            lambda ax: _draw_panel2(ax, pairs, fold_types, test_df, holdout_df, tick_centers, fs=fs),
        ),
        (
            "panel3_relative_inflation",
            lambda ax: _draw_panel3(ax, pairs, fold_types, test_df, holdout_df, tick_centers, fs=fs),
        ),
    ]
    xlim_pad = 0.5
    for slug, draw_fn in panels:
        fig, ax = plt.subplots(1, 1, figsize=(16, 3.0), constrained_layout=True)
        draw_fn(ax)
        _apply_x_axis(ax, tick_centers, tick_labels, group_centers, group_labels, is_bottom=True, fs=fs)
        ax.set_xlim(tick_centers[0] - xlim_pad, tick_centers[-1] + xlim_pad)
        out_path = out_dir / f"approach_comparison_{slug}.png"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_path}")


# ─── LaTeX table helpers ──────────────────────────────────────────────────────


def _f1_cell(val: Optional[float]) -> str:
    if val is None:
        return "---"
    t = max(0.0, min(1.0, val))
    r = int(round(255 + t * (0x44 - 255)))
    g = int(round(255 + t * (0x77 - 255)))
    b = int(round(255 + t * (0xAA - 255)))
    return f"\\cellcolor[HTML]{{{r:02X}{g:02X}{b:02X}}} {val:.2f}"


def _delta_cell(val: Optional[float], scale: float = 0.3) -> str:
    if val is None:
        return "---"
    t = max(-1.0, min(1.0, val / scale))
    if t < 0:
        r, g, b = 229, int(round(229 + t * (229 - 87))), int(round(229 + t * (229 - 87)))
    else:
        r, g, b = int(round(229 - t * (229 - 34))), int(round(229 + t * (139 - 229))), int(round(229 - t * (229 - 34)))
    return f"\\cellcolor[HTML]{{{r:02X}{g:02X}{b:02X}}} {val:+.3f}"


def _pct_cell(val: Optional[float], scale: float = 50.0) -> str:
    if val is None:
        return "---"
    t = max(-1.0, min(1.0, val / scale))
    if t < 0:
        r, g, b = 229, int(round(229 + t * (229 - 87))), int(round(229 + t * (229 - 87)))
    else:
        r, g, b = int(round(229 - t * (229 - 34))), int(round(229 + t * (139 - 229))), int(round(229 - t * (229 - 34)))
    return f"\\cellcolor[HTML]{{{r:02X}{g:02X}{b:02X}}} {val:+.1f}\\%"


def _wrap_latex_table(col_spec: str, header: str, body: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\footnotesize\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        f"{header}"
        "    \\midrule\n"
        f"{body}"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def _save_tables(
    out_dir: Path,
    pairs: list[tuple[str, str, str]],
    fold_types: list[str],
    test_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
) -> None:
    """Write three booktabs LaTeX tables matching the three plot panels."""
    n = len(pairs)
    fold_labels = [_FOLD_LABELS.get(ft, ft) for ft in fold_types]

    def _pair_headers(cols_per_group: int, metric_names: list[str]) -> str:
        pair_row = "    "
        metric_row = "    Fold Type"
        for i, (pair_lbl, _, _) in enumerate(pairs):
            span_fmt = "c|" if i < n - 1 else "c"
            pair_row += f" & \\multicolumn{{{cols_per_group}}}{{{span_fmt}}}{{{pair_lbl}}}"
            for m in metric_names:
                metric_row += f" & {m}"
        return pair_row + " \\\\\n" + metric_row + " \\\\\n"

    # ── Table 1: Holdout F1 ───────────────────────────────────────────────────
    col_spec1 = "l | " + " | ".join("cc" for _ in pairs)
    header1 = _pair_headers(2, ["FT", "OPRO"])
    body1 = ""
    for ft, lbl in zip(fold_types, fold_labels):
        row = f"    {lbl}"
        for _, ft_model, op_model in pairs:
            row += f" & {_f1_cell(_get_f1(holdout_df, ft_model, ft))}"
            row += f" & {_f1_cell(_get_f1(holdout_df, op_model, ft))}"
        body1 += row + " \\\\\n"
    out1 = out_dir / "approach_comparison_table1_holdout_f1.tex"
    out1.write_text(_wrap_latex_table(
        col_spec1, header1, body1,
        caption="Holdout F1 --- Fine-tuning vs OPRO",
        label="tab:approach_comparison_holdout_f1",
    ))
    print(f"Saved → {out1}")

    # ── Table 2: F1 Deltas ────────────────────────────────────────────────────
    col_spec23 = "l | " + " | ".join("ccc" for _ in pairs)
    header2 = _pair_headers(3, ["Adv.", "FT-O", "OP-O"])
    body2 = ""
    for ft, lbl in zip(fold_types, fold_labels):
        row = f"    {lbl}"
        for _, ft_model, op_model in pairs:
            ft_h = _get_f1(holdout_df, ft_model, ft)
            op_h = _get_f1(holdout_df, op_model, ft)
            ft_t = _get_f1(test_df, ft_model, ft)
            op_t = _get_f1(test_df, op_model, ft)
            adv    = (ft_h - op_h) if ft_h is not None and op_h is not None else None
            ft_ov  = (ft_t - ft_h) if ft_t is not None and ft_h is not None else None
            op_ov  = (op_t - op_h) if op_t is not None and op_h is not None else None
            row += f" & {_delta_cell(adv)} & {_delta_cell(ft_ov)} & {_delta_cell(op_ov)}"
        body2 += row + " \\\\\n"
    out2 = out_dir / "approach_comparison_table2_deltas.tex"
    out2.write_text(_wrap_latex_table(
        col_spec23, header2, body2,
        caption="F1 Deltas --- Approach Advantage \\& Overfitting",
        label="tab:approach_comparison_deltas",
    ))
    print(f"Saved → {out2}")

    # ── Table 3: Relative inflation % ─────────────────────────────────────────
    header3 = _pair_headers(3, ["Adv.\\%", "FT-O\\%", "OP-O\\%"])
    body3 = ""
    for ft, lbl in zip(fold_types, fold_labels):
        row = f"    {lbl}"
        for _, ft_model, op_model in pairs:
            ft_h = _get_f1(holdout_df, ft_model, ft)
            op_h = _get_f1(holdout_df, op_model, ft)
            ft_t = _get_f1(test_df, ft_model, ft)
            op_t = _get_f1(test_df, op_model, ft)
            adv   = ((ft_h / op_h - 1) * 100) if ft_h is not None and op_h is not None and op_h > 0 else None
            ft_in = ((ft_t / ft_h - 1) * 100) if ft_t is not None and ft_h is not None and ft_h > 0 else None
            op_in = ((op_t / op_h - 1) * 100) if op_t is not None and op_h is not None and op_h > 0 else None
            row += f" & {_pct_cell(adv)} & {_pct_cell(ft_in)} & {_pct_cell(op_in)}"
        body3 += row + " \\\\\n"
    out3 = out_dir / "approach_comparison_table3_relative_inflation.tex"
    out3.write_text(_wrap_latex_table(
        col_spec23, header3, body3,
        caption="Relative Inflation --- Approach Gain \\& Overfitting (\\%)",
        label="tab:approach_comparison_relative_inflation",
    ))
    print(f"Saved → {out3}")

    # ── Table 4: Avg test-holdout gap - Random vs Temporal ───────────────────
    _RANDOM_FTS = [
        "random_loo_block_cross_validation",
        "random_growing_window_cross_validation",
    ]
    _TEMPORAL_FTS = [
        "temporal_loo_block_cross_validation",
        "temporal_growing_window_cross_validation",
    ]

    def _mean_overfit(ft_type: str, ft_models: list[str]) -> Optional[float]:
        vals = [
            (t / h - 1) * 100
            for m in ft_models
            for h, t in [(_get_f1(holdout_df, m, ft_type), _get_f1(test_df, m, ft_type))]
            if h is not None and t is not None and h > 0
        ]
        return sum(vals) / len(vals) if vals else None

    def _group_avg(ft_types: list[str], ft_models: list[str]) -> Optional[float]:
        vals = [v for ft in ft_types for v in [_mean_overfit(ft, ft_models)] if v is not None]
        return sum(vals) / len(vals) if vals else None

    all_ft_models  = [ft_model  for _, ft_model,  _        in pairs]
    all_op_models  = [op_model  for _, _,          op_model in pairs]

    _T4_FOLDS = [
        ("k_fold_cross_validation",             "K-Fold"),
        ("random_loo_block_cross_validation",    "R-LOO"),
        ("random_growing_window_cross_validation", "R-GW"),
        ("temporal_loo_block_cross_validation",  "T-LOO"),
        ("temporal_growing_window_cross_validation", "T-GW"),
    ]

    col_spec4 = "l | c | c c | c c | c c"
    header4 = (
        "    Approach"
        + "".join(f" & {lbl}" for _, lbl in _T4_FOLDS)
        + " & Rand avg & Temp avg \\\\\n"
    )

    def _approach_row(approach_label: str, models: list[str]) -> str:
        cells = "".join(
            f" & {_pct_cell(_mean_overfit(ft, models))}"
            for ft, _ in _T4_FOLDS
        )
        rand_avg = _pct_cell(_group_avg(_RANDOM_FTS, models))
        temp_avg = _pct_cell(_group_avg(_TEMPORAL_FTS, models))
        return f"    {approach_label}{cells} & {rand_avg} & {temp_avg} \\\\\n"

    body4 = (
        _approach_row("FT", all_ft_models)
        + _approach_row("OPRO", all_op_models)
    )

    out4 = out_dir / "approach_comparison_table4_avg_overfit.tex"
    out4.write_text(_wrap_latex_table(
        col_spec4, header4, body4,
        caption="Average Test/Holdout F1 Inflation (\\%) by Fold Type (mean across paired models: DeepSeek Coder, Mistral 3, Llama 3.2 3B --- models available in both fine-tuned and OPRO variants)",
        label="tab:approach_comparison_avg_overfit",
    ))
    print(f"Saved → {out4}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = ArgumentParser(description="Plot FT vs OPRO approach comparison.")
    parser.add_argument("--results", type=Path, default=INDEX_RESULTS_FILE)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=Path, default=EDA_DIR)
    args = parser.parse_args()

    print(f"Loading results from {args.results} …")
    test_df, holdout_df = load_and_compute(args.results, args.tag, args.dataset)

    fold_types = PRIMARY_FOLD_TYPES
    tick_centers, tick_labels, group_centers, group_labels = build_x_positions(
        MODEL_PAIRS, fold_types
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("Fine-tuning vs OPRO - Direct Approach Comparison", fontsize=13, fontweight="bold")

    _draw_panel1(axes[0], MODEL_PAIRS, fold_types, holdout_df, tick_centers)
    _draw_panel2(axes[1], MODEL_PAIRS, fold_types, test_df, holdout_df, tick_centers)
    _draw_panel3(axes[2], MODEL_PAIRS, fold_types, test_df, holdout_df, tick_centers)

    for i, ax in enumerate(axes):
        _apply_x_axis(
            ax, tick_centers, tick_labels, group_centers, group_labels,
            is_bottom=(i == len(axes) - 1),
        )
        xlim_pad = 0.5
        ax.set_xlim(tick_centers[0] - xlim_pad, tick_centers[-1] + xlim_pad)

    out_path = args.output / "approach_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    _save_individual_panels(
        args.output, MODEL_PAIRS, fold_types, test_df, holdout_df,
        tick_centers, tick_labels, group_centers, group_labels,
    )
    _save_tables(args.output, MODEL_PAIRS, fold_types, test_df, holdout_df)


if __name__ == "__main__":
    main()
