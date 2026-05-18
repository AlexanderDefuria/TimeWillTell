#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Per-(model, run_mode) TOST equivalence analysis.

Replaces the cross-model TOST in plot_pvalue_summary.py. Instead of collapsing
each model to one mean holdout F1 and running n=#models paired TOST, this
module runs a paired TOST per (model, run_mode) using the per-fold F1 series
as the paired observations (aligned by fold_number).

Two Holm-Bonferroni corrections are produced in separate functions:

  - run_per_group_tost()    Holm within each (model, run_mode) family of <=5
                            fold-type pair tests. Answers: "is THIS model
                            equivalent across all pair comparisons?"
  - run_per_run_mode_tost() Holm pooled across all (model x pair) tests inside
                            one run_mode. Answers: "across this entire
                            run_mode family, is equivalence supported?"

See docs/tost_grouped.md for the rationale, math, and how to interpret outputs.

Usage:
    uv run scripts/plot_tost_grouped.py
    uv run scripts/plot_tost_grouped.py --metric f1 --tag fine-tuning --dataset megavul
"""
from __future__ import annotations

import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from scipy import stats as _scipy_stats
from statsmodels.stats.multitest import multipletests

try:
    from scripts import EDA_DIR, INDEX_RESULTS_FILE
    from scripts.plot_results import METRICS, load_results
    from scripts.plot_statistical_tests import MIN_PAIRED_FOLDS, prepare_per_fold_metrics
    from scripts.plot_pvalue_summary import (
        TOST_ALPHA,
        TOST_DELTA,
        TOST_PAIRS,
        RANDOM_PAIRS,
        TEMPORAL_PAIRS,
        CROSS_PAIRS,
        _fmt_pvalue,
        _resolve_fold_type,
        paired_tost,
        extract_mean_f1_matrix,
        run_intragroup_pairs,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import EDA_DIR, INDEX_RESULTS_FILE
    from scripts.plot_results import METRICS, load_results
    from scripts.plot_statistical_tests import MIN_PAIRED_FOLDS, prepare_per_fold_metrics
    from scripts.plot_pvalue_summary import (
        TOST_ALPHA,
        TOST_DELTA,
        TOST_PAIRS,
        RANDOM_PAIRS,
        TEMPORAL_PAIRS,
        CROSS_PAIRS,
        _fmt_pvalue,
        _resolve_fold_type,
        paired_tost,
        extract_mean_f1_matrix,
        run_intragroup_pairs,
    )


# ─── Constants ───────────────────────────────────────────────────────────────

OUTPUT_DIR = EDA_DIR / "statistical_tests"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "data" / "plots" / "6_pvalue_comparison"

RUN_MODE_FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    ("finetuned", ("fine_tuning",)),
    ("gguf",      ("opro", "zero_shot")),
]

_FAMILY_DISPLAY = {"finetuned": "Fine-tuned (LoRA)", "gguf": "Zero-shot (GGUF)"}


# ─── Core: per (model, run_mode, pair) paired TOST ───────────────────────────


def _per_group_pair_results(
    metrics_df: pl.DataFrame,
    metric: str = "f1",
) -> list[dict]:
    """Return one row per (model, run_mode, fold-type pair) - no correction yet.

    For each (model, run_mode) group and each (short_a, short_b) in TOST_PAIRS:
      - Inner-join fold_type A and B rows on fold_number.
      - Skip when either fold_type is absent for the group, or when the paired
        observation count is below MIN_PAIRED_FOLDS.
      - Run paired_tost on the aligned arrays (converted to percentage points).
    """
    if metrics_df.is_empty():
        return []

    rows: list[dict] = []
    groups = (
        metrics_df.select(["model", "run_mode"])
        .unique()
        .sort(["run_mode", "model"])
    )

    for grp in groups.iter_rows(named=True):
        model = grp["model"]
        run_mode = grp["run_mode"]
        sub = metrics_df.filter(
            (pl.col("model") == model) & (pl.col("run_mode") == run_mode)
        )
        available = sub["fold_type"].unique().to_list()

        for short_a, short_b in TOST_PAIRS:
            ft_a = _resolve_fold_type(short_a, available)
            ft_b = _resolve_fold_type(short_b, available)
            if ft_a is None or ft_b is None:
                continue

            vals_a = (
                sub.filter(pl.col("fold_type") == ft_a)
                .select(["fold_number", metric])
                .sort("fold_number")
            )
            vals_b = (
                sub.filter(pl.col("fold_type") == ft_b)
                .select(["fold_number", metric])
                .sort("fold_number")
            )
            paired = vals_a.join(vals_b, on="fold_number", suffix="_b")
            if paired.height < MIN_PAIRED_FOLDS:
                continue

            a = paired[metric].to_numpy().astype(float) * 100.0
            b = paired[f"{metric}_b"].to_numpy().astype(float) * 100.0
            mean_d, ci_lo, ci_hi, p_lower, p_upper, p_tost = paired_tost(a, b)
            # Paired two-sided t-test on the same diff vector - answers
            # "is the per-fold difference significantly non-zero?", which is
            # complementary to the equivalence claim from TOST.
            diff = a - b
            if np.std(diff, ddof=1) < 1e-12:
                p_diff = 1.0
            else:
                p_diff = float(_scipy_stats.ttest_rel(a, b).pvalue)

            rows.append({
                "model":     model,
                "run_mode":  run_mode,
                "short_a":   short_a,
                "short_b":   short_b,
                "ft_a":      ft_a,
                "ft_b":      ft_b,
                "label":     f"{short_a} vs {short_b}",
                "n":         paired.height,
                "mean_d":    mean_d,
                "ci_lo":     ci_lo,
                "ci_hi":     ci_hi,
                "p_lower":   p_lower,
                "p_upper":   p_upper,
                "p_tost":    p_tost,
                "p_diff":    p_diff,
            })

    return rows


def _holm_in_place(rows: list[dict], group_key: str | tuple[str, ...] | None) -> None:
    """Apply Holm-Bonferroni to p_tost and p_diff within groups defined by group_key.

    group_key=None applies one Holm correction over the entire `rows` list.
    Each row gets `p_tost_corrected`, `equivalent`, `p_diff_corrected`,
    `significant_diff` set.
    """
    if not rows:
        return

    if group_key is None:
        groups = {"__all__": list(range(len(rows)))}
    else:
        keys: tuple[str, ...]
        keys = (group_key,) if isinstance(group_key, str) else tuple(group_key)
        groups = {}
        for i, r in enumerate(rows):
            k = tuple(r[k_] for k_ in keys)
            groups.setdefault(k, []).append(i)

    for indices in groups.values():
        p_tost_values = [rows[i]["p_tost"] for i in indices]
        _, p_tost_corr, _, _ = multipletests(p_tost_values, alpha=TOST_ALPHA, method="holm")
        p_diff_values = [rows[i].get("p_diff", float("nan")) for i in indices]
        _, p_diff_corr, _, _ = multipletests(p_diff_values, alpha=TOST_ALPHA, method="holm")
        for j, i in enumerate(indices):
            rows[i]["p_tost_corrected"] = float(p_tost_corr[j])
            rows[i]["equivalent"] = bool(p_tost_corr[j] < TOST_ALPHA)
            rows[i]["p_diff_corrected"] = float(p_diff_corr[j])
            rows[i]["significant_diff"] = bool(p_diff_corr[j] < TOST_ALPHA)


# ─── Holm scopes (the two functions the user asked for) ──────────────────────


def run_per_group_tost(
    metrics_df: pl.DataFrame,
    metric: str = "f1",
) -> list[dict]:
    """Holm-correct within each (model, run_mode) family of <=5 pair tests."""
    rows = _per_group_pair_results(metrics_df, metric=metric)
    _holm_in_place(rows, group_key=("model", "run_mode"))
    return rows


def run_per_run_mode_tost(
    metrics_df: pl.DataFrame,
    metric: str = "f1",
) -> list[dict]:
    """Holm-correct pooled across all (model x pair) tests within each run_mode."""
    rows = _per_group_pair_results(metrics_df, metric=metric)
    _holm_in_place(rows, group_key="run_mode")
    return rows


# ─── Writers: CSV ────────────────────────────────────────────────────────────


_CSV_HEADER = (
    "model,run_mode,short_a,short_b,ft_a,ft_b,n,"
    "mean_d,ci_lo,ci_hi,p_lower,p_upper,p_tost,p_tost_corrected,equivalent,"
    "p_diff,p_diff_corrected,significant_diff\n"
)


def _row_to_csv(r: dict) -> str:
    return (
        f"{r['model']},{r['run_mode']},{r['short_a']},{r['short_b']},"
        f"\"{r['ft_a']}\",\"{r['ft_b']}\",{r['n']},"
        f"{r['mean_d']:.6f},{r['ci_lo']:.6f},{r['ci_hi']:.6f},"
        f"{r['p_lower']:.6e},{r['p_upper']:.6e},{r['p_tost']:.6e},"
        f"{r['p_tost_corrected']:.6e},{r['equivalent']},"
        f"{r['p_diff']:.6e},{r['p_diff_corrected']:.6e},{r['significant_diff']}\n"
    )


def _write_csv(rows: list[dict], path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(_CSV_HEADER)
        for r in rows:
            fh.write(_row_to_csv(r))
    return path


def write_per_group_csv(rows: list[dict], family_label: str) -> Path:
    return _write_csv(rows, OUTPUT_DIR / f"{family_label}_tost_per_group_holdout.csv")


def write_per_run_mode_csv(rows: list[dict], family_label: str) -> Path:
    return _write_csv(rows, OUTPUT_DIR / f"{family_label}_tost_per_run_mode_holdout.csv")


# ─── Writers: LaTeX ──────────────────────────────────────────────────────────


def _latex_table(
    rows: list[dict],
    family_label: str,
    holm_scope_desc: str,
    label: str,
) -> str:
    family_display = _FAMILY_DISPLAY.get(family_label, family_label)
    lines = [
        "% Auto-generated by scripts/plot_tost_grouped.py - do not edit manually.",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\footnotesize",
        "  \\begin{tabular}{llrrrrlrl}",
        "    \\toprule",
        "    Model & Comparison & $n$ & $\\bar{d}$ (pp) & 90\\% CI "
        "& $p_{\\text{TOST}}$ & Equiv. "
        "& $p_{\\text{diff}}$ & Sig.\\,diff \\\\",
        "    \\midrule",
    ]
    last_model: str | None = None
    for r in rows:
        model_cell = "" if r["model"] == last_model else r["model"].replace("_", "\\_")
        last_model = r["model"]
        ci_str = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        equiv_str = "Yes" if r["equivalent"] else "No"
        sig_str = "Yes" if r["significant_diff"] else "No"
        lines.append(
            f"    {model_cell} & {r['label']} & {r['n']} & {r['mean_d']:+.2f} "
            f"& {ci_str} & {_fmt_pvalue(r['p_tost_corrected'])} & {equiv_str} "
            f"& {_fmt_pvalue(r['p_diff_corrected'])} & {sig_str} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{TOST equivalence tests on \\textbf{{holdout}} F1 ({family_display}), "
        f"paired-by-fold within each (model, run\\_mode). {holm_scope_desc} "
        f"$\\bar{{d}}$ is the mean per-fold difference (fold-type-A $-$ fold-type-B, pp); "
        f"90\\% CI corresponds to $\\alpha = {TOST_ALPHA}$ TOST. "
        f"Equiv.: $p_{{\\text{{TOST}}}} < {TOST_ALPHA}$, $\\Delta = {TOST_DELTA}$\\,pp. "
        f"$p_{{\\text{{diff}}}}$ is the Holm-corrected two-sided paired $t$-test p-value "
        f"on the same per-fold difference vector (Sig.\\,diff: $p_{{\\text{{diff}}}} < {TOST_ALPHA}$); "
        f"it answers whether the mean difference is significantly non-zero, complementary to "
        f"the equivalence claim. "
        f"K-Fold vs Rand LOO is the RQ3 inter-commit-leakage comparison (not random vs temporal).}}",
        f"  \\label{{{label}}}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _write_latex(rows: list[dict], path: Path, latex: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(latex)
    return path


def write_per_group_latex(rows: list[dict], family_label: str) -> Path:
    latex = _latex_table(
        rows,
        family_label,
        holm_scope_desc=(
            "Holm--Bonferroni correction applied within each (model, run\\_mode) "
            "family of pair tests."
        ),
        label=f"tab:tost_per_group_{family_label}",
    )
    out = PLOTS_DIR / f"tost_per_group_table_{family_label}.tex"
    return _write_latex(rows, out, latex)


def write_per_run_mode_latex(rows: list[dict], family_label: str) -> Path:
    latex = _latex_table(
        rows,
        family_label,
        holm_scope_desc=(
            "Holm--Bonferroni correction pooled across all (model $\\times$ pair) "
            "tests inside the run\\_mode family."
        ),
        label=f"tab:tost_per_run_mode_{family_label}",
    )
    out = PLOTS_DIR / f"tost_per_run_mode_table_{family_label}.tex"
    return _write_latex(rows, out, latex)


# ─── Writers: forest plots ───────────────────────────────────────────────────


def _plot_forest(rows: list[dict], path: Path, title: str) -> Path:
    if not rows:
        return path

    labels = [f"{r['model']} - {r['label']}" for r in rows]
    means = [r["mean_d"] for r in rows]
    lo    = [r["ci_lo"]  for r in rows]
    hi    = [r["ci_hi"]  for r in rows]
    equiv = [r["equivalent"] for r in rows]

    n = len(labels)
    y_pos = list(range(n))
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * n + 1.5)))

    for i, (m, l, h, eq) in enumerate(zip(means, lo, hi, equiv)):
        color = "#2166ac" if eq else "#d6604d"
        ax.errorbar(
            m, i,
            xerr=[[m - l], [h - m]],
            fmt="o", color=color, ecolor=color,
            capsize=4, markersize=6, linewidth=1.5,
        )

    ax.axvline(-TOST_DELTA, color="black", linestyle="--", linewidth=1.2)
    ax.axvline( TOST_DELTA, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
               markersize=8, label=f"Equivalent (TOST, Δ={TOST_DELTA} pp)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d6604d",
               markersize=8, label="Not equivalent"),
        Line2D([0], [0], color="black", linestyle="--", label=f"±{TOST_DELTA} pp margin"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Per-fold F1 difference (fold-type-A − fold-type-B, pp)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_group_forest(rows: list[dict], family_label: str) -> Path:
    title = (
        f"TOST per (model, run_mode) - Holdout F1 ({_FAMILY_DISPLAY.get(family_label, family_label)}, "
        f"Holm within each model)"
    )
    out = PLOTS_DIR / f"tost_per_group_forest_{family_label}.png"
    return _plot_forest(rows, out, title)


def plot_summary_forest(
    rows_by_family: dict[str, list[dict]],
    metric_label: str = "F1",
) -> Path | None:
    """One summary figure: for each family, one row per fold-type pair whose
    point estimate is the mean across (model, run_mode) of per-group mean_d,
    and whose error bar is the mean's 95% CI (t-distribution, n=#models).

    This is the natural average of the per-(model, run_mode) forest plots:
    every K-Fold-vs-Temp-LOO point in those plots collapses into the one
    summary row for that pair within the family.
    """
    if not rows_by_family:
        return None

    families = [fl for fl, _ in RUN_MODE_FAMILIES if fl in rows_by_family and rows_by_family[fl]]
    if not families:
        return None

    fig, axes = plt.subplots(
        1, len(families),
        figsize=(7 * len(families), max(3.5, 0.7 * len(TOST_PAIRS) + 1.5)),
        sharex=True,
    )
    if len(families) == 1:
        axes = [axes]

    pair_labels = [f"{a} vs {b}" for a, b in TOST_PAIRS]

    for ax, family_label in zip(axes, families):
        rows = rows_by_family[family_label]
        by_pair: dict[str, list[float]] = {lbl: [] for lbl in pair_labels}
        for r in rows:
            if r["label"] in by_pair:
                by_pair[r["label"]].append(r["mean_d"])

        y_pos = []
        means = []
        ci_lo = []
        ci_hi = []
        equivs = []
        used_labels = []
        for i, lbl in enumerate(pair_labels):
            vals = np.asarray(by_pair[lbl], dtype=float)
            if vals.size == 0:
                continue
            m = float(vals.mean())
            if vals.size >= 2:
                se = float(vals.std(ddof=1) / np.sqrt(vals.size))
                from scipy.stats import t as _t
                tcrit = float(_t.ppf(0.975, df=vals.size - 1))
                lo, hi = m - tcrit * se, m + tcrit * se
            else:
                lo = hi = m
            y_pos.append(i)
            means.append(m)
            ci_lo.append(lo)
            ci_hi.append(hi)
            equivs.append(lo > -TOST_DELTA and hi < TOST_DELTA)
            used_labels.append(f"{lbl}  (n={vals.size})")

        for i, m, l, h, eq in zip(y_pos, means, ci_lo, ci_hi, equivs):
            color = "#2166ac" if eq else "#d6604d"
            ax.errorbar(
                m, i,
                xerr=[[m - l], [h - m]],
                fmt="o", color=color, ecolor=color,
                capsize=5, markersize=8, linewidth=2,
            )
        ax.axvline(-TOST_DELTA, color="black", linestyle="--", linewidth=1.2)
        ax.axvline( TOST_DELTA, color="black", linestyle="--", linewidth=1.2)
        ax.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(used_labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean per-fold {metric_label} difference (pp)", fontsize=11)
        ax.set_title(_FAMILY_DISPLAY.get(family_label, family_label),
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
               markersize=9, label=f"95% CI inside ±{TOST_DELTA} pp"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d6604d",
               markersize=9, label=f"95% CI crosses ±{TOST_DELTA} pp"),
        Line2D([0], [0], color="black", linestyle="--", label=f"±{TOST_DELTA} pp margin"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Holdout-F1 difference summary - mean across models, 95% CI of the mean",
        fontsize=12, fontweight="bold", y=1.0,
    )
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "tost_summary_forest.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_run_mode_forest(rows: list[dict], family_label: str) -> Path:
    title = (
        f"TOST per run_mode pool - Holdout F1 ({_FAMILY_DISPLAY.get(family_label, family_label)}, "
        f"Holm across all model×pair tests)"
    )
    out = PLOTS_DIR / f"tost_per_run_mode_forest_{family_label}.png"
    return _plot_forest(rows, out, title)


# ─── Comprehensive table ─────────────────────────────────────────────────────


def write_comprehensive_latex_table(
    per_group_rows: list[dict],
    random_results: list[dict],
    temporal_results: list[dict],
    family_label: str,
) -> Path:
    """Merge per-(model, run_mode) TOST rows with cross-model intra-group rows.

    Top section: per-model holdout TOST (same rows as tost_per_group table).
    Bottom section (after midrule + label): cross-model test-set intra-group
    comparisons from run_intragroup_pairs.
    """
    family_display = _FAMILY_DISPLAY.get(family_label, family_label)
    ncols = 9

    lines = [
        "% Auto-generated by scripts/plot_tost_grouped.py - do not edit manually.",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{llrrrrlrl}",
        "    \\toprule",
        "    Model & Comparison & $n$ & $\\bar{d}$ (pp) & 90\\% CI "
        "& $p_{\\text{TOST}}$ & Equiv. "
        "& $p_{\\text{diff}}$ & Sig.\\,diff \\\\",
        "    \\midrule",
    ]

    # Per-model holdout TOST rows
    last_model: str | None = None
    for r in per_group_rows:
        model_cell = "" if r["model"] == last_model else r["model"].replace("_", "\\_")
        last_model = r["model"]
        ci_str = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        equiv_str = "Yes" if r["equivalent"] else "No"
        sig_str = "Yes" if r["significant_diff"] else "No"
        lines.append(
            f"    {model_cell} & {r['label']} & {r['n']} & {r['mean_d']:+.2f} "
            f"& {ci_str} & {_fmt_pvalue(r['p_tost_corrected'])} & {equiv_str} "
            f"& {_fmt_pvalue(r['p_diff_corrected'])} & {sig_str} \\\\"
        )

    # Intra-group section: within-random then within-temporal
    for section_label, section_rows in [
        ("Within random methods (cross-model, test set)", random_results),
        ("Within temporal methods (cross-model, test set)", temporal_results),
    ]:
        if not section_rows:
            continue
        lines.append("    \\midrule")
        lines.append(f"    \\multicolumn{{{ncols}}}{{l}}{{\\textit{{{section_label}}}}} \\\\")
        for r in section_rows:
            ci_str = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
            equiv_str = "Yes" if r["equivalent"] else "No"
            sig_str = "Yes" if r["significant"] else "No"
            lines.append(
                f"     & {r['label']} & {r['n']} & {r['mean_d']:+.2f} "
                f"& {ci_str} & {_fmt_pvalue(r['p_tost'])} & {equiv_str} "
                f"& {_fmt_pvalue(r['p_corrected'])} & {sig_str} \\\\"
            )

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{TOST equivalence and two-sided paired $t$-tests ({family_display}). "
        f"Top: per-model holdout F1, paired by fold within each (model, run\\_mode), "
        f"Holm--Bonferroni within each model. "
        f"Bottom: cross-model test-set F1, $n$ = number of models, "
        f"Holm--Bonferroni within each section. "
        f"$\\bar{{d}}$: mean difference (A~$-$~B, pp); 90\\% CI for $\\alpha={TOST_ALPHA}$ TOST. "
        f"Equiv.: $p_{{\\text{{TOST}}}} < {TOST_ALPHA}$, $\\Delta={TOST_DELTA}$\\,pp. "
        f"$p_{{\\text{{diff}}}}$: corrected two-sided $p$-value (Sig.\\,diff: $p < {TOST_ALPHA}$).}}",
        f"  \\label{{tab:tost_comprehensive_{family_label}}}",
        "\\end{table}",
    ]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"tost_comprehensive_{family_label}.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")
    return out_path


# ─── Comprehensive summary (no individual models) ────────────────────────────


def write_comprehensive_summary_latex_table(
    holdout_f1: dict[str, dict[str, float]],
    test_f1: dict[str, dict[str, float]],
    family_label: str,
) -> Path:
    """Cross-model summary table with HOLDOUT and TEST sections.

    Each section has three sub-groups:
      Within random, Within temporal, Random vs temporal.
    No per-model rows - each row is aggregated across all models in the family.
    """
    family_display = _FAMILY_DISPLAY.get(family_label, family_label)

    def _collect_rows(f1_dict: dict[str, dict[str, float]]) -> list[dict]:
        models = sorted(f1_dict.keys())
        rand  = run_intragroup_pairs(f1_dict, models, RANDOM_PAIRS)
        temp  = run_intragroup_pairs(f1_dict, models, TEMPORAL_PAIRS)
        cross = run_intragroup_pairs(f1_dict, models, CROSS_PAIRS)
        return [r for group in (rand, temp, cross) for r in group]

    def _append_section(lines: list[str], phase_label: str, rows: list[dict]) -> None:
        n = len(rows)
        for i, r in enumerate(rows):
            ci_str = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
            equiv  = "Yes" if r["equivalent"] else "No"
            sig    = "Yes" if r["significant"] else "No"
            side   = (
                f"\\multirow{{{n}}}{{*}}{{\\rotatebox[origin=c]{{90}}"
                f"{{\\textbf{{{phase_label}}}}}}}"
                if i == 0 else ""
            )
            lines.append(
                f"    {side} & {r['label']} & {r['mean_d']:+.2f} & {ci_str} "
                f"& {_fmt_pvalue(r['p_tost'])} & {equiv} "
                f"& {_fmt_pvalue(r['p_corrected'])} & {sig} \\\\"
            )

    holdout_rows = _collect_rows(holdout_f1)
    test_rows    = _collect_rows(test_f1)

    lines = [
        "% Auto-generated by scripts/plot_tost_grouped.py - do not edit manually.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{graphicx}",
        "\\begin{table}[ht]",
        "  \\centering",
        "  \\scriptsize",
        f"  \\begin{{tabular}}{{llrrrrlrl}}",
        "    \\toprule",
        "    & Comparison & $\\bar{d}$ (pp) & 90\\% CI "
        "& $p_{\\text{TOST}}$ & Equiv. "
        "& $p_{\\text{diff}}$ & Sig.\\,diff \\\\",
        "    \\midrule",
    ]

    _append_section(lines, "Holdout", holdout_rows)
    lines.append("    \\midrule")
    _append_section(lines, "Test", test_rows)

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{Cross-model TOST equivalence and two-sided paired $t$-tests "
        f"({family_display}), aggregated across all models ($n = $ number of models). "
        f"$\\bar{{d}}$: mean difference (A~$-$~B, pp); 90\\% CI for $\\alpha={TOST_ALPHA}$ TOST. "
        f"Equiv.: $p_{{\\text{{TOST}}}} < {TOST_ALPHA}$, $\\Delta={TOST_DELTA}$\\,pp. "
        f"$p_{{\\text{{diff}}}}$: Holm--Bonferroni corrected two-sided $p$-value "
        f"(Sig.\\,diff: $p < {TOST_ALPHA}$). "
        f"Holdout: most-recent 10\\% of data by publish date. "
        f"Test: per-fold held-out test split.}}",
        f"  \\label{{tab:tost_comprehensive_summary_{family_label}}}",
        "\\end{table}",
    ]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"tost_comprehensive_summary_{family_label}.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")
    return out_path


# ─── Entry point ─────────────────────────────────────────────────────────────


def _load_holdout_metrics(results_path: Path, metric: str, tag: str | None, dataset: str | None) -> pl.DataFrame:
    df = load_results(results_path, tag=tag, dataset=dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lf = prepare_per_fold_metrics(df, metric=metric, phase="holdout")
        try:
            return lf.collect(engine="streaming")
        except Exception:
            return lf.collect()


def _print_rows(rows: list[dict], header: str) -> None:
    print(f"\n  {header}")
    if not rows:
        print("    (no rows)")
        return
    for r in rows:
        tag = "EQUIV" if r["equivalent"] else "NOT-EQUIV"
        print(
            f"    {r['model']:30s} {r['label']:24s}  n={r['n']:2d}  "
            f"d={r['mean_d']:+6.2f} pp  90% CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  "
            f"p_TOST={_fmt_pvalue(r['p_tost'])}  p_corr={_fmt_pvalue(r['p_tost_corrected'])}  {tag}"
        )


def main() -> None:
    parser = ArgumentParser(description="Per-(model, run_mode) TOST equivalence on holdout F1.")
    parser.add_argument("--results", type=Path, default=INDEX_RESULTS_FILE)
    parser.add_argument("--metric", type=str, default="f1", choices=METRICS)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        print("Run `uv run scripts/collect_results.py` first.")
        sys.exit(1)

    print(f"Loading holdout per-fold metrics from {args.results} ...")
    metrics = _load_holdout_metrics(args.results, args.metric, args.tag, args.dataset)
    if metrics.is_empty():
        print("  No holdout data. Exiting.")
        sys.exit(0)

    # Load test-set metrics for the intra-group and summary tables
    df_raw = load_results(args.results, tag=args.tag, dataset=args.dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_lf = prepare_per_fold_metrics(df_raw, metric=args.metric, phase="test")
        try:
            test_metrics = test_lf.collect(engine="streaming")
        except Exception:
            test_metrics = test_lf.collect()
    test_f1_all = extract_mean_f1_matrix(test_metrics, metric=args.metric)
    holdout_f1_all = extract_mean_f1_matrix(metrics, metric=args.metric)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    per_group_by_family: dict[str, list[dict]] = {}

    for family_label, run_modes in RUN_MODE_FAMILIES:
        family = metrics.filter(pl.col("run_mode").is_in(list(run_modes)))
        if family.is_empty():
            print(f"\nFamily {family_label}: no rows for run_mode in {run_modes}; skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Family: {family_label}  (run_mode in {run_modes})")
        print(f"{'=' * 60}")

        per_group = run_per_group_tost(family, metric=args.metric)
        _print_rows(per_group, "Per-(model, run_mode) Holm")
        write_per_group_csv(per_group, family_label)
        write_per_group_latex(per_group, family_label)
        plot_per_group_forest(per_group, family_label)
        per_group_by_family[family_label] = per_group

        family_models = sorted({r["model"] for r in per_group})
        family_test_f1    = {m: v for m, v in test_f1_all.items()    if m in family_models}
        family_holdout_f1 = {m: v for m, v in holdout_f1_all.items() if m in family_models}

        rand_intra = run_intragroup_pairs(family_test_f1, family_models, RANDOM_PAIRS)
        temp_intra = run_intragroup_pairs(family_test_f1, family_models, TEMPORAL_PAIRS)
        write_comprehensive_latex_table(per_group, rand_intra, temp_intra, family_label)

        write_comprehensive_summary_latex_table(family_holdout_f1, family_test_f1, family_label)

        per_pool = run_per_run_mode_tost(family, metric=args.metric)
        _print_rows(per_pool, "Per-run_mode pooled Holm")
        write_per_run_mode_csv(per_pool, family_label)
        write_per_run_mode_latex(per_pool, family_label)
        plot_per_run_mode_forest(per_pool, family_label)

    summary_path = plot_summary_forest(per_group_by_family, metric_label=args.metric.upper())
    if summary_path is not None:
        print(f"\nSummary forest → {summary_path}")

    print(f"\nOutputs:\n  CSVs    → {OUTPUT_DIR}\n  Plots   → {PLOTS_DIR}\n  LaTeX   → {PLOTS_DIR}")


if __name__ == "__main__":
    main()
