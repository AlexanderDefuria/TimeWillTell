# Per-(model, run_mode) TOST Equivalence Analysis

Reference for `scripts/plot_tost_grouped.py`, which replaces the cross-model
TOST that previously lived in `scripts/plot_pvalue_summary.py` and
`scripts/plot_statistical_tests.py`.

## 1. Purpose

We claim that random and temporal splitting strategies produce **statistically
equivalent** holdout F1 — i.e. the test-set gap reported in RQ1 is a
test-set-construction artefact, not a real generalisation advantage.
"Equivalent" means: the per-fold F1 difference between two splitting strategies
falls inside ±Δ where Δ = 2 pp F1.

The previous analysis tested this with **one TOST per fold-type pair**,
collapsing each model to its mean holdout F1 across folds. The unit of
analysis was the model (n ≈ 8) and per-fold variation was discarded. That
analysis can only speak to family-of-models behaviour and cannot say whether
*a given model* is equivalent across splitting strategies.

This module re-casts the analysis so the unit is **(model, run\_mode)**, and
the paired observations are the per-fold holdout F1 series joined on
`fold_number`. With n ≈ 5–10 per test the per-test power is lower, but the
question being answered is more specific and the result is reportable per row.

## 2. Inputs

The script consumes the output of `scripts.plot_statistical_tests.prepare_per_fold_metrics`
in the `holdout` phase. Required columns:

| Column | Type | Notes |
|---|---|---|
| `model`        | str | Model alias from `src/model_registry.py`. |
| `run_mode`     | str | `fine_tuning`, `opro`, or `zero_shot`. |
| `fold_type`    | str | Post-normalisation: `K-Fold CV (Default)`, `Random LOO Block CV (Default)`, `Random Growing Window CV (Default)`, `Temporal LOO Block CV (Default)`, `Temporal Growing Window CV (Default)`. |
| `fold_number`  | int | Fold index within the fold_type. **Not** comparable across fold_types in general — the underlying held-out subsets differ; only the holdout *evaluation* set is fixed. |
| `f1`           | float | Holdout F1 in [0, 1]. Converted to pp inside the module. |

See `docs/logging.md` for the metric pipeline that produces these columns.

## 3. Algorithm

For each `(model, run_mode)` group and each `(short_a, short_b)` in
`scripts.plot_pvalue_summary.TOST_PAIRS`:

1. Resolve `short_a`/`short_b` to canonical `fold_type` strings using
   `_resolve_fold_type` (prefix match into the post-normalisation names).
2. Inner-join the per-fold F1 rows for fold_type A and fold_type B on
   `fold_number`. This handles missing/incomplete folds without dropping the
   group entirely.
3. Skip if either fold_type is absent for the group, or if the paired count is
   below `MIN_PAIRED_FOLDS = 5` (the same guard `run_pairwise_tests` uses).
4. Convert both arrays to pp (×100) and call `paired_tost(a, b, delta=2.0)`.
   This is the standard Schuirmann two one-sided tests:
   - `t_lower = (mean_d + Δ) / SE` tests H₀: μ ≤ −Δ
   - `t_upper = (mean_d − Δ) / SE` tests H₀: μ ≥ +Δ
   - `p_TOST = max(p_lower, p_upper)`
   - 90 % CI corresponds to α = 0.05 TOST.

This is a *paired* TOST, not unpaired. The holdout evaluation set is fixed; the
per-fold variation comes from the different training subsets each fold_type
induces, so pairing by `fold_number` controls for shared per-fold idiosyncrasy
of the evaluation procedure.

## 4. Two Holm scopes

Both Holm corrections live in `scripts/plot_tost_grouped.py` as separate
top-level functions. Pick the scope that matches the claim you want to make.

### `run_per_group_tost`

Holm–Bonferroni is applied **within each `(model, run_mode)` family** of ≤5
pair tests. This controls family-wise error for the question:

> For this single model under this run_mode, is equivalence supported across
> all reported fold-type pairs?

This is the right scope when citing a row in the table. With at most 5 tests
per family the correction is mild.

### `run_per_run_mode_tost`

Holm–Bonferroni is applied **pooled across all `(model × pair)` tests inside
one run_mode** (typically ~8 models × 5 pairs ≈ 40 tests). This controls
family-wise error for the broader question:

> Across this entire run_mode family, is equivalence supported for *every*
> (model × pair) reported?

Strictly more conservative than the per-group scope. This is what backs
aggregate prose statements like "for fine-tuned models, equivalence holds
across all (model, pair) comparisons even under pooled correction".

For any individual row, `p_tost_corrected` from `run_per_run_mode_tost` is
≥ `p_tost_corrected` from `run_per_group_tost`.

## 5. Output artefacts

The script writes six files per family (`finetuned` and `gguf`).

| Artefact | Path |
|---|---|
| Per-(model, run_mode) Holm CSV | `data/eda/statistical_tests/{family}_tost_per_group_holdout.csv` |
| Per-run_mode pooled Holm CSV   | `data/eda/statistical_tests/{family}_tost_per_run_mode_holdout.csv` |
| Per-(model, run_mode) LaTeX    | `data/plots/6_pvalue_comparison/tost_per_group_table_{family}.tex` |
| Per-run_mode pooled LaTeX      | `data/plots/6_pvalue_comparison/tost_per_run_mode_table_{family}.tex` |
| Per-(model, run_mode) forest   | `data/plots/6_pvalue_comparison/tost_per_group_forest_{family}.png` |
| Per-run_mode pooled forest     | `data/plots/6_pvalue_comparison/tost_per_run_mode_forest_{family}.png` |

CSV columns: `model, run_mode, short_a, short_b, ft_a, ft_b, n, mean_d, ci_lo,
ci_hi, p_lower, p_upper, p_tost, p_tost_corrected, equivalent, p_diff,
p_diff_corrected, significant_diff`.

The LaTeX table reports both the TOST equivalence verdict (`p_TOST`, `Equiv.`)
**and** the complementary paired two-sided t-test on the same per-fold
difference vector (`p_diff`, `Sig. diff`). The t-test answers
"is the mean difference significantly non-zero?" — a separate question from
TOST's "is the mean difference inside ±Δ?". Both p-values are Holm-corrected
within the same scope as the row's `p_TOST`. Reading the two verdicts
together:

| Equiv. | Sig. diff | Interpretation |
|---|---|---|
| Yes | No  | Strong equivalence claim — bounded *and* indistinguishable from zero. |
| Yes | Yes | Real, but small (<Δ) difference. |
| No  | Yes | Real difference, possibly >Δ; equivalence rejected. |
| No  | No  | Inconclusive — insufficient power for either claim. |

Each table/forest row corresponds to one (model, pair) test. Rows are tagged
"Yes" / blue in the forest when `p_tost_corrected < α (0.05)`.

## 6. Caveats

- **Power**: with n ≈ 5–10 per test the per-test power is modest. Expect more
  "inconclusive" verdicts than the previous cross-model TOST showed. An
  inconclusive row means the data cannot distinguish equivalence from a >Δ
  difference at the chosen α — it is *not* evidence of a real >Δ gap.
- **K-Fold vs Rand LOO**: this pair is in `TOST_PAIRS` because it answers
  RQ3 (inter-commit leakage); both fold_types are random. The table caption
  reflects this and does not claim "(random − temporal)" for that row.
- **Fold_number pairing**: fold_number = i refers to *different* underlying
  subsets across fold_types (e.g. K-Fold fold 0 ≠ Temp LOO fold 0). The
  inner-join is on the *index*, not on subset identity. This matches the
  existing `run_pairwise_tests` convention; the holdout evaluation set, which
  is what F1 is measured on, is identical across fold_types and folds. The
  pairing therefore controls for per-fold *training-procedure* noise, not for
  evaluation-set noise (which is zero by construction).
- **Mixed run_modes per family**: the `gguf` family pools `opro` and
  `zero_shot` run_modes because they share the GGUF backbone in this
  experiment grid. Per-(model, run\_mode) rows keep them separable; the pooled
  Holm correction nonetheless treats them as one family.
- **`MIN_PAIRED_FOLDS = 5`**: a group with fewer than 5 paired observations
  for a pair is silently skipped. Rows missing for a model in the output
  table imply this guard fired (or one of the fold_types is absent).

## 7. Reproducing

```bash
# Regenerate everything
uv run python scripts/plot_tost_grouped.py --metric f1 --tag fine-tuning --dataset megavul

# Inspect outputs
ls data/plots/6_pvalue_comparison/tost_per_*
cat data/eda/statistical_tests/finetuned_tost_per_group_holdout.csv | head

# Run unit tests
uv run pytest tests/test_tost_grouped.py -v
```

The script reads `data/eda/index_results.parquet` (the canonical
`INDEX_RESULTS_FILE` from `scripts/__init__.py`). Refresh it with
`uv run scripts/collect_results.py` if model runs have changed.

## 8. Cross-references

- `scripts/plot_tost_grouped.py` — implementation.
- `scripts/plot_pvalue_summary.py` — source of `paired_tost`, `TOST_DELTA`,
  `TOST_ALPHA`, `TOST_PAIRS`, `_resolve_fold_type`, `_fmt_pvalue`. Also writes
  the superiority-test artefacts for RQ1 (kept).
- `scripts/plot_statistical_tests.py` — `prepare_per_fold_metrics`,
  `MIN_PAIRED_FOLDS`, and the per-model paired t-test/Wilcoxon machinery
  (kept; only the cross-model TOST was retired).
- `docs/logging.md` — upstream metric pipeline.
- `main.tex` RQ2 section — paper text that cites
  `\input{plots/6_pvalue_comparison/tost_per_group_table_finetuned.tex}` and
  the pooled equivalent.
