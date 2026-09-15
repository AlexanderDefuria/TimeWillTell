# Time Will Tell

Replication package for the manuscript:

> **Time Will Tell: Illuminating Causes of Benchmark Inflation in Source Code Vulnerability Detection** 

The repository contains the manuscript (`main.tex`), the experimental
pipeline (LoRA fine-tuning and zero-shot inference for source code
vulnerability detection), the cross-validation and CWE-resampling logic
studied in the paper, and all scripts used to produce the tables and
figures.

A Zenodo mirror of the code and dataset is available at
<https://doi.org/10.5281/zenodo.21542749>

---

## What the paper studies

We evaluate five cross-validation strategies across eight fine-tuned LLMs
and ten zero-shot LLMs on the **MegaVul-C** dataset, all sharing an
identical held-out future test set:

- `K-Fold CV` — random stratified k-fold
- `Random LOO Block CV` — leave-one-block-out, random block assignment
- `Temporal LOO Block CV` — leave-one-block-out, temporally ordered blocks
- `Random Growing Window CV` — expanding-window CV with random ordering
- `Temporal Growing Window CV` — expanding-window CV with temporal ordering

The five research questions investigated:

| RQ  | Question |
|-----|----------|
| RQ1 | Does temporal splitting lower test F1 vs random splitting? |
| RQ2 | Train-set vs test-set effects — does the gap disappear on a shared holdout? |
| RQ3 | Inter-commit leakage — induced vs restricted |
| RQ4 | Time-travel as a source of inflation |
| RQ5 | Evolving vulnerability patterns (CWE drift) and resampling ablations |


---

## Repository layout

| Path | Purpose |
|------|---------|
| `main.tex` | LIPIcs-formatted manuscript source |
| `main.py` | Entry shim — delegates to `src.main.main` |
| `src/` | Pipeline: data, models, training, OPRO |
| `src/datasets/` | `GenericDataset` + splitting / fold / CWE-resampling logic |
| `src/models/` | LoRA fine-tuning (`generic_gpt.py`) and OpenAI-compatible API inference (`api_based.py`) |
| `configs/defaults.toml` | Canonical CLI defaults and named profiles (TOML is the source of truth) |
| `configs/cwe_analysis.toml`, `plot_*.toml` | Per-tool configs |
| `scripts/` | EDA, statistics, table generation, and every plot in the paper |
| `data/datasets/` | Raw MegaVul / BigVul / Devign / PrimeVul / DiverseVul parquets |
| `data/processed/` | Cached tokenized splits (auto-generated) |
| `data/checkpoints/` | Model + OPRO checkpoints (auto-generated) |
| `docs/` | Logging and hyperparameter inventory |
| `create_splits.sh` | Reproduces every fold parquet (LOO × CWE-strategy variants) |
| `cwe_balance_launch.sh`, `launch_many.sh`, `gpu_drac.sh` | Launchers for local and SLURM/DRAC runs |
| `deps/` | Vendored submodules (`llama.cpp`, `wheels_builder`) |

---

## Quick start

The project uses [`uv`](https://github.com/astral-sh/uv) for environment
management; Python 3.12 is required.

```bash
git clone --recurse-submodules <repo-url> TimeWillTell
cd TimeWillTell
uv sync                                                          # create venv and install deps
```

### Environment variables

| Variable | When required | Purpose |
|----------|---------------|---------|
| `HF_TOKEN` | gated models | HuggingFace auth |
| `HF_HOME` | optional | Model cache directory |
| `LLAMABUILDDIR` | zero-shot only | Path to a `llama.cpp` build dir with `llama-server` |

A `.env` is loaded automatically if present.

### Common invocations

```bash
# LoRA fine-tuning
uv run main.py --model llama3.2-1B --dataset megavul

# Zero-shot via local llama-server
uv run main.py --zero-shot --model phi4

# Zero-shot via remote OpenAI-compatible endpoint
uv run main.py --api-url http://host:port --model phi4

# OPRO prompt optimization (requires --zero-shot)
uv run main.py --zero-shot --opro --model phi4 --dataset megavul

# Build/cache splits only (no training)
uv run main.py --processonly --dataset megavul --n 1000

# Fast smoke run: n=100, epochs=1
uv run main.py --dev ...
```

`configs/defaults.toml` defines `[cli]` defaults and named `[profiles.*]`
blocks. Resolution order is **TOML defaults → `--profile` → explicit
CLI flags**. `args.py` keeps all defaults as `None` so that TOML
remains the single source of truth.

Notes:
- `--tag` may not contain underscores (use hyphens).
- `--dev` is for debugging only — do not use for reported runs.
- `SLURM_JOB_ID` triggers DDP and `SLURMEnvironment(auto_requeue=True)`.
- `data/processed/` is keyed off hyperparameters via `storage_structure()`;
  delete it when changing tokenizer or split logic, or pass `--reprocess`.

---

## Reproducing the paper

### 1. Build the splits

`create_splits.sh` enumerates every fold type × CWE-assignment strategy
combination used in the paper and writes the resulting split parquets
into `data/processed/`. It runs up to 8 jobs in parallel:

```bash
./create_splits.sh
```

This covers the resampling fold types
(`{under,over,mixed_resampling}_{random,temporal}_loo_block`) across
strategies `val_most_common`, `primary`, `explode`, plus the
non-resampling baselines (`{random,temporal}_loo_block_cross_validation`).

### 2. Run the experiments

For a single condition, invoke `main.py` directly. To launch the full
sweep on SLURM/DRAC, use the provided launch scripts (`launch_many.sh`,
`cwe_balance_launch.sh`, `gpu_drac.sh`).

### 3. Collect results and produce plots/tables

```bash
uv run scripts/collect_results.py        # aggregate per-run metrics
uv run scripts/plot_all.py               # regenerate every paper figure
uv run scripts/table_all_models.py       # main results table
```

Targeted plotting entry points (one per figure):

- `plot_holdout_results_v2.py` — RQ1/RQ2 holdout vs test contrast
- `plot_test_holdout_contrast.py`, `plot_tost_grouped.py` — equivalence tests
- `plot_cwe_divergence.py`, `plot_jsd_vs_f1.py` — CWE drift and its
  correlation with F1
- `plot_cwe_resampling.py`, `cwe_balance_exploration.py` — RQ5 ablations
- `plot_fold_date_ranges.py`, `extract_fold_date_ranges.py` — temporal
  fold coverage diagnostics
- `plot_loo_fold_trend.py`, `plot_gw_fold_trend.py` — per-fold trends
- `plot_pvalue_summary.py`, `plot_statistical_tests.py` — significance
  summaries
- `plot_approach_comparison.py`, `plot_results.py`,
  `plot_dataset_summary.py` — overview figures

Scripts share canonical path constants (`EDA_DIR`, `DATASETS_DIR`, …)
re-exported from `scripts/__init__.py`. Per-tool documentation lives in
`scripts/plots_README.md`, `scripts/collect_results_README.md`, and
`scripts/cwe_balance_README.md`.

---

## Models

Model aliases are defined in `src/model_registry.py` and grouped into
two families in `src/model_definitions.py`:

- **`generic_gpt`** — LoRA fine-tuning with 4-bit quantization
  (`AutoModelForSequenceClassification` + PEFT). Used for all
  fine-tuned models in the paper.
- **`zero_shot_api`** — OpenAI-compatible inference, either against a
  locally launched `llama-server` (controlled via `LLAMABUILDDIR`) or a
  remote endpoint (`--api-url`).

Adding a model is a one-line `ModelSpec` entry; adding a new GGUF model
auto-routes to the zero-shot family. 

---

## Datasets

The pipeline expects parquet files in `data/datasets/<stem>.parquet` with
columns: `vulnerability` (int), `func_before` (str), `cwe_ids`, `date`,
`publish_date`, `index` (int), `commit_hash`. Passing
`--dataset <stem>` is enough — no code changes required for new
parquets.

MegaVul-C is the primary dataset; BigVul, Devign, PrimeVul and
DiverseVul parquets are included for cross-checks.

