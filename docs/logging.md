# Logging and Metrics Reference

All logging in TimeWillTell uses PyTorch Lightning's built-in mechanisms and direct console output. There is no Python `logging` module usage, no WandB, and no MLflow integration.

---

## Lightning Loggers

Two loggers are configured in `src/main.py` and passed to every `Trainer`:

| Logger | Output directory | Notes |
|--------|-----------------|-------|
| `TensorBoardLogger` | `data/tb_logs/` | `log_graph=False` (LLM graphs too large) |
| `CSVLogger` | `data/csv_logs/` | Plain CSV metric files |

Both use `storage_structure()` to build a deterministic subdirectory path from the experiment hyperparameters (`tag/dataset/dataset_class/fold_type/date/fold/hyperparams/model`).

---

## `self.log()` Metrics

### GenericGPTModel (`src/models/generic_gpt.py`)

Metrics are accumulated per-step into `*_step_outputs` / `*_step_labels` lists and computed in `on_*_epoch_end`.

| Metric | Phase | Granularity | `prog_bar` |
|--------|-------|-------------|------------|
| `train_loss` | train | per step | Yes |
| `train_f1` | train | epoch end | Yes |
| `train_accuracy` | train | epoch end | Yes |
| `train_recall` | train | epoch end | No |
| `train_precision` | train | epoch end | No |
| `train_balance` | train | epoch end | No |
| `val_loss` | val | per step | Yes |
| `val_f1` | val | epoch end | Yes |
| `val_accuracy` | val | epoch end | Yes |
| `val_recall` | val | epoch end | No |
| `val_precision` | val | epoch end | No |
| `val_balance` | val | epoch end | No |
| `test_loss` | test | per step | Yes |
| `test_f1` | test | epoch end | Yes |
| `test_accuracy` | test | epoch end | Yes |
| `test_recall` | test | epoch end | Yes |
| `test_precision` | test | epoch end | No |
| `test_balance` | test | epoch end | Yes |
| `test_size` | test | epoch end | Yes |

### ApiBasedModel (`src/models/api_based.py`)

Test-only model (no training or validation phases).

| Metric | Phase | Granularity | `prog_bar` |
|--------|-------|-------------|------------|
| `test_loss` | test | per step | Yes |
| `test_f1` | test | epoch end | Yes |
| `test_accuracy` | test | epoch end | Yes |
| `test_recall` | test | epoch end | Yes |
| `test_precision` | test | epoch end | No |
| `test_balance` | test | epoch end | Yes |
| `test_size` | test | epoch end | Yes |

---

## Predictions Parquet Files

Both model classes save per-sample predictions at the end of each epoch via `save_predictions_and_labels()`.

**Path pattern:**
```
{logger.log_dir}/{phase}_predictions_labels_epoch{E:03d}_step{S:08d}.parquet
```

Where `phase` is `train`, `val`, `test`, or `holdout`.

**Columns:**

| Column | GenericGPTModel | ApiBasedModel |
|--------|----------------|---------------|
| `index` | Yes | Yes |
| `prediction` | Yes | Yes |
| `label` | Yes | Yes |
| `answer` | No | Yes (raw LLM text) |

---

## Histogram Logging

`GenericGPTModel.on_after_backward()` writes weight and gradient histograms to TensorBoard every 1000 global steps:

- `{param_name}_grad` — gradient histogram
- `{param_name}_weight` — weight histogram

This only applies to fine-tuning (GenericGPTModel). ApiBasedModel does not log histograms.

---

## OPRO Logging (`src/opro.py`)

All OPRO console output is prefixed with `[OPRO]`.

### Console messages
- `[OPRO] Evaluating initial prompt on N samples ...`
- `[OPRO] Initial score: F1=... acc=... rec=... prec=...`
- `[OPRO] Step X/Y — generating candidate ...`
- `[OPRO] Step X: F1=... acc=... rec=... prec=... best=...`
- `[OPRO] New best! score=...`
- `[OPRO] Resumed from ...: N steps already done.`
- `[OPRO] All N steps already completed. Returning best prompt.`
- `[OPRO] Optimization complete. Best score: ...`

### Checkpoint files (under `data/checkpoints/opro/{storage_structure_path}/`)

| File | Format | Contents |
|------|--------|----------|
| `metadata.json` | JSON | Experiment hyperparameters (model, dataset, fold, seed, etc.) |
| `history.json` | JSON | Array of `{step, prompt, score, is_best, metrics}` per evaluated step |
| `best_prompt.txt` | Plain text | The highest-scoring prompt template |
| `train_predictions_labels.parquet` | Parquet | Per-sample predictions for each step (`step`, `index`, `prediction`, `label`, `answer`) |

OPRO resumes automatically: if `history.json` exists at the checkpoint path, previously completed steps are skipped.

---

## Progress Bars

| Context | Implementation | Source |
|---------|---------------|--------|
| Training/validation/test loops | `MyProgBar` (custom `TQDMProgressBar`) | `src/utils.py` |
| Dataset tokenization | `tqdm` (direct) | dataset classes |

`MyProgBar` customises the training bar to include the current epoch in the description and uses `smoothing=0.5`.

---

## Console Output

Key information printed to stdout during a run:

| What | Where | When |
|------|-------|------|
| Resolved run configuration (JSON) | `src/main.py` | After config is built |
| `Saved hyperparameters: {...}` | `src/models/generic_gpt.py` | Model construction |
| `Processing dataset only...` | `src/main.py` | `--processonly` mode |
| `[api_based] Overriding batch_size N -> 1` | `src/utils.py` | Zero-shot with batch_size != 1 |
| `Starting llama-server: ...` | `src/utils.py` | Local GGUF zero-shot |
| `llama-server healthy (port N)` | `src/utils.py` | After server health check passes |

---

## Downstream Statistical Analysis

Per-fold F1 logged here is consumed by `scripts.plot_statistical_tests.prepare_per_fold_metrics`,
which feeds the equivalence analysis in `scripts/plot_tost_grouped.py`.
See [`docs/tost_grouped.md`](tost_grouped.md) for the rewritten TOST design
(per-(model, run\_mode) paired tests with two Holm scopes).

---

## What is NOT Used

- **WandB / MLflow / Neptune** — not integrated; all logging goes through Lightning's TensorBoard and CSV loggers.
- **Python `logging` module** — not used; all output is via `print()`.
- **Log levels / rotation** — no log level configuration or file rotation.
- **Remote logging** — no metrics are sent to external services.

---

## Checkpointing

Configured via `ModelCheckpoint` callbacks in `src/main.py`:

| Callback | Trigger | Config |
|----------|---------|--------|
| Best checkpoint | `val_f1` improves (mode=max) | `save_top_k=2`, filename `{epoch:02d}-{step}` |
| In-progress checkpoint | Every 30 minutes of training time | `save_last=True`, filename `inprogress-{epoch:02d}-{step}` |

The `last.ckpt` symlink is maintained by Lightning and used for automatic resume on restart.
Checkpoint directory is fully deterministic via `storage_structure()` in `src/utils.py`.
