# Finetuning Process Overview

This document describes the end-to-end flow used by `main.py` and `src/main.py`.

## Control Flow

1. Parse CLI arguments in `src/args.py`.
2. Merge defaults, optional profile values, and explicit CLI overrides.
3. Build runtime config objects (`RunConfig`, `DataConfig`, `TrainerConfig`, `ExperimentConfig`).
4. Resolve the selected model alias through the central model registry.
5. Build model/data/trainer definition blocks from `src/model_definitions.py`.
6. Apply runtime overrides:
   - Selected checkpoint to `model_args.model_name`
   - Selected checkpoint to `data_module_args.tokenizer`
7. Instantiate the selected DataModule.
8. If `--processonly`, run `prepare_data()` and `setup()` then exit.
9. Build the Lightning model and trainer from resolved args.
10. Run `trainer.fit(...)` unless `--skip_fit` or `--zero-shot`.
11. Run `trainer.test(...)` on test split.
12. If available, run holdout test split.

## Runtime Branches

- Generic fine-tuning branch: `generic_gpt` family.
- Embedding branch: `embedding` family.
- Reranker branch: `reranker` family.
- Zero-shot branch: selected via `--zero-shot` and model capability.

## Override Precedence

Configuration is applied in this order:

1. Built-in CLI defaults.
2. Profile values loaded from TOML (`--profile`).
3. Explicit CLI options provided on the command line.

## Observability

`src/main.py` prints a resolved configuration snapshot before model/data/trainer instantiation so each run has a deterministic, inspectable configuration.
