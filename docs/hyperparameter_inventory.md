# Hyperparameter Inventory

This maps where effective hyperparameters originate and where they are consumed.

## Global CLI Surface (`src/args.py`)

- Runtime mode:
  - `--processonly`, `--skip_fit`, `--zero-shot`, `--cpu`, `--dev`, `--tag`
- Data split:
  - `--dataset`, `--n`, `--date`, `--fold_type`, `--fold`, `--total_folds`
  - `--no_holdout`, `--templated`
- Training:
  - `--batch_size`, `--grad_acc`, `--epochs`, `--seed`, `--weight_decay`, `--cw`
- Config:
  - `--profile` (TOML profile), `--reprocess` (currently informational)

## Family Definitions (`src/model_definitions.py`)

- `generic_gpt`
  - `model_args`: `lr`, `loss_fn`, LoRA args, `scheduler`, `weight_decay`
  - `data_module_args`: dataset/tokenization/split arguments
  - `trainer_args`: `max_epochs`, `val_check_interval`, `accumulate_grad_batches`
- `zero_shot_api`
  - `model_args`: API URL, model identity, labels
  - `trainer_args`: fixed minimal trainer settings

## Constructor Consumption

- `src/models/generic_gpt.py`
  - Consumes: `model_name`, `lr`, `weight_decay`, LoRA args, scheduler
- `src/models/api_based.py`
  - Consumes zero-shot inference via OpenAI-compatible API (used when `--zero-shot` with `--api-url`)

## Effective Parameter Rules

- `cw` defaults to 16384 from `configs/defaults.toml` (no longer a sentinel value).
- `n` and `fold` are `None` when unset (no longer -1 sentinels).
- `skip_fit` is effectively `skip_fit or zero_shot`.
- `no_holdout` disables holdout test split.
- `weight_decay` comes from resolved config and is propagated into model definitions.
- Numeric defaults (batch_size, epochs, seed, etc.) are defined in `configs/defaults.toml`, not in `args.py`.

## Known Compatibility Notes

- Retrieval runtime has been retired from active `src` and moved to top-level `archive/`.
- Zero-shot runs optionally accept `--api-url`; without it, a local llama-server is auto-spawned for GGUF models. Uses `zero_shot_api` (ApiBasedModel).
- Launcher scripts should pass model aliases only; tune run parameters as separate flags or via profiles.
