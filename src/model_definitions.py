from src.datasets.alpaca import AlpacaDataset
from src.datasets.basic import RawDataset
from src.datasets.datamodule import GenericDataModule
from src.datasets.plaintext import PlainTextDataset
from src.models.api_based import ApiBasedModel
from src.models.generic_gpt import GenericGPTModel
from src.models.prompts import ZERO_SHOT_PROMPT, SYS_INST
from src.model_registry import ZERO_SHOT_LLMS, supports_zero_shot, ModelSpec, get_model_spec
from src.runtime_config import ExperimentConfig
from src.utils import setup_llama_server


def model_definitions(config: ExperimentConfig) -> dict:
    """Factory function that generates model and data module configurations.

    Args:
        config: ExperimentConfig containing all hyperparameters and runtime settings

    Returns:
        Dictionary mapping family names and model aliases to their configurations
    """
    model_name = config.model
    dataset = config.data.dataset
    batch_size = config.data.batch_size
    n = config.data.n
    cw = config.data.cw
    seed = config.run.seed
    date = config.data.date
    fold_type = config.data.fold_type
    fold = config.data.fold
    total_folds = config.data.total_folds
    max_epochs = config.trainer.epochs
    max_steps = config.trainer.max_steps
    accumulate_grad_batches = config.trainer.grad_acc
    tag = config.run.tag
    weight_decay = config.trainer.weight_decay
    templated = config.data.templated
    no_holdout = config.data.no_holdout
    mixed_resampling_multiplier = config.data.mixed_resampling_multiplier
    mixed_resampling_oversample_cap = config.data.mixed_resampling_oversample_cap
    undersampling_multiplier = config.data.undersampling_multiplier
    oversampling_multiplier = config.data.oversampling_multiplier
    oversampling_cap = config.data.oversampling_cap
    cwe_assignment_strategy = config.data.cwe_assignment_strategy
    holdout_prop = config.data.holdout_prop
    test_prop = config.data.test_prop

    base_data_args = {
        "dataset_name": dataset,
        "tokenizer": "generic_gpt_checkpoint",
        "batch_size": batch_size,
        "dataset_size": n,
        "max_length": cw,
        "seed": seed,
        "instruction": "Detect whether or not the following code contains vulnerabilities. Answer with only yes or no. Do not return JSON, punctuation, or explanation.",
        "workers": 4,
        "date": date,
        "fold_type": fold_type,
        "fold": fold,
        "total_folds": total_folds,
        "test_prop": test_prop,
        "holdout_prop": holdout_prop,
        "tag": tag,
        "commit_based_random_splitting": False,
        "hold_out_test_set": not no_holdout,
        "mixed_resampling_multiplier": mixed_resampling_multiplier,
        "mixed_resampling_oversample_cap": mixed_resampling_oversample_cap,
        "undersampling_multiplier": undersampling_multiplier,
        "oversampling_multiplier": oversampling_multiplier,
        "oversampling_cap": oversampling_cap,
        "cwe_assignment_strategy": cwe_assignment_strategy,
    }

    return {
        "generic_gpt": {
            "model": GenericGPTModel,
            "data_module": GenericDataModule,
            "model_args": {
                "lr": 1e-5,  # 1-e5 and 10000 for ~90% F1
                "num_labels": 2,
                "model_name": "generic_gpt_checkpoint",  # Placeholder for generic GPT model
                "loss_fn": "bce",
                "lora_r": 16,
                "weight_decay": weight_decay,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "scheduler": "constant",
                "target_modules": (
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        # "gate_proj",
                        # "up_proj",
                        # "down_proj",
                    ]
                    if "starcoder" not in model_name
                    else ["c_proj"]
                ),
            },
            "data_module_args": {
                **base_data_args,
                "dataset_class": AlpacaDataset if templated else RawDataset,
            },
            "trainer_args": {
                **({"max_epochs": max_epochs} if max_steps is None else {"max_epochs": -1, "max_steps": max_steps}),
                "val_check_interval": 1.0,  # Validate every X batches
                "accumulate_grad_batches": accumulate_grad_batches,
            },
        },
        "zero_shot_api": {
            "model": ApiBasedModel,
            "data_module": GenericDataModule,
            "model_args": {
                "api_url": "api_url_placeholder",
                "model_name": "generic_gpt_checkpoint",
                "num_labels": 2,
                "system_prompt": SYS_INST,
                "user_prompt_template": ZERO_SHOT_PROMPT,
                "seed": seed,
            },
            "data_module_args": {
                **base_data_args,
                "dataset_class": PlainTextDataset,
            },
            "trainer_args": {
                "max_epochs": 1,
            },
        },
    }


def _select_and_configure_family(
    models: dict, config: ExperimentConfig, model_spec: ModelSpec, n_gpu_layers: int = 99
) -> str:
    """Validate the model family, select the right entry, and inject runtime args.

    Args:
        models: Factory dict from model_definitions()
        config: ExperimentConfig with model, zero_shot, api_url, cw settings
        model_spec: ModelSpec for the selected model
        n_gpu_layers: GPU layers for llama-server (default 99 = all)

    Returns:
        The selected family key (e.g., "zero_shot_api" or "generic_gpt")
    """
    model_name = config.model
    zero_shot = config.run.zero_shot
    api_url = config.run.api_url
    ctx_size = config.data.cw

    spawned_server = False
    if zero_shot:
        if not supports_zero_shot(model_name):
            raise ValueError(
                f"Model {model_name} (family: {model_spec.family}) is not available for zero-shot mode. "
                "Zero-shot requires an autoregressive generation-capable model "
                "(currently gguf_gpt models and future API-based models). "
                f"Choose one of {list(ZERO_SHOT_LLMS.keys())}."
            )
        models[model_name] = models["zero_shot_api"]
        selected_family = "zero_shot_api"
        if api_url is None:
            # Branch A: spawn local llama-server and resolve URL + model identifier
            resolved_url, llama_model_name = setup_llama_server(model_name, ctx_size, n_gpu_layers)
            models[model_name]["model_args"]["api_url"] = resolved_url
            models[model_name]["model_args"]["model_name"] = llama_model_name
            spawned_server = True
        else:
            # Branch B: remote / externally-managed server
            models[model_name]["model_args"]["api_url"] = api_url
        if models[model_name]["data_module_args"]["batch_size"] != 1:
            print(f"[api_based] Overriding batch_size {models[model_name]['data_module_args']['batch_size']} → 1 (API models process one sample at a time)")
            models[model_name]["data_module_args"]["batch_size"] = 1
    else:
        if model_spec.family == "gguf_gpt":
            raise ValueError(
                f"GGUF model '{model_name}' requires --zero-shot flag for local inference. "
                "Use: --model {model_name} --zero-shot"
            )
        if model_spec.family not in {"generic_gpt"}:
            raise ValueError(f"Model family '{model_spec.family}' is not available in active runtime.")
        models[model_name] = models[model_spec.family]
        selected_family = model_spec.family
    if not spawned_server:
        models[model_name]["model_args"]["model_name"] = model_spec.checkpoint
    models[model_name]["data_module_args"]["tokenizer"] = model_spec.checkpoint
    if model_spec.gguf_file is not None:
        models[model_name]["model_args"]["gguf_file"] = model_spec.gguf_file
    return selected_family
