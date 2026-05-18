from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    gguf_file: str | list[str] | None = None

    @property
    def family(self) -> str:
        return "gguf_gpt" if self.gguf_file is not None else "generic_gpt"


# Alternatives (removed from _MODEL_SPECS):
# - Generic/LoRA: phi1.5, llama2, llama3, llama3.2-*-instruct, codellama-70B, mistral1,
#   starcoder, codegemma, qwen2.5-3B, tiny, tiny-gemma, tiny-qwen
# - Embedding: unixcoder, qwenembed, qwenembed-8B, qwenembed-4B, codebert, codet5,
#   gemmaembed, salesforce
# - GGUF (superseded by zero-shot list below): llama3.2-1B-gguf, llama3.2-3B-gguf

_MODEL_SPECS: dict[str, ModelSpec] = {
    # Zero Shot (GGUF)
    # Tool Calling
    "devstral-2-small-gguf": ModelSpec(
        "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        gguf_file="Devstral-Small-2-24B-Instruct-2512-Q5_K_M.gguf",
    ),
    "glm-4.7-flash-gguf": ModelSpec(
        "bartowski/zai-org_GLM-4.7-Flash-GGUF",
        gguf_file="zai-org_GLM-4.7-Flash-Q5_K_M.gguf",
    ),
    "deepseekcoder-v2-16B-gguf": ModelSpec("bartowski/Deepeek-Coder-V2-Lite-Instruct-GGUF", gguf_file="DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf"),
    "gpt-oss-20b-gguf": ModelSpec("unsloth/gpt-oss-20b-GGUF", gguf_file="gpt-oss-20b-Q8_0.gguf"),
    "qwen3.5-27B-gguf": ModelSpec("bartowski/Qwen_Qwen3.5-27B-GGUF", gguf_file="Qwen_Qwen3.5-27B-Q8_0.gguf"),
    "qwen3-coder-gguf": ModelSpec("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", gguf_file="Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf"),
    "phi4-gguf": ModelSpec("microsoft/phi-4-gguf", gguf_file="phi-4-Q8_0.gguf"),
    "deepseekcoder-7B-gguf": ModelSpec("mradermacher/deepseek-coder-7b-instruct-v1.5-GGUF", gguf_file="deepseek-coder-7b-instruct-v1.5.Q8_0.gguf"),
    "gemma3-4B-gguf": ModelSpec("bartowski/google_gemma-3-4b-it-GGUF", gguf_file="google_gemma-3-4b-it-Q8_0.gguf"),
    "gemma3-27B-gguf": ModelSpec("bartowski/google_gemma-3-27b-it-GGUF", gguf_file="google_gemma-3-27b-it-Q8_0.gguf"),
    "codellama-7B-gguf": ModelSpec("bartowski/CodeLlama-7B-KStack-clean-GGUF", gguf_file="CodeLlama-7B-KStack-clean-Q8_0.gguf"),
    "mistral3-7B-gguf": ModelSpec("bartowski/Mistral-7B-Instruct-v0.3-GGUF", gguf_file="Mistral-7B-Instruct-v0.3-Q8_0.gguf"),
    "mistral3-24B-gguf": ModelSpec("bartowski/Mistral-Small-24B-Instruct-2501-GGUF", gguf_file="Mistral-Small-24B-Instruct-2501-Q8_0.gguf"),
    "llama3.2-1B-gguf": ModelSpec("bartowski/Llama-3.2-1B-Instruct-GGUF", gguf_file="Llama-3.2-1B-Instruct-f16.gguf"),
    "llama3.2-3B-gguf": ModelSpec("bartowski/Llama-3.2-3B-Instruct-GGUF", gguf_file="Llama-3.2-3B-Instruct-f16.gguf"),
    # ( LORA FINE-TUNE MODELS )
    "llama3.2-1B": ModelSpec("meta-llama/Llama-3.2-1B"),
    "llama3.2-3B": ModelSpec("meta-llama/Llama-3.2-3B"),
    "codellama": ModelSpec("codellama/CodeLlama-7b-hf"),
    "gemma": ModelSpec("google/gemma-3-4b-pt"),
    "qwen3": ModelSpec("Qwen/Qwen3-8B"),
    "llama3.1": ModelSpec("meta-llama/Llama-3.1-8B"),
    "mistral3": ModelSpec("mistralai/Mistral-7B-v0.3"),
    "deepseekcoder": ModelSpec("deepseek-ai/deepseek-coder-7b-base-v1.5"),
}


def all_model_aliases() -> list[str]:
    return sorted(_MODEL_SPECS.keys())


def get_model_spec(alias: str) -> ModelSpec:
    if alias not in _MODEL_SPECS:
        raise ValueError(f"Model alias '{alias}' is not recognized. Available aliases: {', '.join(all_model_aliases())}")
    return _MODEL_SPECS[alias]


ZERO_SHOT_FAMILIES = {"gguf_gpt"}


def supports_zero_shot(alias: str) -> bool:
    return get_model_spec(alias).family in ZERO_SHOT_FAMILIES


def aliases_for_family(family: str) -> dict[str, str]:
    return {alias: spec.checkpoint for alias, spec in _MODEL_SPECS.items() if spec.family == family}


def alias_to_checkpoint_map() -> dict[str, str]:
    return {alias: spec.checkpoint for alias, spec in _MODEL_SPECS.items()}


def alias_to_gguf_map() -> dict[str, ModelSpec]:
    """Return specs for models that need a specific GGUF file download."""
    return {alias: spec for alias, spec in _MODEL_SPECS.items() if spec.gguf_file is not None}


GENERIC_LLMS = aliases_for_family("generic_gpt")
ZERO_SHOT_LLMS = {alias: spec.checkpoint for alias, spec in _MODEL_SPECS.items() if spec.family in ZERO_SHOT_FAMILIES}
