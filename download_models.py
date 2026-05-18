import argparse
import os
import shutil
import sys

import huggingface_hub.constants

huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
from huggingface_hub import hf_hub_download, snapshot_download

from src.model_registry import (
    alias_to_checkpoint_map,
    alias_to_gguf_map,
    all_model_aliases,
    get_model_spec,
)


def download_spec(spec, cache_dir):
    if spec.gguf_file:
        files = spec.gguf_file if isinstance(spec.gguf_file, list) else [spec.gguf_file]
        for filename in files:
            hf_hub_download(
                repo_id=spec.checkpoint,
                filename=filename,
                repo_type="model",
                cache_dir=cache_dir,
            )
    else:
        snapshot_download(
            repo_id=spec.checkpoint,
            repo_type="model",
            cache_dir=cache_dir,
        )


def _repo_id_to_cache_dirname(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def _resolve_hub_dir(cache_dir: str) -> str:
    hub_subdir = os.path.join(cache_dir, "hub")
    if os.path.isdir(hub_subdir):
        return hub_subdir
    return cache_dir


def prune_unlisted_models(cache_dir: str) -> None:
    hub_dir = _resolve_hub_dir(cache_dir)
    print(f"Scanning HuggingFace hub cache: {hub_dir}")
    if not os.path.isdir(hub_dir):
        print("No HuggingFace hub directory found; nothing to prune.")
        return

    allowed_repo_dirs = {
        _repo_id_to_cache_dirname(get_model_spec(alias).checkpoint) for alias in all_model_aliases()
    }

    removed_count = 0
    model_repo_count = 0
    for entry in os.listdir(hub_dir):
        if not entry.startswith("models--"):
            continue
        model_path = os.path.join(hub_dir, entry)
        if not os.path.isdir(model_path):
            continue
        model_repo_count += 1
        if entry in allowed_repo_dirs:
            continue
        print(f"Removing unlisted cached model: {entry}")
        shutil.rmtree(model_path)
        removed_count += 1

    if model_repo_count == 0:
        print("No cached model repositories found to evaluate.")
        return
    if removed_count == 0:
        print("Prune complete: 0 unlisted cached model repositories removed.")
        return
    print(f"Prune complete: removed {removed_count} unlisted cached model repositories.")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace models to local cache.")
    parser.add_argument(
        "--model",
        type=str,
        choices=all_model_aliases(),
        help="Download only one model by alias.",
    )
    parser.add_argument(
        "--zero-shot-only",
        action="store_true",
        help="Download only the zero-shot (GGUF) models. Ignored when --model is set.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Only remove cached models not listed in model_registry and exit.",
    )
    args = parser.parse_args()

    cache_dir = os.getenv("HF_HOME")
    if not cache_dir:
        print("HF_HOME is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Using HuggingFace cache: {cache_dir}\n")

    if args.prune:
        if args.model or args.zero_shot_only:
            print("--prune cannot be combined with --model or --zero-shot-only.", file=sys.stderr)
            sys.exit(2)
        prune_unlisted_models(cache_dir)
        return

    if args.model:
        spec = get_model_spec(args.model)
        gguf_display = ", ".join(spec.gguf_file) if isinstance(spec.gguf_file, list) else spec.gguf_file
        print(f"\n{spec.checkpoint}" + (f" / {gguf_display}" if spec.gguf_file else "") + "\n")
        download_spec(spec, cache_dir)
        return

    for spec in alias_to_gguf_map().values():
        gguf_display = ", ".join(spec.gguf_file) if isinstance(spec.gguf_file, list) else spec.gguf_file
        print(f"\n{spec.checkpoint} / {gguf_display}\n")
        download_spec(spec, cache_dir)

    if args.zero_shot_only:
        return

    for alias, checkpoint in alias_to_checkpoint_map().items():
        spec = get_model_spec(alias)
        if spec.gguf_file is None:
            print(f"\n{checkpoint}\n")
            download_spec(spec, cache_dir)

if __name__ == "__main__":
    main()
