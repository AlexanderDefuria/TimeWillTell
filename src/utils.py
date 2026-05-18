from pathlib import Path
from typing import Any, Optional, Tuple
import sys
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
import torch
import os
import contextlib
import socket
import time
import urllib.request
import shutil
import json
import subprocess
import atexit
from src.models import get_hf_cache_dir
from huggingface_hub import hf_hub_download


from src.model_registry import ModelSpec, get_model_spec


class MyProgBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = Tqdm(
            desc=f"{self.train_description}-{self.trainer.current_epoch}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0.5,
            bar_format=self.BAR_FORMAT,
        )
        bar.total = self.trainer.num_training_batches
        return bar


def data_dir():
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    return data_dir


_CWE_RESAMPLING_PREFIXES = ("mixed_resampling_", "undersampling_", "oversampling_")


def storage_structure(
    root: Path,
    tag,
    dataset,
    dataset_class,
    fold_type,
    date,
    fold,
    total_fold,
    model_name,
    hyperparameter_string: Optional[str] = None,
    cwe_assignment_strategy: Optional[str] = None,
):
    effective_fold_type = fold_type
    if cwe_assignment_strategy and fold_type.startswith(_CWE_RESAMPLING_PREFIXES):
        effective_fold_type = f"{fold_type}__{cwe_assignment_strategy}"
    path = root / tag / dataset / dataset_class / effective_fold_type / date
    if fold is not None and total_fold is not None:
        if "single" not in fold_type:
            path = path / f"fold_{fold}_of_{total_fold}"
    if hyperparameter_string:
        path = path / hyperparameter_string
    path = path / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def resolve_llama_server() -> str:
    build_dir = os.environ.get("LLAMABUILDDIR", "")
    if build_dir:
        candidate = os.path.join(build_dir, "bin", "llama-server")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("llama-server")
    if found:
        return found
    raise RuntimeError("llama-server not found. Set LLAMABUILDDIR to the llama.cpp build directory " "(done automatically by gpu_drac.sh for GGUF models).")


def start_llama_server(
    gguf_path: str,
    port: int,
    n_gpu_layers: int | None = None,
    ctx_size: int | None = None,
    chat_template_kwargs: dict | None = None,
    extra_args: list | None = None,
) -> subprocess.Popen:
    server = resolve_llama_server()
    cmd = [server, "--model", gguf_path, "--host", "127.0.0.1", "--port", str(port)]
    if n_gpu_layers is not None:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]
    if ctx_size is not None and ctx_size > 0:
        cmd += ["--ctx-size", str(ctx_size)]
    if chat_template_kwargs:
        cmd += ["--chat-template-kwargs", json.dumps(chat_template_kwargs)]
    if extra_args:
        cmd += extra_args
    print(f"Starting llama-server: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def wait_for_server(proc: subprocess.Popen, port: int, timeout: int = 300) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(f"llama-server died before becoming healthy.\n{stderr}")
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"llama-server healthy (port {port})")
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"llama-server did not become healthy within {timeout}s")


def resolve_gguf_path(model_alias: str, gguf_model: Optional[str] = None) -> tuple[str, str]:
    """Resolve GGUF model path and model name.

    Returns (gguf_path, model_name) where model_name is the
    repo_id:filename string for the OpenAI API.
    """
    if gguf_model:
        # Raw path override -- derive a model name from the filename
        model_name = Path(gguf_model).stem
        return gguf_model, model_name

    spec = get_model_spec(model_alias)
    if spec.gguf_file is None:
        raise ValueError(f"Model '{model_alias}' is not a GGUF model.")

    gguf_files = spec.gguf_file if isinstance(spec.gguf_file, list) else [spec.gguf_file]
    gguf_path = hf_hub_download(
        repo_id=spec.checkpoint,
        filename=gguf_files[0],
        cache_dir=get_hf_cache_dir(),
        local_files_only=True,
    )

    # Model name format expected by llama-server OpenAI-compatible API
    model_name = f"{spec.checkpoint}:{gguf_files[0]}"
    return gguf_path, model_name

def _configure_model(models, model_name: str, model_spec, zero_shot: bool, api_url: str | None, ctx_size: int = 1024, n_gpu_layers: int = 99, timeout: int = 120) -> str:
    """Validate the model family, select the right entry, and inject runtime args.

    Returns the selected family key.
    """
    from src.model_registry import ZERO_SHOT_LLMS, supports_zero_shot

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
            resolved_url, llama_model_name = setup_llama_server(model_name, ctx_size, n_gpu_layers, timeout)
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




def setup_llama_server(
    model_alias: str,
    ctx_size: int,
    n_gpu_layers: int,
    timeout: int = 300,
    chat_template_kwargs: dict | None = None,
    extra_args: list | None = None,
) -> Tuple[str, str]:
    """Spawn a local llama-server for GGUF models when no explicit API URL is given.

    Returns the resolved API URL, or None if no server was spawned.
    """
    gguf_path, model_name = resolve_gguf_path(model_alias)

    port = find_free_port()
    proc = start_llama_server(
        gguf_path=gguf_path,
        port=port,
        n_gpu_layers=n_gpu_layers,
        ctx_size=ctx_size,
        chat_template_kwargs=chat_template_kwargs,
        extra_args=extra_args,
    )

    def _cleanup_server():
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup_server)
    try:
        wait_for_server(proc, port, timeout=timeout)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        proc.kill()
        proc.wait()
        sys.exit(1)
    return f"http://127.0.0.1:{port}", model_name

def check_server_health(api_url: str) -> bool:
    try:
        urllib.request.urlopen(f"{api_url}/health", timeout=2)
        return True
    except Exception:
        return False

def handle_llama_server_failure(api_url: str, model_alias: str, ctx_size: int, n_gpu_layers: int, chat_template_kwargs: dict | None = None, extra_args: list | None = None) -> str:
    """Check server health; restart and return new api_url if unhealthy. Raises if restart fails."""
    if check_server_health(api_url):
        return api_url
    attempts = 2
    for attempt in range(attempts):
        print(f"Attempting to restart llama-server (attempt {attempt + 1}/{attempts})...")
        new_url, _ = setup_llama_server(model_alias=model_alias, ctx_size=ctx_size, n_gpu_layers=n_gpu_layers, chat_template_kwargs=chat_template_kwargs, extra_args=extra_args)
        if check_server_health(new_url):
            return new_url
    raise RuntimeError("llama-server is not responding after multiple attempts to restart.")
