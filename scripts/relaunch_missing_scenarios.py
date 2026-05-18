#!/usr/bin/env python
"""
Relaunch missing scenarios from missing_monolithic.csv via SLURM sbatch.

Usage:
    uv run scripts/relaunch_missing_scenarios.py                    # Dry-run: print commands
    uv run scripts/relaunch_missing_scenarios.py --real             # Actually submit to SLURM
    uv run scripts/relaunch_missing_scenarios.py --csv <path>       # Use custom CSV path
"""
from __future__ import annotations

import socket
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import polars as pl

try:
    from scripts import PROJECT_ROOT
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import PROJECT_ROOT


# ─── SLURM Configuration (matching launch_many.sh) ───────────────────────────

def get_cpu_config(hostname: Optional[str] = None) -> tuple[str, str, str]:
    """Return (SMALL_CPU, NORMAL_CPU, LARGE_CPU) based on hostname."""
    if hostname is None:
        hostname = socket.gethostname()

    if "rorqual" in hostname:
        return (
            "--cpus-per-task=4 --mem=31GB",
            "--cpus-per-task=8 --mem=62GB",
            "--cpus-per-task=16 --mem=124GB",
        )
    else:
        # FIR or other cluster
        return (
            "--cpus-per-task=3 --mem=70GB",
            "--cpus-per-task=6 --mem=140GB",
            "--cpus-per-task=12 --mem=280GB",
        )


def get_gpu_flags(small_cpu: str, normal_cpu: str, large_cpu: str) -> tuple[str, str, str]:
    """Return (SMALL_GPU, NORMAL_GPU, LARGE_GPU) combined with CPU flags."""
    small_gpu = f"--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 {small_cpu}"
    normal_gpu = f"--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 {normal_cpu}"
    large_gpu = f"--gpus=h100:1 {large_cpu}"
    return small_gpu, normal_gpu, large_gpu


def is_small_model(model: str) -> bool:
    """Check if model is in the small category."""
    return any(model.endswith(suffix) for suffix in ["4B-gguf", "1B-gguf", "3B-gguf"])


def is_zero_shot_model(model: str) -> bool:
    """Check if model is zero-shot (GGUF)."""
    return model.endswith("-gguf")


def is_large_model(model: str) -> bool:
    """Check if model is in the large category."""
    return "large" in model


def get_gpu_flag(model: str, gpu_flags: tuple[str, str, str]) -> str:
    """Select appropriate GPU flag based on model size."""
    small_gpu, normal_gpu, large_gpu = gpu_flags

    if is_large_model(model):
        return large_gpu
    elif is_small_model(model):
        return small_gpu
    else:
        return normal_gpu


# ─── Command Building ────────────────────────────────────────────────────────

def infer_cw(model: str, existing_cw: str) -> str:
    """Infer cw value from model type if not provided."""
    if existing_cw and str(existing_cw).strip():
        return str(existing_cw)
    # GGUF models use 16384, finetuned models use 1024
    return "16384" if model.endswith("-gguf") else "1024"


def build_sbatch_command(
    row: dict,
    script_path: Path,
    total_folds: int,
    account: str = "def-pbranco",
    time_limit: str = "12:00:00",
    gpu_flags: Optional[tuple[str, str, str]] = None,
) -> list[str]:
    """Build sbatch command for a single scenario from CSV row."""
    if gpu_flags is None:
        cpu_flags = get_cpu_config()
        gpu_flags = get_gpu_flags(*cpu_flags)

    model = row["model"]
    fold_type = row["fold_type"]
    fold_number = row["fold_number"]
    cw = infer_cw(model, row.get("cw", ""))
    flags_cli = row["flags"]

    # Construct sbatch directives
    job_name = f"{int(fold_number) + 1}/{total_folds}-{model}-{cw}-{fold_type}"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--account={account}",
        f"--time={time_limit}",
    ]

    # Add array flag only for non-zero-shot models
    if not is_zero_shot_model(model):
        cmd.append("--array=0-5%1")

    # Add GPU flags
    gpu_flag = get_gpu_flag(model, gpu_flags)
    cmd.extend(gpu_flag.split())

    # Log file naming
    tag = row.get("tag", "test-holdout")
    log_file = f"{model}-{int(fold_number) + 1}-of-{total_folds}-{cw}-{fold_type}-{tag}-%a.log"
    cmd.extend(["-o", log_file])

    # Add script and main.py flags
    cmd.append(str(script_path))
    cmd.extend(flags_cli.split())

    return cmd


def main() -> None:
    parser = ArgumentParser(description="Relaunch missing scenarios from missing_monolithic.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "missing_monolithic.csv",
        help="Path to missing_monolithic.csv (default: PROJECT_ROOT/missing_monolithic.csv)",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=PROJECT_ROOT / "gpu_drac.sh",
        help="Path to gpu_drac.sh script (default: PROJECT_ROOT/gpu_drac.sh)",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="def-pbranco",
        help="SLURM account (default: def-pbranco)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="12:00:00",
        help="Time limit (default: 12:00:00)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model name (substring match, e.g. 'llama', 'phi4', 'gguf')",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Actually submit jobs to SLURM (default: dry-run, just print)",
    )
    args = parser.parse_args()

    # Validate CSV exists
    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        print("Run `uv run scripts/plot_results.py` first to generate missing_monolithic.csv")
        sys.exit(1)

    # Validate script exists
    if not args.script.exists():
        print(f"ERROR: Script not found: {args.script}")
        sys.exit(1)

    # Load CSV
    try:
        df = pl.read_csv(args.csv)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        sys.exit(1)

    # Limit to primary fold types
    primary_fold_types = [
        "k_fold_cross_validation",
        "temporal_growing_window_cross_validation",
        "temporal_loo_block_cross_validation",
        "random_growing_window_cross_validation",
        "random_loo_block_cross_validation",
    ]
    df = df.filter(pl.col("fold_type").is_in(primary_fold_types))
    if df.is_empty():
        print("No scenarios found in primary fold types. Exiting.")
        sys.exit(0)

    # Apply model filter if specified
    if args.model:
        df = df.filter(pl.col("model").str.contains(args.model, literal=False))
        if df.is_empty():
            print(f"No scenarios found matching model filter: '{args.model}'")
            sys.exit(0)
        print(f"Filtered to {df.height} scenario(s) matching model: '{args.model}'")
        print()
    elif df.is_empty():
        print("No missing scenarios in CSV. Exiting.")
        sys.exit(0)

    # Pre-compute GPU flags based on hostname
    cpu_flags = get_cpu_config()
    gpu_flags = get_gpu_flags(*cpu_flags)

    # Calculate total number of folds
    total_folds = int(df.select(pl.col("fold_number").max()).item()) + 1

    print(f"Found {df.height} missing scenario(s) to relaunch")
    print(f"Mode: {'SUBMITTING to SLURM (--real)' if args.real else 'DRY-RUN (use --real to submit)'}")
    print()

    submitted = 0
    failed = 0

    for row in df.iter_rows(named=True):
        try:
            cmd = build_sbatch_command(row, args.script, total_folds, args.account, args.time, gpu_flags)
            cmd_str = " ".join(cmd)

            print(f"[{row['model']:<25} {row['fold_type']:<35} fold={row['fold_number']} {row['missing_phase']:<10}]")
            print(f"  {cmd_str}")

            if args.real:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    job_id = result.stdout.strip().split()[-1]
                    print(f"  ✓ Submitted as job {job_id}")
                    submitted += 1
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Failed: {e.stderr}")
                    failed += 1
            print()

        except Exception as e:
            print(f"✗ Error processing row: {e}")
            failed += 1

    # Summary
    print("─" * 80)
    if args.real:
        print(f"Summary: {submitted} submitted, {failed} failed")
        if failed > 0:
            sys.exit(1)
    else:
        print(f"Dry-run complete. {df.height} command(s) would be submitted.")
        print("Use --real to actually submit to SLURM.")


if __name__ == "__main__":
    main()
