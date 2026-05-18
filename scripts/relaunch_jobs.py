#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Relaunch SLURM training jobs that are FAILED or RUNNING based on log analysis.
Jobs marked COMPLETE are never relaunched.

Job type is inferred from the model name (mirrors launch_many.sh):
  - "gguf" in model  → zero-shot + OPRO, no --array
  - otherwise        → fine-tuning, --array=0-5%1
  - "large" in model → large GPU tier

Log naming convention (must match track_progress.py regex):
  {model}-{fold}-of-{total_folds}--{fold_type}-{tag}-{array_idx}.log

Note: fold in log filename is 1-based; --fold CLI arg is 0-based.
"""

import sys
import socket
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from track_progress import (
    collect_logs,
    infer_expected_matrix,
    build_report,
    print_summary,
    STATUS_FAILED,
    STATUS_RUNNING,
    GroupKey,
)

_ACCOUNT = "def-pbranco"
_DATASET  = "megavul"


def _resource_flags(model: str) -> list[str]:
    is_rorqual = "rorqual" in socket.gethostname()
    if is_rorqual:
        small_cpu = ["--cpus-per-task=4",  "--mem=31GB"]
        normal_cpu = ["--cpus-per-task=8",  "--mem=62GB"]
        large_cpu  = ["--cpus-per-task=16", "--mem=124GB"]
    else:
        small_cpu = ["--cpus-per-task=3",  "--mem=70GB"]
        normal_cpu = ["--cpus-per-task=6",  "--mem=140GB"]
        large_cpu  = ["--cpus-per-task=12", "--mem=280GB"]

    small_gpu = ["--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1"] + small_cpu
    normal_gpu = ["--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1"] + normal_cpu
    large_gpu  = ["--gpus=h100:1"] + large_cpu

    if "3B-gguf" in model or "4B-gguf" in model or "1B-gguf" in model:
        return small_gpu
    return large_gpu if "large" in model else normal_gpu


def _build_sbatch_cmd(key: GroupKey, fold: int) -> list[str]:
    """Build sbatch command. fold is 1-based (from log filename)."""
    zero_shot = "gguf" in key.model
    log_name  = f"{key.model}-{fold}-of-{key.total_folds}--{key.fold_type}-{key.tag}-%a.log"
    job_name  = f"{fold}/{key.total_folds}-{key.model}-{key.fold_type}"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--account={_ACCOUNT}",
        "--time=12:00:00",
        *_resource_flags(key.model),
    ]

    if not zero_shot:
        cmd.append("--array=0-5%1")

    cmd += ["-o", log_name, "SCVD/gpu_drac.sh"]

    if zero_shot:
        cmd += ["--zero-shot", "--opro"]

    cmd += [
        "--batch_size=2",
        "--grad_acc=32",
        f"--model={key.model}",
        "--epochs=5",
        f"--dataset={_DATASET}",
        f"--fold={fold - 1}",      # log filename is 1-based; CLI arg is 0-based
        f"--total_folds={key.total_folds}",
        f"--tag={key.tag}",
        f"--fold_type={key.fold_type}",
    ]

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relaunch FAILED/RUNNING SLURM jobs inferred from .log files."
    )
    parser.add_argument("--dir", type=Path, default=Path("."),
                        help="Directory containing .log files (default: current directory).")
    parser.add_argument("--tag", default=None,
                        help="Filter to a specific tag (e.g. test-holdout).")
    parser.add_argument("--model", default=None,
                        help="Filter to a single model name.")
    parser.add_argument("--failed", action="store_true",
                        help="Relaunch FAILED folds.")
    parser.add_argument("--running", action="store_true",
                        help="Relaunch RUNNING folds.")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print sbatch commands without executing.")
    args = parser.parse_args()

    if not args.failed and not args.running:
        parser.error("Specify at least one of --failed or --running.")

    directory = args.dir.resolve()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory.", file=sys.stderr)
        sys.exit(1)

    target_statuses = set()
    if args.failed:
        target_statuses.add(STATUS_FAILED)
    if args.running:
        target_statuses.add(STATUS_RUNNING)

    logs   = collect_logs(directory, tag_filter=args.tag)
    matrix = infer_expected_matrix(logs)
    report = build_report(logs, matrix)

    print_summary(report)
    print()

    to_relaunch = [
        (key, fold)
        for key, data in report.items()
        for fold, status in data["folds"].items()
        if status in target_statuses
        if args.model is None or key.model == args.model
    ]

    if not to_relaunch:
        labels = " / ".join(sorted(target_statuses))
        print(f"Nothing to relaunch — no {labels} jobs found.")
        return

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Relaunching {len(to_relaunch)} job(s):\n")

    for key, fold in sorted(to_relaunch, key=lambda x: (x[0].model, x[0].fold_type, x[1])):
        cmd   = _build_sbatch_cmd(key, fold)
        label = f"  {key.model} | {key.fold_type} | fold {fold}/{key.total_folds} | {key.tag}"

        if args.dry_run:
            print(label)
            print(f"    {' '.join(cmd)}\n")
        else:
            print(label, end=" ... ", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"ERROR: {result.stderr.strip()}", file=sys.stderr)


if __name__ == "__main__":
    main()
