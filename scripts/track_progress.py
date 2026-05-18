#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
SLURM training progress tracker.

Scans a directory for .log files matching the naming convention:
  {model}-{fold}-of-{total_folds}--{fold_type}-{tag}-{array_idx}.log

Infers the expected run matrix from filenames and reports which jobs are
complete, failed, running, or missing.
"""
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

# Regex: model may contain hyphens (e.g. deepseekcoder-v2-16B-gguf).
# fold_type uses underscores only — this is the anchor that disambiguates.
LOG_RE = re.compile(
    r"^(.+?)-(\d+)-of-(\d+)--([a-z_]+)-(.+)-([^-]+)\.log$"
)

STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED   = "FAILED"
STATUS_RUNNING  = "RUNNING"
STATUS_MISSING  = "MISSING"


@dataclass(frozen=True)
class GroupKey:
    model: str
    fold_type: str
    tag: str
    total_folds: int


@dataclass
class LogFile:
    path: Path
    model: str
    fold: int
    total_folds: int
    fold_type: str
    tag: str
    array_idx: str
    status: str


def parse_log_filename(path: Path) -> Optional[dict]:
    m = LOG_RE.match(path.name)
    if not m:
        return None
    return {
        "model":       m.group(1),
        "fold":        int(m.group(2)),
        "total_folds": int(m.group(3)),
        "fold_type":   m.group(4),
        "tag":         m.group(5),
        "array_idx":   m.group(6),
    }


def check_log_status(path: Path) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return STATUS_RUNNING
    if "Error" in text:
        return STATUS_FAILED
    if "SUCCESS" in text:
        return STATUS_SUCCESS
    return STATUS_RUNNING


def collect_logs(directory: Path, tag_filter: Optional[str] = None) -> list[LogFile]:
    logs = []
    for path in sorted(directory.glob("*.log")):
        parsed = parse_log_filename(path)
        if parsed is None:
            continue
        if tag_filter and parsed["tag"] != tag_filter:
            continue
        status = check_log_status(path)
        logs.append(LogFile(path=path, status=status, **parsed))
    return logs


def infer_expected_matrix(logs: list[LogFile]) -> dict[GroupKey, set[int]]:
    """For each (model, fold_type, tag, total_folds) group, expect folds 1..total_folds."""
    matrix: dict[GroupKey, set[int]] = {}
    for log in logs:
        key = GroupKey(log.model, log.fold_type, log.tag, log.total_folds)
        if key not in matrix:
            matrix[key] = set(range(1, log.total_folds + 1))
    return matrix


def build_report(
    logs: list[LogFile],
    matrix: dict[GroupKey, set[int]],
) -> dict[GroupKey, dict]:
    """
    Returns a dict keyed by GroupKey. Each value is:
      {
        "folds": {fold_num: status, ...},   # found logs
        "expected": set[int],               # all expected fold numbers
      }
    """
    found: dict[GroupKey, dict[int, str]] = defaultdict(dict)
    for log in logs:
        key = GroupKey(log.model, log.fold_type, log.tag, log.total_folds)
        found[key][log.fold] = log.status

    report = {}
    for key, expected_folds in matrix.items():
        fold_statuses = {}
        for fold in expected_folds:
            fold_statuses[fold] = found[key].get(fold, STATUS_MISSING)
        report[key] = {"folds": fold_statuses, "expected": expected_folds}
    return report


def _count_statuses(folds: dict[int, str]) -> dict[str, int]:
    counts = {STATUS_SUCCESS: 0, STATUS_FAILED: 0, STATUS_RUNNING: 0, STATUS_MISSING: 0}
    for s in folds.values():
        counts[s] += 1
    return counts


def print_summary(report: dict[GroupKey, dict]) -> None:
    if not report:
        print("No log files found.")
        return

    # Determine column widths dynamically.
    col_model     = max(len("Model"),     max(len(k.model)     for k in report))
    col_fold_type = max(len("Fold Type"), max(len(k.fold_type) for k in report))
    col_tag       = max(len("Tag"),       max(len(k.tag)       for k in report))

    header = (
        f"{'Model'.ljust(col_model)} | "
        f"{'Fold Type'.ljust(col_fold_type)} | "
        f"{'Tag'.ljust(col_tag)} | "
        f"{'Complete':>10} | {'Failed':>6} | {'Running':>7} | {'Missing':>7}"
    )
    print(header)
    print("-" * len(header))

    for key in sorted(report, key=lambda k: (k.model, k.fold_type, k.tag)):
        counts = _count_statuses(report[key]["folds"])
        total  = key.total_folds
        complete_str = f"{counts[STATUS_SUCCESS]}/{total}"
        print(
            f"{key.model.ljust(col_model)} | "
            f"{key.fold_type.ljust(col_fold_type)} | "
            f"{key.tag.ljust(col_tag)} | "
            f"{complete_str:>10} | "
            f"{counts[STATUS_FAILED]:>6} | "
            f"{counts[STATUS_RUNNING]:>7} | "
            f"{counts[STATUS_MISSING]:>7}"
        )


def print_verbose(report: dict[GroupKey, dict]) -> None:
    status_symbols = {
        STATUS_SUCCESS: "✓",
        STATUS_FAILED:   "✗",
        STATUS_RUNNING:  "~",
        STATUS_MISSING:  "?",
    }

    for key in sorted(report, key=lambda k: (k.model, k.fold_type, k.tag)):
        folds = report[key]["folds"]
        print(f"\nModel: {key.model}  |  Fold Type: {key.fold_type}  |  Tag: {key.tag}")
        fold_width = len(str(key.total_folds))
        for fold_num in sorted(folds):
            status = folds[fold_num]
            sym    = status_symbols[status]
            print(f"  Fold {str(fold_num).rjust(fold_width)}: {sym} {status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track SLURM training job progress from .log files."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("."),
        help="Directory containing .log files (default: current directory).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-fold detail in addition to the summary table.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Filter results to a specific tag (e.g. test-holdout).",
    )
    args = parser.parse_args()

    directory = args.dir.resolve()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory.", file=sys.stderr)
        sys.exit(1)

    logs   = collect_logs(directory, tag_filter=args.tag)
    matrix = infer_expected_matrix(logs)
    report = build_report(logs, matrix)

    print_summary(report)
    if args.verbose:
        print_verbose(report)


if __name__ == "__main__":
    main()
