#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""
Master plotting runner — executes all canonical data/plots/ scripts in sequence.

Steps (in order):
  results  →  scripts/plot_results.py
  pvalue   →  scripts/plot_pvalue_summary.py
  jsd      →  scripts/plot_jsd_vs_f1.py
  contrast →  scripts/plot_test_holdout_contrast.py

Usage:
    uv run scripts/plot_all.py                          # full run
    uv run scripts/plot_all.py --dry-run                # show commands only
    uv run scripts/plot_all.py --skip jsd               # skip one step
    uv run scripts/plot_all.py --only pvalue            # run one step
    uv run scripts/plot_all.py --tag fine-tuning --dataset megavul
"""
from __future__ import annotations

import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path


# ─── Step registry ────────────────────────────────────────────────────────────


@dataclass
class Step:
    name: str
    script: str
    common: list[str] = field(default_factory=list)


STEPS: list[Step] = [
    Step("summary",  "scripts/plot_dataset_summary.py",     common=[]),
    Step("results",  "scripts/plot_results.py",             common=["--results", "--tag", "--dataset"]),
Step("pvalue",   "scripts/plot_pvalue_summary.py",      common=["--results", "--tag", "--dataset"]),
    Step("jsd",        "scripts/plot_jsd_vs_f1.py",             common=["--tag", "--dataset"]),
    Step("resampling", "scripts/plot_cwe_resampling.py",       common=["--tag", "--dataset"]),
    Step("contrast",   "scripts/plot_test_holdout_contrast.py", common=["--results", "--tag", "--dataset"]),
    Step("approach-cmp", "scripts/plot_approach_comparison.py", common=["--results", "--tag", "--dataset"]),
]

STEP_NAMES = [s.name for s in STEPS]


# ─── Argument forwarding ──────────────────────────────────────────────────────


def _build_cmd(step: Step, args) -> list[str]:
    """Build the full subprocess command for a step, forwarding only declared args."""
    cmd = ["uv", "run", step.script]
    if "--results" in step.common and args.results:
        cmd += ["--results", str(args.results)]
    if "--tag" in step.common and args.tag:
        cmd += ["--tag", args.tag]
    if "--dataset" in step.common and args.dataset:
        cmd += ["--dataset", args.dataset]
    return cmd


# ─── Display helpers ──────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _print_summary(results: list[tuple[str, str, float]]) -> None:
    col_w = [max(len("Step"), max(len(n) for n, _, _ in results)),
             max(len("Status"), max(len(st) for _, st, _ in results)),
             max(len("Duration"), max(len(_fmt_duration(d)) for _, _, d in results))]

    def row(a: str, b: str, c: str, sep: str = "║") -> str:
        return f"{sep} {a:<{col_w[0]}} {sep} {b:<{col_w[1]}} {sep} {c:>{col_w[2]}} {sep}"

    h_top  = "╔" + "╦".join("═" * (w + 2) for w in col_w) + "╗"
    h_mid  = "╠" + "╬".join("═" * (w + 2) for w in col_w) + "╣"
    h_bot  = "╚" + "╩".join("═" * (w + 2) for w in col_w) + "╝"
    title  = "plot_all  —  run summary"
    width  = len(h_top)
    banner = "║" + title.center(width - 2) + "║"

    print()
    print(h_top)
    print(banner)
    print(h_mid)
    print(row("Step", "Status", "Duration"))
    print(h_mid)
    for name, status, duration in results:
        print(row(name, status, _fmt_duration(duration)))
    print(h_bot)

    n_failed = sum(1 for _, st, _ in results if st != "OK")
    if n_failed:
        print(f"{n_failed} step(s) failed.")
    else:
        print("All steps completed successfully.")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = ArgumentParser(description="Run all canonical data/plots/ plotting scripts.")
    parser.add_argument("--results", type=Path, default=None,
                        help="Override path to index_results.parquet")
    parser.add_argument("--tag", type=str, default=None,
                        help="Filter by experiment tag (forwarded to all scripts)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter by dataset name (forwarded to compatible scripts)")
    parser.add_argument("--skip", dest="skip", action="append", default=[],
                        metavar="NAME", choices=STEP_NAMES,
                        help=f"Skip a step by name; may be repeated. Choices: {STEP_NAMES}")
    parser.add_argument("--only", dest="only", action="append", default=[],
                        metavar="NAME", choices=STEP_NAMES,
                        help="Run only the named step(s); may be repeated")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands that would run, then exit")
    args = parser.parse_args()

    # Resolve which steps to run
    active = STEPS
    if args.only:
        active = [s for s in STEPS if s.name in args.only]
    elif args.skip:
        active = [s for s in STEPS if s.name not in args.skip]

    if not active:
        print("No steps to run (check --skip / --only arguments).")
        sys.exit(0)

    # Build commands
    cmds = {step.name: _build_cmd(step, args) for step in active}

    if args.dry_run:
        print("Dry run — commands that would execute:")
        for name, cmd in cmds.items():
            print(f"  [{name}]  {' '.join(cmd)}")
        return

    # Execute
    summary: list[tuple[str, str, float]] = []
    for step in active:
        cmd = cmds[step.name]
        print(f"\n{'─' * 60}")
        print(f"  Step: {step.name}  →  {step.script}")
        print(f"  Cmd : {' '.join(cmd)}")
        print(f"{'─' * 60}")

        t0 = time.monotonic()
        result = subprocess.run(cmd, text=True, stderr=subprocess.PIPE)
        duration = time.monotonic() - t0

        if result.returncode != 0:
            status = "FAILED"
            stderr_lines = (result.stderr or "").splitlines()
            tail = stderr_lines[-20:] if len(stderr_lines) > 20 else stderr_lines
            print(f"\n[{step.name}] FAILED (exit {result.returncode}) — last stderr lines:")
            for line in tail:
                print(f"    {line}")
        else:
            status = "OK"

        summary.append((step.name, status, duration))

    _print_summary(summary)

    if any(st != "OK" for _, st, _ in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
