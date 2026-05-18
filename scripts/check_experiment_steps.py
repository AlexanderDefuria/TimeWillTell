"""
Report max epoch and total steps for each fine-tuning experiment.

Parses filenames of the form:
  {phase}_predictions_labels_epoch{EEE}_step{SSSSSSSS}.parquet

Groups by the experiment directory (everything above the phase-parquet file)
and prints the highest epoch and step seen per experiment.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts import TB_LOGS_DIR

FILENAME_RE = re.compile(r"_epoch(\d+)_step(\d+)\.parquet$")
FT_DIR = TB_LOGS_DIR / "fine_tuning"


def main():
    records: dict[Path, dict] = {}

    holdout_dirs = {
        f.parent
        for f in FT_DIR.rglob("holdout_predictions_labels_epoch*_step*.parquet")
    }

    for f in sorted(FT_DIR.rglob("*_predictions_labels_epoch*_step*.parquet")):
        exp_dir = f.parent
        if exp_dir not in holdout_dirs:
            continue
        m = FILENAME_RE.search(f.name)
        if not m:
            continue
        epoch, step = int(m.group(1)), int(m.group(2))

        if exp_dir not in records:
            records[exp_dir] = {"max_epoch": epoch, "max_step": step}
        else:
            records[exp_dir]["max_epoch"] = max(records[exp_dir]["max_epoch"], epoch)
            records[exp_dir]["max_step"] = max(records[exp_dir]["max_step"], step)

    if not records:
        print("No fine-tuning prediction files found.")
        return

    # Print relative to FT_DIR for readability
    rows = sorted(
        [(path.relative_to(FT_DIR), v["max_epoch"], v["max_step"]) for path, v in records.items()],
        key=lambda r: r[0],
    )

    col_w = max(len(str(r[0])) for r in rows)
    print(f"{'Experiment':<{col_w}}  {'Epochs':>6}  {'Max step':>10}")
    print("-" * (col_w + 20))
    for path, epoch, step in rows:
        print(f"{str(path):<{col_w}}  {epoch:>6}  {step:>10,}")


if __name__ == "__main__":
    main()
