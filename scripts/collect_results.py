#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import polars as pl

try:
    from scripts import DATA_DIR, EDA_DIR, INDEX_RESULTS_FILE, MEGAVUL_DATASET_PARQUET, TB_LOGS_DIR
except ModuleNotFoundError:
    # Support running as a file path: `uv run scripts/collect_results.py`.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts import DATA_DIR, EDA_DIR, INDEX_RESULTS_FILE, MEGAVUL_DATASET_PARQUET, TB_LOGS_DIR


def collect_all_files(
    base_tb_logs_dir: Path,
    phase: List[str] = ["train", "test", "val", "holdout"],
) -> List[Path]:
    all_dirs = []
    for root, dirs, files in os.walk(base_tb_logs_dir):
        for dir in dirs:
            dir_path = Path(root) / dir
            # if "single" in str(dir_path):
            # continue
            # if any(f.endswith(".parquet") for f in os.listdir(dir_path)):
            # all_dirs.append(dir_path / str(f))
            all_dirs.extend(
                [
                    dir_path / f
                    for f in os.listdir(dir_path)
                    if f.endswith(".parquet")
                    and any(
                        [p in f for p in phase],
                    )
                ]
            )
    return all_dirs


def extract_metadata_from_path(path: Path) -> Dict:
    # Example parent directory
    # tag/dataset/datamodule/fold_type/date/fold_number/hyperparameters/model/
    filename_match = re.match(
        r"^(?P<phase>[a-z]+)_predictions_labels_epoch(?P<epoch>\d+)_step(?P<step>\d+)$",
        path.stem,
    )
    if filename_match is None:
        return {}

    parts = path.parts
    if "tb_logs" not in parts:
        return {}
    tb_logs_idx = parts.index("tb_logs")
    if tb_logs_idx + 1 >= len(parts):
        return {}
    run_mode = parts[tb_logs_idx + 1]
    if run_mode not in {"zero_shot", "fine_tuning", "opro"}:
        return {}

    try:
        parts = path.parts
        hyperparameters = parts[-3].split("-")
        dataset_size_token = hyperparameters[4]
        if not dataset_size_token.startswith("n"):
            return {}
        dataset_size_value = dataset_size_token[1:]
        dataset_size = int(dataset_size_value) if dataset_size_value.isdigit() else dataset_size_value
        fold_segment = parts[-4]
        if fold_segment.startswith("fold_") and "_of_" in fold_segment:
            fold_number = int(fold_segment.replace("fold_", "").split("_of_")[0])
            total_folds = int(fold_segment.split("_of_")[-1])
            date = parts[-5]
            fold_type = parts[-6]
            datamodule = parts[-7]
            dataset = parts[-8]
            tag = parts[-9]
        else:
            fold_number = 0
            total_folds = 1
            date = parts[-4]
            fold_type = parts[-5]
            datamodule = parts[-6]
            dataset = parts[-7]
            tag = parts[-8]

        # Parse optional OPRO-specific hyperparameters (steps{X}-trainsize{Y})
        opro_steps = None
        opro_train_size = None
        for token in hyperparameters[5:]:
            if token.startswith("steps"):
                opro_steps = int(token.replace("steps", ""))
            elif token.startswith("trainsize"):
                opro_train_size = int(token.replace("trainsize", ""))

        metadata = {
            "phase": filename_match.group("phase"),
            "epoch": int(filename_match.group("epoch")),
            "step": int(filename_match.group("step")),
            "run_mode": run_mode,
            "model": parts[-2],
            "fold_number": fold_number,
            "total_folds": total_folds,
            "date": date,
            "fold_type": fold_type,
            "datamodule": datamodule,
            "dataset": dataset,
            "tag": tag,
            "batch_size": int(hyperparameters[0].replace("bs", "")),
            "accumulation_steps": int(hyperparameters[1].replace("acc", "")),
            "cw": int(hyperparameters[2].replace("cw", "")),
            "seed": int(hyperparameters[3].replace("seed", "")),
            "dataset_size": dataset_size,
            "opro_steps": opro_steps,
            "opro_train_size": opro_train_size,
        }
    except (IndexError, ValueError):
        return {}

    return metadata


def select_latest_prediction_files(paths: List[Path]) -> List[Path]:
    latest_files: Dict[tuple, tuple[tuple[str, int, int], Path]] = {}
    for path in paths:
        metadata = extract_metadata_from_path(path)
        if metadata == {}:
            continue
        if metadata["run_mode"] == "zero_shot":
            continue
        if metadata["model"].endswith("-gguf") and metadata["run_mode"] != "opro":
            continue
        if metadata["model"] == "codellama-7B-gguf":
            continue
        if metadata["phase"] not in {"test", "holdout"}:
            continue

        # Key covers the full experiment config (excluding date) so that if the same
        # config was run on multiple dates, only the most recent survives.
        key = (
            metadata["model"], metadata["fold_type"], metadata["fold_number"],
            metadata["phase"], metadata["tag"], metadata["dataset"],
            metadata["datamodule"], metadata["seed"], metadata["cw"],
            metadata["total_folds"], metadata["run_mode"], metadata["dataset_size"],
        )
        recency = (metadata["date"], metadata["epoch"], metadata["step"])
        current = latest_files.get(key)
        if current is None or recency > current[0]:
            latest_files[key] = (recency, path)

    return [selected_path for _, selected_path in latest_files.values()]


if __name__ == "__main__":
    # Setup args for argparse
    argparser = ArgumentParser(description="Collect TensorBoard results from remote server.")
    argparser.add_argument(
        "--fir",
        action="store_true",
        help="Rsync from fir.alliancecan.ca",
    )
    argparser.add_argument(
        "--rorqual",
        action="store_true",
        help="Rsync from rorqual.alliancecan.ca",
    )
    argparser.add_argument(
        "--exclude-full",
        action="store_true",
        help="Drop rows where dataset_size is the string 'full' (e.g. nfull runs). Default keeps all sizes.",
    )
    args = argparser.parse_args()
    remotes = []
    if args.rorqual:
        remotes.append("adefu020@rorqual.alliancecan.ca:/home/adefu020/links")
    if args.fir:
        remotes.append("adefu020@fir.alliancecan.ca:/home/adefu020")

    for remote in remotes:
        filter = '--include="*/" --include="holdout*.parquet" --include="test*.parquet" --exclude="*"'
        os.system(
            f"rsync -av --info=progress2 {filter} {remote}/projects/def-pbranco/adefu020/SCVD/data/tb_logs"
            f" {DATA_DIR}/"
        )
    all_results_paths = collect_all_files(TB_LOGS_DIR.parent, phase=["test", "holdout"])
    latest_results_paths = select_latest_prediction_files(all_results_paths)
    print(f"Collected {len(all_results_paths)} result files.")
    print(f"Selected {len(latest_results_paths)} latest result files.")
    print(f"Dropped {len(all_results_paths) - len(latest_results_paths)} older result files.")

    dataset = pl.read_parquet(MEGAVUL_DATASET_PARQUET).with_columns([pl.col("index").alias("sample_index")]).select(
        [
            "sample_index",
            "project",
            "commit_hash",
            "publish_date",
            "cwe_ids",
        ]
    )

    groups: set[tuple[str, int, str, str]] = set()
    timing_stats: Dict[tuple[str, int, str], Dict[str, object]] = {}
    chunk_paths: List[Path] = []
    rows_written = 0

    with TemporaryDirectory(prefix="collect_results_chunks_", dir=EDA_DIR) as temp_dir:
        for idx, path in enumerate(latest_results_paths):
            metadata = extract_metadata_from_path(path)
            if metadata == {}:
                continue
            if args.exclude_full and metadata["dataset_size"] == "full":
                continue

            df = pl.read_parquet(path)
            if df.height == 0:
                continue

            df = df.with_columns([pl.lit(v).alias(k) for k, v in metadata.items()])
            df = df.with_columns([pl.col("index").cast(pl.Int64).alias("sample_index")])
            df = df.join(dataset, left_on="sample_index", right_on="sample_index", how="left")
            if df.height == 0:
                continue

            chunk_path = Path(temp_dir) / f"chunk_{idx:05d}.parquet"
            df.write_parquet(chunk_path)
            chunk_paths.append(chunk_path)
            rows_written += df.height

            groups.add((metadata["fold_type"], metadata["fold_number"], metadata["phase"], metadata["model"]))
            timing_key = (metadata["fold_type"], metadata["fold_number"], metadata["phase"])
            start_date = df.select(pl.col("publish_date").min()).item()
            end_date = df.select(pl.col("publish_date").max()).item()
            existing = timing_stats.get(timing_key)
            if existing is None:
                timing_stats[timing_key] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "samples": df.height,
                }
            else:
                timing_stats[timing_key] = {
                    "start_date": min(existing["start_date"], start_date),
                    "end_date": max(existing["end_date"], end_date),
                    "samples": existing["samples"] + df.height,
                }

        if rows_written == 0:
            print("No rows after filtering; skipping index file. Exiting.")
            sys.exit(0)

        pl.concat([pl.scan_parquet(path) for path in chunk_paths], how="diagonal_relaxed").sink_parquet(INDEX_RESULTS_FILE)

    for row in sorted(groups, key=lambda g: (g[3], g[0], g[1])):
        print(row)

    if not timing_stats:
        print("No timing groups.")
        sys.exit(0)

    for (fold_type, fold_number, phase), stats in sorted(timing_stats.items(), key=lambda item: item[0][0]):
        print(
            f" {phase} | {fold_type} | Fold {fold_number} "
            f"| {stats['start_date']} to {stats['end_date']} | Samples: {stats['samples']}"
        )
