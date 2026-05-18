"""CWE-based pure undersampling and pure oversampling using val_most_common strategy."""
from __future__ import annotations

from math import floor

import polars as pl

from src.datasets.mixed_resampling import (
    _assign_cwe,
    _assigned_cwe_counts,
    _capped_oversample_indices,
    _exploded_cwe_counts,
    _filter_train_for_val_cwes,
    _greedy_explode_oversample_capped,
    _greedy_explode_undersample,
    _prune_cwe_ancestors,
)


def undersample_train_indices(
    df: pl.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
    multiplier: float,
    seed: int,
    cwe_assignment_strategy: str = "val_most_common",
) -> tuple[list[int], list[int]]:
    """Return (kept_indices, []) using pure CWE undersampling.

    For each CWE in the train/val intersection:
      scale = (val_freq * multiplier) / train_freq
      - scale <= 1 (overrepresented): undersample to floor(train_count * scale)
      - scale > 1 (underrepresented): keep all rows unchanged
    CWE buckets absent from val are dropped. Never adds rows.

    ``cwe_assignment_strategy`` controls how rows are bucketed; see
    :func:`mixed_resample_train_indices` for the available strategies.
    """
    train_df = _prune_cwe_ancestors(df.filter(pl.col("index").is_in(train_indices)))
    val_df = _prune_cwe_ancestors(df.filter(pl.col("index").is_in(val_indices)))

    if cwe_assignment_strategy == "explode":
        return _undersample_explode(train_df, val_df, multiplier, seed), []

    val_exploded = (
        _exploded_cwe_counts(val_df) if cwe_assignment_strategy == "val_most_common" else None
    )
    train_df = _assign_cwe(train_df, cwe_assignment_strategy, val_exploded)
    val_df = _assign_cwe(val_df, cwe_assignment_strategy, val_exploded)
    val_dist = _assigned_cwe_counts(val_df)

    train_df = train_df.filter(pl.col("assigned_cwe").is_in(list(val_dist.keys())))
    train_dist = _assigned_cwe_counts(train_df)

    total_val = sum(v for cwe, v in val_dist.items() if cwe in train_dist) or 1
    total_train = sum(train_dist.values()) or 1

    kept: list[int] = []
    for cwe, val_count in val_dist.items():
        train_count = train_dist.get(cwe, 0)
        if train_count == 0:
            continue
        bucket = train_df.filter(pl.col("assigned_cwe") == cwe)
        scale = (val_count / total_val * multiplier) / (train_count / total_train)
        if scale <= 1.0:
            n_keep = floor(train_count * scale)
            if n_keep > 0:
                kept.extend(bucket.sample(n=n_keep, seed=seed)["index"].to_list())
        else:
            kept.extend(bucket["index"].to_list())
    return kept, []


def _undersample_explode(
    train_df: pl.DataFrame, val_df: pl.DataFrame, multiplier: float, seed: int
) -> list[int]:
    """Pure undersample with explode strategy.

    Per-CWE budgets are floor(train_count * scale) with scale capped at 1.0;
    CWEs absent from val have val_count=0 and are skipped (their rows are
    dropped via the greedy bucket allocator).
    """
    val_dist = _exploded_cwe_counts(val_df)
    train_dist = _exploded_cwe_counts(train_df)
    total_val = sum(v for cwe, v in val_dist.items() if cwe in train_dist) or 1
    total_train = sum(train_dist.values()) or 1

    budgets: dict[str, int] = {}
    for cwe, train_count in train_dist.items():
        val_count = val_dist.get(cwe, 0)
        if val_count == 0:
            continue
        scale = min((val_count / total_val * multiplier) / (train_count / total_train), 1.0)
        budgets[cwe] = floor(train_count * scale)
    return _greedy_explode_undersample(train_df, budgets, seed)


def oversample_train_indices(
    df: pl.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
    multiplier: float,
    seed: int,
    oversample_cap: float = 2.0,
    cwe_assignment_strategy: str = "val_most_common",
) -> tuple[list[int], list[int]]:
    """Return (kept_indices, extra_indices) using pure CWE oversampling.

    For each CWE in the train/val intersection:
      scale = (val_freq * multiplier) / train_freq
      - scale < 1 (overrepresented): keep all rows unchanged
      - scale >= 1 (underrepresented): keep all rows and add
        min(floor(train_count*(scale-1)), floor(train_count*oversample_cap)) extra copies
    CWE buckets absent from val are dropped.
    extra_indices is a flat list of original indices (with repetitions) to append as train rows.

    ``cwe_assignment_strategy`` controls how rows are bucketed; see
    :func:`mixed_resample_train_indices` for the available strategies.
    """
    train_df = _prune_cwe_ancestors(df.filter(pl.col("index").is_in(train_indices)))
    val_df = _prune_cwe_ancestors(df.filter(pl.col("index").is_in(val_indices)))

    if cwe_assignment_strategy == "explode":
        kept, extras = _oversample_explode(train_df, val_df, multiplier, seed, oversample_cap)
        return kept, extras

    val_exploded = (
        _exploded_cwe_counts(val_df) if cwe_assignment_strategy == "val_most_common" else None
    )
    train_df = _assign_cwe(train_df, cwe_assignment_strategy, val_exploded)
    val_df = _assign_cwe(val_df, cwe_assignment_strategy, val_exploded)
    val_dist = _assigned_cwe_counts(val_df)

    train_df = train_df.filter(pl.col("assigned_cwe").is_in(list(val_dist.keys())))
    train_dist = _assigned_cwe_counts(train_df)

    total_val = sum(v for cwe, v in val_dist.items() if cwe in train_dist) or 1
    total_train = sum(train_dist.values()) or 1

    kept: list[int] = []
    extras: list[int] = []
    for cwe, val_count in val_dist.items():
        train_count = train_dist.get(cwe, 0)
        if train_count == 0:
            continue
        bucket = train_df.filter(pl.col("assigned_cwe") == cwe)
        scale = (val_count / total_val * multiplier) / (train_count / total_train)
        kept.extend(bucket["index"].to_list())
        if scale > 1.0 and oversample_cap > 0.0:
            n_extra = min(
                floor(train_count * (scale - 1)),
                floor(train_count * oversample_cap),
            )
            extras.extend(_capped_oversample_indices(bucket, n_extra, oversample_cap, seed))
    return kept, extras


def _oversample_explode(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    multiplier: float,
    seed: int,
    oversample_cap: float,
) -> tuple[list[int], list[int]]:
    """Pure oversample with explode strategy.

    All filtered train rows are kept; underrepresented CWEs (scale > 1) get
    capped duplicates via the greedy multi-bucket oversampler with global
    per-row cap of ``ceil(oversample_cap)`` extras.
    """
    val_dist = _exploded_cwe_counts(val_df)
    train_df = _filter_train_for_val_cwes(train_df, val_dist)
    train_dist = _exploded_cwe_counts(train_df)
    total_val = sum(v for cwe, v in val_dist.items() if cwe in train_dist) or 1
    total_train = sum(train_dist.values()) or 1

    kept = train_df["index"].to_list()
    if oversample_cap <= 0.0:
        return kept, []

    budgets: dict[str, int] = {}
    for cwe, val_count in val_dist.items():
        train_count = train_dist.get(cwe, 0)
        if train_count == 0:
            continue
        scale = (val_count / total_val * multiplier) / (train_count / total_train)
        if scale > 1.0:
            capped = min(scale, 1.0 + oversample_cap)
            n_extra = floor(train_count * (capped - 1))
            if n_extra > 0:
                budgets[cwe] = n_extra
    extras = _greedy_explode_oversample_capped(train_df, budgets, seed, oversample_cap)
    return kept, extras
