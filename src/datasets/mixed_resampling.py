"""CWE-based train resampling (undersample + optional oversample) using val_most_common strategy."""
from __future__ import annotations

import math
import re
import sys
from math import floor
from pathlib import Path

import polars as pl

NO_CWE = "no-cwe"

_NUM_RE = re.compile(r"(\d+)")


def _prune_cwe_ancestors(df: pl.DataFrame) -> pl.DataFrame:
    """Drop any CWE from cwe_ids that is an ancestor of another CWE in the same list."""
    try:
        from scripts.cwe_mapping import prune_redundant_ancestors
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from scripts.cwe_mapping import prune_redundant_ancestors

    def _prune(cwe_list):
        items = cwe_list.to_list() if hasattr(cwe_list, "to_list") else list(cwe_list)
        if not items or len(items) <= 1:
            return items
        parsed = {}
        for s in items:
            if s is None:
                continue
            m = _NUM_RE.search(str(s))
            if m:
                parsed[int(m.group(1))] = s
        if len(parsed) <= 1:
            return list(parsed.values()) if parsed else items
        pruned = prune_redundant_ancestors(frozenset(parsed.keys()))
        return [parsed[c] for c in parsed if c in pruned]

    return df.with_columns(
        pl.col("cwe_ids").map_elements(_prune, return_dtype=pl.List(pl.Utf8))
    )


def mixed_resample_train_indices(
    df: pl.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
    multiplier: float,
    seed: int,
    oversample_cap: float = 2.0,
    cwe_assignment_strategy: str = "val_most_common",
) -> tuple[list[int], list[int]]:
    """Return (kept_indices, extra_indices) using the chosen CWE assignment strategy.

    All rows (vul and nonvul) are bucketed by assigned_cwe. Rows with empty
    cwe_ids get NO_CWE. CWE buckets absent from val are dropped.

    ``cwe_assignment_strategy``:
      - ``val_most_common``: pick the CWE from the row's list most common in val.
      - ``primary``: pick the first CWE in the (pruned) list.

    RF (relative frequency) is computed over the intersection of train and val
    CWEs; val CWEs absent from train are excluded from total_val so the
    denominator only covers matchable buckets.

    For each CWE in the intersection:
      scale = (val_freq * multiplier) / train_freq
      - If scale <= 1 (overrepresented): undersample to floor(train_count * scale).
      - If scale > 1 (underrepresented) and oversample_cap > 0: keep all train
        rows and add min(floor(train_count*(scale-1)),
        floor(train_count*oversample_cap)) extra copies sampled with replacement.
      - If scale > 1 and oversample_cap == 0: keep all train rows unchanged.

    extra_indices is a flat list of original indices (with repetitions); each
    entry represents one additional duplicate row to append to the train split.
    """
    train_df = _prune_cwe_ancestors(df.filter(pl.col("index").is_in(train_indices)))
    val_df = _prune_cwe_ancestors(df.filter(pl.col("index").is_in(val_indices)))

    if cwe_assignment_strategy == "explode":
        return _mixed_resample_explode(train_df, val_df, multiplier, seed, oversample_cap)

    val_exploded = (
        _exploded_cwe_counts(val_df) if cwe_assignment_strategy == "val_most_common" else None
    )
    train_df = _assign_cwe(train_df, cwe_assignment_strategy, val_exploded)
    val_df = _assign_cwe(val_df, cwe_assignment_strategy, val_exploded)
    val_dist = _assigned_cwe_counts(val_df)

    # Drop train rows whose CWE is absent from val
    train_df = train_df.filter(pl.col("assigned_cwe").is_in(list(val_dist.keys())))
    train_dist = _assigned_cwe_counts(train_df)

    # RF denominators: total_val excludes val CWEs absent from train
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

        if scale <= 1.0:
            n_keep = floor(train_count * scale)
            if n_keep > 0:
                kept.extend(bucket.sample(n=n_keep, seed=seed)["index"].to_list())
        else:
            kept.extend(bucket["index"].to_list())
            if oversample_cap > 0.0:
                n_extra = min(
                    floor(train_count * (scale - 1)),
                    floor(train_count * oversample_cap),
                )
                extras.extend(_capped_oversample_indices(bucket, n_extra, oversample_cap, seed))
    return kept, extras


def _capped_oversample_indices(
    bucket: pl.DataFrame, n_extra: int, oversample_cap: float, seed: int
) -> list[int]:
    """Sample ``n_extra`` indices from ``bucket`` with each row appearing at most
    ``ceil(oversample_cap)`` times. Uses without-replacement passes (each shuffled
    independently) so the per-row duplicate count is hard-bounded — not just the
    bucket total.
    """
    if n_extra <= 0 or bucket.is_empty():
        return []
    train_count = bucket.height
    max_passes = math.ceil(oversample_cap)
    capped_total = min(n_extra, max_passes * train_count)
    result: list[int] = []
    remaining = capped_total
    pass_idx = 0
    while remaining > 0:
        n_this = min(remaining, train_count)
        sampled = bucket.sample(n=n_this, seed=seed + pass_idx)["index"].to_list()
        result.extend(sampled)
        remaining -= n_this
        pass_idx += 1
    return result


def _exploded_cwe_counts(df: pl.DataFrame) -> dict[str, int]:
    """Explode cwe_ids across all rows; empty lists count as NO_CWE."""
    has_cwe = df.filter(pl.col("cwe_ids").list.len() > 0)
    no_cwe_count = df.filter(pl.col("cwe_ids").list.len() == 0).height
    result: dict[str, int] = {}
    if not has_cwe.is_empty():
        exploded = has_cwe.explode("cwe_ids").drop_nulls("cwe_ids")
        counts = exploded.group_by("cwe_ids").agg(pl.len().alias("count"))
        result = dict(zip(counts["cwe_ids"].to_list(), counts["count"].to_list()))
    if no_cwe_count > 0:
        result[NO_CWE] = no_cwe_count
    return result


def _assign_val_most_common(df: pl.DataFrame, val_exploded_counts: dict[str, int]) -> pl.DataFrame:
    """Assign each row to its best-matching CWE by val frequency."""
    def _pick(row: dict) -> str:
        cwe_list = row["cwe_ids"] or []
        if not cwe_list:
            return NO_CWE
        return max(cwe_list, key=lambda c: val_exploded_counts.get(c, 0))

    assigned = [_pick(r) for r in df.iter_rows(named=True)]
    return df.with_columns(pl.Series("assigned_cwe", assigned, dtype=pl.Utf8))


def _assign_primary_cwe(df: pl.DataFrame) -> pl.DataFrame:
    """Assign the first CWE in the (pruned) cwe_ids list, or NO_CWE if the list is empty."""
    return df.with_columns(
        pl.when(pl.col("cwe_ids").list.len() > 0)
        .then(pl.col("cwe_ids").list.first())
        .otherwise(pl.lit(NO_CWE))
        .alias("assigned_cwe")
    )


VALID_CWE_STRATEGIES = ("val_most_common", "primary", "explode")


def _assign_cwe(
    df: pl.DataFrame,
    strategy: str,
    val_exploded_counts: dict[str, int] | None,
) -> pl.DataFrame:
    """Dispatch CWE assignment by strategy.

    ``primary`` ignores ``val_exploded_counts``; ``val_most_common`` requires it.
    ``explode`` is a no-op (caller uses the raw ``cwe_ids`` list directly).
    """
    if strategy == "explode":
        return df
    if strategy == "val_most_common":
        assert val_exploded_counts is not None, "val_exploded_counts required for val_most_common"
        return _assign_val_most_common(df, val_exploded_counts)
    if strategy == "primary":
        return _assign_primary_cwe(df)
    raise ValueError(
        f"Unknown CWE assignment strategy: {strategy!r}. Valid: {VALID_CWE_STRATEGIES}"
    )


def _greedy_explode_undersample(
    train_df: pl.DataFrame, budgets: dict[str, int], seed: int
) -> list[int]:
    """Multi-CWE greedy undersample: each row may claim at most one bucket.

    Iterates CWEs from largest budget to smallest. The ``NO_CWE`` bucket uses
    the empty-list condition since ``list.contains(NO_CWE)`` would not match a
    string list. Returns the list of kept row indices.
    """
    used: set[int] = set()
    kept: list[int] = []
    for cwe, budget in sorted(budgets.items(), key=lambda x: -x[1]):
        if budget <= 0:
            continue
        used_list = list(used)
        if cwe == NO_CWE:
            candidates = train_df.filter(
                (pl.col("cwe_ids").list.len() == 0) & ~pl.col("index").is_in(used_list)
            )
        else:
            candidates = train_df.filter(
                pl.col("cwe_ids").list.contains(cwe) & ~pl.col("index").is_in(used_list)
            )
        n_keep = min(budget, candidates.height)
        if n_keep > 0:
            selected = candidates.sample(n=n_keep, seed=seed)["index"].to_list()
            kept.extend(selected)
            used.update(selected)
    return kept


def _greedy_explode_oversample_capped(
    train_df: pl.DataFrame,
    budgets: dict[str, int],
    seed: int,
    oversample_cap: float,
) -> list[int]:
    """Multi-CWE greedy oversample with strict per-row global cap.

    Iterates CWEs from largest budget to smallest. Within each bucket, samples
    in without-replacement passes (so a single bucket pass can't itself
    duplicate a row). Across buckets, tracks a global per-row extras counter
    and skips rows that have already hit ``ceil(oversample_cap)`` extras —
    later buckets may fall short of their budget but the per-row cap is never
    violated. Returns a flat list of indices (with repetitions) for extras.
    """
    if oversample_cap <= 0.0:
        return []
    max_extras_per_row = math.ceil(oversample_cap)
    extras: list[int] = []
    row_count: dict[int, int] = {}

    for cwe, n_extra in sorted(budgets.items(), key=lambda x: -x[1]):
        if n_extra <= 0:
            continue
        if cwe == NO_CWE:
            bucket = train_df.filter(pl.col("cwe_ids").list.len() == 0)
        else:
            bucket = train_df.filter(pl.col("cwe_ids").list.contains(cwe))
        if bucket.is_empty():
            continue
        bucket_indices = bucket["index"].to_list()
        remaining = n_extra
        pass_idx = 0
        while remaining > 0:
            eligible = [
                i for i in bucket_indices if row_count.get(i, 0) < max_extras_per_row
            ]
            if not eligible:
                break
            n_this = min(remaining, len(eligible))
            sampled = (
                pl.DataFrame({"index": eligible}, schema={"index": train_df.schema["index"]})
                .sample(n=n_this, seed=seed + pass_idx)["index"]
                .to_list()
            )
            extras.extend(sampled)
            for s in sampled:
                row_count[s] = row_count.get(s, 0) + 1
            remaining -= n_this
            pass_idx += 1
    return extras


def _assigned_cwe_counts(df: pl.DataFrame) -> dict[str, int]:
    """Count assigned_cwe values across all rows."""
    if df.is_empty():
        return {}
    counts = df.group_by("assigned_cwe").agg(pl.len().alias("count"))
    return dict(zip(counts["assigned_cwe"].to_list(), counts["count"].to_list()))


def _filter_train_for_val_cwes(train_df: pl.DataFrame, val_dist: dict[str, int]) -> pl.DataFrame:
    """For explode strategy: keep only train rows whose cwe_ids overlap with val.

    Rows whose entire CWE list is absent from val (and with NO_CWE absent from val)
    are dropped, since their explode contribution would all map to dropped buckets.
    """
    val_cwes = set(val_dist.keys())
    non_no_cwe_val = list(val_cwes - {NO_CWE})
    has_val_cwe = pl.col("cwe_ids").list.eval(pl.element().is_in(non_no_cwe_val)).list.any()
    if NO_CWE in val_cwes:
        return train_df.filter(has_val_cwe | (pl.col("cwe_ids").list.len() == 0))
    return train_df.filter(has_val_cwe)


def _mixed_resample_explode(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    multiplier: float,
    seed: int,
    oversample_cap: float,
) -> tuple[list[int], list[int]]:
    """Mixed resample (undersample + capped oversample) using explode strategy.

    Three-phase algorithm:
      1. Compute under/over budgets from the pre-resample exploded distribution.
         For over_budget CWEs, set their budget to full train count so their rows
         survive the greedy undersample (which uses exclusive bucket ownership).
      2. Re-measure the exploded distribution post-undersample.
      3. Compute fresh oversample budgets from the re-measured distribution and
         apply the per-row-capped greedy oversample.
    """
    val_dist = _exploded_cwe_counts(val_df)
    train_df = _filter_train_for_val_cwes(train_df, val_dist)
    train_dist = _exploded_cwe_counts(train_df)
    total_val = sum(v for cwe, v in val_dist.items() if cwe in train_dist) or 1
    total_train = sum(train_dist.values()) or 1

    under_budgets: dict[str, int] = {}
    over_cwes: set[str] = set()
    for cwe, train_count in train_dist.items():
        val_count = val_dist.get(cwe, 0)
        if val_count == 0:
            continue
        scale = (val_count / total_val * multiplier) / (train_count / total_train)
        if scale < 1.0:
            under_budgets[cwe] = floor(train_count * scale)
        else:
            over_cwes.add(cwe)

    # Phase 1: combined undersample — set over_cwes to full count to protect them.
    combined_budgets = dict(under_budgets)
    for cwe in over_cwes:
        combined_budgets[cwe] = train_dist[cwe]
    kept = _greedy_explode_undersample(train_df, combined_budgets, seed)
    undersampled = train_df.filter(pl.col("index").is_in(kept))

    # Phase 2: re-measure.
    post_dist = _exploded_cwe_counts(undersampled)
    post_total = sum(post_dist.values()) or 1
    total_val_fresh = sum(v for cwe, v in val_dist.items() if cwe in post_dist) or 1

    # Phase 3: fresh oversample budgets.
    if oversample_cap <= 0.0:
        return kept, []
    fresh_over: dict[str, int] = {}
    for cwe, post_count in post_dist.items():
        val_count = val_dist.get(cwe, 0)
        if val_count == 0:
            continue
        fresh_scale = (val_count / total_val_fresh * multiplier) / (post_count / post_total)
        if fresh_scale > 1.0:
            capped = min(fresh_scale, 1.0 + oversample_cap)
            n_extra = floor(post_count * (capped - 1))
            if n_extra > 0:
                fresh_over[cwe] = n_extra
    extras = _greedy_explode_oversample_capped(undersampled, fresh_over, seed, oversample_cap)
    return kept, extras
