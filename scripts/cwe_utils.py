"""Shared CWE helpers for analysis scripts (test/holdout metrics only in downstream code)."""

from __future__ import annotations

import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import polars as pl


def explode_with_cwe_id(df: pl.DataFrame) -> pl.DataFrame:
    """Explode cwe_ids and parse numeric CWE id into cwe_id."""
    return (
        df.explode("cwe_ids")
        .drop_nulls("cwe_ids")
        .with_columns(
            pl.col("cwe_ids")
            .cast(pl.Utf8)
            .str.extract(r"(\d+)")
            .cast(pl.Int64, strict=False)
            .alias("cwe_id")
        )
        .drop_nulls("cwe_id")
    )


def with_single_cwe_id(df: pl.DataFrame) -> pl.DataFrame:
    """Assign one CWE id per sample (max numeric id from cwe_ids)."""
    return (
        df.with_columns(
            pl.col("cwe_ids")
            .list.eval(
                pl.element()
                .cast(pl.Utf8)
                .str.extract(r"(\d+)")
                .cast(pl.Int64, strict=False)
            )
            .list.max()
            .alias("cwe_id")
        )
        .drop_nulls("cwe_id")
    )


def sanitize_for_path(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def apply_cwe_label_lazy(lf: pl.LazyFrame, mode: str) -> pl.LazyFrame:
    """Apply explode/primary CWE transform on a LazyFrame (no collect yet).

    Use this for a single fused ``.collect()`` after ``scan_parquet`` + filters so Polars
    can optimize the full plan and use its thread pool for decode/explode work. For
    ``prune_ancestors=True`` use :func:`prepare_cwe_labels` on an eager frame instead
    (Python UDF path).
    """
    if mode == "primary":
        return (
            lf.with_columns(
                pl.col("cwe_ids")
                .list.eval(
                    pl.element()
                    .cast(pl.Utf8)
                    .str.extract(r"(\d+)")
                    .cast(pl.Int64, strict=False)
                )
                .list.max()
                .alias("cwe_id")
            )
            .drop_nulls("cwe_id")
        )
    if mode == "explode":
        return (
            lf.explode("cwe_ids")
            .drop_nulls("cwe_ids")
            .with_columns(
                pl.col("cwe_ids")
                .cast(pl.Utf8)
                .str.extract(r"(\d+)")
                .cast(pl.Int64, strict=False)
                .alias("cwe_id")
            )
            .drop_nulls("cwe_id")
        )
    raise ValueError(f"Unknown cwe_label mode: {mode!r} (use 'explode' or 'primary')")


def _resolve_prune_process_workers(n: int) -> int:
    """1 = sequential in-process prune; 0 or negative = auto min(8, cpus-1); else cap at n."""
    if n == 1:
        return 1
    if n <= 0:
        return min(8, max(1, (os.cpu_count() or 1) - 1))
    return max(1, n)


def _effective_prune_max_workers(prune_process_workers: int | None) -> int:
    """None or omit → 0 (auto); 1 = sequential; N>1 explicit cap."""
    return 0 if prune_process_workers is None else prune_process_workers


def _row_slices_for_chunks(n_rows: int, num_chunks: int) -> list[tuple[int, int]]:
    """Return (offset, length) slices covering all rows; num_chunks capped by n_rows."""
    if n_rows <= 0:
        return []
    k = max(1, min(num_chunks, n_rows))
    base, rem = divmod(n_rows, k)
    out: list[tuple[int, int]] = []
    off = 0
    for i in range(k):
        ln = base + (1 if i < rem else 0)
        if ln > 0:
            out.append((off, ln))
            off += ln
    return out


def _prune_cwe_ids_chunk_worker(chunk: pl.DataFrame) -> pl.DataFrame:
    """Module-level worker for multiprocessing (must be picklable)."""
    return with_pruned_cwe_ids(chunk)


def with_pruned_cwe_ids_parallel(
    df: pl.DataFrame,
    *,
    max_workers: int,
    log_prune_parallel: bool = False,
) -> pl.DataFrame:
    """Row-independent ``with_pruned_cwe_ids`` via process pool; falls back when one worker or one chunk."""
    nw = _resolve_prune_process_workers(max_workers)
    n = df.height
    if nw <= 1:
        return with_pruned_cwe_ids(df)
    slices = _row_slices_for_chunks(n, nw)
    if len(slices) <= 1:
        return with_pruned_cwe_ids(df)
    chunks = [df.slice(off, ln) for off, ln in slices]
    use_workers = min(nw, len(chunks))
    if log_prune_parallel:
        print(
            f"       [prune] multiprocessing: {n:,} rows, "
            f"{use_workers} worker(s), {len(chunks)} chunk(s)"
        )
    # spawn avoids fork-after-thread issues in callers (e.g. pytest) on Linux.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=use_workers, mp_context=ctx) as pool:
        parts = list(pool.map(_prune_cwe_ids_chunk_worker, chunks))
    return pl.concat(parts, how="vertical")


def prepare_cwe_labels(
    df: pl.DataFrame,
    mode: str,
    prune_ancestors: bool = False,
    *,
    prune_process_workers: int | None = None,
    log_prune_parallel: bool = False,
) -> pl.DataFrame:
    if prune_ancestors:
        df = with_pruned_cwe_ids_parallel(
            df,
            max_workers=_effective_prune_max_workers(prune_process_workers),
            log_prune_parallel=log_prune_parallel,
        )
    if mode == "primary":
        return with_single_cwe_id(df)
    if mode == "explode":
        return explode_with_cwe_id(df)
    raise ValueError(f"Unknown cwe_label mode: {mode!r} (use 'explode' or 'primary')")


def with_pruned_cwe_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Prune cwe_ids lists: drop any CWE that is an ancestor of another in the same list."""
    try:
        from scripts.cwe_mapping import prune_redundant_ancestors
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from scripts.cwe_mapping import prune_redundant_ancestors

    _NUM_RE = re.compile(r"(\d+)")

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


@lru_cache(maxsize=1)
def _cwe_pillar_lookup() -> pl.DataFrame:
    """One-row-per-CWE lookup frame: cwe_id → Research Concepts pillar id."""
    try:
        from scripts.cwe_mapping import (
            RESEARCH_CONCEPTS_VIEW_ID,
            get_all_research_concepts_paths,
        )
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parents[1]))
        from scripts.cwe_mapping import (
            RESEARCH_CONCEPTS_VIEW_ID,
            get_all_research_concepts_paths,
        )

    paths = get_all_research_concepts_paths()
    # path format: (1000, pillar, ..., cwe_id); len>=2 → pillar at index 1; pillars self-map.
    src = list(paths.keys()) + [RESEARCH_CONCEPTS_VIEW_ID]
    dst = [p[1] if len(p) >= 2 else p[0] for p in paths.values()] + [RESEARCH_CONCEPTS_VIEW_ID]
    return pl.DataFrame(
        {"__src__": src, "__dst__": dst},
        schema={"__src__": pl.Int64, "__dst__": pl.Int64},
    )


def with_cwe_rollup_pillar(
    df: pl.DataFrame, source_col: str = "cwe_id", target_col: str = "cwe_rollup_id"
) -> pl.DataFrame:
    """Map CWE IDs to View 1000 Research Concepts pillar via a vectorized left join."""
    lookup = _cwe_pillar_lookup().rename({"__src__": source_col, "__dst__": target_col})
    src_dtype = df.schema.get(source_col)
    if src_dtype is not None and src_dtype != pl.Int64:
        lookup = lookup.with_columns(pl.col(source_col).cast(src_dtype))
    return df.join(lookup, on=source_col, how="left")
