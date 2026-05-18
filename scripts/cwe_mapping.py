"""Map CWE IDs to the CWE View 1000 (Research Concepts) pillars.

Reference: https://cwe.mitre.org/data/definitions/1000.html
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

_LOCAL_CSV = Path(__file__).resolve().parents[1] / "data" / "cwe_1000.csv"
_VIEW_1000_CSV_ZIP = "https://cwe.mitre.org/data/csv/1000.csv.zip"

RESEARCH_CONCEPTS_VIEW_ID = 1000
RESEARCH_CONCEPTS_PILLARS = {
    284,  # Improper Access Control
    435,  # Improper Interaction Between Multiple Correctly-Behaving Entities
    664,  # Improper Control of a Resource Through its Lifetime
    682,  # Incorrect Calculation
    691,  # Insufficient Control Flow Management
    693,  # Protection Mechanism Failure
    697,  # Incorrect Comparison
    703,  # Improper Check or Handling of Exceptional Conditions
    707,  # Improper Neutralization
    710,  # Improper Adherence to Coding Standards
}

PILLAR_NAMES: dict[int, str] = {
    284: "Improper Access Control",
    435: "Improper Interaction Between Entities",
    664: "Improper Resource Control",
    682: "Incorrect Calculation",
    691: "Insufficient Control Flow",
    693: "Protection Mechanism Failure",
    697: "Incorrect Comparison",
    703: "Improper Exception Handling",
    707: "Improper Neutralization",
    710: "Improper Coding Standards",
}

_CHILD_OF_RE = re.compile(r"::NATURE:ChildOf:CWE ID:(\d+):VIEW ID:1000(?::|::)")


@lru_cache(maxsize=1)
def _view_1000_parent_map() -> dict[int, tuple[int, ...]]:
    """Build {cwe_id: (parent_ids...)} using ChildOf edges in View 1000.

    Loads from the bundled local CSV (data/cwe_1000.csv) when available;
    falls back to a live download from cwe.mitre.org otherwise.
    """
    if _LOCAL_CSV.exists():
        csv_text = _LOCAL_CSV.read_text(encoding="utf-8", errors="ignore")
    else:
        raw_zip = urllib.request.urlopen(_VIEW_1000_CSV_ZIP, timeout=30).read()
        archive = zipfile.ZipFile(io.BytesIO(raw_zip))
        csv_text = archive.read("1000.csv").decode("utf-8", errors="ignore")
    rows = csv.DictReader(io.StringIO(csv_text))

    parent_map: dict[int, tuple[int, ...]] = {}
    for row in rows:
        cwe_id = int(row["CWE-ID"])
        related = row.get("Related Weaknesses", "")
        parent_ids = tuple(int(match) for match in _CHILD_OF_RE.findall(related))
        parent_map[cwe_id] = parent_ids
    return parent_map


def get_research_concepts_category(cwe_id: int) -> int:
    """Return the Research Concepts pillar for a CWE ID.

    Args:
        cwe_id: CWE integer to map.

    Returns:
        A pillar CWE integer (one of 284, 435, 664, 682, 691, 693, 697, 703,
        707, 710). If cwe_id is one of these pillars, it is returned as-is.

    Raises:
        ValueError: If cwe_id is invalid or has no path to a View 1000 pillar.
    """
    if not isinstance(cwe_id, int) or cwe_id <= 0:
        raise ValueError(f"cwe_id must be a positive integer, got: {cwe_id!r}")

    if cwe_id in RESEARCH_CONCEPTS_PILLARS:
        return cwe_id
    if cwe_id == RESEARCH_CONCEPTS_VIEW_ID:
        return RESEARCH_CONCEPTS_VIEW_ID

    parent_map = _view_1000_parent_map()
    if cwe_id not in parent_map:
        raise ValueError(f"CWE-{cwe_id} not found in MITRE View 1000 dataset.")

    visited: set[int] = set()
    stack = [cwe_id]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        if current in RESEARCH_CONCEPTS_PILLARS:
            return current

        stack.extend(parent_map.get(current, ()))

    raise ValueError(
        f"CWE-{cwe_id} does not map to a configured View 1000 pillar category."
    )


@lru_cache(maxsize=None)
def get_research_concepts_path(cwe_id: int) -> tuple[int, ...]:
    """Return a View 1000 path like (1000, 284, ..., cwe_id).

    For CWEs with multiple parent chains, this returns the shortest path and
    then breaks ties deterministically by smaller parent IDs first.
    """
    if not isinstance(cwe_id, int) or cwe_id <= 0:
        raise ValueError(f"cwe_id must be a positive integer, got: {cwe_id!r}")
    if cwe_id == RESEARCH_CONCEPTS_VIEW_ID:
        return (RESEARCH_CONCEPTS_VIEW_ID,)

    parent_map = _view_1000_parent_map()
    if cwe_id not in parent_map and cwe_id not in RESEARCH_CONCEPTS_PILLARS:
        raise ValueError(f"CWE-{cwe_id} not found in MITRE View 1000 dataset.")

    queue: list[tuple[int, tuple[int, ...]]] = [(cwe_id, (cwe_id,))]
    while queue:
        current, upward_path = queue.pop(0)
        if current in RESEARCH_CONCEPTS_PILLARS:
            return (RESEARCH_CONCEPTS_VIEW_ID,) + tuple(reversed(upward_path))

        for parent in sorted(parent_map.get(current, ())):
            if parent in upward_path:
                continue
            queue.append((parent, upward_path + (parent,)))

    raise ValueError(
        f"CWE-{cwe_id} does not map to a configured View 1000 pillar category."
    )


def get_all_research_concepts_paths() -> dict[int, tuple[int, ...]]:
    """Return {cwe_id: (1000, ..., cwe_id)} for all mappable View 1000 CWEs."""
    all_cwes = sorted(_view_1000_parent_map())
    return {cwe_id: get_research_concepts_path(cwe_id) for cwe_id in all_cwes}


def prune_redundant_ancestors(cwe_ids: frozenset[int]) -> frozenset[int]:
    """Remove any CWE that is an ancestor of another CWE in the same set.

    If cwe_ids contains both a parent and a descendant on the same branch,
    the parent is dropped and only the most specific descendant is kept.
    CWEs not present in the View 1000 parent map are always kept.
    """
    if len(cwe_ids) <= 1:
        return cwe_ids
    parent_map = _view_1000_parent_map()

    def _all_ancestors(cwe_id: int) -> set[int]:
        result: set[int] = set()
        stack = list(parent_map.get(cwe_id, ()))
        while stack:
            p = stack.pop()
            if p not in result:
                result.add(p)
                stack.extend(parent_map.get(p, ()))
        return result

    ancestor_sets = {c: _all_ancestors(c) for c in cwe_ids}
    return frozenset(
        c for c in cwe_ids
        if not any(c in ancestor_sets[other] for other in cwe_ids if other != c)
    )


def print_research_concepts_tree() -> None:
    """Print one path per CWE, e.g. 1000 -> 284 -> 269."""
    for _cwe_id, path in get_all_research_concepts_paths().items():
        print(" -> ".join(str(node) for node in path))


if __name__ == "__main__":
    print_research_concepts_tree()
