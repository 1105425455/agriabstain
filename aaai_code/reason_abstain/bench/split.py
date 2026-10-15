"""Leakage-safe assignment of source images to the train, development and test splits.

Mechanism:
- The unit of assignment is the **source image**, keyed by ``(dataset, source_image_id)``.
  Every metamorphic group and sample derived from one image must land in the same split;
  otherwise the model sees an image during training and its transformation at test time,
  which leaks and invalidates any generalization claim.
- **Stratification** is by ``(dataset, stratum_label)``, where the stratum is the
  representative category of the source image. This gives every category representation in
  all three splits instead of confining a category to one split.
- **Determinism**: a given ``seed`` reproduces the assignment exactly, because image keys are
  sorted before slicing and the result does not depend on dict ordering.

See ``scripts/assign_splits.py`` for the entry point.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

SPLIT_NAMES = ("train", "dev", "test")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    dev: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.dev + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if min(self.train, self.dev, self.test) < 0:
            raise ValueError("Split ratios must be non-negative")


def image_key(record: dict[str, Any]) -> tuple[str, str]:
    """Leakage-safe split unit: (dataset, source_image_id).

    Falls back to parsing the group_id prefix when source_image_id is absent, and finally
    to the group_id itself, so a record always maps to a stable unit.
    """
    dataset = str(record.get("dataset", "unknown"))
    source_image_id = record.get("source_image_id")
    if source_image_id is not None and str(source_image_id) != "":
        return dataset, str(source_image_id)
    return dataset, str(record.get("group_id", record.get("sample_id", "unknown")))


def _stratum_label(records: list[dict[str, Any]]) -> str:
    """Representative class for an image-unit, used for stratification.

    Uses the label of the answerable base sample (the ground-truth target class). If no base
    sample, falls back to the most common non-empty label, else "_unlabeled".
    """
    base_labels = [
        str(r.get("label"))
        for r in records
        if isinstance(r.get("metadata"), dict)
        and r["metadata"].get("relation") == "base"
        and r.get("label")
    ]
    if base_labels:
        return base_labels[0]
    labels = [str(r.get("label")) for r in records if r.get("label")]
    if labels:
        return max(set(labels), key=labels.count)
    return "_unlabeled"


def _assign_units_in_stratum(
    units: list[tuple[str, str]],
    ratios: SplitRatios,
    seed: int,
    stratum: str,
) -> dict[tuple[str, str], str]:
    """Deterministically assign image-units within one stratum to splits.

    Largest-remainder allocation on a deterministically shuffled order. When a stratum is too
    small to give dev/test a whole unit, dev/test get priority for at least one unit each
    (so no class is missing from dev/test when there are >=2 / >=3 units).
    """
    import random

    rng = random.Random(f"{seed}:{stratum}")
    ordered = sorted(units)
    rng.shuffle(ordered)

    n = len(ordered)
    assignment: dict[tuple[str, str], str] = {}
    if n == 0:
        return assignment
    if n == 1:
        assignment[ordered[0]] = "train"
        return assignment
    if n == 2:
        assignment[ordered[0]] = "train"
        assignment[ordered[1]] = "dev"
        return assignment

    # n >= 3: guarantee >=1 in each split, allocate the rest by ratio (largest remainder).
    n_dev = max(1, round(n * ratios.dev))
    n_test = max(1, round(n * ratios.test))
    if n_dev + n_test >= n:  # keep at least 1 for train
        n_dev = 1
        n_test = 1
    n_train = n - n_dev - n_test

    idx = 0
    for _ in range(n_train):
        assignment[ordered[idx]] = "train"
        idx += 1
    for _ in range(n_dev):
        assignment[ordered[idx]] = "dev"
        idx += 1
    for _ in range(n_test):
        assignment[ordered[idx]] = "test"
        idx += 1
    return assignment


def assign_splits(
    records: list[dict[str, Any]],
    ratios: SplitRatios | None = None,
    seed: int = 20260622,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign a leakage-safe, stratified split to every record (mutates split field).

    Returns (records_with_split, summary). All records sharing an image_key get the same split.
    """
    ratios = ratios or SplitRatios()

    records_by_unit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_unit[image_key(record)].append(record)

    # Group image-units by (dataset, stratum) for stratified assignment.
    units_by_stratum: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    unit_stratum: dict[tuple[str, str], str] = {}
    for unit, unit_records in records_by_unit.items():
        dataset = unit[0]
        stratum = _stratum_label(unit_records)
        unit_stratum[unit] = stratum
        units_by_stratum[(dataset, stratum)].append(unit)

    unit_split: dict[tuple[str, str], str] = {}
    for (dataset, stratum), units in units_by_stratum.items():
        unit_split.update(_assign_units_in_stratum(units, ratios, seed, f"{dataset}:{stratum}"))

    for unit, unit_records in records_by_unit.items():
        split = unit_split[unit]
        for record in unit_records:
            record["split"] = split

    summary = _summarize(records, records_by_unit, unit_split, unit_stratum)
    return records, summary


def _summarize(
    records: list[dict[str, Any]],
    records_by_unit: dict[tuple[str, str], list[dict[str, Any]]],
    unit_split: dict[tuple[str, str], str],
    unit_stratum: dict[tuple[str, str], str],
) -> dict[str, Any]:
    record_counts = {name: 0 for name in SPLIT_NAMES}
    unit_counts = {name: 0 for name in SPLIT_NAMES}
    for unit, unit_records in records_by_unit.items():
        split = unit_split[unit]
        record_counts[split] += len(unit_records)
        unit_counts[split] += 1

    # Per-dataset record counts.
    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: {n: 0 for n in SPLIT_NAMES})
    for record in records:
        by_dataset[str(record.get("dataset"))][str(record["split"])] += 1

    # Leakage check: every image_key maps to exactly one split.
    unit_split_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        unit_split_sets[image_key(record)].add(str(record["split"]))
    leaked_units = [u for u, s in unit_split_sets.items() if len(s) > 1]

    return {
        "split_unit": "(dataset, source_image_id)",
        "total_records": len(records),
        "total_image_units": len(records_by_unit),
        "record_counts": record_counts,
        "image_unit_counts": unit_counts,
        "by_dataset_records": {k: by_dataset[k] for k in sorted(by_dataset)},
        "leaked_image_units": len(leaked_units),
        "leakage_safe": len(leaked_units) == 0,
    }
