#!/usr/bin/env python3
"""Create the conservative L3 hard-evaluation subset from exact visibility endpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def endpoint_ratio(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") or {}
    value = metadata.get("achieved_visible_ratio")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    ratio = float(value)
    if math.isclose(ratio, 0.0, abs_tol=1e-9):
        return 0.0
    if math.isclose(ratio, 1.0, abs_tol=1e-9):
        return 1.0
    return None


def validate_endpoint(row: dict[str, Any], ratio: float) -> None:
    metadata = row.get("metadata") or {}
    expected_decision = "answer" if ratio == 1.0 else "abstain"
    if metadata.get("gt_decision") != expected_decision:
        raise ValueError(f"{metadata.get('sample_id')}: invalid endpoint decision")
    solution = json.loads(row["solution"])
    if solution.get("decision") != expected_decision:
        raise ValueError(f"{metadata.get('sample_id')}: solution/metadata decision mismatch")
    if ratio == 0.0 and solution.get("reason") != "no_target":
        raise ValueError(f"{metadata.get('sample_id')}: zero-visibility reason must be no_target")
    if ratio == 1.0:
        label = solution.get("label")
        box = solution.get("box")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{metadata.get('sample_id')}: supported endpoint needs a label")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{metadata.get('sample_id')}: supported endpoint needs a box")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1000.0
            for value in box
        ):
            raise ValueError(f"{metadata.get('sample_id')}: invalid supported endpoint box")
        if float(box[0]) >= float(box[2]) or float(box[1]) >= float(box[3]):
            raise ValueError(f"{metadata.get('sample_id')}: degenerate supported endpoint box")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    endpoints: list[dict[str, Any]] = []
    total = 0
    counts: Counter[str] = Counter()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            ratio = endpoint_ratio(row)
            if ratio is None:
                continue
            validate_endpoint(row, ratio)
            endpoints.append(row)
            counts[str(ratio)] += 1

    if not endpoints or counts["0.0"] == 0 or counts["1.0"] == 0:
        raise ValueError("L3 endpoint subset must contain both exact 0.0 and 1.0 levels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in endpoints:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "protocol": "l3_exact_endpoint_hard_evaluation_v1",
                "input_records": total,
                "output_records": len(endpoints),
                "visibility_counts": dict(sorted(counts.items())),
                "excluded_intermediate_records": total - len(endpoints),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
