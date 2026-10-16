#!/usr/bin/env python3
"""汇总三个 seed 的 development-selected operating point，并与论文主结果对齐比较。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

SEEDS = (7, 21, 42)
PAPER_RESULT = (
    "new_reason/outputs/matched_retention_dev_20260725/analysis/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def extract(result: dict[str, Any]) -> dict[str, Any]:
    dev = result["development_selection"]
    test = result["frozen_test_evaluation"]
    return {
        "selected_threshold": dev["selected_threshold"],
        "dev_target_retention": dev["target_supported_query_retention"],
        "dev_selected_retention": dev["selected_empirical_point"][
            "supported_query_retention"
        ],
        "test_native_retention": test["sft_native"]["supported_query_retention"],
        "test_native_fca": test["sft_native"]["unsupported_hallucination_rate"],
        "test_regate_retention": test["regate"]["supported_query_retention"],
        "test_regate_fca": test["regate"]["unsupported_hallucination_rate"],
        "test_regate_strict3": test["regate"]["strict_3_accuracy"],
        "absolute_fca_reduction": test["absolute_fca_reduction"],
        "relative_fca_reduction": test["relative_fca_reduction"],
    }


def main() -> int:
    args = parse_args()
    repo = args.root.resolve().parent

    per_seed: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        path = args.root / f"runs/seed-{seed}/analysis/result.json"
        if not path.exists():
            raise SystemExit(f"missing result for seed {seed}: {path}")
        per_seed[str(seed)] = extract(json.loads(path.read_text(encoding="utf-8")))

    fields = list(next(iter(per_seed.values())))
    aggregate: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [per_seed[str(seed)][field] for seed in SEEDS]
        aggregate[field] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }

    paper_path = repo / PAPER_RESULT
    paper = extract(json.loads(paper_path.read_text(encoding="utf-8")))

    comparison = {}
    for field in fields:
        lo, hi = aggregate[field]["min"], aggregate[field]["max"]
        mean, std = aggregate[field]["mean"], aggregate[field]["std"]
        comparison[field] = {
            "paper": paper[field],
            "multiseed_mean": mean,
            "multiseed_std": std,
            "paper_within_seed_range": lo <= paper[field] <= hi,
            "paper_within_one_std": abs(paper[field] - mean) <= std if std else None,
            "paper_minus_mean": paper[field] - mean,
        }

    summary = {
        "protocol": (
            "Development-selected operating point, identical to the paper: choose the "
            "highest empirical p_answer threshold whose dev supported-query retention is "
            "at least the SFT-native dev retention on the same 2,408-record canonical "
            "development split, freeze it, then apply to that run's own 2,646-record "
            "comprehensive test predictions."
        ),
        "condition": "grouped",
        "seeds": list(SEEDS),
        "dev_records": 2408,
        "test_records": 2646,
        "paper_reference": PAPER_RESULT,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paper_comparison": comparison,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n=== development-selected operating point, grouped, {len(SEEDS)} seeds ===")
    header = f"{'metric':<28}"
    for seed in SEEDS:
        header += f"{'seed-' + str(seed):>12}"
    header += f"{'mean':>12}{'std':>10}{'paper':>12}"
    print(header)
    for field in fields:
        line = f"{field:<28}"
        for seed in SEEDS:
            line += f"{per_seed[str(seed)][field]:>12.4f}"
        line += f"{aggregate[field]['mean']:>12.4f}{aggregate[field]['std']:>10.4f}"
        line += f"{paper[field]:>12.4f}"
        print(line)
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
