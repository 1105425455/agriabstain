#!/usr/bin/env python3
"""Aggregate all frozen multi-seed runs without selecting runs by test results."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
AAAI_ROOT = EXPERIMENT_ROOT.parent / "aaai_code"
sys.path.insert(0, str(AAAI_ROOT))

from experiments.grouping_ablation.summarize_results import extract  # noqa: E402

SEEDS = (7, 21, 42)
CONDITIONS = ("grouped", "independent")
REGATE_DATASETS = (
    "comprehensive",
    "L1_discrim",
    "L2_evidence",
    "L3_dose",
    "refcocog",
    "plantwild",
)
NATIVE_DATASETS = ("comprehensive", "refcocog", "plantwild")
METRICS = ("FCA", "A_FU", "MCC", "Strict_3", "E2E_mIoU", "MTS_macro", "T6")


def linear_extract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "FCA": value["FCA"],
        "A_FU": value["A_FU"],
        "MCC": value["MCC"],
        "Strict_3": value["strict_3_accuracy"],
        "E2E_mIoU": None,
        "MTS_macro": None,
        "T6": None,
        "state_accuracy": value["state_accuracy"],
    }


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=EXPERIMENT_ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "summary")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_root = args.runs_root / condition / f"seed-{seed}"
            for dataset in REGATE_DATASETS:
                mlp_path = run_root / "evaluation/regate" / dataset / "metrics.json"
                linear_path = run_root / "linear_evaluation" / dataset / "metrics.json"
                if mlp_path.is_file():
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "interface": "mlp_regate",
                            "dataset": dataset,
                            "metrics": extract(mlp_path),
                        }
                    )
                else:
                    missing.append(str(mlp_path))
                if linear_path.is_file():
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "interface": "linear_gate",
                            "dataset": dataset,
                            "metrics": linear_extract(linear_path),
                        }
                    )
                else:
                    missing.append(str(linear_path))
            for dataset in NATIVE_DATASETS:
                native_path = run_root / "evaluation/native" / dataset / "metrics.json"
                if native_path.is_file():
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "interface": "native",
                            "dataset": dataset,
                            "metrics": extract(native_path),
                        }
                    )
                else:
                    missing.append(str(native_path))

    if missing and not args.allow_incomplete:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} expected result files:\n{preview}")

    grouped_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    by_run: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["condition"], row["interface"], row["dataset"])
        by_run[(row["condition"], row["seed"], row["interface"], row["dataset"])] = row["metrics"]
        for metric in METRICS:
            value = row["metrics"].get(metric)
            if isinstance(value, (int, float)):
                grouped_values[(*key, metric)].append(float(value))

    aggregate = [
        {
            "condition": condition,
            "interface": interface,
            "dataset": dataset,
            "metric": metric,
            **mean_std(values),
        }
        for (condition, interface, dataset, metric), values in sorted(grouped_values.items())
    ]

    paired: list[dict[str, Any]] = []
    for interface in ("native", "mlp_regate", "linear_gate"):
        datasets = NATIVE_DATASETS if interface == "native" else REGATE_DATASETS
        for dataset in datasets:
            for metric in METRICS:
                deltas: list[float] = []
                per_seed: dict[str, float] = {}
                for seed in SEEDS:
                    grouped = by_run.get(("grouped", seed, interface, dataset), {}).get(metric)
                    independent = by_run.get(("independent", seed, interface, dataset), {}).get(
                        metric
                    )
                    if isinstance(grouped, (int, float)) and isinstance(independent, (int, float)):
                        delta = float(grouped) - float(independent)
                        deltas.append(delta)
                        per_seed[str(seed)] = delta
                if deltas:
                    paired.append(
                        {
                            "interface": interface,
                            "dataset": dataset,
                            "metric": metric,
                            "direction": "grouped_minus_independent",
                            "per_seed": per_seed,
                            **mean_std(deltas),
                        }
                    )

    result = {
        "schema_version": 1,
        "frozen_seeds": list(SEEDS),
        "all_runs_reported": not missing,
        "missing": missing,
        "rows": rows,
        "aggregate": aggregate,
        "paired_differences": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "multiseed_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[aggregate] wrote {output}; rows={len(rows)} missing={len(missing)}", flush=True)


if __name__ == "__main__":
    main()
