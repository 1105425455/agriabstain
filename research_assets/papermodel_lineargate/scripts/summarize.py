#!/usr/bin/env python3
"""Collect the gate comparison and check it against the published ablation table.

The MLP column of this run must reproduce the published numbers, because it reuses the
paper's adapter and the paper's calibrated head on the paper's test split at the same
three-class-argmax operating point. A mismatch there means the re-evaluation drifted and
the linear-gate column cannot be trusted either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Published grouping-ablation row, regate interface, comprehensive test split.
PUBLISHED = {
    "grouped": {
        "FCA": 0.1323,
        "supported_query_retention": 0.9758,
        "MCC": 0.8485,
        "strict_3_accuracy": 0.8560,
        "MTS_macro": 0.8195,
    },
    "independent": {
        "FCA": 0.4860,
        "supported_query_retention": 0.9554,
        "MCC": 0.5231,
        "strict_3_accuracy": 0.6633,
        "MTS_macro": 0.5267,
    },
}
REPORTED = ("FCA", "supported_query_retention", "MCC", "strict_3_accuracy", "MTS_macro")
SANITY_TOLERANCE = 0.02


def read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="comprehensive")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "operating_point": "three_class_argmax",
        "published_reference": PUBLISHED,
        "arms": {},
        "sanity": {},
        "construction_effect": {},
    }

    for arm in ("grouped", "independent"):
        arm_root = args.runs_root / arm
        mlp = read_metrics(arm_root / "mlp_evaluation" / args.dataset / "metrics.json")
        linear = read_metrics(arm_root / "linear_evaluation" / args.dataset / "metrics.json")
        refit = read_metrics(arm_root / "mlp_refit_evaluation" / args.dataset / "metrics.json")
        result["arms"][arm] = {"mlp": mlp, "mlp_refit": refit, "linear": linear}

        if mlp is None:
            result["sanity"][arm] = {"status": "missing"}
            continue
        deviations = {
            metric: (
                round(float(mlp[metric]) - PUBLISHED[arm][metric], 4)
                if isinstance(mlp.get(metric), (int, float))
                else None
            )
            for metric in REPORTED
        }
        worst = max(
            (abs(value) for value in deviations.values() if value is not None),
            default=None,
        )
        result["sanity"][arm] = {
            "status": (
                "unknown"
                if worst is None
                else ("ok" if worst <= SANITY_TOLERANCE else "deviates")
            ),
            "max_abs_deviation": worst,
            "deviation_vs_published": deviations,
        }

    for gate in ("mlp", "mlp_refit", "linear"):
        grouped = (result["arms"].get("grouped") or {}).get(gate)
        independent = (result["arms"].get("independent") or {}).get(gate)
        if not grouped or not independent:
            continue
        result["construction_effect"][gate] = {
            metric: round(float(grouped[metric]) - float(independent[metric]), 4)
            for metric in REPORTED
            if isinstance(grouped.get(metric), (int, float))
            and isinstance(independent.get(metric), (int, float))
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gates = ("mlp", "mlp_refit", "linear")
    columns = [f"{arm[:5]}-{gate}" for arm in ("grouped", "independent") for gate in gates]
    header = f"{'metric':26s}" + "".join(f"{name:>16s}" for name in columns)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for metric in REPORTED:
        row = f"{metric:26s}"
        for arm in ("grouped", "independent"):
            for gate in gates:
                payload = (result["arms"].get(arm) or {}).get(gate) or {}
                row += f"{fmt(payload.get(metric)):>16s}"
        print(row, flush=True)

    print(flush=True)
    for arm, sanity in result["sanity"].items():
        print(
            f"[sanity] {arm}: status={sanity.get('status')} "
            f"max_abs_deviation={sanity.get('max_abs_deviation')}",
            flush=True,
        )
    for gate, effect in result["construction_effect"].items():
        print(f"[construction-effect grouped-minus-independent] {gate}: {effect}", flush=True)
    print(f"[summarize] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
