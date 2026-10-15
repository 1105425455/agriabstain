#!/usr/bin/env python3
"""Collect grouping-ablation metrics beside the frozen grouped baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


# Both roots hold evaluation outputs kept outside the release; override them with the
# environment variables below or with --baseline-root / --final-root.
DEFAULT_BASELINE_ROOT = Path(
    os.environ.get("BASE_NATIVE_REGATE_ROOT", "outputs/base_native_regate")
)
DEFAULT_FINAL_ROOT = Path(
    os.environ.get("GROUPED_EVAL_ROOT", "outputs/grouped_final/eval_faithful_v1")
)


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "n": value.get("n_samples"),
        "n_evaluated": nested(value, "metrics", "counts", "total"),
        "n_ambiguous_skipped": value.get("n_ambiguous_skipped"),
        "decision_accuracy": nested(value, "gate", "decision_accuracy"),
        "FCA": nested(value, "metrics", "headline", "FCA"),
        "A_FU": nested(value, "metrics", "safeguard", "A_FU"),
        "U_Abs": nested(value, "metrics", "safeguard", "U_Abs"),
        "AAC": nested(value, "metrics", "safeguard", "AAC"),
        "MCC": nested(value, "metrics", "safeguard", "MCC"),
        # Do not substitute gate accuracy when an older metrics file lacks
        # Strict_3: the two quantities can use different denominators (L3).
        "Strict_3": nested(value, "metrics", "semantic", "Strict_3"),
        "E2E_mIoU": (
            nested(value, "metrics", "grounding_reason", "E2E_mIoU")
            if nested(value, "metrics", "grounding_reason", "E2E_mIoU") is not None
            else nested(value, "metrics", "grounding_reason", "A_Ground")
        ),
        "U_Reason": nested(value, "metrics", "grounding_reason", "U_Reason"),
        "U_Reason_c": nested(value, "metrics", "grounding_reason", "U_Reason_c"),
        "invalid": nested(value, "metrics", "counts", "invalid"),
        "invalid_answerable": nested(value, "metrics", "counts", "invalid_answerable"),
        "invalid_unanswerable": nested(value, "metrics", "counts", "invalid_unanswerable"),
        "MTS_micro": nested(value, "mts", "MTS_micro"),
        "MTS_macro": nested(value, "mts", "MTS_macro"),
        "T6": nested(value, "mts", "per_relation", "T6"),
        "OOD_confusion": nested(value, "mts", "per_relation", "OOD_confusion"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    args = parser.parse_args()

    external_datasets = {
        "comprehensive": "comprehensive",
        "refcocog": "refcocog",
        "plantwild": "plantwild",
    }
    controlled_datasets = {
        "L1_discrim": "L1_discrim",
        "L2_evidence": "L2_evidence",
        "L3_dose": "L3_dose",
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "grouped baseline vs independently sampled negatives",
        "rows": [],
    }
    for interface in ("native", "regate"):
        datasets = dict(external_datasets)
        if interface == "regate":
            datasets.update(controlled_datasets)
        for dataset, folder in datasets.items():
            ablation_path = args.eval_root / interface / folder / "metrics.json"
            if not ablation_path.is_file():
                raise FileNotFoundError(f"Missing ablation metrics: {ablation_path}")

            if dataset in controlled_datasets:
                baseline_path = args.final_root / folder / "metrics.json"
            else:
                baseline_path = args.baseline_root / interface
                if interface == "native":
                    baseline_path = baseline_path / "sft"
                baseline_path = baseline_path / folder / "metrics.json"
            if not baseline_path.is_file():
                raise FileNotFoundError(f"Missing grouped baseline metrics: {baseline_path}")

            grouped = extract(baseline_path)
            independent = extract(ablation_path)
            delta = {
                key: (
                    independent[key] - grouped[key]
                    if isinstance(independent.get(key), (int, float))
                    and isinstance(grouped.get(key), (int, float))
                    else None
                )
                for key in independent
                if key
                not in {
                    "n",
                    "n_evaluated",
                    "n_ambiguous_skipped",
                    "invalid",
                    "invalid_answerable",
                    "invalid_unanswerable",
                }
            }
            result["rows"].append(
                {
                    "interface": interface,
                    "dataset": dataset,
                    "grouped": grouped,
                    "independent_negative": independent,
                    "delta_independent_minus_grouped": delta,
                    "grouped_metrics_path": str(baseline_path.resolve()),
                    "independent_metrics_path": str(ablation_path.resolve()),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[summary] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
