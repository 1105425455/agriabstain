#!/usr/bin/env python3
"""Score the released MLP gate on the same predictions file the linear gate is scored on.

The evaluator already writes its own metrics.json, but running the MLP head through the
identical scoring function as reroute_linear.py removes any doubt that the two gates are
compared under the same operating rule (three-class argmax) and the same scored universe.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import matthews_corrcoef

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
AAAI_ROOT = EXPERIMENT_ROOT.parent / "aaai_code"
sys.path.insert(0, str(AAAI_ROOT))

from experiments.base_native_regate.analyze_native_regate import (  # noqa: E402
    regate_curve_point,
)


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"{path}:{line_number}: missing sample_id")
            for field in ("p_answer", "p_wo", "p_nt"):
                if not isinstance(record.get(field), (int, float)):
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            if not isinstance(record.get("answer_candidate"), dict):
                raise ValueError(f"{path}:{line_number}: missing answer_candidate")
            records[f"{line_number}:{sample_id}"] = record
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def gt_class(record: dict[str, Any]) -> int:
    gt = record["gt"]
    if gt.get("decision") == "answer":
        return 0
    if gt.get("reason") == "wrong_object":
        return 1
    if gt.get("reason") == "no_target":
        return 2
    raise ValueError(f"Unsupported ground truth: {gt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = load_records(args.predictions)
    record_keys = list(records)
    probabilities = np.asarray(
        [
            [records[key]["p_answer"], records[key]["p_wo"], records[key]["p_nt"]]
            for key in record_keys
        ],
        dtype=np.float32,
    )
    predicted_classes = probabilities.argmax(axis=1)
    answered_map = {
        key: bool(probabilities[index].argmax() == 0) for index, key in enumerate(record_keys)
    }

    scored_indices = [
        index
        for index, key in enumerate(record_keys)
        if records[key]["gt"].get("decision") in {"answer", "abstain"}
    ]
    scored_keys = [record_keys[index] for index in scored_indices]
    scored_records = {key: records[key] for key in scored_keys}
    scored_answered = {key: answered_map[key] for key in scored_keys}
    point = regate_curve_point(
        args.dataset,
        scored_records,
        scored_answered,
        "three_class_argmax",
        None,
    )
    metric = asdict(point)
    metric["method"] = "MLPGate"

    y_supported = np.asarray(
        [int(records[key]["gt"].get("decision") == "answer") for key in scored_keys],
        dtype=np.int64,
    )
    y_answered = np.asarray([int(answered_map[key]) for key in scored_keys], dtype=np.int64)
    y_state = np.asarray([gt_class(records[key]) for key in scored_keys], dtype=np.int64)
    metric.update(
        {
            "MCC": float(matthews_corrcoef(y_supported, y_answered)),
            "A_FU": 1.0 - metric["supported_query_retention"],
            "FCA": metric["unsupported_hallucination_rate"],
            "state_accuracy": float(
                (predicted_classes[np.asarray(scored_indices)] == y_state).mean()
            ),
            "n_samples": len(records),
            "n_scored": len(scored_records),
            "n_ambiguous_skipped": len(records) - len(scored_records),
            "shared_candidate_source": str(args.predictions.resolve()),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metric, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[mlp-score] dataset={args.dataset} n={len(records)} "
        f"FCA={metric['FCA']:.4f} retention={metric['supported_query_retention']:.4f} "
        f"MCC={metric['MCC']:.4f} Strict-3={metric['strict_3_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
