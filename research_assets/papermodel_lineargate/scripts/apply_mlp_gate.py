#!/usr/bin/env python3
"""Score a locally fitted MLP gate on the evaluator's dumped hidden states.

Uses the same scoring function and operating rule as score_mlp.py and reroute_linear.py so
that the released MLP head, a refitted MLP head and the linear gate are all measured
identically on one shared forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn

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
            if not isinstance(record.get("head_hidden"), list):
                raise ValueError(f"{path}:{line_number}: missing head_hidden")
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


def build_mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4096, 512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", default="MLPGateRefit")
    args = parser.parse_args()

    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    model = build_mlp()
    model.load_state_dict(payload["head_state"])
    model.eval()
    mu = payload["mu"].float()
    sd = payload["sd"].float()

    records = load_records(args.predictions)
    record_keys = list(records)
    hidden = torch.tensor(
        np.asarray([records[key]["head_hidden"] for key in record_keys], dtype=np.float32)
    )
    if hidden.shape[1] != 4096:
        raise ValueError(f"Unexpected hidden width: {tuple(hidden.shape)}")

    with torch.inference_mode():
        logits = model((hidden - mu) / sd)
        probabilities = torch.softmax(logits, dim=1).numpy()
    predicted_classes = probabilities.argmax(axis=1)

    answered_map = {
        key: bool(predicted_classes[index] == 0) for index, key in enumerate(record_keys)
    }
    probability_rows = [
        {
            "record_index": index,
            "sample_id": records[key]["sample_id"],
            "p_answer": float(probabilities[index][0]),
            "p_wo": float(probabilities[index][1]),
            "p_nt": float(probabilities[index][2]),
            "predicted_class": int(predicted_classes[index]),
        }
        for index, key in enumerate(record_keys)
    ]

    scored_indices = [
        index
        for index, key in enumerate(record_keys)
        if records[key]["gt"].get("decision") in {"answer", "abstain"}
    ]
    scored_keys = [record_keys[index] for index in scored_indices]
    scored_records = {key: records[key] for key in scored_keys}
    scored_answered = {key: answered_map[key] for key in scored_keys}
    point = regate_curve_point(
        args.dataset, scored_records, scored_answered, "three_class_argmax", None
    )
    metric = asdict(point)
    metric["method"] = args.method

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
            "gate_model": str(args.model.resolve()),
            "shared_candidate_source": str(args.predictions.resolve()),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gate_probabilities.jsonl").open("w", encoding="utf-8") as handle:
        for row in probability_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metric, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[mlp-refit] dataset={args.dataset} n={len(records)} "
        f"FCA={metric['FCA']:.4f} retention={metric['supported_query_retention']:.4f} "
        f"MCC={metric['MCC']:.4f} Strict-3={metric['strict_3_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
