#!/usr/bin/env python
"""Aggregate raw-VLM calibration predictions into a per-class capability gate.

Reads the predictions JSONL written by eval_qwen3vl_3reason.py and computes per-label
answer_rate / label_match_rate / correct_rate / mean_iou from the ``pred`` / ``gt``
sub-objects.  Classes whose base correct_rate is below ``--threshold`` are flagged for
exclusion (the model can neither be relied on to ground them nor to be measured fairly),
mirroring the PlantDoc capability-gate decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reason_abstain.bench.io import read_jsonl
from reason_abstain.eval.metrics import box_iou
from reason_abstain.eval.parser import normalize_label


def _extract_label(row: dict[str, Any]) -> str:
    """Return the ground-truth label from a predictions JSONL record."""
    gt = row.get("gt", {})
    label = gt.get("label")
    if isinstance(label, str) and label:
        return label
    # fallback: flat label key (calibration-style)
    label = row.get("label")
    if isinstance(label, str) and label:
        return label
    return "UNKNOWN"


def _summarize(rows: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "answer_rate": 0.0,
            "label_match_rate": 0.0,
            "correct_rate": 0.0,
            "mean_iou": 0.0,
        }

    # Collect per-sample fields from the standard predictions JSONL schema
    # (pred / gt sub-objects) or the flat calibration schema.
    answers = 0
    label_matches = 0
    corrects = 0
    ious: list[float] = []

    for r in rows:
        gt = r.get("gt", {})
        pred = r.get("pred", {})

        # --- answer (model committed to an answer) ---
        parsed = pred if pred.get("valid") is True else {}
        decision = parsed.get("decision") or r.get("parsed", {}).get("decision")
        if decision == "answer":
            answers += 1

        # --- label_match ---
        gt_label = gt.get("label")
        pred_label = parsed.get("label")
        if isinstance(gt_label, str) and isinstance(pred_label, str):
            lm = normalize_label(pred_label) == normalize_label(gt_label)
            label_matches += int(lm)

        # --- correct (decision + label correct) ---
        gt_decision = gt.get("decision")
        if gt_decision == "answer" and decision == "answer":
            if isinstance(gt_label, str) and isinstance(pred_label, str):
                corrects += int(normalize_label(pred_label) == normalize_label(gt_label))
        elif gt_decision == "abstain" and decision == "abstain":
            gt_reason = gt.get("reason")
            pred_reason = parsed.get("reason")
            if isinstance(gt_reason, str) and isinstance(pred_reason, str):
                corrects += int(pred_reason == gt_reason)
        # fallback: flat correct field (calibration-style)
        elif r.get("correct") is not None:
            corrects += int(bool(r["correct"]))

        # --- IoU (label-gated, only when both answer and GT answer) ---
        if decision == "answer" and gt_decision == "answer":
            if isinstance(gt_label, str) and isinstance(pred_label, str):
                if normalize_label(pred_label) == normalize_label(gt_label):
                    gt_box = gt.get("box")
                    pred_box = parsed.get("box")
                    if isinstance(gt_box, list) and len(gt_box) == 4:
                        if isinstance(pred_box, list) and len(pred_box) == 4:
                            try:
                                ious.append(box_iou(pred_box, gt_box))
                            except (TypeError, ValueError, ZeroDivisionError):
                                pass
        # fallback: flat iou field (calibration-style)
        if not gt and r.get("iou") is not None:
            try:
                ious.append(float(r["iou"]))
            except (TypeError, ValueError):
                pass

    return {
        "n": n,
        "answer_rate": answers / n,
        "label_match_rate": label_matches / n,
        "correct_rate": corrects / n,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", required=True, help="predictions JSONL (eval_qwen3vl_3reason.py output)"
    )
    parser.add_argument("--output", required=True, help="capability gate JSON output")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[_extract_label(row)].append(row)

    by_label_stats: dict[str, dict[str, Any]] = {}
    for label, group in sorted(by_label.items()):
        by_label_stats[label] = _summarize(group, args.iou_threshold)

    overall = _summarize(rows, args.iou_threshold)
    excluded = sorted(
        label for label, stats in by_label_stats.items() if stats["correct_rate"] < args.threshold
    )

    result = {
        "num_samples": len(rows),
        "threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "match": "normalized(underscore/space/case-insensitive)",
        "overall": overall,
        "excluded_class_names": excluded,
        "by_label": by_label_stats,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote capability gate to {output}", flush=True)
    print(
        f"overall correct_rate={overall['correct_rate']:.3f} mean_iou={overall['mean_iou']:.3f}",
        flush=True,
    )
    print(f"=== per-label (threshold {args.threshold}) ===", flush=True)
    for label, stats in sorted(by_label_stats.items(), key=lambda kv: kv[1]["correct_rate"]):
        flag = "  DROP" if stats["correct_rate"] < args.threshold else ""
        print(
            f"  {label:30s} n={stats['n']:3d} correct={stats['correct_rate']:.2f} "
            f"answer={stats['answer_rate']:.2f} iou={stats['mean_iou']:.2f}{flag}",
            flush=True,
        )
    print(f"=== excluded_class_names ({len(excluded)}) ===", flush=True)
    for label in excluded:
        print(f"  - {label}", flush=True)


if __name__ == "__main__":
    main()
