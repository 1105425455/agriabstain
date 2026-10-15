#!/usr/bin/env python3
"""Score GPT-5.6 Sol with the existing ReGate metric implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from reason_abstain.eval.metrics import (  # noqa: E402
    aggregate_metrics,
    box_iou,
    metamorphic_test_satisfaction,
)
from reason_abstain.eval.parser import (  # noqa: E402
    normalize_label,
    parse_structured_output,
)

DECISION_SYSTEM_PROMPT = (
    "You are a vision-language grounding assistant for high-stakes plant disease diagnosis. "
    "Locate the queried target. If the queried target is present and localizable, return "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]} with box in [0,1000] relative coords. '
    "Otherwise abstain: "
    '{"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image, or cropped out of view)."
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index(
    records: list[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label}: missing string {key}")
        if value in indexed:
            raise ValueError(f"{label}: duplicate {key}={value}")
        indexed[value] = record
    return indexed


def _record_keys(records: list[dict[str, Any]]) -> list[str]:
    occurrences: dict[str, int] = {}
    keys: list[str] = []
    for record in records:
        sample_id = str(record["metadata"]["sample_id"])
        occurrence = occurrences.get(sample_id, 0)
        occurrences[sample_id] = occurrence + 1
        keys.append(f"{sample_id}::occurrence={occurrence}")
    return keys


def _gt(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "decision": metadata.get("decision"),
        "label": metadata.get("label"),
        "box": metadata.get("box"),
        "reason": metadata.get("reason"),
        "relation": metadata.get("relation"),
        "sample_id": metadata.get("sample_id"),
        "metadata": {
            "relation": metadata.get("relation"),
            "achieved_visible_ratio": metadata.get("achieved_visible_ratio"),
            "target_visible_ratio": metadata.get("target_visible_ratio"),
            "sample_id": metadata.get("sample_id"),
        },
    }


def _decision_correct(prediction: dict[str, Any], gt: dict[str, Any]) -> bool:
    if not prediction.get("valid") or prediction.get("decision") != gt.get("decision"):
        return False
    if gt.get("decision") == "answer":
        return normalize_label(prediction.get("label")) == normalize_label(gt.get("label"))
    if gt.get("decision") == "abstain":
        return prediction.get("reason") == gt.get("reason")
    return False


def _grounding_correct(
    prediction: dict[str, Any], gt: dict[str, Any], iou_threshold: float
) -> bool:
    if (
        prediction.get("valid") is not True
        or prediction.get("decision") != "answer"
        or gt.get("decision") != "answer"
        or normalize_label(prediction.get("label")) != normalize_label(gt.get("label"))
    ):
        return False
    box = prediction.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        return box_iou(box, gt["box"]) >= iou_threshold
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _label_gated_iou(prediction: dict[str, Any], gt: dict[str, Any]) -> float:
    if (
        prediction.get("valid") is not True
        or prediction.get("decision") != "answer"
        or gt.get("decision") != "answer"
        or normalize_label(prediction.get("label")) != normalize_label(gt.get("label"))
        or not prediction.get("box")
        or not gt.get("box")
    ):
        return 0.0
    try:
        return box_iou(prediction["box"], gt["box"])
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_bootstrap_difference(
    pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    statistic: Callable[[dict[str, Any], dict[str, Any]], float],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    differences = [
        statistic(gpt_prediction, gt) - statistic(regate_prediction, gt)
        for gpt_prediction, regate_prediction, gt in pairs
    ]
    rng = random.Random(seed)
    estimates = [
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(replicates)
    ]
    return {
        "difference_gpt_minus_regate": sum(differences) / len(differences),
        "ci_low": _percentile(estimates, 0.025),
        "ci_high": _percentile(estimates, 0.975),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--api-output", type=Path, required=True)
    parser.add_argument("--regate-predictions", type=Path, required=True)
    parser.add_argument("--regate-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = _load_jsonl(args.data)
    record_keys = _record_keys(data)
    api_records = _index(_load_jsonl(args.api_output), "record_key", "API output")
    expected_keys = set(record_keys)
    if set(api_records) != expected_keys:
        missing = sorted(expected_keys - set(api_records))[:10]
        extra = sorted(set(api_records) - expected_keys)[:10]
        raise RuntimeError(f"API output mismatch: missing={missing}, extra={extra}")

    predictions: list[dict[str, Any]] = []
    hard_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    ratio_states: dict[str, list[str]] = defaultdict(list)
    for record_key, record in zip(record_keys, data, strict=True):
        sample_id = str(record["metadata"]["sample_id"])
        raw = str(api_records[record_key]["raw"])
        prediction = parse_structured_output(raw)
        gt = _gt(record)
        predictions.append(
            {
                "record_key": record_key,
                "sample_id": sample_id,
                "relation": gt["relation"],
                "gt": {
                    key: gt.get(key)
                    for key in ("decision", "label", "box", "reason", "metadata")
                },
                "pred": prediction,
                "raw": raw,
                "api_model": api_records[record_key].get("api_model"),
                "usage": api_records[record_key].get("usage"),
            }
        )
        ratio_bin = record["metadata"].get("visible_ratio_bin")
        if ratio_bin is not None:
            state = (
                str(prediction.get("decision"))
                if prediction.get("valid")
                else "invalid"
            )
            ratio_states[str(ratio_bin)].append(state)
        if gt.get("decision") is not None:
            hard_pairs.append((prediction, gt))

    metrics = aggregate_metrics(hard_pairs, iou_threshold=args.iou_threshold)
    mts = metamorphic_test_satisfaction(
        hard_pairs,
        iou_threshold=args.iou_threshold,
    )
    hard_valid = sum(prediction.get("valid") is True for prediction, _ in hard_pairs)
    hard_decision_correct = sum(
        _decision_correct(prediction, gt) for prediction, gt in hard_pairs
    )
    all_valid = sum(record["pred"].get("valid") is True for record in predictions)

    regate_rows = _load_jsonl(args.regate_predictions)
    expected_sample_ids = [
        str(record["metadata"]["sample_id"])
        for record in data
    ]
    regate_sample_ids = [str(record.get("sample_id")) for record in regate_rows]
    if regate_sample_ids != expected_sample_ids:
        raise RuntimeError(
            "ReGate predictions do not match evaluation data in row order"
        )
    regate_records = dict(zip(record_keys, regate_rows, strict=True))
    paired: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for record in predictions:
        gt = record["gt"]
        if gt["decision"] is None:
            continue
        paired.append((record["pred"], regate_records[record["record_key"]]["pred"], gt))

    answerable = [item for item in paired if item[2]["decision"] == "answer"]
    unsupported = [item for item in paired if item[2]["decision"] == "abstain"]
    regate_hard_pairs = [
        (regate_prediction, gt)
        for _gpt_prediction, regate_prediction, gt in paired
    ]
    regate_canonical_metrics = aggregate_metrics(
        regate_hard_pairs,
        iou_threshold=args.iou_threshold,
    )
    regate_mts = metamorphic_test_satisfaction(
        regate_hard_pairs,
        iou_threshold=args.iou_threshold,
    )
    comparisons = {
        "FCA": _paired_bootstrap_difference(
            unsupported,
            lambda prediction, _gt_value: float(
                prediction.get("valid") is True
                and prediction.get("decision") == "answer"
            ),
            args.bootstrap_replicates,
            args.seed,
        ),
        "A_FU": _paired_bootstrap_difference(
            answerable,
            lambda prediction, _gt_value: float(
                prediction.get("valid") is not True
                or prediction.get("decision") != "answer"
            ),
            args.bootstrap_replicates,
            args.seed + 100,
        ),
        "supported_grounding_accuracy": _paired_bootstrap_difference(
            answerable,
            lambda prediction, gt_value: float(
                _grounding_correct(prediction, gt_value, args.iou_threshold)
            ),
            args.bootstrap_replicates,
            args.seed + 200,
        ),
        "Strict_3": _paired_bootstrap_difference(
            paired,
            lambda prediction, gt_value: float(
                _decision_correct(prediction, gt_value)
            ),
            args.bootstrap_replicates,
            args.seed + 300,
        ),
        "E2E_mIoU": _paired_bootstrap_difference(
            answerable,
            _label_gated_iou,
            args.bootstrap_replicates,
            args.seed + 400,
        ),
    }

    ratio_rows: list[dict[str, Any]] = []
    for ratio_bin, states in sorted(
        ratio_states.items(),
        key=lambda item: float(item[0]),
    ):
        counts = Counter(states)
        ratio_rows.append(
            {
                "visible_ratio_bin": ratio_bin,
                "n": len(states),
                "answer_rate": counts["answer"] / len(states),
                "abstain_rate": counts["abstain"] / len(states),
                "invalid_rate": counts["invalid"] / len(states),
            }
        )

    run_config_path = args.api_output.with_suffix(".run_config.json")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    api_model_counts = Counter(
        str(record.get("api_model"))
        for record in api_records.values()
    )
    attempt_counts = Counter(
        int(record.get("attempts", 1))
        for record in api_records.values()
    )
    access_date_utc = datetime.fromtimestamp(
        args.api_output.stat().st_mtime,
        tz=timezone.utc,
    ).date().isoformat()
    result = {
        "protocol": {
            "data": str(args.data.resolve()),
            "records": len(data),
            "hard_endpoint_records": len(hard_pairs),
            "ambiguous_intermediate_records": len(data) - len(hard_pairs),
            "prompt_policy": (
                "Exact ReGate decision system override, per-record query, and image. "
                "GPT output is parsed once with the same ReGate parser; "
                "no JSON mode or semantic repair."
            ),
            "reason_policy": "Labels and reasons use the frozen evaluation GT.",
            "iou_threshold": args.iou_threshold,
            "metric_denominator_policy": {
                "FCA": "all GT unsupported records; invalid is not a committed answer",
                "A_FU": "all GT supported records; invalid is a false rejection",
                "Strict_3": "all hard-label records; invalid is incorrect",
                "E2E_mIoU": (
                    "all GT supported records; invalid, abstention, or wrong label "
                    "contributes zero"
                ),
                "paired_CI": "paired bootstrap over the same fixed GT sample universe",
            },
            "api": {
                "provider": "OpenAI-compatible commercial API gateway",
                "endpoint": run_config["endpoint"],
                "requested_model_id": run_config["model_requested"],
                "returned_model_ids": dict(sorted(api_model_counts.items())),
                "provider_model_revision": None,
                "version_note": (
                    "The provider returned model ID gpt-5.6-sol but exposed no "
                    "dated snapshot or revision identifier."
                ),
                "access_date_utc": access_date_utc,
                "temperature": run_config["temperature"],
                "max_completion_tokens": run_config["max_completion_tokens"],
                "image_detail": run_config["image_detail"],
                "system_prompt": DECISION_SYSTEM_PROMPT,
                "user_prompt_policy": (
                    "The per-record user query is unchanged except that the "
                    "literal <image> placeholder is removed when attaching the image."
                ),
                "evaluation_runs": 1,
                "accepted_responses_per_record": 1,
                "request_attempt_histogram": {
                    str(key): value
                    for key, value in sorted(attempt_counts.items())
                },
                "repeat_policy": (
                    "One accepted response per record in one evaluation run; "
                    "no repeated model runs or majority voting. Additional attempts, "
                    "if present, are transport retries only."
                ),
            },
        },
        "gpt56_sol": {
            "all_record_format_valid_rate": all_valid / len(predictions),
            "hard_endpoint_format_valid_rate": hard_valid / len(hard_pairs),
            "hard_endpoint_decision_accuracy": hard_decision_correct / len(hard_pairs),
            "metrics": metrics,
            "mts": mts,
            "answer_rate_by_visibility": ratio_rows,
        },
        "regate_reference": {
            "source": str(args.regate_metrics.resolve()),
            "metrics": regate_canonical_metrics,
            "mts": regate_mts,
        },
        "paired_gpt_minus_regate": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if ratio_rows:
        _write_csv(args.output_dir / "answer_rate_by_visibility.csv", ratio_rows)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
