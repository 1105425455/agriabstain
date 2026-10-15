#!/usr/bin/env python3
"""仅用 dev 选择 matched-retention threshold，并冻结后评估已有 test 产物。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aaai-code", type=Path, required=True)
    parser.add_argument("--dev-native", type=Path, required=True)
    parser.add_argument("--dev-probabilities", type=Path, required=True)
    parser.add_argument("--test-native", type=Path, required=True)
    parser.add_argument("--test-regate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.aaai_code.resolve()))
    from experiments.base_native_regate.analyze_native_regate import (  # noqa: PLC0415
        load_jsonl,
        native_point,
        regate_curve_point,
    )

    dev_native = load_jsonl(args.dev_native)
    dev_probs_rows = load_lines(args.dev_probabilities)
    test_native = load_jsonl(args.test_native)
    test_regate = load_jsonl(args.test_regate)

    if len(dev_native) != 2408 or len(dev_probs_rows) != 2408:
        raise ValueError(
            f"Expected 2408 dev records, got native={len(dev_native)}, "
            f"probabilities={len(dev_probs_rows)}"
        )
    if len(test_native) != 2646 or len(test_regate) != 2646:
        raise ValueError(
            f"Expected 2646 test records, got native={len(test_native)}, regate={len(test_regate)}"
        )
    if set(test_native) != set(test_regate):
        raise ValueError("SFT-native test and ReGate test sample IDs do not match")

    dev_probs: dict[str, dict[str, Any]] = {}
    for row in dev_probs_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Dev probability record has missing sample_id")
        if sample_id in dev_probs:
            raise ValueError(f"Duplicate dev probability sample_id: {sample_id}")
        if row.get("variant_id") != "original":
            raise ValueError(f"{sample_id}: expected original variant only")
        probs = [row.get("p_answer"), row.get("p_wo"), row.get("p_nt")]
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in probs
        ):
            raise ValueError(f"{sample_id}: invalid probabilities {probs}")
        if not math.isclose(sum(float(value) for value in probs), 1.0, abs_tol=1e-5):
            raise ValueError(f"{sample_id}: probabilities do not sum to one: {probs}")
        dev_probs[sample_id] = row

    if set(dev_native) != set(dev_probs):
        missing = sorted(set(dev_native) - set(dev_probs))[:10]
        extra = sorted(set(dev_probs) - set(dev_native))[:10]
        raise ValueError(f"Dev sample IDs do not match: missing={missing}, extra={extra}")

    for sample_id, native_record in dev_native.items():
        gt_decision = native_record["gt"].get("decision")
        prob_decision = dev_probs[sample_id].get("original_gt_decision")
        if gt_decision != prob_decision:
            raise ValueError(
                f"{sample_id}: GT mismatch native={gt_decision}, probability={prob_decision}"
            )

    dev_native_point = native_point(
        "comprehensive_dev",
        "SFT_native_observed",
        dev_native,
        iou_threshold=0.5,
    )
    supported_ids = {
        sample_id
        for sample_id, record in dev_native.items()
        if record["gt"].get("decision") == "answer"
    }
    unsupported_ids = set(dev_native) - supported_ids
    p_answer = {sample_id: float(record["p_answer"]) for sample_id, record in dev_probs.items()}
    unique_thresholds = sorted(set(p_answer.values()), reverse=True)
    thresholds = [math.nextafter(max(unique_thresholds), math.inf), *unique_thresholds]
    target_retention = dev_native_point.supported_query_retention

    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        answered_supported = sum(p_answer[sample_id] >= threshold for sample_id in supported_ids)
        answered_unsupported = sum(
            p_answer[sample_id] >= threshold for sample_id in unsupported_ids
        )
        sweep.append(
            {
                "threshold": threshold,
                "answered_supported": answered_supported,
                "supported": len(supported_ids),
                "supported_query_retention": answered_supported / len(supported_ids),
                "answered_unsupported": answered_unsupported,
                "unsupported": len(unsupported_ids),
                "unsupported_hallucination_rate": (answered_unsupported / len(unsupported_ids)),
                "total_coverage": ((answered_supported + answered_unsupported) / len(dev_native)),
            }
        )

    eligible = [row for row in sweep if row["supported_query_retention"] >= target_retention]
    if not eligible:
        raise ValueError(f"Cannot reach dev target retention {target_retention}")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["threshold"]),
            float(row["supported_query_retention"]),
        ),
    )
    frozen_threshold = float(selected["threshold"])

    test_native_point = native_point(
        "comprehensive",
        "SFT_native_observed",
        test_native,
        iou_threshold=0.5,
    )
    test_answered = {
        sample_id: float(record["p_answer"]) >= frozen_threshold
        for sample_id, record in test_regate.items()
    }
    test_regate_point = regate_curve_point(
        "comprehensive",
        test_regate,
        test_answered,
        policy="dev_selected_p_supported_threshold",
        threshold=frozen_threshold,
        iou_threshold=0.5,
    )
    baseline_fca = test_native_point.unsupported_hallucination_rate
    frozen_fca = test_regate_point.unsupported_hallucination_rate
    relative_fca_reduction = (
        (baseline_fca - frozen_fca) / baseline_fca if baseline_fca else math.nan
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "dev_threshold_sweep.csv", sweep)
    comparison_rows = [
        asdict(test_native_point),
        asdict(test_regate_point),
    ]
    write_csv(args.output_dir / "test_comparison.csv", comparison_rows)

    result = {
        "protocol": {
            "selection_split": "development",
            "evaluation_split": "test",
            "selection_rule": (
                "Choose the highest empirical p_answer threshold whose dev supported-query "
                "retention is at least the SFT-native dev retention; freeze it before test."
            ),
            "test_rematching": False,
            "interpolation_used_for_claim": False,
        },
        "integrity": {
            "dev_native_sha256": sha256_file(args.dev_native),
            "dev_probabilities_sha256": sha256_file(args.dev_probabilities),
            "test_native_sha256": sha256_file(args.test_native),
            "test_regate_sha256": sha256_file(args.test_regate),
            "dev_id_sets_equal": True,
            "test_id_sets_equal": True,
            "n_dev": len(dev_native),
            "n_test": len(test_native),
        },
        "development_selection": {
            "sft_native": asdict(dev_native_point),
            "target_supported_query_retention": target_retention,
            "selected_threshold": frozen_threshold,
            "selected_empirical_point": selected,
            "retention_overshoot": (
                float(selected["supported_query_retention"]) - target_retention
            ),
        },
        "frozen_test_evaluation": {
            "sft_native": asdict(test_native_point),
            "regate": asdict(test_regate_point),
            "supported_retention_difference_regate_minus_sft": (
                test_regate_point.supported_query_retention
                - test_native_point.supported_query_retention
            ),
            "absolute_fca_reduction": baseline_fca - frozen_fca,
            "relative_fca_reduction": relative_fca_reduction,
        },
        "claim_text": (
            f"Using a threshold selected exclusively on the development set "
            f"(tau={frozen_threshold:.9g}) and frozen before test evaluation, ReGate "
            f"achieves test supported retention "
            f"{test_regate_point.supported_query_retention:.4f} versus "
            f"{test_native_point.supported_query_retention:.4f} for SFT native, while "
            f"reducing test FCA from {baseline_fca:.4f} to {frozen_fca:.4f} "
            f"({relative_fca_reduction * 100:.1f}% relative reduction)."
        ),
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "claim.txt").write_text(
        result["claim_text"] + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
