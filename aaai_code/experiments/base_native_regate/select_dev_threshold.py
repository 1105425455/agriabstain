#!/usr/bin/env python3
"""Select a ReGate answer threshold on development data and freeze it for test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .analyze_native_regate import load_jsonl, native_point, regate_curve_point


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_probabilities(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in rows:
                raise ValueError(f"Invalid or duplicate sample_id: {sample_id!r}")
            probs = [row.get("p_answer"), row.get("p_wo"), row.get("p_nt")]
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in probs
            ):
                raise ValueError(f"{sample_id}: invalid probability vector {probs}")
            if not math.isclose(sum(map(float, probs)), 1.0, abs_tol=1e-5):
                raise ValueError(f"{sample_id}: probabilities do not sum to one")
            rows[sample_id] = row
    return rows


def select_threshold(
    native: dict[str, dict[str, Any]],
    probabilities: dict[str, dict[str, Any]],
) -> tuple[float, list[dict[str, Any]], Any]:
    if set(native) != set(probabilities):
        raise ValueError("Native and probability sample IDs differ")
    for sample_id, native_row in native.items():
        probability_gt = probabilities[sample_id].get("original_gt_decision")
        if probability_gt != native_row["gt"].get("decision"):
            raise ValueError(f"{sample_id}: development GT decision mismatch")
    native_result = native_point("comprehensive_dev", "SFT_native_observed", native)
    supported = {sid for sid, row in native.items() if row["gt"].get("decision") == "answer"}
    unsupported = set(native) - supported
    p_answer = {sid: float(row["p_answer"]) for sid, row in probabilities.items()}
    values = sorted(set(p_answer.values()), reverse=True)
    thresholds = [math.nextafter(max(values), math.inf), *values]
    sweep = []
    for threshold in thresholds:
        answered_supported = sum(p_answer[sid] >= threshold for sid in supported)
        answered_unsupported = sum(p_answer[sid] >= threshold for sid in unsupported)
        sweep.append(
            {
                "threshold": threshold,
                "answered_supported": answered_supported,
                "supported": len(supported),
                "supported_query_retention": answered_supported / len(supported),
                "answered_unsupported": answered_unsupported,
                "unsupported": len(unsupported),
                "unsupported_hallucination_rate": answered_unsupported / len(unsupported),
                "total_coverage": (answered_supported + answered_unsupported) / len(native),
            }
        )
    eligible = [
        row
        for row in sweep
        if row["supported_query_retention"] >= native_result.supported_query_retention
    ]
    if not eligible:
        raise ValueError("No threshold reaches the native development retention")
    selected = max(eligible, key=lambda row: row["threshold"])
    return float(selected["threshold"]), sweep, native_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-native", type=Path, required=True)
    parser.add_argument("--dev-probabilities", type=Path, required=True)
    parser.add_argument("--test-native", type=Path, required=True)
    parser.add_argument("--test-regate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dev_native = load_jsonl(args.dev_native)
    dev_probabilities = load_probabilities(args.dev_probabilities)
    test_native = load_jsonl(args.test_native)
    test_regate = load_jsonl(args.test_regate)
    if set(test_native) != set(test_regate):
        raise ValueError("Native and ReGate test sample IDs differ")
    for sample_id, native_row in test_native.items():
        if native_row.get("gt") != test_regate[sample_id].get("gt"):
            raise ValueError(f"{sample_id}: native and ReGate test GT differ")
    threshold, sweep, dev_result = select_threshold(dev_native, dev_probabilities)
    selected = next(row for row in sweep if row["threshold"] == threshold)

    test_native_result = native_point("comprehensive", "SFT_native_observed", test_native)
    test_answered = {
        sid: float(row["p_answer"]) >= threshold for sid, row in test_regate.items()
    }
    test_regate_result = regate_curve_point(
        "comprehensive",
        test_regate,
        test_answered,
        "dev_selected_p_supported_threshold",
        threshold,
    )
    relative_reduction = (
        (test_native_result.unsupported_hallucination_rate
         - test_regate_result.unsupported_hallucination_rate)
        / test_native_result.unsupported_hallucination_rate
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dev_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    with (args.output_dir / "test_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(test_native_result)))
        writer.writeheader()
        writer.writerow(asdict(test_native_result))
        writer.writerow(asdict(test_regate_result))

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
            "sft_native": asdict(dev_result),
            "target_supported_query_retention": dev_result.supported_query_retention,
            "selected_threshold": threshold,
            "selected_empirical_point": selected,
            "retention_overshoot": (
                selected["supported_query_retention"] - dev_result.supported_query_retention
            ),
        },
        "frozen_test_evaluation": {
            "sft_native": asdict(test_native_result),
            "regate": asdict(test_regate_result),
            "supported_retention_difference_regate_minus_sft": (
                test_regate_result.supported_query_retention
                - test_native_result.supported_query_retention
            ),
            "absolute_fca_reduction": (
                test_native_result.unsupported_hallucination_rate
                - test_regate_result.unsupported_hallucination_rate
            ),
            "relative_fca_reduction": relative_reduction,
        },
        "claim_text": (
            f"Using a threshold selected exclusively on the development set "
            f"(tau={threshold:.9g}) and frozen before test evaluation, ReGate achieves test "
            f"supported retention {test_regate_result.supported_query_retention:.4f} versus "
            f"{test_native_result.supported_query_retention:.4f} for SFT native, while reducing "
            f"test FCA from {test_native_result.unsupported_hallucination_rate:.4f} to "
            f"{test_regate_result.unsupported_hallucination_rate:.4f} "
            f"({relative_reduction * 100:.1f}% relative reduction)."
        ),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "claim.txt").write_text(result["claim_text"] + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
