#!/usr/bin/env python3
"""Build a path-free canonical artifact for the grouping ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "FCA",
    "A_FU",
    "U_Abs",
    "AAC",
    "MCC",
    "Strict_3",
    "E2E_mIoU",
    "U_Reason",
    "U_Reason_c",
    "MTS_macro",
    "MTS_micro",
    "T6",
    "OOD_confusion",
    "invalid",
    "invalid_answerable",
    "invalid_unanswerable",
    "n",
    "n_evaluated",
    "n_ambiguous_skipped",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_calibration(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8")

    def value(pattern: str) -> float:
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"missing calibration field: {pattern}")
        return float(match.group(1))

    return {
        "train_records": int(value(r"total_train=(\d+)")),
        "dev_records": int(value(r"total_dev=(\d+)")),
        "best_dev_loss": value(r"最佳 dev loss: ([0-9.]+)"),
        "train_mcc": value(r"train MCC \(multi-class\): ([0-9.]+)"),
        "dev_mcc": value(r"dev\s+MCC \(multi-class\): ([0-9.]+)"),
        "dev_accuracy": value(r"accuracy\s+([0-9.]+)\s+100"),
        "dev_answer_recall": value(r"answer\(0\)\s+[0-9.]+\s+([0-9.]+)"),
        "dev_wrong_object_recall": value(r"wrong_object\(1\)\s+[0-9.]+\s+([0-9.]+)"),
        "dev_no_target_recall": value(r"no_target\(2\)\s+[0-9.]+\s+([0-9.]+)"),
    }


def compact_metrics(metrics: dict[str, Any], dataset: str) -> dict[str, Any]:
    compact = {key: metrics.get(key) for key in METRIC_KEYS}
    if dataset == "plantwild":
        compact["E2E_mIoU"] = None
    return compact


def selection_evidence(
    final: dict[str, Any],
    checkpoints: dict[str, Any],
    log_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if final["selection_policy"] != manifest["selection_policy"]:
        raise ValueError("checkpoint selection policy differs from the dev manifest")
    log_hashes: dict[str, str] = {}
    observed_records: set[int] = set()
    recovered_steps: list[int] = []
    for step, expected_loss in checkpoints["dev_losses"].items():
        log_path = log_dir / f"dev_loss_checkpoint-{step}.log"
        if not log_path.is_file():
            recovery_path = log_dir / f"dev_loss_checkpoint-{step}-recovery.log"
            if recovery_path.is_file():
                log_path = recovery_path
                recovered_steps.append(int(step))
        text = log_path.read_text(encoding="utf-8")
        loss_matches = re.findall(r"^DEV_LOSS=([0-9.eE+-]+)$", text, re.MULTILINE)
        if len(loss_matches) != 1 or float(loss_matches[0]) != expected_loss:
            raise ValueError(f"dev-loss log mismatch for checkpoint {step}")
        completed = [
            int(done)
            for done, total in re.findall(r"(?:^|\s)(\d+)/(\d+)(?:\s|$)", text)
            if done == total
        ]
        if not completed:
            raise ValueError(f"no completed-record count in {log_path}")
        observed_records.add(max(completed))
        log_hashes[str(step)] = file_sha256(log_path)
    if observed_records != {manifest["dev_records"]}:
        raise ValueError(f"unexpected dev record counts: {observed_records}")
    if recovered_steps:
        expected_task_ids = {
            str(step): manifest["grouped_epoch1_recovery_task_id"] for step in recovered_steps
        }
        if checkpoints.get("extra_checkpoint_task_ids") != expected_task_ids:
            raise ValueError("recovered-checkpoint task provenance mismatch")
    return {
        "policy": final["selection_policy"],
        "dev_records": manifest["dev_records"],
        "dev_subset_sha256": manifest["dev_subset_sha256"],
        "evaluator_sha256": manifest["evaluator_sha256"],
        "candidate_steps": sorted(int(step) for step in checkpoints["dev_losses"]),
        "best_step": final["best_step"],
        "best_epoch": final["best_epoch"],
        "best_val_loss": final["best_val_loss"],
        "dev_losses": checkpoints["dev_losses"],
        "dev_loss_log_sha256": log_hashes,
        "recovered_candidates": (
            {
                "steps": recovered_steps,
            }
            if recovered_steps
            else None
        ),
        "train_runtime_seconds": final["train_runtime_seconds"],
    }


def build_artifact(artifact_root: Path) -> dict[str, Any]:
    audit_path = artifact_root / "data" / "audit.json"
    comparison_path = artifact_root / "summary" / "comparison.json"
    independent_selection_path = artifact_root / "selection" / "final_metrics.json"
    independent_checkpoints_path = artifact_root / "selection" / "checkpoint_selection.json"
    grouped_selection_path = (
        artifact_root / "grouped_dev_selection" / "selection" / "final_metrics.json"
    )
    grouped_checkpoints_path = (
        artifact_root / "grouped_dev_selection" / "selection" / "checkpoint_selection.json"
    )
    selection_manifest_path = artifact_root / "data" / "dev_selection_manifest.json"
    calibration_path = artifact_root / "logs" / "calibrate_train.log"

    audit = load_json(audit_path)
    comparison = load_json(comparison_path)
    independent_selection = load_json(independent_selection_path)
    independent_checkpoints = load_json(independent_checkpoints_path)
    grouped_selection = load_json(grouped_selection_path)
    grouped_checkpoints = load_json(grouped_checkpoints_path)
    selection_manifest = load_json(selection_manifest_path)
    calibration = extract_calibration(calibration_path)

    if not all(audit["checks"].values()):
        failed = [key for key, passed in audit["checks"].items() if not passed]
        raise ValueError(f"grouping data audit failed: {failed}")
    if audit["counts"]["rows"] != 7072:
        raise ValueError("unexpected grouping-ablation row count")
    if audit["counts"]["positive_rows"] != 2703:
        raise ValueError("unexpected positive count")
    if audit["counts"]["replaced_negative_rows"] != 4369:
        raise ValueError("unexpected negative count")
    if calibration["train_records"] != 7072 or calibration["dev_records"] != 100:
        raise ValueError("unexpected calibrated-head record counts")

    rows = []
    for row in comparison["rows"]:
        rows.append(
            {
                "interface": row["interface"],
                "dataset": row["dataset"],
                "metric_applicability": {"E2E_mIoU": row["dataset"] != "plantwild"},
                "grouped": compact_metrics(row["grouped"], row["dataset"]),
                "independent_negative": compact_metrics(
                    row["independent_negative"], row["dataset"]
                ),
                "delta_independent_minus_grouped": {
                    key: (
                        None
                        if key == "E2E_mIoU" and row["dataset"] == "plantwild"
                        else row["delta_independent_minus_grouped"].get(key)
                    )
                    for key in METRIC_KEYS
                    if key in row["delta_independent_minus_grouped"]
                },
            }
        )

    return {
        "schema_version": 2,
        "protocol": "same_source_vs_independent_negative_full_retrain_seed42_v2",
        "primary_mts": "MTS_macro",
        "metric_policy": {
            "denominators": "fixed_gt_universe_invalid_is_incorrect",
            "AAC": (
                "correct supported outcomes divided by all supported GT records; "
                "requires label and IoU>=0.5 when GT boxes exist, and label when box-free"
            ),
            "E2E_mIoU_applicability": "not applicable to probes without GT boxes",
            "U_Abs_complement_note": (
                "FCA + U_Abs equals one only when unsupported invalid count is zero"
            ),
            "U_Reason": "correct-reason abstentions divided by all unsupported GT records",
            "U_Reason_c": "correct reasons divided by predicted abstentions on unsupported GT",
        },
        "source_artifacts": {
            "data_audit_sha256": file_sha256(audit_path),
            "comparison_sha256": file_sha256(comparison_path),
            "grouped_selection_sha256": file_sha256(grouped_selection_path),
            "grouped_checkpoint_losses_sha256": file_sha256(grouped_checkpoints_path),
            "independent_selection_sha256": file_sha256(independent_selection_path),
            "independent_checkpoint_losses_sha256": file_sha256(
                independent_checkpoints_path
            ),
            "dev_selection_manifest_sha256": file_sha256(selection_manifest_path),
            "calibration_log_sha256": file_sha256(calibration_path),
        },
        "data_control": {
            "seed": audit["seed"],
            "counts": audit["counts"],
            "category_state_strata": audit["category_state_strata"],
            "candidate_inventory": audit["candidate_inventory"],
            "checks": audit["checks"],
            "baseline_sft_sha256": audit["inputs"]["baseline_sft_sha256"],
            "independent_sft_sha256": audit["outputs"]["sft_train_sha256"],
            "negative_assignments_sha256": audit["outputs"]["assignments_sha256"],
        },
        "training": {
            "base_model": "Qwen3-VL-8B-Instruct",
            "seed": 42,
            "gpu_count": 8,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "epochs": 6,
            "learning_rate": 0.0001,
            "scheduler": "cosine",
            "warmup_ratio": 0.05,
            "precision": "bf16",
            "deepspeed": "ZeRO-2",
            "lora": {"rank": 16, "alpha": 32, "dropout": 0.05},
            "head_lambda": 0.4,
            "head_warmup_steps": 10,
            "head_state_weights": [1.0, 1.5, 1.0],
            "checkpoint_selection": {
                "grouped": selection_evidence(
                    grouped_selection,
                    grouped_checkpoints,
                    artifact_root / "grouped_dev_selection" / "logs",
                    selection_manifest,
                ),
                "independent_negative": selection_evidence(
                    independent_selection,
                    independent_checkpoints,
                    artifact_root / "logs",
                    selection_manifest,
                ),
            },
        },
        "independent_negative_calibrated_head": calibration,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifact = build_artifact(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
