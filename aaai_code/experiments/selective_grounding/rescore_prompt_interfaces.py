#!/usr/bin/env python3
"""Build the canonical fixed-GT prompt-interface metric artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reason_abstain.eval.metrics import aggregate_metrics


METHOD_DIRS = {
    "regate": None,
    "native_lm": "native_lm",
    "direct": "prompt_basic",
    "reward": "prompt_reward",
    "reward_evidence": "prompt_reward_norms",
    "structured_cot": "cot_structured",
}
DATASET_DIRS = {
    "comprehensive": ("test_full", "comprehensive"),
    "refcocog": ("ood_refcocog", "refcocog"),
    "plantwild_closed12": ("plantwild_real_closed12_v1", "plantwild"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"duplicate sample_id in {path}")
    return records


def build_artifact(*, prompt_root: Path, regate_root: Path) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "prompt_interface_fixed_gt_v1",
        "evaluator": "reason_abstain.eval.metrics.aggregate_metrics",
        "denominator_policy": "fixed_gt_universe_invalid_is_incorrect",
        "datasets": {},
    }

    for dataset, (prompt_dir, regate_dir) in DATASET_DIRS.items():
        regate_path = regate_root / regate_dir / "predictions.jsonl"
        reference = load_predictions(regate_path)
        reference_gt = {record["sample_id"]: record["gt"] for record in reference}
        dataset_result: dict[str, Any] = {"methods": {}}

        for method, method_dir in METHOD_DIRS.items():
            path = (
                regate_path
                if method_dir is None
                else prompt_root / method_dir / prompt_dir / "predictions.jsonl"
            )
            records = reference if method_dir is None else load_predictions(path)
            candidate_gt = {record["sample_id"]: record["gt"] for record in records}
            if candidate_gt != reference_gt:
                raise ValueError(f"sample/GT mismatch for {method}/{dataset}: {path}")
            metrics = aggregate_metrics([(record["pred"], record["gt"]) for record in records])
            dataset_result["methods"][method] = {
                "n_samples": len(records),
                "predictions_sha256": file_sha256(path),
                "metrics": metrics,
            }
        artifact["datasets"][dataset] = dataset_result
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--regate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifact = build_artifact(prompt_root=args.prompt_root, regate_root=args.regate_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
