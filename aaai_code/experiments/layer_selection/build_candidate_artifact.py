from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = (9, 18, 27)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    manifest_path: Path,
    dev_scan_path: Path,
    min_score_curve_path: Path,
    candidate_layers: tuple[int, ...] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scan = json.loads(dev_scan_path.read_text(encoding="utf-8"))
    curve = json.loads(min_score_curve_path.read_text(encoding="utf-8"))

    results = {int(row["transformer_layer"]): row for row in scan["results"]}
    curve_results = {int(row["transformer_layer"]): row for row in curve}
    missing = sorted(set(candidate_layers) - set(results))
    if missing:
        raise ValueError(f"Candidate layers missing from scan: {missing}")

    candidates = []
    split_counts: set[tuple[int, int]] = set()
    for layer in candidate_layers:
        row = results[layer]
        curve_row = curve_results.get(layer)
        if curve_row is None:
            raise ValueError(f"Candidate layer {layer} missing from min-score curve")
        if abs(float(row["min_score"]) - float(curve_row["min_score"])) > 1e-8:
            raise ValueError(f"Layer {layer} differs between scan and min-score curve")
        n_train = int(row["n_train"])
        n_dev = int(row["n_dev"])
        split_counts.add((n_train, n_dev))
        answer_accuracy = 1.0 - float(row["answer_fa"])
        expected_min = min(
            answer_accuracy,
            float(row["wo_recall"]),
            float(row["nt_accuracy"]),
        )
        if abs(expected_min - float(row["min_score"])) > 1e-4:
            raise ValueError(f"Layer {layer} has inconsistent min_score")
        candidates.append(
            {
                "api_index": layer,
                "npz_index": int(row["npz_index"]),
                "classifier": row["classifier"],
                "best_C": float(row["best_C"]),
                "answer_accuracy": round(answer_accuracy, 4),
                "visual_contradiction_accuracy": round(float(row["wo_recall"]), 4),
                "target_unavailable_accuracy": round(float(row["nt_accuracy"]), 4),
                "min_per_state_accuracy": round(float(row["min_score"]), 4),
            }
        )

    if len(split_counts) != 1:
        raise ValueError(f"Candidate layers use different splits: {sorted(split_counts)}")
    n_train, n_dev = split_counts.pop()
    n_extracted = int(manifest["n_train"])
    if n_extracted != n_train + n_dev:
        raise ValueError(
            f"Extracted record count {n_extracted} != internal split {n_train}+{n_dev}"
        )

    selected = max(candidates, key=lambda row: row["min_per_state_accuracy"])
    return {
        "schema_version": 1,
        "protocol": "pre_adaptation_frozen_base_candidate_comparison_v1",
        "model": "Qwen3-VL-8B-Instruct",
        "adapter": None,
        "readout": "pre_response_prompt_token",
        "candidate_policy": "compare_fixed_api_indices_9_18_27",
        "selection_metric": "maximum minimum per-state development accuracy",
        "split": {
            "group_disjoint": True,
            "seed": int(scan["dev_split_seed"]),
            "extracted_records": n_extracted,
            "probe_fit_records": n_train,
            "probe_development_records": n_dev,
        },
        "candidates": candidates,
        "selected_api_index": int(selected["api_index"]),
        "selected_min_per_state_accuracy": float(selected["min_per_state_accuracy"]),
        "index_mapping": "API index L is hidden_states[L]; API index 27 is block-26 output",
        "source_provenance": {
            "manifest_sha256": file_sha256(manifest_path),
            "dev_scan_sha256": file_sha256(dev_scan_path),
            "min_score_curve_sha256": file_sha256(min_score_curve_path),
            "input_sha256": manifest["input_sha256"],
            "selection_records_sha256": manifest["rl_samples_sha256"],
            "split_assignment_sha256": manifest["split_assignment_sha256"],
            "extraction_script_sha256": manifest["script_sha256"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dev-scan", type=Path, required=True)
    parser.add_argument("--min-score-curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = build_artifact(args.manifest, args.dev_scan, args.min_score_curve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
