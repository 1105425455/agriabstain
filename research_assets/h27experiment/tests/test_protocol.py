from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INDEPENDENT_SHA256 = "76709bd2012e0bd0e48f0f01eb05b50466000e49b028ccbc3d48113e301e8555"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_training_data() -> None:
    manifest = json.loads((EXPERIMENT_ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    audit = manifest["training_data_audit"]
    assert manifest["seeds"] == {
        "dev_subset_sampling": 42,
        "full_pipeline": [7, 21, 42],
    }
    assert audit["grouped"]["rows"] == 7072
    assert audit["independent"]["rows"] == 7072
    assert audit["grouped"]["state_counts"] == audit["independent"]["state_counts"]
    independent = EXPERIMENT_ROOT / "data/sft_train_independent.json"
    assert sha256_file(independent) == EXPECTED_INDEPENDENT_SHA256


def test_pipeline_does_not_rebuild_independent_data() -> None:
    script = (EXPERIMENT_ROOT / "code/run_condition.sh").read_text(encoding="utf-8")
    assert "build_independent_negatives.py" not in script
    assert "--dump-hidden" in script
    assert "--dump-all-answer-candidates" in script
    assert 'RUN_SEED" != "7"' in script
    assert "dev_subset_sampling_seed=42" in script


def test_linear_reroute_smoke(tmp_path: Path) -> None:
    model_path = tmp_path / "linear_gate.npz"
    coef = np.zeros((3, 4096), dtype=np.float32)
    coef[0, 0] = 2.0
    coef[1, 0] = -2.0
    intercept = np.asarray([0.0, 0.0, 0.5], dtype=np.float32)
    np.savez_compressed(
        model_path,
        coef=coef,
        intercept=intercept,
        classes=np.asarray([0, 1, 2], dtype=np.int64),
        mu=np.zeros(4096, dtype=np.float32),
        sd=np.ones(4096, dtype=np.float32),
        run_seed=np.asarray(7, dtype=np.int64),
        selected_c=np.asarray(0.1, dtype=np.float64),
    )

    rows = [
        {
            "sample_id": "answer",
            "gt": {
                "decision": "answer",
                "label": "leaf",
                "boxes": [[0.0, 0.0, 100.0, 100.0]],
            },
            "head_hidden": [2.0, *([0.0] * 4095)],
            "answer_candidate": {
                "valid": True,
                "label": "leaf",
                "box": [0.0, 0.0, 100.0, 100.0],
            },
        },
        {
            # Duplicate semantic IDs are valid in the released probe annotations.
            "sample_id": "answer",
            "gt": {"decision": "abstain", "reason": "wrong_object"},
            "head_hidden": [-2.0, *([0.0] * 4095)],
            "answer_candidate": {"valid": False},
        },
        {
            "sample_id": "missing",
            "gt": {"decision": "abstain", "reason": "no_target"},
            "head_hidden": [0.0, *([0.0] * 4095)],
            "answer_candidate": {"valid": False},
        },
        {
            "sample_id": "ambiguous",
            "gt": {"decision": None, "reason": None},
            "head_hidden": [0.0, *([0.0] * 4095)],
            "answer_candidate": {"valid": False},
        },
    ]
    predictions = tmp_path / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_ROOT / "code/reroute_linear.py"),
            "--model",
            str(model_path),
            "--predictions",
            str(predictions),
            "--dataset",
            "comprehensive",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["state_accuracy"] == 1.0
    assert metrics["FCA"] == 0.0
    assert metrics["supported_query_retention"] == 1.0
    assert metrics["MCC"] == 1.0
    assert metrics["n_samples"] == 4
    assert metrics["n_scored"] == 3
    assert metrics["n_ambiguous_skipped"] == 1
    probabilities = [
        json.loads(line)
        for line in (output_dir / "linear_probabilities.jsonl").read_text().splitlines()
    ]
    assert len(probabilities) == 4
    assert [row["record_index"] for row in probabilities] == [0, 1, 2, 3]
