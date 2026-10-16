#!/usr/bin/env python3
"""Copy and audit the frozen inputs used by the H27 follow-up experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
AAAI_ROOT = REPO_ROOT / "aaai_code"
DATA_ROOT = EXPERIMENT_ROOT / "data"

SOURCES = {
    "sft_train_grouped.json": REPO_ROOT / "new_reason/outputs/sft_3reason/sft_train.json",
    "sft_train_independent.json": (
        AAAI_ROOT
        / "experiments/grouping_ablation/artifacts/data/sft_train_independent_negatives.json"
    ),
    "sft_dev.json": REPO_ROOT / "new_reason/outputs/sft_3reason/sft_dev.json",
    "dev.jsonl": REPO_ROOT / "new_reason/outputs/splits/dev.jsonl",
    "comprehensive.jsonl": REPO_ROOT / "new_reason/outputs/sft_3reason/sft_test.jsonl",
    "L1_discrim.jsonl": REPO_ROOT / "new_reason/outputs/eval_probes/test_L1_discrim.jsonl",
    "L2_evidence.jsonl": REPO_ROOT / "new_reason/outputs/eval_probes/test_L2_evidence.jsonl",
    "L3_dose.jsonl": REPO_ROOT / "new_reason/outputs/eval_probes/test_L3_dose.jsonl",
    "refcocog.jsonl": REPO_ROOT / "new_reason/outputs/eval_probes/ood_refcocog.jsonl",
    "plantwild.jsonl": (
        REPO_ROOT / "new_reason/outputs/eval_probes/plantwild_real_closed12_v1/"
        "plantwild_real_closed12_paired_408.jsonl"
    ),
    "discrim_musd.pt": REPO_ROOT / "final_story/outputs/discrim_musd.pt",
}

EXPECTED_GROUPED_SHA256 = "f6a0f922be69a5f5a585c89937ced60101a413c33bac7c6f98c6ff7967472663"
EXPECTED_INDEPENDENT_SHA256 = "76709bd2012e0bd0e48f0f01eb05b50466000e49b028ccbc3d48113e301e8555"
EXPECTED_STATE_COUNTS = {"0": 2703, "1": 1500, "2": 2869}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_train(path: Path) -> dict[str, Any]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a JSON list")
    counts = Counter(str(int(row["head_label"])) for row in rows)
    result = {
        "rows": len(rows),
        "state_counts": dict(sorted(counts.items())),
        "sha256": sha256_file(path),
    }
    if result["rows"] != 7072:
        raise ValueError(f"{path}: expected 7072 rows, found {result['rows']}")
    if result["state_counts"] != EXPECTED_STATE_COUNTS:
        raise ValueError(
            f"{path}: unexpected state counts {result['state_counts']}; "
            f"expected {EXPECTED_STATE_COUNTS}"
        )
    return result


def image_paths_from_json(path: Path) -> list[str]:
    rows = load_json(path)
    return [
        str(image)
        for row in rows
        for image in (
            row.get("image") if isinstance(row.get("image"), list) else [row.get("image")]
        )
        if image
    ]


def image_paths_from_jsonl(path: Path) -> list[str]:
    result: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            images = row.get("images")
            if not isinstance(images, list):
                images = [row.get("image_path")]
            result.extend(str(image) for image in images if image)
    return result


def audit_media(paths: list[str], relative_root: Path) -> dict[str, Any]:
    resolved = {Path(path) if Path(path).is_absolute() else relative_root / path for path in paths}
    missing = sorted(str(path) for path in resolved if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} media files; first entries: {missing[:5]}")
    return {"references": len(paths), "unique_files": len(resolved), "missing": 0}


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "fixed-data full-pipeline seeds 7/21/42",
        "files": {},
    }
    for destination_name, source in SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(f"Missing source input: {source}")
        destination = DATA_ROOT / destination_name
        shutil.copy2(source, destination)
        manifest["files"][destination_name] = {
            "source": str(source),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    grouped_audit = audit_train(DATA_ROOT / "sft_train_grouped.json")
    independent_audit = audit_train(DATA_ROOT / "sft_train_independent.json")
    if grouped_audit["sha256"] != EXPECTED_GROUPED_SHA256:
        raise ValueError(
            f"Grouped data hash changed: {grouped_audit['sha256']} != {EXPECTED_GROUPED_SHA256}"
        )
    if independent_audit["sha256"] != EXPECTED_INDEPENDENT_SHA256:
        raise ValueError(
            "Independent data hash changed: "
            f"{independent_audit['sha256']} != {EXPECTED_INDEPENDENT_SHA256}"
        )
    manifest["training_data_audit"] = {
        "grouped": grouped_audit,
        "independent": independent_audit,
        "independent_assignment_is_fixed_across_run_seeds": True,
    }
    manifest["media_audit"] = {
        "sft_train_grouped.json": audit_media(
            image_paths_from_json(DATA_ROOT / "sft_train_grouped.json"),
            Path("/"),
        ),
        "sft_train_independent.json": audit_media(
            image_paths_from_json(DATA_ROOT / "sft_train_independent.json"),
            Path("/"),
        ),
    }
    for name in (
        "comprehensive.jsonl",
        "L1_discrim.jsonl",
        "L2_evidence.jsonl",
        "L3_dose.jsonl",
        "refcocog.jsonl",
        "plantwild.jsonl",
    ):
        manifest["media_audit"][name] = audit_media(
            image_paths_from_jsonl(DATA_ROOT / name),
            Path("/"),
        )
    manifest["seeds"] = {
        "full_pipeline": [7, 21, 42],
        "dev_subset_sampling": 42,
    }

    output = DATA_ROOT / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[prepare-data] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
