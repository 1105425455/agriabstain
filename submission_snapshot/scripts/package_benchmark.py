#!/usr/bin/env python
"""
package_benchmark.py — Data-only packaging of the ReGate benchmark subset.

This script copies the curated benchmark data (annotations, source images,
OOD images) from the full research repository into a self-contained
``benchmark/`` directory with all paths rewritten to relative POSIX paths.

It does NOT include code (reason_abstain/), configs, scripts, pyproject.toml,
or README — only the data and annotation files needed to run evaluation.

Usage:
    python scripts/package_benchmark.py \
        --source-root /path/to/research-repo \
        --output-root /path/to/benchmark

Set PLANTDOC_ROOT if the PlantDoc release does not sit at <source-root>/datasets/plantdoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Constants / expected counts
# ──────────────────────────────────────────────────────────────────────

EXPECTED_IMAGE_COUNTS = {
    "source_plantdoc": 726,
    "source_coco": 203,
    "ood_refcocog": 181,
    "ood_plantwild_closed12": 144,
}

EXPECTED_ANNOTATION_COUNTS = {
    "sft_train": 7072,
    "sft_dev": 2408,
    "sft_test": 2646,
    "closed12": 408,
    "refcocog": 362,
}

# Files to copy from new_reason/outputs/splits/
SPLIT_FILES = [
    "train.jsonl",
    "dev.jsonl",
    "test.jsonl",
    "dose_pool.jsonl",
    "split_manifest.json",
    "SPLIT_METADATA_FIX_AUDIT.json",
]

# Files to copy from new_reason/outputs/sft_3reason/
SFT_FILES = [
    "sft_train.json",
    "sft_train_with_source.json",
    "sft_dev.json",
    "sft_test.jsonl",
]

# Raw benchmark files
RAW_BENCHMARK_FILES = [
    "plantdoc_full.raw.jsonl",
    "plantdoc_full.raw.summary.json",
    "plantdoc_full.raw.config.snapshot.yaml",
    "coco_full.raw.jsonl",
    "coco_full.raw.summary.json",
    "coco_full.raw.config.snapshot.yaml",
]

# Eval probe files
EVAL_PROBE_FILES = [
    "dev_L1_discrim.jsonl",
    "dev_L2_evidence.jsonl",
    "dev_L3_dose.jsonl",
    "test_L1_discrim.jsonl",
    "test_L2_evidence.jsonl",
    "test_L3_dose.jsonl",
    "ood_refcocog.jsonl",
    "REFCOCOG_IMAGE_ID_ISOLATION_AUDIT.json",
]

# Closed12 files
CLOSED12_FILES = [
    "plantwild_real_closed12_paired_408.jsonl",
    "BUILD_AUDIT.json",
]

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def count_jsonl(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def normalize_capability_label(value: Any) -> str:
    """Normalize labels exactly for the capability-gate audit."""
    return re.sub(r"[_\s]+", " ", str(value or "").strip().casefold())


def build_plantdoc_capability_audit(
    predictions_path: Path,
    calibration_path: Path,
    historical_gate_path: Path,
    *,
    threshold: float = 0.40,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Recompute the pre-split PlantDoc capability audit from saved predictions."""
    rows = load_jsonl(predictions_path)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("label")), []).append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group)
        answered = 0
        label_matches = 0
        correct = 0
        ious: list[float] = []
        for row in group:
            parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
            is_answer = parsed.get("valid") is True and parsed.get("decision") == "answer"
            answered += int(is_answer)
            label_match = is_answer and normalize_capability_label(
                parsed.get("label")
            ) == normalize_capability_label(row.get("label"))
            label_matches += int(label_match)
            if row.get("iou") is not None:
                iou = float(row["iou"])
                ious.append(iou)
                correct += int(label_match and iou >= iou_threshold)
        return {
            "n": n,
            "answer_rate": answered / n if n else 0.0,
            "label_match_rate": label_matches / n if n else 0.0,
            "correct_rate": correct / n if n else 0.0,
            "mean_iou_on_answered": sum(ious) / len(ious) if ious else 0.0,
        }

    per_label = {label: summarize(group) for label, group in sorted(by_label.items())}
    excluded = sorted(
        label for label, stats in per_label.items() if stats["correct_rate"] < threshold
    )
    retained = sorted(label for label in per_label if label not in excluded)
    return {
        "audit_name": "PlantDoc pre-split capability-gate audit",
        "version": "1.0",
        "selection_timing": (
            "Model-informed label-space curation was performed on the complete pre-split "
            "PlantDoc base-record pool before source-image assignment to benchmark train/dev/test."
        ),
        "scope_note": (
            "The `train` token in source sample IDs denotes the upstream PlantDoc directory; "
            "it is not the later leakage-safe benchmark train split."
        ),
        "model": "Qwen3-VL-8B-Instruct (raw base, no task adapter)",
        "relation_filter": sorted({str(row.get("relation")) for row in rows}),
        "num_instances": len(rows),
        "observed_label_count": len(per_label),
        "thresholds": {"correct_rate": threshold, "iou": iou_threshold},
        "match": "case-insensitive after underscore/whitespace normalization",
        "capability_excluded_labels": excluded,
        "capability_retained_labels": retained,
        "capability_retained_label_count": len(retained),
        "additional_config_exclusion": {
            "label": "plantdoc_unverified_class_14",
            "reason": "unverified placeholder class; absent from the calibration pool",
        },
        "configured_but_unobserved_label": "tomato_leaf_yellow_virus",
        "final_observed_agricultural_label_count": len(retained),
        "per_label": per_label,
        "provenance": {
            "calibration_records": {
                "path": "outputs/benchmarks/sampled/calibration/plantdoc_answerable_calibration.jsonl",
            },
            "base_predictions": {
                "path": "outputs/evals/qwen3vl8b_raw/plantdoc_answerable_predictions.jsonl",
            },
            "historical_gate_summary": {
                "path": "outputs/evals/qwen3vl8b_raw/plantdoc_capability_gate.json",
            },
        },
        "limitation": (
            "Because the capability filter precedes benchmark splitting and uses base-model "
            "predictions, the agricultural label space is capability-controlled rather than "
            "a model-independent sample of all PlantDoc classes."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Path rewriting
# ──────────────────────────────────────────────────────────────────────


def rewrite_path_in_value(
    value: str,
    plantdoc_images_old: str,
    plantdoc_images_new: str,
    coco_images_old: str,
    coco_images_new: str,
    derived_old: str,
    derived_new: str,
    refcocog_old: str,
    refcocog_new: str,
    plantwild_old: str,
    plantwild_new: str,
) -> str:
    """Rewrite a single string value if it looks like a path with old prefix."""

    # 1) PlantDoc source images
    if plantdoc_images_old in value:
        basename = os.path.basename(value)
        return f"{plantdoc_images_new}/{basename}"

    # 2) COCO source images (train2017)
    if coco_images_old in value:
        basename = os.path.basename(value)
        return f"{coco_images_new}/{basename}"

    # 3) Derived (transformed) images — rewrite path but don't copy
    if derived_old in value:
        rel = value.split(derived_old, 1)[1]
        if rel.startswith("/"):
            rel = rel[1:]
        return f"{derived_new}/{rel}"

    # 4) RefCOCOg OOD images
    if refcocog_old in value:
        basename = os.path.basename(value)
        return f"{refcocog_new}/{basename}"

    # 5) PlantWild OOD images — preserve category subdirectory
    if plantwild_old in value:
        marker = "plantwild/images/"
        if marker in value:
            rel = value.split(marker, 1)[1]
            return f"{plantwild_new}/{rel}"
        return f"{plantwild_new}/{os.path.basename(value)}"

    # 6) Paths stored in the RefCOCOg provenance audit
    audit_path_map = {
        "/data/ood/refcocog/data/val-00000-of-00003.parquet": (
            "data/ood/refcocog/metadata/val-00000-of-00003.parquet"
        ),
        "/new_reason/outputs/splits/train.jsonl": "annotations/splits/train.jsonl",
        "/new_reason/outputs/eval_probes/ood_refcocog.jsonl": (
            "annotations/eval/ood_refcocog.jsonl"
        ),
    }
    for old_suffix, new_path in audit_path_map.items():
        if value.endswith(old_suffix):
            return new_path

    return value


def deep_rewrite(
    obj: Any,
    plantdoc_images_old: str,
    plantdoc_images_new: str,
    coco_images_old: str,
    coco_images_new: str,
    derived_old: str,
    derived_new: str,
    refcocog_old: str,
    refcocog_new: str,
    plantwild_old: str,
    plantwild_new: str,
) -> Any:
    """Recursively rewrite all string values containing old paths."""
    if isinstance(obj, str):
        return rewrite_path_in_value(
            obj,
            plantdoc_images_old,
            plantdoc_images_new,
            coco_images_old,
            coco_images_new,
            derived_old,
            derived_new,
            refcocog_old,
            refcocog_new,
            plantwild_old,
            plantwild_new,
        )
    elif isinstance(obj, dict):
        return {
            k: deep_rewrite(
                v,
                plantdoc_images_old,
                plantdoc_images_new,
                coco_images_old,
                coco_images_new,
                derived_old,
                derived_new,
                refcocog_old,
                refcocog_new,
                plantwild_old,
                plantwild_new,
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [
            deep_rewrite(
                item,
                plantdoc_images_old,
                plantdoc_images_new,
                coco_images_old,
                coco_images_new,
                derived_old,
                derived_new,
                refcocog_old,
                refcocog_new,
                plantwild_old,
                plantwild_new,
            )
            for item in obj
        ]
    return obj


def rewrite_image_path_in_record(
    record: dict,
    plantdoc_images_old: str,
    plantdoc_images_new: str,
    coco_images_old: str,
    coco_images_new: str,
    derived_old: str,
    derived_new: str,
    refcocog_old: str,
    refcocog_new: str,
    plantwild_old: str,
    plantwild_new: str,
) -> dict:
    """Deep-rewrite all image_path/image/images fields in a record."""
    return deep_rewrite(
        record,
        plantdoc_images_old,
        plantdoc_images_new,
        coco_images_old,
        coco_images_new,
        derived_old,
        derived_new,
        refcocog_old,
        refcocog_new,
        plantwild_old,
        plantwild_new,
    )


# ──────────────────────────────────────────────────────────────────────
# Main packaging logic
# ──────────────────────────────────────────────────────────────────────


def package_benchmark(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Execute data-only packaging (images + annotations) and return a summary dict.

    This script only copies data/ and annotations/ into benchmark/. It does NOT
    copy configs, scripts, reason_abstain code, pyproject.toml, or README.
    """
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    release_root = source_root / "aaai_code"
    output_inside_source = source_root == output_root or source_root in output_root.parents
    output_inside_release = release_root == output_root or release_root in output_root.parents
    if output_root == source_root or output_root in source_root.parents:
        raise ValueError("Output root must not equal or contain the source root")
    if output_inside_source and not output_inside_release:
        raise ValueError("Output root inside source root must be under <source_root>/aaai_code")

    report: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "steps": {},
        "errors": [],
        "warnings": [],
    }

    # Resolve key sub-paths
    NR = source_root / "new_reason"
    NR_OUT = NR / "outputs"
    NR_SPLITS = NR_OUT / "splits"
    NR_SFT = NR_OUT / "sft_3reason"
    NR_BENCHMARKS = NR_OUT / "benchmarks"
    NR_EVAL = NR_OUT / "eval_probes"

    plantdoc_root = Path(os.environ.get("PLANTDOC_ROOT", source_root / "datasets" / "plantdoc"))
    QFORMER_PLANTDOC_IMAGES = plantdoc_root / "train" / "images"
    QFORMER_PLANTDOC_LABELS = plantdoc_root / "train" / "labels"
    COCO_IMAGES = source_root / "datasets" / "train2017"
    COCO_ANNOTATIONS = source_root / "datasets" / "annotations" / "instances_train2017.json"

    REFCOCOG_IMAGES_SRC = source_root / "data" / "ood" / "refcocog" / "images"
    REFCOCOG_PARQUET = (
        source_root / "data" / "ood" / "refcocog" / "data" / "val-00000-of-00003.parquet"
    )
    REFCOCOG_GATE = source_root / "outputs" / "ood" / "cal" / "gate_refcocog.json"

    PLANTDOC_CAPABILITY_RECORDS = (
        source_root
        / "outputs"
        / "benchmarks"
        / "sampled"
        / "calibration"
        / "plantdoc_answerable_calibration.jsonl"
    )
    PLANTDOC_CAPABILITY_PREDICTIONS = (
        source_root
        / "outputs"
        / "evals"
        / "qwen3vl8b_raw"
        / "plantdoc_answerable_predictions.jsonl"
    )
    PLANTDOC_CAPABILITY_GATE = (
        source_root / "outputs" / "evals" / "qwen3vl8b_raw" / "plantdoc_capability_gate.json"
    )

    PLANTWILD_IMAGES_SRC = (
        NR / "data" / "plantwild_source_disjoint_v1" / "raw" / "plantwild" / "images"
    )

    # ── Paths for rewriting ──
    plantdoc_images_old = str(QFORMER_PLANTDOC_IMAGES)
    coco_images_old = str(COCO_IMAGES)
    derived_old = str(NR_OUT / "benchmarks" / "transformed_images")
    refcocog_old = str(REFCOCOG_IMAGES_SRC)
    plantwild_old = str(PLANTWILD_IMAGES_SRC)

    plantdoc_images_new = "data/source/plantdoc/images"
    coco_images_new = "data/source/coco/images"
    derived_new = "data/derived"
    refcocog_new = "data/ood/refcocog/images"
    plantwild_new = "data/ood/plantwild_closed12/images"

    # ── Output subpaths ──
    OUT_ANNOTATIONS_RAW = output_root / "annotations" / "raw"
    OUT_ANNOTATIONS_SPLITS = output_root / "annotations" / "splits"
    OUT_ANNOTATIONS_SFT = output_root / "annotations" / "sft"
    OUT_ANNOTATIONS_EVAL = output_root / "annotations" / "eval"
    OUT_ANNOTATIONS_AUDITS = output_root / "annotations" / "audits"

    # ══════════════════════════════════════════════════════════════════
    # Step 0: Clean up any legacy artifacts under benchmark/ that should
    #         not be in a data-only package (e.g. from previous runs of
    #         an older version of this script).
    # ══════════════════════════════════════════════════════════════════
    print("Step 0: Cleaning legacy artifacts from output root...")

    legacy_dirs = [
        output_root / "reason_abstain",
        output_root / "scripts",
        output_root / "configs",
        output_root / ".ruff_cache",
        output_root / "outputs",
    ]
    for d in legacy_dirs:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed legacy: {d.relative_to(output_root)}")

    legacy_files = [
        output_root / "pyproject.toml",
        output_root / "README.md",
    ]
    for f in legacy_files:
        if f.exists():
            f.unlink()
            print(f"  Removed legacy: {f.relative_to(output_root)}")

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Create output directories (data-only)
    # ══════════════════════════════════════════════════════════════════
    print("Step 1: Creating output directories...")

    managed_dirs = [
        output_root / "data",
        output_root / "annotations",
    ]
    for managed_dir in managed_dirs:
        if managed_dir.exists():
            shutil.rmtree(managed_dir)

    dirs = [
        output_root / "data" / "source" / "plantdoc" / "images",
        output_root / "data" / "source" / "plantdoc" / "labels",
        output_root / "data" / "source" / "coco" / "images",
        output_root / "data" / "source" / "coco" / "annotations",
        output_root / "data" / "ood" / "refcocog" / "images",
        output_root / "data" / "ood" / "refcocog" / "metadata",
        output_root / "data" / "ood" / "plantwild_closed12" / "images",
        output_root / "data" / "derived",
        OUT_ANNOTATIONS_RAW,
        OUT_ANNOTATIONS_SPLITS,
        OUT_ANNOTATIONS_SFT,
        OUT_ANNOTATIONS_EVAL,
        OUT_ANNOTATIONS_AUDITS,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Determine source images from split base items
    # ══════════════════════════════════════════════════════════════════
    print("Step 2: Determining source images from split base items...")

    plantdoc_images_set: set[str] = set()
    coco_images_set: set[str] = set()

    for split_name in ["train", "dev", "test"]:
        split_path = NR_SPLITS / f"{split_name}.jsonl"
        if not split_path.exists():
            report["errors"].append(f"Split file not found: {split_path}")
            continue
        items = load_jsonl(split_path)
        for item in items:
            rel = item.get("metadata", {}).get("relation", "")
            if rel != "base":
                continue
            img_path = item.get("image_path", "")
            if "plantdoc" in img_path.lower():
                plantdoc_images_set.add(img_path)
            elif "train2017" in img_path or "coco" in img_path.lower():
                coco_images_set.add(img_path)

    n_plantdoc = len(plantdoc_images_set)
    n_coco = len(coco_images_set)

    report["steps"]["source_images_found"] = {
        "plantdoc": n_plantdoc,
        "coco": n_coco,
    }

    if n_plantdoc != EXPECTED_IMAGE_COUNTS["source_plantdoc"]:
        report["warnings"].append(
            f"PlantDoc source images: found {n_plantdoc}, expected {EXPECTED_IMAGE_COUNTS['source_plantdoc']}"
        )
    if n_coco != EXPECTED_IMAGE_COUNTS["source_coco"]:
        report["warnings"].append(
            f"COCO source images: found {n_coco}, expected {EXPECTED_IMAGE_COUNTS['source_coco']}"
        )

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Copy PlantDoc source images + YOLO labels
    # ══════════════════════════════════════════════════════════════════
    print("Step 3: Copying PlantDoc source images and YOLO labels...")

    plantdoc_img_dst = output_root / "data" / "source" / "plantdoc" / "images"
    plantdoc_label_dst = output_root / "data" / "source" / "plantdoc" / "labels"

    plantdoc_images_copied = 0
    plantdoc_labels_missing: list[str] = []
    plantdoc_labels_copied = 0

    for src_path_str in sorted(plantdoc_images_set):
        src_path = Path(src_path_str)
        if not src_path.exists():
            report["warnings"].append(f"PlantDoc source image missing: {src_path}")
            continue
        dst_path = plantdoc_img_dst / src_path.name
        shutil.copy2(src_path, dst_path)
        plantdoc_images_copied += 1

        # YOLO label: same basename but .txt
        label_name = src_path.stem + ".txt"
        label_src = QFORMER_PLANTDOC_LABELS / label_name
        label_dst = plantdoc_label_dst / label_name
        if label_src.exists():
            shutil.copy2(label_src, label_dst)
            plantdoc_labels_copied += 1
        else:
            plantdoc_labels_missing.append(label_name)

    report["steps"]["plantdoc_images"] = {
        "found": n_plantdoc,
        "copied": plantdoc_images_copied,
        "labels_copied": plantdoc_labels_copied,
        "labels_missing": plantdoc_labels_missing,
    }
    if plantdoc_labels_missing:
        report["warnings"].append(
            f"PlantDoc YOLO labels missing for {len(plantdoc_labels_missing)} images: "
            + ", ".join(plantdoc_labels_missing[:10])
            + (
                f" ... and {len(plantdoc_labels_missing) - 10} more"
                if len(plantdoc_labels_missing) > 10
                else ""
            )
        )

    # ══════════════════════════════════════════════════════════════════
    # Step 4: Copy COCO source images
    # ══════════════════════════════════════════════════════════════════
    print("Step 4: Copying COCO source images...")

    coco_img_dst = output_root / "data" / "source" / "coco" / "images"
    coco_images_copied = 0
    coco_image_ids: list[int] = []

    for src_path_str in sorted(coco_images_set):
        src_path = Path(src_path_str)
        if not src_path.exists():
            report["warnings"].append(f"COCO source image missing: {src_path}")
            continue
        dst_path = coco_img_dst / src_path.name
        shutil.copy2(src_path, dst_path)
        coco_images_copied += 1
        try:
            coco_id = int(src_path.stem)
            coco_image_ids.append(coco_id)
        except ValueError:
            pass

    report["steps"]["coco_images"] = {
        "found": n_coco,
        "copied": coco_images_copied,
        "image_ids": sorted(coco_image_ids),
    }

    # ══════════════════════════════════════════════════════════════════
    # Step 5: Generate COCO annotation subset
    # ══════════════════════════════════════════════════════════════════
    print("Step 5: Generating COCO annotation subset...")

    coco_ann_dst = (
        output_root / "data" / "source" / "coco" / "annotations" / "instances_selected.json"
    )

    selected_image_ids = set(coco_image_ids)
    coco_full = load_json(COCO_ANNOTATIONS)

    selected_images = [img for img in coco_full["images"] if img["id"] in selected_image_ids]
    selected_annotations = [
        ann for ann in coco_full["annotations"] if ann["image_id"] in selected_image_ids
    ]
    all_categories = coco_full["categories"]

    found_image_ids = {img["id"] for img in selected_images}
    missing_ids = selected_image_ids - found_image_ids
    if missing_ids:
        report["warnings"].append(f"COCO image IDs not found in annotations: {sorted(missing_ids)}")

    coco_selected = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": all_categories,
    }
    write_json(coco_ann_dst, coco_selected)

    report["steps"]["coco_annotation_subset"] = {
        "selected_image_ids_count": len(selected_image_ids),
        "images_in_annotation": len(selected_images),
        "annotations_in_subset": len(selected_annotations),
        "categories_included": len(all_categories),
        "missing_image_ids": sorted(missing_ids) if missing_ids else [],
    }

    # ══════════════════════════════════════════════════════════════════
    # Step 6: Copy OOD RefCOCOg images + metadata
    # ══════════════════════════════════════════════════════════════════
    print("Step 6: Copying RefCOCOg OOD images and metadata...")

    refcocog_img_dst = output_root / "data" / "ood" / "refcocog" / "images"
    refcocog_meta_dst = output_root / "data" / "ood" / "refcocog" / "metadata"

    refcocog_records = load_jsonl(NR_EVAL / "ood_refcocog.jsonl")
    refcocog_image_paths: set[str] = set()
    for rec in refcocog_records:
        for img in rec.get("images", []):
            if isinstance(img, str):
                refcocog_image_paths.add(img)

    n_refcocog = len(refcocog_image_paths)
    report["steps"]["refcocog_images_found"] = {"unique_images": n_refcocog}

    if n_refcocog != EXPECTED_IMAGE_COUNTS["ood_refcocog"]:
        report["warnings"].append(
            f"RefCOCOg unique images: found {n_refcocog}, expected {EXPECTED_IMAGE_COUNTS['ood_refcocog']}"
        )

    refcocog_copied = 0
    refcocog_missing: list[str] = []
    for p in sorted(refcocog_image_paths):
        src = Path(p)
        if not src.exists():
            refcocog_missing.append(p)
            continue
        dst = refcocog_img_dst / src.name
        shutil.copy2(src, dst)
        refcocog_copied += 1

    report["steps"]["refcocog_images_copied"] = {
        "copied": refcocog_copied,
        "missing": refcocog_missing,
    }

    if REFCOCOG_PARQUET.exists():
        shutil.copy2(REFCOCOG_PARQUET, refcocog_meta_dst / "val-00000-of-00003.parquet")
    else:
        report["warnings"].append(f"RefCOCOg parquet not found: {REFCOCOG_PARQUET}")

    if REFCOCOG_GATE.exists():
        shutil.copy2(REFCOCOG_GATE, refcocog_meta_dst / "gate_refcocog.json")
    else:
        report["warnings"].append(f"RefCOCOg gate not found: {REFCOCOG_GATE}")

    # ══════════════════════════════════════════════════════════════════
    # Step 7: Copy PlantWild closed12 images (with category subdirs)
    # ══════════════════════════════════════════════════════════════════
    print("Step 7: Copying PlantWild closed12 OOD images...")

    closed12_path = (
        NR_EVAL / "plantwild_real_closed12_v1" / "plantwild_real_closed12_paired_408.jsonl"
    )
    closed12_records = load_jsonl(closed12_path)
    plantwild_image_paths: set[str] = set()
    for rec in closed12_records:
        for img in rec.get("images", []):
            if isinstance(img, str):
                plantwild_image_paths.add(img)

    n_plantwild = len(plantwild_image_paths)
    report["steps"]["plantwild_images_found"] = {"unique_images": n_plantwild}

    if n_plantwild != EXPECTED_IMAGE_COUNTS["ood_plantwild_closed12"]:
        report["warnings"].append(
            f"PlantWild closed12 unique images: found {n_plantwild}, expected {EXPECTED_IMAGE_COUNTS['ood_plantwild_closed12']}"
        )

    plantwild_img_dst = output_root / "data" / "ood" / "plantwild_closed12" / "images"
    plantwild_copied = 0
    plantwild_missing: list[str] = []

    for p in sorted(plantwild_image_paths):
        src = Path(p)
        if not src.exists():
            plantwild_missing.append(p)
            continue
        marker = "plantwild/images/"
        if marker in p:
            rel = p.split(marker, 1)[1]
            dst = plantwild_img_dst / rel
        else:
            dst = plantwild_img_dst / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        plantwild_copied += 1

    report["steps"]["plantwild_images_copied"] = {
        "copied": plantwild_copied,
        "missing": plantwild_missing,
    }

    # ══════════════════════════════════════════════════════════════════
    # Step 8: Copy annotation files with path rewriting
    # ══════════════════════════════════════════════════════════════════
    print("Step 8: Copying and rewriting annotation files...")

    annotation_files: list[dict[str, Any]] = []

    # 8a: Split files
    for fname in SPLIT_FILES:
        src = NR_SPLITS / fname
        if not src.exists():
            report["errors"].append(f"Split file not found: {src}")
            continue
        if fname.endswith(".jsonl"):
            records = load_jsonl(src)
            rewritten = [
                rewrite_image_path_in_record(
                    r,
                    plantdoc_images_old,
                    plantdoc_images_new,
                    coco_images_old,
                    coco_images_new,
                    derived_old,
                    derived_new,
                    refcocog_old,
                    refcocog_new,
                    plantwild_old,
                    plantwild_new,
                )
                for r in records
            ]
            write_jsonl(OUT_ANNOTATIONS_SPLITS / fname, rewritten)
            annotation_files.append(
                {
                    "path": f"annotations/splits/{fname}",
                    "records": len(rewritten),
                    "format": "jsonl",
                }
            )
        elif fname.endswith(".json"):
            data = load_json(src)
            rewritten = deep_rewrite(
                data,
                plantdoc_images_old,
                plantdoc_images_new,
                coco_images_old,
                coco_images_new,
                derived_old,
                derived_new,
                refcocog_old,
                refcocog_new,
                plantwild_old,
                plantwild_new,
            )
            write_json(OUT_ANNOTATIONS_SPLITS / fname, rewritten)
            annotation_files.append({"path": f"annotations/splits/{fname}", "format": "json"})

    # 8b: SFT files
    for fname in SFT_FILES:
        src = NR_SFT / fname
        if not src.exists():
            report["errors"].append(f"SFT file not found: {src}")
            continue
        if fname.endswith(".jsonl"):
            records = load_jsonl(src)
            rewritten = [
                rewrite_image_path_in_record(
                    r,
                    plantdoc_images_old,
                    plantdoc_images_new,
                    coco_images_old,
                    coco_images_new,
                    derived_old,
                    derived_new,
                    refcocog_old,
                    refcocog_new,
                    plantwild_old,
                    plantwild_new,
                )
                for r in records
            ]
            write_jsonl(OUT_ANNOTATIONS_SFT / fname, rewritten)
            annotation_files.append(
                {"path": f"annotations/sft/{fname}", "records": len(rewritten), "format": "jsonl"}
            )
        elif fname.endswith(".json"):
            records = load_json(src)
            rewritten = [
                rewrite_image_path_in_record(
                    r,
                    plantdoc_images_old,
                    plantdoc_images_new,
                    coco_images_old,
                    coco_images_new,
                    derived_old,
                    derived_new,
                    refcocog_old,
                    refcocog_new,
                    plantwild_old,
                    plantwild_new,
                )
                for r in records
            ]
            write_json(OUT_ANNOTATIONS_SFT / fname, rewritten)
            annotation_files.append(
                {"path": f"annotations/sft/{fname}", "records": len(rewritten), "format": "json"}
            )

    # 8c: Raw benchmark files
    for fname in RAW_BENCHMARK_FILES:
        src = NR_BENCHMARKS / fname
        if not src.exists():
            report["errors"].append(f"Raw benchmark file not found: {src}")
            continue
        if fname.endswith(".jsonl"):
            records = load_jsonl(src)
            rewritten = [
                rewrite_image_path_in_record(
                    r,
                    plantdoc_images_old,
                    plantdoc_images_new,
                    coco_images_old,
                    coco_images_new,
                    derived_old,
                    derived_new,
                    refcocog_old,
                    refcocog_new,
                    plantwild_old,
                    plantwild_new,
                )
                for r in records
            ]
            write_jsonl(OUT_ANNOTATIONS_RAW / fname, rewritten)
            annotation_files.append(
                {"path": f"annotations/raw/{fname}", "records": len(rewritten), "format": "jsonl"}
            )
        elif fname.endswith(".json"):
            data = load_json(src)
            rewritten = deep_rewrite(
                data,
                plantdoc_images_old,
                plantdoc_images_new,
                coco_images_old,
                coco_images_new,
                derived_old,
                derived_new,
                refcocog_old,
                refcocog_new,
                plantwild_old,
                plantwild_new,
            )
            write_json(OUT_ANNOTATIONS_RAW / fname, rewritten)
            annotation_files.append({"path": f"annotations/raw/{fname}", "format": "json"})
        elif fname.endswith(".yaml"):
            content = src.read_text(encoding="utf-8")
            content = content.replace(plantdoc_images_old, plantdoc_images_new)
            content = content.replace(str(QFORMER_PLANTDOC_LABELS), "data/source/plantdoc/labels")
            content = content.replace(coco_images_old, coco_images_new)
            content = content.replace(derived_old, derived_new)
            content = content.replace(
                str(COCO_ANNOTATIONS), "data/source/coco/annotations/instances_selected.json"
            )
            (OUT_ANNOTATIONS_RAW / fname).write_text(content, encoding="utf-8")
            annotation_files.append({"path": f"annotations/raw/{fname}", "format": "yaml"})

    # 8d: Eval probe files
    for fname in EVAL_PROBE_FILES:
        src = NR_EVAL / fname
        if not src.exists():
            print(f"  (eval probe not found, skipping: {src})")
            continue
        if fname.endswith(".jsonl"):
            records = load_jsonl(src)
            rewritten = [
                rewrite_image_path_in_record(
                    r,
                    plantdoc_images_old,
                    plantdoc_images_new,
                    coco_images_old,
                    coco_images_new,
                    derived_old,
                    derived_new,
                    refcocog_old,
                    refcocog_new,
                    plantwild_old,
                    plantwild_new,
                )
                for r in records
            ]
            write_jsonl(OUT_ANNOTATIONS_EVAL / fname, rewritten)
            annotation_files.append(
                {"path": f"annotations/eval/{fname}", "records": len(rewritten), "format": "jsonl"}
            )
        elif fname.endswith(".json"):
            data = load_json(src)
            rewritten = deep_rewrite(
                data,
                plantdoc_images_old,
                plantdoc_images_new,
                coco_images_old,
                coco_images_new,
                derived_old,
                derived_new,
                refcocog_old,
                refcocog_new,
                plantwild_old,
                plantwild_new,
            )
            write_json(OUT_ANNOTATIONS_EVAL / fname, rewritten)
            annotation_files.append({"path": f"annotations/eval/{fname}", "format": "json"})

    # 8e: Closed12 files
    for fname in CLOSED12_FILES:
        src = NR_EVAL / "plantwild_real_closed12_v1" / fname
        if not src.exists():
            report["errors"].append(f"Closed12 file not found: {src}")
            continue
        if fname.endswith(".jsonl"):
            records = load_jsonl(src)
            rewritten = [
                rewrite_image_path_in_record(
                    r,
                    plantdoc_images_old,
                    plantdoc_images_new,
                    coco_images_old,
                    coco_images_new,
                    derived_old,
                    derived_new,
                    refcocog_old,
                    refcocog_new,
                    plantwild_old,
                    plantwild_new,
                )
                for r in records
            ]
            write_jsonl(OUT_ANNOTATIONS_EVAL / fname, rewritten)
            annotation_files.append(
                {"path": f"annotations/eval/{fname}", "records": len(rewritten), "format": "jsonl"}
            )
        elif fname.endswith(".json"):
            data = load_json(src)
            rewritten = deep_rewrite(
                data,
                plantdoc_images_old,
                plantdoc_images_new,
                coco_images_old,
                coco_images_new,
                derived_old,
                derived_new,
                refcocog_old,
                refcocog_new,
                plantwild_old,
                plantwild_new,
            )
            write_json(OUT_ANNOTATIONS_EVAL / fname, rewritten)
            annotation_files.append({"path": f"annotations/eval/{fname}", "format": "json"})

    # 8f: Pre-split PlantDoc capability-gate audit
    capability_sources = (
        PLANTDOC_CAPABILITY_RECORDS,
        PLANTDOC_CAPABILITY_PREDICTIONS,
        PLANTDOC_CAPABILITY_GATE,
    )
    missing_capability_sources = [str(path) for path in capability_sources if not path.exists()]
    if missing_capability_sources:
        report["errors"].append(
            "PlantDoc capability audit source(s) missing: " + ", ".join(missing_capability_sources)
        )
    else:
        capability_audit = build_plantdoc_capability_audit(
            PLANTDOC_CAPABILITY_PREDICTIONS,
            PLANTDOC_CAPABILITY_RECORDS,
            PLANTDOC_CAPABILITY_GATE,
            threshold=0.40,
            iou_threshold=0.5,
        )
        capability_audit_name = "PLANTDOC_CAPABILITY_GATE_AUDIT.json"
        write_json(OUT_ANNOTATIONS_AUDITS / capability_audit_name, capability_audit)
        annotation_files.append(
            {"path": f"annotations/audits/{capability_audit_name}", "format": "json"}
        )

    report["steps"]["annotation_files_copied"] = len(annotation_files)

    # ── Note: configs, scripts, and code are NOT copied into the benchmark
    #    directory. This is a data-only package. Config files live at the
    #    aaai_code/configs/ level; scripts at aaai_code/scripts/; code at
    #    aaai_code/reason_abstain/.

    # ══════════════════════════════════════════════════════════════════
    # Step 9: Build MANIFEST.json
    # ══════════════════════════════════════════════════════════════════
    print("Step 12: Building MANIFEST.json...")

    image_manifest: list[dict[str, Any]] = []

    for img_path in sorted(plantdoc_img_dst.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image_manifest.append(
                {
                    "path": f"data/source/plantdoc/images/{img_path.name}",
                    "sha256": sha256_file(img_path),
                    "bytes": file_size(img_path),
                    "domain": "source_plantdoc",
                }
            )

    for img_path in sorted(coco_img_dst.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image_manifest.append(
                {
                    "path": f"data/source/coco/images/{img_path.name}",
                    "sha256": sha256_file(img_path),
                    "bytes": file_size(img_path),
                    "domain": "source_coco",
                }
            )

    for img_path in sorted(refcocog_img_dst.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image_manifest.append(
                {
                    "path": f"data/ood/refcocog/images/{img_path.name}",
                    "sha256": sha256_file(img_path),
                    "bytes": file_size(img_path),
                    "domain": "ood_refcocog",
                }
            )

    for img_path in sorted(plantwild_img_dst.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            rel = img_path.relative_to(output_root)
            image_manifest.append(
                {
                    "path": str(rel),
                    "sha256": sha256_file(img_path),
                    "bytes": file_size(img_path),
                    "domain": "ood_plantwild_closed12",
                }
            )

    domain_counts: dict[str, int] = Counter()
    domain_bytes: dict[str, int] = Counter()
    for entry in image_manifest:
        domain_counts[entry["domain"]] += 1
        domain_bytes[entry["domain"]] += entry["bytes"]

    source_split_unique = {
        "plantdoc": {"train": 0, "dev": 0, "test": 0},
        "coco": {"train": 0, "dev": 0, "test": 0},
    }
    for split_name in ["train", "dev", "test"]:
        split_path = NR_SPLITS / f"{split_name}.jsonl"
        if split_path.exists():
            items = load_jsonl(split_path)
            for item in items:
                if item.get("metadata", {}).get("relation") != "base":
                    continue
                img_path = item.get("image_path", "")
                if "plantdoc" in img_path.lower():
                    source_split_unique["plantdoc"][split_name] += 1
                else:
                    source_split_unique["coco"][split_name] += 1
            # Deduplicate: count unique basenames
            plantdoc_imgs: set[str] = set()
            coco_imgs: set[str] = set()
            for item in items:
                if item.get("metadata", {}).get("relation") != "base":
                    continue
                img_path = item.get("image_path", "")
                if "plantdoc" in img_path.lower():
                    plantdoc_imgs.add(os.path.basename(img_path))
                else:
                    coco_imgs.add(os.path.basename(img_path))
            source_split_unique["plantdoc"][split_name] = len(plantdoc_imgs)
            source_split_unique["coco"][split_name] = len(coco_imgs)

    # Build annotation file section
    manifest_annotations: list[dict[str, Any]] = []
    for af in annotation_files:
        af_path = output_root / af["path"]
        if af_path.exists():
            entry = {
                "path": af["path"],
                "format": af["format"],
                "sha256": sha256_file(af_path),
                "bytes": file_size(af_path),
            }
            if "records" in af:
                entry["records"] = af["records"]
            else:
                if af["format"] == "jsonl":
                    entry["records"] = count_jsonl(af_path)
                elif af["format"] == "json":
                    try:
                        data = json.loads(af_path.read_text(encoding="utf-8"))
                        entry["records"] = len(data) if isinstance(data, list) else "N/A (dict)"
                    except Exception:
                        entry["records"] = "N/A"
            manifest_annotations.append(entry)

    # Verify expected counts
    sft_train_records = next(
        (e["records"] for e in manifest_annotations if e["path"].endswith("sft_train.json")), None
    )
    sft_dev_records = next(
        (e["records"] for e in manifest_annotations if e["path"].endswith("sft_dev.json")), None
    )
    sft_test_records = next(
        (e["records"] for e in manifest_annotations if e["path"].endswith("sft_test.jsonl")), None
    )
    closed12_records = next(
        (
            e["records"]
            for e in manifest_annotations
            if e["path"].endswith("plantwild_real_closed12_paired_408.jsonl")
        ),
        None,
    )
    refcocog_records = next(
        (e["records"] for e in manifest_annotations if e["path"].endswith("ood_refcocog.jsonl")),
        None,
    )

    count_checks = {}
    for label, actual, expected in [
        ("sft_train", sft_train_records, EXPECTED_ANNOTATION_COUNTS["sft_train"]),
        ("sft_dev", sft_dev_records, EXPECTED_ANNOTATION_COUNTS["sft_dev"]),
        ("sft_test", sft_test_records, EXPECTED_ANNOTATION_COUNTS["sft_test"]),
        ("closed12", closed12_records, EXPECTED_ANNOTATION_COUNTS["closed12"]),
        ("refcocog", refcocog_records, EXPECTED_ANNOTATION_COUNTS["refcocog"]),
    ]:
        if actual is not None and actual != expected:
            report["warnings"].append(f"Count mismatch: {label}: got {actual}, expected {expected}")
            count_checks[label] = {"actual": actual, "expected": expected, "match": False}
        elif actual is not None:
            count_checks[label] = {"actual": actual, "expected": expected, "match": True}
        else:
            count_checks[label] = {"actual": None, "expected": expected, "match": False}

    # Path audit: count old absolute prefixes
    old_prefixes = [
        str(source_root),
        str(QFORMER_PLANTDOC_IMAGES),
        str(COCO_IMAGES),
        str(NR_OUT / "benchmarks" / "transformed_images"),
        str(REFCOCOG_IMAGES_SRC),
        str(PLANTWILD_IMAGES_SRC),
    ]
    path_audit: dict[str, int] = {}
    for af in manifest_annotations:
        af_path = output_root / af["path"]
        if af_path.exists():
            content = af_path.read_text(encoding="utf-8")
            for prefix in old_prefixes:
                count = content.count(prefix)
                if count > 0:
                    path_audit[f"{prefix} in {af['path']}"] = count

    # Check referenced images vs actual
    source_image_refs: dict[str, set[str]] = {
        "source_plantdoc": set(),
        "source_coco": set(),
        "ood_refcocog": set(),
        "ood_plantwild_closed12": set(),
    }
    for af in manifest_annotations:
        af_path = output_root / af["path"]
        if not af_path.exists():
            continue
        content = af_path.read_text(encoding="utf-8")
        for domain, prefix in [
            ("source_plantdoc", "data/source/plantdoc/images/"),
            ("source_coco", "data/source/coco/images/"),
            ("ood_refcocog", "data/ood/refcocog/images/"),
            ("ood_plantwild_closed12", "data/ood/plantwild_closed12/images/"),
        ]:
            for match in re.finditer(re.escape(prefix) + r"[^\"\'\n]+", content):
                source_image_refs[domain].add(match.group(0))

    existing_images: dict[str, set[str]] = {}
    for domain, img_dir_name in [
        ("source_plantdoc", "data/source/plantdoc/images"),
        ("source_coco", "data/source/coco/images"),
        ("ood_refcocog", "data/ood/refcocog/images"),
        ("ood_plantwild_closed12", "data/ood/plantwild_closed12/images"),
    ]:
        d = output_root / img_dir_name
        if d.exists():
            existing = {
                str(f.relative_to(output_root))
                for f in d.rglob("*")
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            }
            existing_images[domain] = existing

    missing_image_refs: dict[str, list[str]] = {}
    for domain, refs in source_image_refs.items():
        missing = [r for r in refs if r not in existing_images.get(domain, set())]
        if missing:
            missing_image_refs[domain] = missing

    derived_refs: list[str] = []
    for af in manifest_annotations:
        af_path = output_root / af["path"]
        if not af_path.exists():
            continue
        content = af_path.read_text(encoding="utf-8")
        for match in re.finditer(r"data/derived/[^\"\'\n]+", content):
            derived_refs.append(match.group(0))

    # Build MANIFEST
    manifest = {
        "benchmark_name": "ReGate Benchmark Subset",
        "version": "1.0",
        "description": (
            "Curated subset of the ReGate benchmark for an AAAI submission. "
            "Contains source images, OOD evaluation images, and rewritten annotations. "
            "This is a data-only package (no code/scripts/configs included). "
            "Generate derived images using scripts/build_benchmark.py from aaai_code/."
        ),
        "image_counts": {
            "source_plantdoc": {
                "total": domain_counts.get("source_plantdoc", 0),
                "expected": EXPECTED_IMAGE_COUNTS["source_plantdoc"],
                "match": domain_counts.get("source_plantdoc", 0)
                == EXPECTED_IMAGE_COUNTS["source_plantdoc"],
                "by_split": source_split_unique["plantdoc"],
            },
            "source_coco": {
                "total": domain_counts.get("source_coco", 0),
                "expected": EXPECTED_IMAGE_COUNTS["source_coco"],
                "match": domain_counts.get("source_coco", 0)
                == EXPECTED_IMAGE_COUNTS["source_coco"],
                "by_split": source_split_unique["coco"],
            },
            "ood_refcocog": {
                "total": domain_counts.get("ood_refcocog", 0),
                "expected": EXPECTED_IMAGE_COUNTS["ood_refcocog"],
                "match": domain_counts.get("ood_refcocog", 0)
                == EXPECTED_IMAGE_COUNTS["ood_refcocog"],
            },
            "ood_plantwild_closed12": {
                "total": domain_counts.get("ood_plantwild_closed12", 0),
                "expected": EXPECTED_IMAGE_COUNTS["ood_plantwild_closed12"],
                "match": domain_counts.get("ood_plantwild_closed12", 0)
                == EXPECTED_IMAGE_COUNTS["ood_plantwild_closed12"],
            },
        },
        "annotation_counts": count_checks,
        "annotations": manifest_annotations,
        "path_audit": {
            "old_prefix_occurrences": path_audit,
            "old_prefix_occurrences_total": sum(path_audit.values()),
            "missing_image_refs": missing_image_refs,
            "missing_image_refs_total": sum(len(v) for v in missing_image_refs.values()),
            "derived_image_refs_count": len(derived_refs),
            "derived_image_refs": derived_refs[:20],
        },
        "image_manifest": "IMAGE_MANIFEST.json",
    }

    write_json(output_root / "MANIFEST.json", manifest)
    write_json(output_root / "IMAGE_MANIFEST.json", image_manifest)

    report["final"] = {
        "total_images": len(image_manifest),
        "total_image_bytes": sum(e["bytes"] for e in image_manifest),
        "total_annotation_files": len(manifest_annotations),
        "total_annotation_bytes": sum(e["bytes"] for e in manifest_annotations),
        "old_prefix_occurrences": sum(path_audit.values()),
        "missing_image_refs": sum(len(v) for v in missing_image_refs.values()),
        "derived_refs_count": len(derived_refs),
    }

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PACKAGING COMPLETE")
    print("=" * 60)
    print(f"  Source: {source_root}")
    print(f"  Output: {output_root}")
    print(f"  Total images: {report['final']['total_images']}")
    print(f"  Total image bytes: {report['final']['total_image_bytes']:,}")
    print(f"  Annotation files: {report['final']['total_annotation_files']}")
    print(f"  Old prefix occurrences: {report['final']['old_prefix_occurrences']}")
    print(f"  Missing image refs: {report['final']['missing_image_refs']}")
    print(f"  Derived refs (expected missing): {report['final']['derived_refs_count']}")
    if report["warnings"]:
        print(f"\n  Warnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"    ⚠ {w}")
    if report["errors"]:
        print(f"\n  Errors ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"    ❌ {e}")

    return report


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data-only packaging of ReGate benchmark subset. "
        "Copies data/ and annotations/ into benchmark/; does NOT include code, configs, or scripts."
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root of the source research repository",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory for the benchmark package (default: <source_root>/aaai_code/benchmark)",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else source_root / "aaai_code" / "benchmark"
    )

    print("package_benchmark.py — Data-only packaging of ReGate benchmark subset")
    print(f"  Source root: {source_root}")
    print(f"  Output root: {output_root}")
    print()

    report = package_benchmark(source_root, output_root)

    report_path = output_root / "PACKAGE_REPORT.json"
    write_json(report_path, report)
    print(f"\nReport saved to: {report_path}")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
