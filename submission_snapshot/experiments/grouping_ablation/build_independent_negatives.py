#!/usr/bin/env python3
"""Build the independently sampled-negative grouping ablation.

The positive rows are copied byte-for-byte at the record level from the SFT_FINAL
training set. Every negative row keeps its prompt, target response, state, and queried
category, but receives an independently sampled image from an annotated source that:

1. does not occur in any benchmark train/dev/test/dose split;
2. does not contain the queried category; and
3. for ``wrong_object``, contains the same visible confusable category as the original
   T6 record.

Only files below the requested output directory are written. Source annotations and
images are read-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


AAAI_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AAAI_ROOT.parent
EXPERIMENT_ROOT = Path(__file__).resolve().parent

DEFAULT_BASELINE_SFT = REPO_ROOT / "new_reason/outputs/sft_3reason/sft_train_with_source.json"
DEFAULT_RAW_TRAIN = REPO_ROOT / "new_reason/outputs/splits/train.jsonl"
DEFAULT_SPLIT_DIR = REPO_ROOT / "new_reason/outputs/splits"
_PLANTDOC_ROOT = Path(os.environ.get("PLANTDOC_ROOT", "plantdoc"))
DEFAULT_PLANTDOC_IMAGES = _PLANTDOC_ROOT / "train" / "images"
DEFAULT_PLANTDOC_LABELS = _PLANTDOC_ROOT / "train" / "labels"
DEFAULT_COCO_IMAGES = REPO_ROOT / "datasets/train2017"
DEFAULT_COCO_ANNOTATIONS = REPO_ROOT / "datasets/annotations/instances_train2017.json"
DEFAULT_PLANTDOC_CONFIG = AAAI_ROOT / "configs/plantdoc_full.yaml"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "artifacts/data"

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
STATE_NAMES = {0: "answer", 1: "wrong_object", 2: "no_target"}


@dataclass(frozen=True)
class CandidateImage:
    dataset: str
    source_id: str
    image_path: str
    labels: frozenset[str]


@dataclass(frozen=True)
class NegativeRequest:
    row_index: int
    state: int
    dataset: str
    queried_label: str
    visible_label: str | None
    relation: str
    original_source_id: str
    original_image_path: str


def normalize_label(value: str | None) -> str:
    return " ".join(str(value or "").replace("_", " ").lower().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_state(row: dict[str, Any]) -> int:
    if row["decision"] == "answer":
        return 0
    if row.get("reason") == "wrong_object":
        return 1
    if row.get("reason") == "no_target":
        return 2
    raise ValueError(f"Unsupported decision/reason in {row.get('sample_id')}")


def raw_queried_label(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    source_target = row.get("source_target") or {}
    value = (
        metadata.get("mismatch_label")
        or metadata.get("absent_label")
        or row.get("label")
        or source_target.get("label")
    )
    label = normalize_label(value)
    if not label:
        raise ValueError(f"Cannot derive queried label for {row.get('sample_id')}")
    return label


def raw_visible_label(row: dict[str, Any]) -> str | None:
    value = (row.get("source_target") or {}).get("label")
    label = normalize_label(value)
    return label or None


def sft_query(row: dict[str, Any]) -> str:
    for turn in row.get("conversations", []):
        if turn.get("from") == "human":
            return str(turn.get("value", "")).removeprefix("<image>")
    raise ValueError("SFT row has no human turn")


def sft_image(row: dict[str, Any]) -> str:
    images = row.get("image")
    if isinstance(images, list) and len(images) == 1:
        return str(images[0])
    if isinstance(images, str):
        return images
    raise ValueError("SFT row must contain exactly one image")


def provenance_key(image_path: str, query: str, state: int) -> tuple[str, str, int]:
    return image_path, query, state


def join_sft_to_raw(
    sft_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        index[provenance_key(str(row["image_path"]), str(row["query"]), raw_state(row))].append(row)

    joined: list[dict[str, Any]] = []
    missing: list[int] = []
    for row_index, row in enumerate(sft_rows):
        key = provenance_key(sft_image(row), sft_query(row), int(row["head_label"]))
        matches = index.get(key, [])
        if not matches:
            missing.append(row_index)
            continue
        # Multiple raw rows can share an image/query/state. Their relevant state/category
        # provenance is equivalent for this ablation; use stable sample_id order.
        joined.append(sorted(matches, key=lambda item: str(item["sample_id"]))[0])

    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise ValueError(f"Failed to join {len(missing)} SFT rows to raw train: {preview}")
    if len(joined) != len(sft_rows):
        raise AssertionError("SFT/raw join changed row count")
    return joined


def benchmark_source_ids(split_dir: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "dose_pool.jsonl"):
        path = split_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark split: {path}")
        for row in read_jsonl(path):
            result[str(row["dataset"])].add(str(row["source_image_id"]))
    return result


def _read_yolo_class_ids(path: Path) -> list[int]:
    class_ids: list[int] = []
    if not path.exists():
        return class_ids
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row at {path}:{line_number}")
        class_ids.append(int(parts[0]))
    return class_ids


def load_plantdoc_candidates(
    images_dir: Path,
    labels_dir: Path,
    config_path: Path,
    excluded_source_ids: set[str],
) -> list[CandidateImage]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    class_names = [normalize_label(name) for name in config["class_names"]]
    excluded_class_ids = {int(value) for value in config.get("excluded_class_ids", [])}

    candidates: list[CandidateImage] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        source_id = image_path.stem
        if source_id in excluded_source_ids:
            continue
        class_ids = _read_yolo_class_ids(labels_dir / f"{source_id}.txt")
        valid_ids = {
            class_id
            for class_id in class_ids
            if class_id not in excluded_class_ids and 0 <= class_id < len(class_names)
        }
        labels = frozenset(class_names[class_id] for class_id in valid_ids)
        if not labels:
            continue
        candidates.append(
            CandidateImage(
                dataset="plantdoc_vg",
                source_id=source_id,
                image_path=str(image_path.resolve()),
                labels=labels,
            )
        )
    return candidates


def load_coco_candidates(
    images_dir: Path,
    annotation_path: Path,
    excluded_source_ids: set[str],
) -> list[CandidateImage]:
    with annotation_path.open(encoding="utf-8") as handle:
        coco = json.load(handle)

    categories = {
        int(category["id"]): normalize_label(category["name"]) for category in coco["categories"]
    }
    image_rows = {int(image["id"]): image for image in coco["images"]}
    labels_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0) or annotation.get("bbox") is None:
            continue
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if category_id in categories:
            labels_by_image[image_id].add(categories[category_id])

    candidates: list[CandidateImage] = []
    for image_id, labels in labels_by_image.items():
        source_id = str(image_id)
        if source_id in excluded_source_ids:
            continue
        image_row = image_rows.get(image_id)
        if image_row is None:
            continue
        image_path = images_dir / str(image_row["file_name"])
        if not image_path.exists():
            continue
        candidates.append(
            CandidateImage(
                dataset="coco_ground_abstain",
                source_id=source_id,
                image_path=str(image_path.resolve()),
                labels=frozenset(labels),
            )
        )
    return candidates


class IndependentNegativeSampler:
    """Deterministic, diversity-aware sampler over unused annotated sources."""

    def __init__(self, candidates: Iterable[CandidateImage], seed: int) -> None:
        self._rng = random.Random(seed)
        self._all: dict[str, list[CandidateImage]] = defaultdict(list)
        self._contains: dict[tuple[str, str], list[CandidateImage]] = defaultdict(list)
        self.usage: Counter[tuple[str, str]] = Counter()
        for candidate in candidates:
            self._all[candidate.dataset].append(candidate)
            for label in sorted(candidate.labels):
                self._contains[(candidate.dataset, label)].append(candidate)
        for key in sorted(self._all):
            self._rng.shuffle(self._all[key])
        for key in sorted(self._contains):
            self._rng.shuffle(self._contains[key])

    def choose(self, request: NegativeRequest) -> CandidateImage:
        if request.state == 1:
            if not request.visible_label:
                raise ValueError(f"wrong_object row {request.row_index} lacks visible label")
            base_pool = self._contains.get((request.dataset, request.visible_label), [])
        else:
            base_pool = self._all.get(request.dataset, [])

        eligible = [
            candidate
            for candidate in base_pool
            if request.queried_label not in candidate.labels
            and (
                request.state != 1
                or request.visible_label is None
                or request.visible_label in candidate.labels
            )
        ]
        if not eligible:
            raise ValueError(
                "No independent candidate for "
                f"dataset={request.dataset}, state={STATE_NAMES[request.state]}, "
                f"query={request.queried_label}, visible={request.visible_label}"
            )

        # Inspect a deterministic random window and prefer the globally least-used
        # source. This avoids concentrating thousands of negatives on a few images.
        window = eligible if len(eligible) <= 128 else self._rng.sample(eligible, k=128)
        min_usage = min(
            self.usage[(candidate.dataset, candidate.source_id)] for candidate in window
        )
        least_used = [
            candidate
            for candidate in window
            if self.usage[(candidate.dataset, candidate.source_id)] == min_usage
        ]
        selected = self._rng.choice(least_used)
        self.usage[(selected.dataset, selected.source_id)] += 1
        return selected


def make_request(
    row_index: int, sft_row: dict[str, Any], raw_row: dict[str, Any]
) -> NegativeRequest:
    state = int(sft_row["head_label"])
    if state not in (1, 2):
        raise ValueError("Independent-negative requests must be negative states")
    return NegativeRequest(
        row_index=row_index,
        state=state,
        dataset=str(raw_row["dataset"]),
        queried_label=raw_queried_label(raw_row),
        visible_label=raw_visible_label(raw_row),
        relation=str((raw_row.get("metadata") or {}).get("relation", "")),
        original_source_id=str(raw_row["source_image_id"]),
        original_image_path=str(raw_row["image_path"]),
    )


def replace_negative(
    row: dict[str, Any], request: NegativeRequest, candidate: CandidateImage
) -> dict[str, Any]:
    output = copy.deepcopy(row)
    output["image"] = [candidate.image_path]
    output["source_image_id"] = f"independent:{candidate.dataset}:{candidate.source_id}"
    output["grouping_ablation"] = {
        "condition": "independent_negative",
        "state": STATE_NAMES[request.state],
        "queried_label": request.queried_label,
        "required_visible_label": request.visible_label if request.state == 1 else None,
        "original_relation": request.relation,
        "original_source_image_id": request.original_source_id,
        "original_image_path": request.original_image_path,
        "sampled_dataset": candidate.dataset,
        "sampled_source_image_id": candidate.source_id,
        "sampled_labels": sorted(candidate.labels),
    }
    return output


def category_state_distribution(
    sft_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> Counter[tuple[int, str]]:
    return Counter(
        (int(sft_row["head_label"]), raw_queried_label(raw_row))
        for sft_row, raw_row in zip(sft_rows, raw_rows, strict=True)
    )


def build_ablation(
    sft_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    sampler: IndependentNegativeSampler,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for row_index, (sft_row, raw_row) in enumerate(zip(sft_rows, raw_rows, strict=True)):
        state = int(sft_row["head_label"])
        if state == 0:
            output_rows.append(copy.deepcopy(sft_row))
            continue
        request = make_request(row_index, sft_row, raw_row)
        candidate = sampler.choose(request)
        output_rows.append(replace_negative(sft_row, request, candidate))
        assignments.append(
            {
                "row_index": row_index,
                "state": STATE_NAMES[state],
                "queried_label": request.queried_label,
                "required_visible_label": request.visible_label if state == 1 else None,
                "original_relation": request.relation,
                "original_source_image_id": request.original_source_id,
                "sampled_dataset": candidate.dataset,
                "sampled_source_image_id": candidate.source_id,
                "sampled_image_path": candidate.image_path,
                "sampled_labels": sorted(candidate.labels),
            }
        )
    return output_rows, assignments


def audit_output(
    baseline_rows: list[dict[str, Any]],
    output_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    excluded_sources: dict[str, set[str]],
) -> dict[str, Any]:
    baseline_states = Counter(int(row["head_label"]) for row in baseline_rows)
    output_states = Counter(int(row["head_label"]) for row in output_rows)
    baseline_categories = category_state_distribution(baseline_rows, raw_rows)
    output_categories = category_state_distribution(output_rows, raw_rows)

    positive_sources = {
        str(row.get("source_image_id")) for row in output_rows if int(row["head_label"]) == 0
    }
    independent_sources = {
        (str(row["sampled_dataset"]), str(row["sampled_source_image_id"])) for row in assignments
    }
    independent_prefixed_sources = {
        f"independent:{row['sampled_dataset']}:{row['sampled_source_image_id']}"
        for row in assignments
    }
    output_negative_sources = {
        str(row.get("source_image_id")) for row in output_rows if int(row["head_label"]) in (1, 2)
    }
    missing_images = [sft_image(row) for row in output_rows if not Path(sft_image(row)).is_file()]

    invalid_assignments = []
    benchmark_collisions = []
    for assignment in assignments:
        labels = set(assignment["sampled_labels"])
        if assignment["queried_label"] in labels:
            invalid_assignments.append(
                {"row_index": assignment["row_index"], "error": "queried_label_present"}
            )
        required = assignment["required_visible_label"]
        if assignment["state"] == "wrong_object" and required not in labels:
            invalid_assignments.append(
                {"row_index": assignment["row_index"], "error": "visible_label_missing"}
            )
        dataset = str(assignment["sampled_dataset"])
        source_id = str(assignment["sampled_source_image_id"])
        if source_id in excluded_sources.get(dataset, set()):
            benchmark_collisions.append({"dataset": dataset, "source_image_id": source_id})

    positive_rows_unchanged = all(
        baseline == output
        for baseline, output in zip(baseline_rows, output_rows, strict=True)
        if int(baseline["head_label"]) == 0
    )
    negative_prompt_target_unchanged = all(
        baseline["conversations"] == output["conversations"]
        and baseline["head_label"] == output["head_label"]
        and baseline["decision_label"] == output["decision_label"]
        and baseline["reason_label"] == output["reason_label"]
        for baseline, output in zip(baseline_rows, output_rows, strict=True)
        if int(baseline["head_label"]) in (1, 2)
    )

    checks = {
        "row_count_equal": len(baseline_rows) == len(output_rows),
        "state_distribution_equal": baseline_states == output_states,
        "category_state_distribution_equal": baseline_categories == output_categories,
        "positive_rows_unchanged": positive_rows_unchanged,
        "negative_prompt_target_unchanged": negative_prompt_target_unchanged,
        "all_negative_sources_replaced": output_negative_sources == independent_prefixed_sources,
        "positive_negative_source_overlap_empty": not (
            positive_sources & independent_prefixed_sources
        ),
        "benchmark_source_collision_empty": not benchmark_collisions,
        "semantic_assignment_errors_empty": not invalid_assignments,
        "all_media_paths_exist": not missing_images,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Grouping-ablation audit failed: {failed}")

    usage = Counter(
        (str(row["sampled_dataset"]), str(row["sampled_source_image_id"])) for row in assignments
    )
    return {
        "schema_version": 1,
        "condition": "independent_negative",
        "checks": checks,
        "counts": {
            "rows": len(output_rows),
            "states": {STATE_NAMES[key]: value for key, value in sorted(output_states.items())},
            "positive_rows": output_states[0],
            "replaced_negative_rows": len(assignments),
            "independent_negative_sources": len(independent_sources),
            "independent_negative_sources_by_dataset": dict(
                sorted(Counter(dataset for dataset, _ in independent_sources).items())
            ),
            "maximum_rows_per_independent_source": max(usage.values(), default=0),
        },
        "category_state_strata": len(output_categories),
        "benchmark_collisions": benchmark_collisions,
        "invalid_assignments": invalid_assignments,
        "missing_images": missing_images,
        "source_usage_histogram": dict(sorted(Counter(usage.values()).items())),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sft", type=Path, default=DEFAULT_BASELINE_SFT)
    parser.add_argument("--raw-train", type=Path, default=DEFAULT_RAW_TRAIN)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--plantdoc-images", type=Path, default=DEFAULT_PLANTDOC_IMAGES)
    parser.add_argument("--plantdoc-labels", type=Path, default=DEFAULT_PLANTDOC_LABELS)
    parser.add_argument("--plantdoc-config", type=Path, default=DEFAULT_PLANTDOC_CONFIG)
    parser.add_argument("--coco-images", type=Path, default=DEFAULT_COCO_IMAGES)
    parser.add_argument("--coco-annotations", type=Path, default=DEFAULT_COCO_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] baseline={args.baseline_sft}", flush=True)
    baseline_rows = json.loads(args.baseline_sft.read_text(encoding="utf-8"))
    raw_train_rows = read_jsonl(args.raw_train)
    raw_joined = join_sft_to_raw(baseline_rows, raw_train_rows)
    excluded_sources = benchmark_source_ids(args.split_dir)

    print("[build] indexing unused PlantDoc sources", flush=True)
    plantdoc_candidates = load_plantdoc_candidates(
        args.plantdoc_images,
        args.plantdoc_labels,
        args.plantdoc_config,
        excluded_sources["plantdoc_vg"],
    )
    print(f"[build] PlantDoc candidates={len(plantdoc_candidates)}", flush=True)

    print("[build] indexing unused COCO sources", flush=True)
    coco_candidates = load_coco_candidates(
        args.coco_images,
        args.coco_annotations,
        excluded_sources["coco_ground_abstain"],
    )
    print(f"[build] COCO candidates={len(coco_candidates)}", flush=True)

    sampler = IndependentNegativeSampler(
        [*plantdoc_candidates, *coco_candidates],
        seed=args.seed,
    )
    output_rows, assignments = build_ablation(baseline_rows, raw_joined, sampler)
    audit = audit_output(
        baseline_rows,
        output_rows,
        raw_joined,
        assignments,
        excluded_sources,
    )

    output_path = output_dir / "sft_train_independent_negatives.json"
    assignment_path = output_dir / "negative_assignments.jsonl"
    audit_path = output_dir / "audit.json"
    write_json(output_path, output_rows)
    write_jsonl(assignment_path, assignments)
    audit.update(
        {
            "seed": args.seed,
            "inputs": {
                "baseline_sft": str(args.baseline_sft.resolve()),
                "baseline_sft_sha256": sha256_file(args.baseline_sft),
                "raw_train": str(args.raw_train.resolve()),
                "raw_train_sha256": sha256_file(args.raw_train),
                "split_dir": str(args.split_dir.resolve()),
                "plantdoc_config": str(args.plantdoc_config.resolve()),
                "coco_annotations": str(args.coco_annotations.resolve()),
            },
            "outputs": {
                "sft_train": str(output_path),
                "sft_train_sha256": sha256_file(output_path),
                "assignments": str(assignment_path),
                "assignments_sha256": sha256_file(assignment_path),
            },
            "candidate_inventory": {
                "plantdoc": len(plantdoc_candidates),
                "coco": len(coco_candidates),
            },
        }
    )
    write_json(audit_path, audit)
    print(json.dumps(audit["counts"], ensure_ascii=False, indent=2), flush=True)
    print(f"[build] wrote {output_path}", flush=True)
    print(f"[build] audit {audit_path}", flush=True)


if __name__ == "__main__":
    main()
