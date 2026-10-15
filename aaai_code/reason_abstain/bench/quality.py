from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VALID_REASONS_BY_DATASET = {
    "coco_ground_abstain": {"target_absent", "visual_evidence_insufficient"},
    "plantdoc_vg": {
        "target_absent",
        "visual_evidence_insufficient",
        "query_image_mismatch",
    },
}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_samples": len(records),
        "num_groups": len({record["group_id"] for record in records}),
        "by_dataset": dict(Counter(record["dataset"] for record in records)),
        "by_split": dict(Counter(record["split"] for record in records)),
        "by_sample_type": dict(Counter(record["sample_type"] for record in records)),
        "by_decision": dict(Counter(record["decision"] for record in records)),
        "by_reason": dict(
            Counter(record.get("reason") for record in records if record.get("reason"))
        ),
        "by_transform": dict(Counter(record.get("transform", "none") for record in records)),
        "by_relation": dict(
            Counter(record.get("metadata", {}).get("relation", "unknown") for record in records)
        ),
    }


def review_records(records: list[dict[str, Any]], check_files: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_split: dict[str, str] = {}
    source_answer_labels: dict[str, set[str]] = defaultdict(set)

    for idx, record in enumerate(records):
        sample_id = str(record.get("sample_id", f"row-{idx}"))
        group_id = str(record.get("group_id", ""))
        groups[group_id].append(record)

        if sample_id in seen_sample_ids:
            issues.append(
                {"severity": "error", "sample_id": sample_id, "issue": "duplicate_sample_id"}
            )
        seen_sample_ids.add(sample_id)

        source_key = f"{record.get('dataset')}::{record.get('source_image_id')}"
        split = str(record.get("split"))
        if source_key in source_split and source_split[source_key] != split:
            issues.append(
                {
                    "severity": "error",
                    "sample_id": sample_id,
                    "issue": "source_image_split_leakage",
                    "source_key": source_key,
                }
            )
        source_split[source_key] = split
        if record.get("decision") == "answer" and isinstance(record.get("label"), str):
            source_answer_labels[source_key].add(str(record["label"]))

        issues.extend(_review_record(record, check_files=check_files))

    for group_id, group_records in groups.items():
        issues.extend(_review_group(group_id, group_records))

    for idx, record in enumerate(records):
        sample_id = str(record.get("sample_id", f"row-{idx}"))
        issues.extend(_review_cross_record(record, sample_id, source_answer_labels))

    severity_counts = Counter(issue["severity"] for issue in issues)
    return {
        "summary": summarize_records(records),
        "severity_counts": dict(severity_counts),
        "passed": severity_counts.get("error", 0) == 0,
        "issues": issues,
    }


def _review_record(record: dict[str, Any], check_files: bool) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    sample_id = record.get("sample_id")
    dataset = record.get("dataset")
    decision = record.get("decision")

    if check_files and not Path(str(record.get("image_path", ""))).exists():
        issues.append({"severity": "error", "sample_id": sample_id, "issue": "missing_image_file"})

    if decision == "answer":
        if not isinstance(record.get("label"), str):
            issues.append(
                {"severity": "error", "sample_id": sample_id, "issue": "answer_missing_label"}
            )
        if not _valid_box(record.get("box")):
            issues.append(
                {"severity": "error", "sample_id": sample_id, "issue": "invalid_answer_box"}
            )
        if record.get("reason") is not None:
            issues.append(
                {"severity": "error", "sample_id": sample_id, "issue": "answer_has_reason"}
            )
    elif decision == "abstain":
        reason = record.get("reason")
        valid_reasons = VALID_REASONS_BY_DATASET.get(str(dataset), set())
        if reason not in valid_reasons:
            issues.append(
                {
                    "severity": "error",
                    "sample_id": sample_id,
                    "issue": "invalid_or_disallowed_reason",
                    "reason": reason,
                    "dataset": dataset,
                }
            )
        if record.get("box") is not None:
            issues.append({"severity": "error", "sample_id": sample_id, "issue": "abstain_has_box"})
    else:
        issues.append({"severity": "error", "sample_id": sample_id, "issue": "invalid_decision"})

    return issues


def _review_cross_record(
    record: dict[str, Any],
    sample_id: str,
    source_answer_labels: dict[str, set[str]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return issues
    if metadata.get("relation") != "T6":
        return issues

    mismatch_label = metadata.get("mismatch_label")
    source_key = f"{record.get('dataset')}::{record.get('source_image_id')}"
    if isinstance(mismatch_label, str) and mismatch_label in source_answer_labels[source_key]:
        issues.append(
            {
                "severity": "error",
                "sample_id": sample_id,
                "issue": "query_image_mismatch_label_present_in_source_image",
                "mismatch_label": mismatch_label,
                "source_key": source_key,
            }
        )
    return issues


def _review_group(group_id: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    relations = {record.get("metadata", {}).get("relation") for record in records}
    required_core = {"base", "T1", "T2", "T3", "T4", "T5"}
    missing_core = sorted(required_core - relations)
    for relation in missing_core:
        issues.append(
            {
                "severity": "warning",
                "group_id": group_id,
                "issue": "missing_core_relation",
                "relation": relation,
            }
        )
    if not any(record["decision"] == "answer" for record in records):
        issues.append({"severity": "error", "group_id": group_id, "issue": "group_has_no_answer"})
    if not any(record["decision"] == "abstain" for record in records):
        issues.append(
            {"severity": "warning", "group_id": group_id, "issue": "group_has_no_abstain"}
        )
    if "T4" in relations and "T4_control" not in relations:
        issues.append({"severity": "warning", "group_id": group_id, "issue": "missing_t4_control"})
    # T7 region monotonicity: an include (answer) should be paired with at least one occlusion
    # signal — either the new severity ladder (T7_ladder) or the legacy binary exclude.
    if "T7_include" in relations and "T7_ladder" not in relations and "T7_exclude" not in relations:
        issues.append(
            {"severity": "warning", "group_id": group_id, "issue": "missing_t7_occlusion"}
        )
    return issues


def _valid_box(box: object) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(value, int | float) for value in box):
        return False
    x1, y1, x2, y2 = [float(value) for value in box]
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
