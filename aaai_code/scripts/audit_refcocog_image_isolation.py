#!/usr/bin/env python3
"""MUTATION TOOL: rewrites `ood_refcocog.jsonl` in place to backfill metadata fields.

Audits image isolation for the RefCOCOg probe and backfills four metadata fields. **This is
not a read-only check.** For read-only verification use the commands in REVIEWER_README.md,
section 4.

Steps:
  1. Read the parquet and precompute `coco_image_id`, `refcocog_row_index`, `coco_file_name`
     and `question_id` for every RefCOCOg row.
  2. Read `ood_refcocog.jsonl`, map each `sample_id` of the form `refg-{row}-ans` or
     `refg-{row}-absent` back to its parquet row and backfill those four fields, leaving the
     record count, the ordering and all other fields unchanged. The write is atomic.
  3. Read `train.jsonl`, restricted to `dataset=coco_ground_abstain`, and collect the
     `source_image_id` values as integers.
  4. Audit: the intersection between the COCO ids used by the probe and those used in
     training must be empty, otherwise the audit fails.
  5. Audit: the answer and absent records of one RefCOCOg row must carry the same COCO id.
  6. Write the audit JSON.

Usage:
  # from the package root
  python scripts/audit_refcocog_image_isolation.py

  # help only, reads and writes nothing
  python scripts/audit_refcocog_image_isolation.py --help
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[1]  # aaai_code/

if (BASE_DIR / "benchmark").is_dir():
    # package mode: paths under benchmark/
    PARQUET_PATH = BASE_DIR / "benchmark/data/ood/refcocog/metadata/val-00000-of-00003.parquet"
    TRAIN_PATH = BASE_DIR / "benchmark/annotations/splits/train.jsonl"
    PROBE_PATH = BASE_DIR / "benchmark/annotations/eval/ood_refcocog.jsonl"
    AUDIT_PATH = BASE_DIR / "benchmark/annotations/eval/REFCOCOG_IMAGE_ID_ISOLATION_AUDIT.json"
else:
    # source mode: fallback to old research repo layout
    PARQUET_PATH = BASE_DIR / "data/ood/refcocog/data/val-00000-of-00003.parquet"
    TRAIN_PATH = BASE_DIR / "outputs/splits/train.jsonl"
    PROBE_PATH = BASE_DIR / "outputs/eval_probes/ood_refcocog.jsonl"
    AUDIT_PATH = BASE_DIR / "outputs/eval_probes/REFCOCOG_IMAGE_ID_ISOLATION_AUDIT.json"

SAMPLE_ID_PATTERN = re.compile(r"^refg-(\d+)-(ans|absent)$")
METADATA_FIELDS = (
    "coco_image_id",
    "refcocog_row_index",
    "coco_file_name",
    "question_id",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_jsonl_records(records: list[dict]) -> str:
    """SHA256 over the JSON lines, in order."""
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def remove_added_metadata(records: list[dict]) -> list[dict]:
    """Deep-copy a record and drop the metadata fields this script adds."""
    cleaned = deepcopy(records)
    for record in cleaned:
        metadata = record.get("metadata", {})
        for field in METADATA_FIELDS:
            metadata.pop(field, None)
    return cleaned


def sha256_serialized_jsonl(records: list[dict]) -> str:
    """SHA256 of the content, serialised in the JSONL format the probe builder emits."""
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sample_id_order_hash(records: list[dict]) -> str:
    """SHA256 over the `sample_id` values, concatenated in order."""
    parts = [r["metadata"]["sample_id"] for r in records]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_atomic(records: list[dict], path: Path) -> None:
    """Write JSONL atomically."""
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp_path, str(path))
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def build_parquet_index(parquet_path: Path) -> dict[int, dict[str, Any]]:
    """Read the parquet and build, for each row:
      row_index -> {coco_image_id, refcocog_row_index, coco_file_name, question_id}
    """
    tbl = pq.read_table(parquet_path)
    img_col = tbl.column("image")
    qid_col = tbl.column("question_id")
    n_rows = tbl.num_rows

    index: dict[int, dict[str, Any]] = {}
    parse_failures = 0

    for idx in range(n_rows):
        img_path_str: str = img_col[idx].as_py()["path"]
        coco_fn = Path(img_path_str).name
        coco_match = re.match(r"COCO_(train|val)(2014|2017)_(\d+)\.jpg", coco_fn)
        if not coco_match:
            print(f"[WARN] 第 {idx} 行 image.path 格式异常: {coco_fn!r}，跳过")
            parse_failures += 1
            continue

        coco_image_id = int(coco_match.group(3))
        question_id_val: int = int(qid_col[idx].as_py())
        index[idx] = {
            "coco_image_id": coco_image_id,
            "refcocog_row_index": idx,
            "coco_file_name": coco_fn,
            "question_id": question_id_val,
        }

    print(f"  Parquet 索引: {len(index)}/{n_rows} 行成功, 解析失败={parse_failures}")
    return index


def load_train_coco_ids(train_path: Path) -> set[int]:
    """Read `train.jsonl`, keep `dataset=coco_ground_abstain`, and return `source_image_id` as integers.

    """
    records = load_jsonl(train_path)
    coco_ids: set[int] = set()
    for r in records:
        if r.get("dataset") == "coco_ground_abstain":
            sid = r.get("source_image_id")
            if sid is not None:
                coco_ids.add(int(sid))
    n_coco = sum(1 for r in records if r.get("dataset") == "coco_ground_abstain")
    print(f"  训练 coco_ground_abstain 记录: {n_coco}")
    print(f"  训练 unique COCO IDs: {len(coco_ids)}")
    return coco_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MUTATION: audit RefCOCOg image isolation and backfill metadata. Atomically rewrites ood_refcocog.jsonl; do not use for read-only verification."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=PARQUET_PATH,
        help="Path to the RefCOCOg parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=TRAIN_PATH,
        help="Path to train.jsonl, used to extract COCO IDs (default: %(default)s)",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=PROBE_PATH,
        help="Path to ood_refcocog.jsonl (will be atomically rewritten) (default: %(default)s)",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=AUDIT_PATH,
        help="Output path for the audit JSON (default: %(default)s)",
    )
    args = parser.parse_args()

    parquet_path = args.parquet.resolve()
    train_path = args.train.resolve()
    probe_path = args.probe.resolve()
    audit_path = args.audit_output.resolve()

    print("=" * 60)
    print("audit_refcocog_image_isolation: 审计 & 回填 metadata")
    print("=" * 60)

    if not parquet_path.exists():
        print(f"[ERROR] Parquet 不存在: {parquet_path}")
        sys.exit(1)
    print(f"\n[1] Parquet: {parquet_path}")

    parquet_index = build_parquet_index(parquet_path)

    print(f"\n[2] 训练数据: {train_path}")
    train_coco_ids = load_train_coco_ids(train_path)

    print(f"\n[3] Probe: {probe_path}")
    probe_records = load_jsonl(probe_path)
    probe_before_hash = sha256_file(probe_path)
    probe_before_records_hash = sha256_jsonl_records(probe_records)
    normalized_before_hash = sha256_jsonl_records(remove_added_metadata(probe_records))
    reconstructed_pre_backfill_hash = sha256_serialized_jsonl(remove_added_metadata(probe_records))
    missing_fields_before = sum(
        1
        for record in probe_records
        if any(field not in record.get("metadata", {}) for field in METADATA_FIELDS)
    )
    print(f"  Probe 记录数: {len(probe_records)}")
    print(f"  Probe SHA256 (before): {probe_before_hash}")

    print("\n[4] 回填 metadata...")
    n_parse_fail = 0
    probe_coco_ids: set[int] = set()
    row_to_coco_id: dict[int, int] = {}
    row_to_coco_id_consistent = True
    row_to_coco_id_conflicts: list[dict] = []

    filled_records: list[dict] = []
    for r in probe_records:
        sid = r["metadata"]["sample_id"]
        m = SAMPLE_ID_PATTERN.match(sid)
        if not m:
            print(f"[WARN] sample_id 格式异常: {sid!r}，跳过")
            n_parse_fail += 1
            filled_records.append(r)
            continue

        row_idx = int(m.group(1))
        # m.group(2) = 'ans' or 'absent' — used for grouping validation
        row_info = parquet_index.get(row_idx)
        if row_info is None:
            print(f"[WARN] parquet 索引中无 row {row_idx} (sample_id={sid!r})，跳过")
            n_parse_fail += 1
            filled_records.append(r)
            continue

        r["metadata"]["coco_image_id"] = row_info["coco_image_id"]
        r["metadata"]["refcocog_row_index"] = row_info["refcocog_row_index"]
        r["metadata"]["coco_file_name"] = row_info["coco_file_name"]
        r["metadata"]["question_id"] = row_info["question_id"]

        probe_coco_ids.add(row_info["coco_image_id"])

        if row_idx in row_to_coco_id:
            expected = row_to_coco_id[row_idx]
            if row_info["coco_image_id"] != expected:
                row_to_coco_id_consistent = False
                row_to_coco_id_conflicts.append(
                    {
                        "row": row_idx,
                        "sample_id": sid,
                        "expected_coco_id": expected,
                        "actual_coco_id": row_info["coco_image_id"],
                    }
                )
        else:
            row_to_coco_id[row_idx] = row_info["coco_image_id"]

        filled_records.append(r)

    assert len(filled_records) == len(probe_records), "行数变化！"
    normalized_after_hash = sha256_jsonl_records(remove_added_metadata(filled_records))
    if normalized_after_hash != normalized_before_hash:
        raise RuntimeError("除新增 RefCOCOg metadata 字段外，probe 内容发生变化")

    print(f"  解析失败: {n_parse_fail}")
    print(f"  Probe unique COCO IDs: {len(probe_coco_ids)}")
    print(f"  同 row answer/absent ID 一致: {row_to_coco_id_consistent}")

    print("\n[5] 审计交集...")
    intersection = train_coco_ids & probe_coco_ids
    intersection_count = len(intersection)
    print(f"  Train COCO IDs: {len(train_coco_ids)}")
    print(f"  Probe COCO IDs: {len(probe_coco_ids)}")
    print(f"  交集: {intersection_count}")

    if intersection_count > 0:
        print(f"  [ERROR] 交集不为 0! 交集列表: {sorted(intersection)[:20]}")
        sys.exit(1)
    else:
        print("  [PASS] 交集为 0。")

    probe_after_hash = sha256_jsonl_records(filled_records)
    write_atomic(filled_records, probe_path)
    probe_after_file_hash = sha256_file(probe_path)
    print(f"\n[6] 写入: {probe_path}")
    print(f"  After SHA256 (file): {probe_after_file_hash}")

    sid_hash = sample_id_order_hash(filled_records)
    print(f"  sample_id 顺序 hash: {sid_hash}")

    audit: dict[str, Any] = {
        "task": "audit_refcocog_image_isolation",
        "input_files": {
            "parquet": str(parquet_path),
            "parquet_sha256": sha256_file(parquet_path),
            "train_jsonl": str(train_path),
            "train_jsonl_sha256": sha256_file(train_path),
            "probe_jsonl": str(probe_path),
        },
        "train": {
            "coco_ground_abstain_records": sum(
                1 for r in load_jsonl(train_path) if r.get("dataset") == "coco_ground_abstain"
            ),
            "unique_coco_image_ids": len(train_coco_ids),
        },
        "probe": {
            "total_records": len(probe_records),
            "unique_refcocog_rows": len(
                set(
                    int(m.group(1))
                    for r in probe_records
                    if (m := SAMPLE_ID_PATTERN.match(r["metadata"]["sample_id"]))
                )
            ),
            "unique_coco_image_ids": len(probe_coco_ids),
            "parse_failures": n_parse_fail,
            "sample_id_order_hash": sid_hash,
        },
        "intersection": {
            "count": intersection_count,
            "list": sorted(intersection) if intersection_count > 0 else [],
        },
        "row_consistency": {
            "consistent": row_to_coco_id_consistent,
            "conflicts": row_to_coco_id_conflicts,
        },
        "file_hashes": {
            "probe_before_file": probe_before_hash,
            "probe_after_file": probe_after_file_hash,
            "probe_before_records": probe_before_records_hash,
            "probe_after_records": probe_after_hash,
            "normalized_without_added_metadata": normalized_after_hash,
            "reconstructed_pre_backfill_file": reconstructed_pre_backfill_hash,
        },
        "metadata_fields_added": list(METADATA_FIELDS),
        "backfill_history": {
            "records_missing_any_field_before_this_run": missing_fields_before,
            "note": (
                "后续幂等运行可能显示缺失数为0；reconstructed_pre_backfill_file "
                "通过移除四个新增字段按原JSONL格式重建。"
            ),
        },
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[AUDIT] 审计 JSON 已写入: {audit_path}")

    print("\n[DONE] audit_refcocog_image_isolation 完成。")


if __name__ == "__main__":
    main()
