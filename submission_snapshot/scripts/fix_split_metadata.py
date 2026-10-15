#!/usr/bin/env python3
"""MUTATION TOOL: rewrites the top-level `split` field of `dev.jsonl` and `test.jsonl` in place.

The field was written as 'train' in every record; this script corrects it. **This is not a
read-only check.** For read-only verification use the commands in REVIEWER_README.md,
section 4.

Behaviour:
  - dev.jsonl  -> split="dev"
  - test.jsonl -> split="test"
  - only the top-level `split` field changes; the line order, the sample ids and every other
    field keep their value and meaning
  - the SHA256 of each record, normalised with `split` removed, is identical before and after
  - the write is atomic, through a temporary file and a rename
  - afterwards it verifies the record count, the `split` values, the order hash of the sample
    ids, and the pairwise intersections of train, development and test keyed by
    (dataset, source_image_id)

Usage:
  # from the package root
  python scripts/fix_split_metadata.py

  # explicit splits directory
  python scripts/fix_split_metadata.py --splits-dir benchmark/annotations/splits

  # help only, reads and writes nothing
  python scripts/fix_split_metadata.py --help
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]  # aaai_code/
if (BASE_DIR / "benchmark").is_dir():
    SPLITS_DIR = BASE_DIR / "benchmark" / "annotations" / "splits"
else:
    SPLITS_DIR = BASE_DIR / "outputs" / "splits"

EXPECTED_LINES = {
    "dev.jsonl": 2408,
    "test.jsonl": 2646,
}

SPLIT_MAP = {
    "dev.jsonl": "dev",
    "test.jsonl": "test",
}


def sha256_file(path: Path) -> str:
    """SHA256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_normalized(records: list[dict]) -> str:
    """SHA256 of the record serialised without its `split` field."""
    lines: list[str] = []
    for r in records:
        r_copy = {k: v for k, v in r.items() if k != "split"}
        lines.append(json.dumps(r_copy, sort_keys=True, ensure_ascii=False))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def sha256_serialized_jsonl(records: list[dict]) -> str:
    """SHA256 of the content, serialised in the JSONL format the generator emits."""
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_records(path: Path) -> list[dict]:
    """Load a JSONL file."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fix_split(records: list[dict], target_split: str) -> list[dict]:
    """Change the top-level `split` field and nothing else."""
    fixed: list[dict] = []
    for r in records:
        r["split"] = target_split
        fixed.append(r)
    return fixed


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


def component_hash(records: list[dict]) -> str:
    """SHA256 over the `source_image_id` values, concatenated in order."""
    parts = [r["sample_id"] for r in records]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def compute_intersections(all_records: dict[str, list[dict]]) -> dict:
    """Pairwise intersections of train, development and test, keyed by (dataset, source_image_id).

    """
    keysets: dict[str, set] = {}
    for split_name, recs in all_records.items():
        ks: set[tuple[str, str]] = set()
        for r in recs:
            ds = r.get("dataset", "")
            sid = r.get("source_image_id", "")
            ks.add((ds, sid))
        keysets[split_name] = ks

    result: dict[str, int | list] = {}
    pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for a, b in pairs:
        inter = keysets.get(a, set()) & keysets.get(b, set())
        result[f"{a}_vs_{b}_intersection_count"] = len(inter)
        result[f"{a}_vs_{b}_intersection_list"] = sorted(inter)[:20]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MUTATION: fix the top-level split field of dev.jsonl / test.jsonl. Atomically rewrites the files; do not use for read-only verification."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory containing train/dev/test.jsonl (default: auto-detected %(default)s)",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="Output path for the audit JSON (default: <splits-dir>/SPLIT_METADATA_FIX_AUDIT.json)",
    )
    args = parser.parse_args()

    splits_dir = args.splits_dir.resolve()
    audit_path = (args.audit_output or splits_dir / "SPLIT_METADATA_FIX_AUDIT.json").resolve()

    print("=" * 60)
    print("fix_split_metadata: 修复 dev/test split 字段")
    print("=" * 60)

    train_path = splits_dir / "train.jsonl"
    train_records = load_records(train_path)
    print(f"\n[train] {train_path}: {len(train_records)} 行")

    before_sha: dict[str, str] = {}
    before_normalized_sha: dict[str, str] = {}
    before_wrong_split: dict[str, int] = {}
    reconstructed_buggy_sha: dict[str, str] = {}
    records_by_file: dict[str, list[dict]] = {}

    for fname in ["dev.jsonl", "test.jsonl"]:
        path = splits_dir / fname
        if not path.exists():
            print(f"[ERROR] {path} 不存在，跳过")
            sys.exit(1)

        records = load_records(path)
        expected = EXPECTED_LINES.get(fname)
        if expected and len(records) != expected:
            print(f"[ERROR] {fname}: 行数 {len(records)} != 期望 {expected}")
            sys.exit(1)

        before_sha[fname] = sha256_file(path)
        before_normalized_sha[fname] = sha256_normalized(records)
        wrong_count = sum(1 for r in records if r.get("split") != SPLIT_MAP[fname])
        before_wrong_split[fname] = wrong_count
        buggy_records = [{**r, "split": "train"} for r in records]
        reconstructed_buggy_sha[fname] = sha256_serialized_jsonl(buggy_records)
        records_by_file[fname] = records

        print(f"\n[{fname}] {len(records)} 行, 错误 split 数={wrong_count}")

        target_split = SPLIT_MAP[fname]
        fixed = fix_split(records, target_split)

        after_normalized_sha = sha256_normalized(fixed)
        if after_normalized_sha != before_normalized_sha[fname]:
            print(
                f"[ERROR] {fname}: 规范化 SHA256 不一致！"
                f" 修复前={before_normalized_sha[fname]}, 修复后={after_normalized_sha}"
            )
            print("       这意味着除 split 外字段发生了变化，退出。")
            sys.exit(1)

        write_atomic(fixed, path)
        after_sha = sha256_file(path)

        re_records = load_records(path)
        assert len(re_records) == len(records), f"{fname}: 行数变化！"

        wrong_after = sum(1 for r in re_records if r.get("split") != SPLIT_MAP[fname])
        if wrong_after > 0:
            print(f"[ERROR] {fname}: 修复后仍有 {wrong_after} 条 split 错误！")
            sys.exit(1)

        sid_hash = component_hash(re_records)

        print(f"  -> 修复完成: split={target_split}, sample_id_hash={sid_hash}")
        print(f"  -> before SHA256: {before_sha[fname]}")
        print(f"  -> after SHA256:  {after_sha}")
        print(f"  -> normalized SHA256 (无split) 一致: {after_normalized_sha}")

        records_by_file[fname] = re_records

    all_records = {
        "train": train_records,
        "dev": records_by_file["dev.jsonl"],
        "test": records_by_file["test.jsonl"],
    }
    intersections = compute_intersections(all_records)

    train_dev_inter = intersections.get("train_vs_dev_intersection_count", -1)
    train_test_inter = intersections.get("train_vs_test_intersection_count", -1)
    dev_test_inter = intersections.get("dev_vs_test_intersection_count", -1)

    print("\n" + "=" * 60)
    print("交集验证 (dataset, source_image_id)")
    print("=" * 60)
    print(f"  Train ∩ Dev:  {train_dev_inter}")
    print(f"  Train ∩ Test: {train_test_inter}")
    print(f"  Dev ∩ Test:   {dev_test_inter}")

    if train_dev_inter > 0 or train_test_inter > 0 or dev_test_inter > 0:
        print("[ERROR] 存在交集！泄漏检测失败。")
        sys.exit(1)
    else:
        print("[PASS] 所有交集为 0。")

    audit: dict[str, Any] = {
        "task": "fix_split_metadata",
        "files_processed": ["dev.jsonl", "test.jsonl"],
        "file_sha256_before": before_sha,
        "file_sha256_after": {
            fname: sha256_file(splits_dir / fname) for fname in ["dev.jsonl", "test.jsonl"]
        },
        "normalized_without_split_sha256": before_normalized_sha,
        "line_counts": {
            fname: len(records_by_file[fname]) for fname in ["dev.jsonl", "test.jsonl"]
        },
        "wrong_split_before": before_wrong_split,
        "wrong_split_after": {
            "dev.jsonl": 0,
            "test.jsonl": 0,
        },
        "split_values_after": {fname: SPLIT_MAP[fname] for fname in ["dev.jsonl", "test.jsonl"]},
        "sample_id_order_hash": {
            fname: component_hash(records_by_file[fname]) for fname in ["dev.jsonl", "test.jsonl"]
        },
        "source_image_intersections": intersections,
        "historical_bug_reconstruction": {
            "note": (
                "首次修复前 dev/test 的顶层 split 均为 train。后续幂等运行会看到 "
                "wrong_split_before=0；以下 SHA256 由当前记录仅将 split 还原为 train 后，"
                "按原生成格式重建。"
            ),
            "wrong_split_count": {
                "dev.jsonl": EXPECTED_LINES["dev.jsonl"],
                "test.jsonl": EXPECTED_LINES["test.jsonl"],
            },
            "reconstructed_pre_fix_file_sha256": reconstructed_buggy_sha,
        },
    }

    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[AUDIT] 审计 JSON 已写入: {audit_path}")

    print("\n[DONE] fix_split_metadata 完成。")


if __name__ == "__main__":
    main()
