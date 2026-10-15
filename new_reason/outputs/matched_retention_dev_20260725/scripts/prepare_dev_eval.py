#!/usr/bin/env python3
"""将 canonical dev split 转为评估 JSONL，并执行严格的数据一致性审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

DECISION_SYSTEM_PROMPT = (
    "You are AgriAbstain, a vision-language grounding assistant for high-stakes plant "
    "disease diagnosis. Locate the queried target. If the queried target is present and "
    "localizable, return "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]} with box in [0,1000] '
    "relative coords. Otherwise abstain: "
    '{"decision":"abstain","reason":"wrong_object|no_target"}. wrong_object: the image '
    "shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image, or cropped "
    "out of view)."
)
EXPECTED_DEV_ROWS = 2408
EXPECTED_TEST_ROWS = 2646
VALID_REASONS = {"wrong_object", "no_target"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def normalized_solution(row: dict[str, Any]) -> dict[str, Any]:
    decision = row.get("decision")
    if decision == "answer":
        label = row.get("label")
        box = row.get("box")
        if not isinstance(label, str) or not label or not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{row.get('sample_id')}: invalid supported annotation")
        return {"decision": "answer", "label": label, "box": box}
    if decision == "abstain":
        reason = row.get("reason")
        if reason not in VALID_REASONS:
            raise ValueError(f"{row.get('sample_id')}: invalid abstention reason {reason!r}")
        return {"decision": "abstain", "reason": reason}
    raise ValueError(f"{row.get('sample_id')}: invalid decision {decision!r}")


def source_key(row: dict[str, Any]) -> tuple[str, str]:
    dataset = str(row.get("dataset") or "")
    source_id = row.get("source_image_id")
    if not source_id:
        source_id = row.get("group_id")
    if not source_id:
        raise ValueError(f"{row.get('sample_id')}: missing source image/group identity")
    return dataset, str(source_id)


def resolve_image(raw: str, benchmark_root: Path, derived_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        candidate = path
    else:
        candidate = benchmark_root / path
        if not candidate.is_file() and path.parts[:2] == ("data", "derived"):
            candidate = derived_root / Path(*path.parts[2:])
    if not candidate.is_file():
        raise FileNotFoundError(f"Cannot resolve image {raw!r}: tried {candidate}")
    return candidate.resolve()


def make_eval_record(
    row: dict[str, Any],
    benchmark_root: Path,
    derived_root: Path,
) -> dict[str, Any]:
    solution = normalized_solution(row)
    metadata = row.get("metadata") or {}
    image = resolve_image(str(row["image_path"]), benchmark_root, derived_root)
    return {
        "messages": [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{row['query']}"},
        ],
        "images": [str(image)],
        "solution": {
            "decision": solution["decision"],
            "label": solution.get("label"),
            "box": solution.get("box"),
            "reason": solution.get("reason"),
        },
        "metadata": {
            "decision": solution["decision"],
            "label": solution.get("label"),
            "box": solution.get("box"),
            "reason": solution.get("reason"),
            "relation": metadata.get("relation"),
            "achieved_visible_ratio": metadata.get("achieved_visible_ratio"),
            "target_visible_ratio": metadata.get("target_visible_ratio"),
            "dose_only": metadata.get("dose_only", False),
            "sample_id": row["sample_id"],
            "source_image_id": row.get("source_image_id"),
            "group_id": row.get("group_id"),
        },
    }


def raw_semantic_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["image_path"]),
        str(row["query"]).replace("<image>", "").strip(),
        json.dumps(normalized_solution(row), sort_keys=True, separators=(",", ":")),
    )


def sft_semantic_key(row: dict[str, Any]) -> tuple[str, str, str]:
    human: str | None = None
    answer: dict[str, Any] | None = None
    for turn in row.get("conversations", []):
        if turn.get("from") == "human":
            human = str(turn.get("value", "")).replace("<image>", "").strip()
        elif turn.get("from") == "gpt":
            answer = json.loads(turn["value"])
    images = row.get("image")
    if not isinstance(images, list) or len(images) != 1 or human is None or answer is None:
        raise ValueError("Malformed sft_dev record")
    return (
        str(images[0]),
        human,
        json.dumps(answer, sort_keys=True, separators=(",", ":")),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--dev-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--sft-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dev_rows = load_jsonl(args.dev_split)
    test_rows = load_jsonl(args.test_split)
    if len(dev_rows) != EXPECTED_DEV_ROWS:
        raise ValueError(f"dev: expected {EXPECTED_DEV_ROWS}, got {len(dev_rows)}")
    if len(test_rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"test: expected {EXPECTED_TEST_ROWS}, got {len(test_rows)}")

    sample_ids = [row.get("sample_id") for row in dev_rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("dev contains missing sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("dev contains duplicate sample_id")
    if any((row.get("metadata") or {}).get("dose_only") for row in dev_rows):
        raise ValueError("dev unexpectedly contains dose_only records")

    dev_sources = {source_key(row) for row in dev_rows}
    test_sources = {source_key(row) for row in test_rows}
    source_overlap = sorted(dev_sources & test_sources)
    if source_overlap:
        raise ValueError(f"dev/test source-image overlap: {source_overlap[:20]}")

    with args.sft_dev.open(encoding="utf-8") as handle:
        sft_dev_rows = json.load(handle)
    if not isinstance(sft_dev_rows, list) or len(sft_dev_rows) != EXPECTED_DEV_ROWS:
        raise ValueError(
            f"sft_dev: expected JSON list of {EXPECTED_DEV_ROWS}, got "
            f"{len(sft_dev_rows) if isinstance(sft_dev_rows, list) else type(sft_dev_rows)}"
        )
    raw_counter = Counter(raw_semantic_key(row) for row in dev_rows)
    sft_counter = Counter(sft_semantic_key(row) for row in sft_dev_rows)
    if raw_counter != sft_counter:
        missing = list((raw_counter - sft_counter).elements())[:5]
        extra = list((sft_counter - raw_counter).elements())[:5]
        raise ValueError(f"canonical dev and sft_dev differ: missing={missing}, extra={extra}")

    eval_rows = [
        make_eval_record(row, args.benchmark_root.resolve(), args.derived_root.resolve())
        for row in dev_rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        dir=args.output.parent,
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            for row in eval_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp_name, args.output)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise

    decision_counts = Counter(row["decision"] for row in dev_rows)
    reason_counts = Counter(row.get("reason") for row in dev_rows if row["decision"] == "abstain")
    relation_counts = Counter((row.get("metadata") or {}).get("relation") for row in dev_rows)
    manifest = {
        "protocol": "development-only threshold selection for matched retention",
        "inputs": {
            "dev_split": str(args.dev_split.resolve()),
            "dev_split_sha256": sha256_file(args.dev_split),
            "test_split": str(args.test_split.resolve()),
            "test_split_sha256": sha256_file(args.test_split),
            "sft_dev": str(args.sft_dev.resolve()),
            "sft_dev_sha256": sha256_file(args.sft_dev),
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "n_dev": len(dev_rows),
        "n_test_audited": len(test_rows),
        "unique_dev_sample_ids": len(set(sample_ids)),
        "unique_dev_source_images": len(dev_sources),
        "unique_test_source_images": len(test_sources),
        "dev_test_source_overlap": len(source_overlap),
        "sft_dev_semantic_multiset_equal": True,
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "relation_counts": {
            str(key): value
            for key, value in sorted(relation_counts.items(), key=lambda x: str(x[0]))
        },
        "benchmark_root": str(args.benchmark_root.resolve()),
        "derived_root": str(args.derived_root.resolve()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
