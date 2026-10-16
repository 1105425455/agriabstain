#!/usr/bin/env python3
"""Prepare a fixed, target-size-filtered OpenRef single-pair diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


EXPECTED_TOTAL = 1760
EXPECTED_RELATIONS = {"single": 904, "multi": 612, "none": 244}
SYSTEM_PROMPT = (
    "You are a vision-language grounding assistant. Determine whether the referring "
    "description identifies one visible, localizable target in the image. Every stated "
    "attribute, identity, text, state, and relation must match the same target. If it "
    "does, return exactly one JSON object: "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]}. '
    "The box must use relative coordinates in [0,1000]. Otherwise return exactly one "
    'JSON object: {"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object means that a visually plausible candidate is present but contradicts "
    "one or more details in the description. no_target means that the described object "
    "category has no corresponding target in the image. A small or partially occluded "
    "target should still be answered when it remains localizable. Do not guess a box."
)
USER_TEMPLATE = (
    "<image>Locate the single object described below. Every detail must match the same "
    "target.\nDescription: {description}"
)


def _load_json(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("OpenRef val.json must contain a list")
    return records


def _target_area(record: dict[str, Any]) -> float:
    proposal = record["proposal"][0]
    return float(proposal["width"]) * float(proposal["height"]) / 10_000.0


def _eligible(record: dict[str, Any], minimum_area: float) -> bool:
    return (
        record.get("relation_type") == "single"
        and record.get("bbox_num") == 1
        and isinstance(record.get("proposal"), list)
        and len(record["proposal"]) == 1
        and isinstance(record.get("positive"), str)
        and bool(record["positive"].strip())
        and isinstance(record.get("negative"), str)
        and bool(record["negative"].strip())
        and record["positive"].strip().casefold() != record["negative"].strip().casefold()
        and _target_area(record) >= minimum_area
    )


def _normalized_box(proposal: dict[str, Any]) -> list[int]:
    x1 = round(10 * float(proposal["x"]))
    y1 = round(10 * float(proposal["y"]))
    x2 = round(10 * (float(proposal["x"]) + float(proposal["width"])))
    y2 = round(10 * (float(proposal["y"]) + float(proposal["height"])))
    box = [min(1000, max(0, value)) for value in (x1, y1, x2, y2)]
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError(f"degenerate normalized box: {box}")
    return box


def _messages(description: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(description=description)},
    ]


def _zip_member_index(archive: zipfile.ZipFile) -> dict[str, str]:
    by_basename: dict[str, str] = {}
    collisions: set[str] = set()
    for member in archive.namelist():
        if member.endswith("/"):
            continue
        basename = PurePosixPath(member).name
        if basename in by_basename and by_basename[basename] != member:
            collisions.add(basename)
        by_basename[basename] = member
    if collisions:
        raise RuntimeError(f"duplicate image basenames in archive: {sorted(collisions)[:5]}")
    return by_basename


def _extract_selected_images(
    archive_path: Path,
    selected: list[dict[str, Any]],
    image_dir: Path,
) -> dict[str, dict[str, Any]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_info: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        member_index = _zip_member_index(archive)
        for record in selected:
            filename = str(record["image"])
            member = member_index.get(filename)
            if member is None:
                raise FileNotFoundError(f"{filename} is missing from {archive_path}")
            data = archive.read(member)
            output = image_dir / filename
            if output.exists() and output.read_bytes() != data:
                raise RuntimeError(f"existing extracted image differs: {output}")
            if not output.exists():
                output.write_bytes(data)
            with Image.open(output) as image:
                actual_width, actual_height = image.size
            proposal = record["proposal"][0]
            declared = (
                int(proposal["original_width"]),
                int(proposal["original_height"]),
            )
            if (actual_width, actual_height) != declared:
                raise RuntimeError(
                    f"{filename}: decoded size {(actual_width, actual_height)} "
                    f"does not match declared {declared}"
                )
            image_info[filename] = {
                "path": str(output.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": actual_width,
                "height": actual_height,
            }
    return image_info


def _record(
    source: dict[str, Any],
    image: dict[str, Any],
    pair_index: int,
    role: str,
    vision_max_pixels: int,
) -> dict[str, Any]:
    if role not in {"positive", "negative"}:
        raise ValueError(role)
    description = str(source[role]).strip()
    is_positive = role == "positive"
    box = _normalized_box(source["proposal"][0]) if is_positive else None
    decision = "answer" if is_positive else "abstain"
    solution = (
        {"decision": "answer", "label": description, "box": box}
        if is_positive
        else {"decision": "abstain", "reason": "wrong_object"}
    )
    pair_id = f"openref-single-{pair_index:04d}"
    sample_id = f"{pair_id}-{role}"
    return {
        "messages": _messages(description),
        "images": [image["path"]],
        "solution": json.dumps(solution, ensure_ascii=False, separators=(",", ":")),
        "metadata": {
            "sample_id": sample_id,
            "source_image_id": str(source["image"]),
            "dataset": "openref_val_single_size_filtered",
            "split": "validation",
            "pair_id": pair_id,
            "pair_role": role,
            "decision": decision,
            "reason": None if is_positive else "wrong_object",
            "reason_scored": False,
            "label": description if is_positive else None,
            "label_scored": False,
            "box": box,
            "relation_type": "single",
            "description": description,
            "positive_description": str(source["positive"]).strip(),
            "negative_description": str(source["negative"]).strip(),
            "target_relative_area": _target_area(source),
            "image_width": image["width"],
            "image_height": image["height"],
            "image_sha256": image["sha256"],
            "source_filename": str(source["image"]),
            "official_negative": not is_positive,
            "vision_max_pixels": vision_max_pixels,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--minimum-target-area", type=float, default=0.01)
    parser.add_argument("--vision-max-pixels", type=int, default=16_777_216)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load_json(args.val_json)
    relation_counts = Counter(str(record.get("relation_type")) for record in records)
    if len(records) != EXPECTED_TOTAL or dict(relation_counts) != EXPECTED_RELATIONS:
        raise RuntimeError(
            f"unexpected OpenRef validation distribution: n={len(records)}, "
            f"relations={dict(relation_counts)}"
        )
    filenames = [str(record["image"]) for record in records]
    if len(filenames) != len(set(filenames)):
        raise RuntimeError("OpenRef validation image filenames are not unique")

    eligible = [
        record for record in records if _eligible(record, args.minimum_target_area)
    ]
    eligible.sort(key=lambda record: str(record["image"]))
    if len(eligible) < args.pairs:
        raise RuntimeError(f"only {len(eligible)} eligible pairs for requested {args.pairs}")
    rng = random.Random(args.seed)
    selected_names = {
        str(record["image"]) for record in rng.sample(eligible, args.pairs)
    }
    selected = [
        record for record in eligible if str(record["image"]) in selected_names
    ]

    data_dir = args.output_root / "data"
    artifact_dir = args.output_root / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    image_info = _extract_selected_images(
        args.images_zip, selected, artifact_dir / "images"
    )

    paired_records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for pair_index, source in enumerate(selected):
        filename = str(source["image"])
        paired_records.extend(
            _record(
                source,
                image_info[filename],
                pair_index,
                role,
                args.vision_max_pixels,
            )
            for role in ("positive", "negative")
        )
    for source in eligible:
        manifest.append(
            {
                "source_filename": str(source["image"]),
                "selected": str(source["image"]) in selected_names,
                "target_relative_area": _target_area(source),
                "positive": str(source["positive"]),
                "negative": str(source["negative"]),
            }
        )

    data_path = data_dir / f"openref_single_{args.pairs}_pairs.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for record in paired_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_path = data_dir / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "source_split": "OpenRef validation",
                    "relation_type": "single",
                    "minimum_target_relative_area": args.minimum_target_area,
                    "eligible_pairs": len(eligible),
                    "selected_pairs": len(selected),
                    "vision_max_pixels": args.vision_max_pixels,
                    "vision_preprocessing": (
                        "The per-image max_pixels field is passed to qwen_vl_utils, "
                        "preserving aspect ratio and preventing the processor-only "
                        "max_pixels argument from being silently ignored."
                    ),
                    "selection": (
                        "Python random.Random(seed).sample over records sorted by image "
                        "filename; selection is independent of all model outputs"
                    ),
                    "seed": args.seed,
                    "negative_scoring": (
                        "Any valid abstention is correct. The benchmark negative does not "
                        "provide a reason label, so wrong_object versus no_target is not scored."
                    ),
                },
                "records": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_records": len(records),
        "relation_counts": dict(relation_counts),
        "eligible_pairs": len(eligible),
        "selected_pairs": len(selected),
        "evaluation_records": len(paired_records),
        "unique_selected_images": len(image_info),
        "data_path": str(data_path.resolve()),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (data_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
