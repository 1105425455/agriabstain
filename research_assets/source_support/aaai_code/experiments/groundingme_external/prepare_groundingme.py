#!/usr/bin/env python3
"""Prepare an auditable three-state GroundingME evaluation set.

The script never uses model predictions. It:
1. converts all 1,005 official records to the AgriAbstain JSONL interface;
2. extracts the embedded images without recompression;
3. creates deterministic cross-image no-target candidates; and
4. finalizes only visually reviewed candidates into the derived probe.

GroundingME Rejection records are mapped to ``wrong_object`` because their
descriptions contradict a visible candidate's appearance/component/text/state.
Derived ``no_target`` records transplant a valid supported description to a
recipient image in which the described object category has been manually
verified absent by contact-sheet inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
IMAGE_DIR = ARTIFACTS_DIR / "images"
SHEET_DIR = ARTIFACTS_DIR / "review_sheets"

EXPECTED_TOTAL = 1005
EXPECTED_L1 = {
    "Discriminative": 204,
    "Spatial": 300,
    "Limited": 300,
    "Rejection": 201,
}

SYSTEM_PROMPT = (
    "You are a vision-language grounding assistant. Determine whether the referring "
    "description identifies one visible, localizable target in the image. Every stated "
    "attribute, component, text, state, and relation must match the same target. If it "
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

# These broad or non-object labels are poor no-target donor heads: their absence is
# difficult to verify exhaustively from a single image.
NO_TARGET_DONOR_BLOCKLIST = {
    "animal",
    "art",
    "athlete",
    "blue",
    "building",
    "child",
    "clothe",
    "clothing",
    "competitor",
    "customer",
    "design",
    "equipment",
    "food",
    "girl",
    "hand",
    "hair",
    "individual",
    "light",
    "livestock",
    "logo",
    "man",
    "object",
    "participant",
    "pedestrian",
    "person",
    "player",
    "product",
    "signage",
    "spectator",
    "student",
    "tourist",
    "uniform",
    "vehicle",
    "wear",
    "woman",
    "worker",
    "writing",
}


def _load_frame() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("test-*.parquet"))
    if len(files) != 4:
        raise RuntimeError(f"expected 4 parquet shards under {DATA_DIR}, found {len(files)}")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    if len(frame) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} records, found {len(frame)}")
    counts = frame["subtask_l1"].value_counts().to_dict()
    if counts != EXPECTED_L1:
        raise RuntimeError(f"unexpected L1 distribution: {counts}")
    if frame["id"].duplicated().any():
        raise RuntimeError("duplicate GroundingME id")
    return frame.sort_values("id").reset_index(drop=True)


def _image_bytes(image_value: Any) -> tuple[bytes, str]:
    if not isinstance(image_value, dict):
        raise TypeError(f"unsupported image value: {type(image_value)}")
    data = image_value.get("bytes")
    source_name = image_value.get("path")
    if not isinstance(data, bytes) or not isinstance(source_name, str):
        raise TypeError("image record must contain bytes and path")
    return data, source_name


def _extract_images(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    index: dict[int, dict[str, Any]] = {}
    seen_names: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        data, source_name = _image_bytes(row.image)
        digest = hashlib.sha256(data).hexdigest()
        suffix = Path(source_name).suffix.lower() or ".jpg"
        output_name = f"gme_{int(row.id):04d}_{digest[:12]}{suffix}"
        output_path = IMAGE_DIR / output_name
        if not output_path.exists():
            output_path.write_bytes(data)
        elif hashlib.sha256(output_path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"existing image hash mismatch: {output_path}")

        with Image.open(io.BytesIO(data)) as image:
            actual_size = image.size
        declared_size = (int(row.width), int(row.height))
        if actual_size != declared_size:
            raise RuntimeError(
                f"id={row.id}: declared image size {declared_size} != decoded {actual_size}"
            )
        if source_name in seen_names and seen_names[source_name] != digest:
            raise RuntimeError(f"source filename collision with different bytes: {source_name}")
        seen_names[source_name] = digest
        index[int(row.id)] = {
            "path": str(output_path.resolve()),
            "source_name": source_name,
            "sha256": digest,
            "width": declared_size[0],
            "height": declared_size[1],
        }
    return index


def _normalize_box(box: Any, width: int, height: int) -> list[int] | None:
    if box is None:
        return None
    values = list(box)
    if len(values) != 4:
        raise ValueError(f"invalid box length: {values}")
    x1, y1, x2, y2 = (float(value) for value in values)
    normalized = [
        round(1000 * x1 / width),
        round(1000 * y1 / height),
        round(1000 * x2 / width),
        round(1000 * y2 / height),
    ]
    normalized = [min(1000, max(0, value)) for value in normalized]
    if not (normalized[0] < normalized[2] and normalized[1] < normalized[3]):
        raise ValueError(f"degenerate normalized box: source={values}, normalized={normalized}")
    return normalized


def _messages(description: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(description=description)},
    ]


def _official_record(row: Any, image_info: dict[str, Any]) -> dict[str, Any]:
    is_rejection = row.subtask_l1 == "Rejection"
    box = None if is_rejection else _normalize_box(row.bbox, row.width, row.height)
    decision = "abstain" if is_rejection else "answer"
    reason = "wrong_object" if is_rejection else None
    label = None if is_rejection else str(row.detection_type)
    sample_id = f"groundingme-{int(row.id):04d}"
    solution: dict[str, Any]
    if is_rejection:
        solution = {"decision": "abstain", "reason": "wrong_object"}
    else:
        solution = {"decision": "answer", "label": label, "box": box}
    return {
        "messages": _messages(str(row.description)),
        "images": [image_info["path"]],
        "solution": json.dumps(solution, ensure_ascii=False, separators=(",", ":")),
        "metadata": {
            "sample_id": sample_id,
            "source_image_id": sample_id,
            "dataset": "groundingme_official",
            "split": "test",
            "decision": decision,
            "reason": reason,
            "label": label,
            "label_scored": False,
            "box": box,
            "box_pixel": None if is_rejection else [int(v) for v in row.bbox],
            "relation": "official_rejection" if is_rejection else "official_supported",
            "subtask_l1": str(row.subtask_l1),
            "subtask_l2": str(row.subtask_l2),
            "detection_type": str(row.detection_type),
            "description": str(row.description),
            "image_width": int(row.width),
            "image_height": int(row.height),
            "image_sha256": image_info["sha256"],
            "source_filename": image_info["source_name"],
            "official_id": int(row.id),
            "derived": False,
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_official(frame: pd.DataFrame, image_index: dict[int, dict[str, Any]]) -> list[dict]:
    records = [
        _official_record(row, image_index[int(row.id)])
        for row in frame.itertuples(index=False)
    ]
    output = ARTIFACTS_DIR / "official_1005.jsonl"
    _write_jsonl(output, records)
    audit = {
        "records": len(records),
        "state_counts": Counter(record["metadata"]["decision"] for record in records),
        "reason_counts": Counter(
            record["metadata"]["reason"]
            for record in records
            if record["metadata"]["reason"] is not None
        ),
        "l1_counts": Counter(record["metadata"]["subtask_l1"] for record in records),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "output": str(output),
    }
    (ARTIFACTS_DIR / "official_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=dict) + "\n",
        encoding="utf-8",
    )
    return records


def _positive_stratum(row: Any) -> str:
    return f"{row.subtask_l1}/{row.subtask_l2}"


def build_candidates(
    frame: pd.DataFrame,
    image_index: dict[int, dict[str, Any]],
    *,
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positives = frame[frame["subtask_l1"] != "Rejection"].copy()
    donor_pool = positives[
        ~positives["detection_type"].str.lower().isin(NO_TARGET_DONOR_BLOCKLIST)
    ].copy()

    by_stratum: dict[str, list[Any]] = defaultdict(list)
    for row in positives.itertuples(index=False):
        by_stratum[_positive_stratum(row)].append(row)
    donor_rows = list(donor_pool.itertuples(index=False))
    rng.shuffle(donor_rows)
    used_donors: set[int] = set()
    used_recipients: set[int] = set()
    candidates: list[dict[str, Any]] = []

    for stratum in sorted(by_stratum):
        recipients = list(by_stratum[stratum])
        rng.shuffle(recipients)
        made = 0
        for recipient in recipients:
            if int(recipient.id) in used_recipients:
                continue
            recipient_text = (
                f"{recipient.description} {recipient.detection_type}"
            ).lower()
            compatible = [
                donor
                for donor in donor_rows
                if int(donor.id) not in used_donors
                and int(donor.id) != int(recipient.id)
                and str(donor.detection_type).lower() != str(recipient.detection_type).lower()
                and str(donor.detection_type).lower() not in recipient_text
            ]
            if not compatible:
                raise RuntimeError(f"no donor available for recipient id={recipient.id}")
            donor = compatible[rng.randrange(len(compatible))]
            candidate_id = f"gme-nt-cand-{len(candidates):03d}"
            candidate = {
                "candidate_id": candidate_id,
                "status": "pending_manual_review",
                "review_absent": None,
                "review_notes": "",
                "recipient_official_id": int(recipient.id),
                "recipient_source_image_id": f"groundingme-{int(recipient.id):04d}",
                "recipient_image": image_index[int(recipient.id)]["path"],
                "recipient_image_sha256": image_index[int(recipient.id)]["sha256"],
                "recipient_l1": str(recipient.subtask_l1),
                "recipient_l2": str(recipient.subtask_l2),
                "recipient_original_detection_type": str(recipient.detection_type),
                "recipient_original_description": str(recipient.description),
                "donor_official_id": int(donor.id),
                "donor_detection_type": str(donor.detection_type),
                "donor_description": str(donor.description),
                "donor_l1": str(donor.subtask_l1),
                "donor_l2": str(donor.subtask_l2),
                "construction": "cross_image_supported_description_transplant",
                "seed": seed,
            }
            candidates.append(candidate)
            used_recipients.add(int(recipient.id))
            used_donors.add(int(donor.id))
            made += 1
            if made == per_stratum:
                break
        if made != per_stratum:
            raise RuntimeError(f"stratum {stratum}: requested {per_stratum}, made {made}")

    output = ARTIFACTS_DIR / "no_target_candidates.jsonl"
    _write_jsonl(output, candidates)
    return candidates


def _fit_text(draw: ImageDraw.ImageDraw, text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposed = f"{current} {word}".strip()
        if draw.textbbox((0, 0), proposed, font=font)[2] <= width:
            current = proposed
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_review_sheets(candidates: list[dict[str, Any]], *, per_sheet: int = 12) -> None:
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_w, tile_h, header_h = 520, 430, 78
    cols = 3
    rows = (per_sheet + cols - 1) // cols
    for sheet_index, start in enumerate(range(0, len(candidates), per_sheet)):
        subset = candidates[start : start + per_sheet]
        canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
        for offset, candidate in enumerate(subset):
            col = offset % cols
            row = offset // cols
            x0, y0 = col * tile_w, row * tile_h
            with Image.open(candidate["recipient_image"]) as source:
                image = source.convert("RGB")
                image.thumbnail((tile_w - 12, tile_h - header_h - 12))
            tile = Image.new("RGB", (tile_w, tile_h), "white")
            tile.paste(image, ((tile_w - image.width) // 2, header_h + 4))
            draw = ImageDraw.Draw(tile)
            title = (
                f"{candidate['candidate_id']} | absent? "
                f"{candidate['donor_detection_type']}"
            )
            draw.text((6, 4), title, fill="red", font=font)
            detail = (
                f"recipient={candidate['recipient_official_id']} "
                f"({candidate['recipient_l1']}/{candidate['recipient_l2']}); "
                f"own={candidate['recipient_original_detection_type']}"
            )
            for line_index, line in enumerate(_fit_text(draw, detail, tile_w - 12, font)[:3]):
                draw.text((6, 20 + 14 * line_index), line, fill="black", font=font)
            canvas.paste(tile, (x0, y0))
        canvas.save(SHEET_DIR / f"sheet_{sheet_index:02d}.jpg", quality=92)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def finalize(review_path: Path) -> None:
    official = _load_jsonl(ARTIFACTS_DIR / "official_1005.jsonl")
    candidates = {
        record["candidate_id"]: record
        for record in _load_jsonl(ARTIFACTS_DIR / "no_target_candidates.jsonl")
    }
    review = json.loads(review_path.read_text(encoding="utf-8"))
    rejected_ids = set(review.get("rejected_candidate_ids", []))
    unknown = rejected_ids - set(candidates)
    if unknown:
        raise RuntimeError(f"review references unknown candidates: {sorted(unknown)}")
    approved_ids = sorted(set(candidates) - rejected_ids)
    if len(approved_ids) != 200:
        raise RuntimeError(
            f"exactly 200 visually approved no-target candidates required, found {len(approved_ids)}"
        )

    derived: list[dict[str, Any]] = []
    for candidate_id in approved_ids:
        item = candidates[candidate_id]
        sample_id = f"groundingme-derived-{candidate_id}"
        solution = {"decision": "abstain", "reason": "no_target"}
        derived.append(
            {
                "messages": _messages(item["donor_description"]),
                "images": [item["recipient_image"]],
                "solution": json.dumps(solution, separators=(",", ":")),
                "metadata": {
                    "sample_id": sample_id,
                    "source_image_id": item["recipient_source_image_id"],
                    "dataset": "groundingme_derived_no_target",
                    "split": "test",
                    "decision": "abstain",
                    "reason": "no_target",
                    "label": None,
                    "label_scored": False,
                    "box": None,
                    "relation": "derived_no_target",
                    "subtask_l1": "DerivedNoTarget",
                    "subtask_l2": item["recipient_l2"],
                    "detection_type": item["donor_detection_type"],
                    "description": item["donor_description"],
                    "image_sha256": item["recipient_image_sha256"],
                    "recipient_official_id": item["recipient_official_id"],
                    "donor_official_id": item["donor_official_id"],
                    "candidate_id": candidate_id,
                    "construction": item["construction"],
                    "visual_review": {
                        "status": "approved",
                        "review_absent": True,
                        "protocol": review.get("protocol", ""),
                        "reviewer": review.get("reviewer", ""),
                    },
                    "derived": True,
                },
            }
        )

    combined = official + derived
    _write_jsonl(ARTIFACTS_DIR / "derived_no_target_200.jsonl", derived)
    _write_jsonl(ARTIFACTS_DIR / "evaluation_1205.jsonl", combined)
    state_counts = Counter(
        (
            record["metadata"]["decision"]
            if record["metadata"]["decision"] == "answer"
            else record["metadata"]["reason"]
        )
        for record in combined
    )
    audit = {
        "official_records": len(official),
        "derived_no_target_records": len(derived),
        "total_records": len(combined),
        "state_counts": dict(state_counts),
        "unique_sample_ids": len({record["metadata"]["sample_id"] for record in combined}),
        "unique_source_images": len(
            {record["metadata"]["source_image_id"] for record in combined}
        ),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "review_file": str(review_path),
        "review_protocol": review.get("protocol"),
        "reviewer": review.get("reviewer"),
        "rejected_candidate_ids": sorted(rejected_ids),
    }
    if audit["unique_sample_ids"] != len(combined):
        raise RuntimeError("combined dataset contains duplicate sample ids")
    (ARTIFACTS_DIR / "evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["official", "candidates", "finalize", "all-pre-review"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates-per-stratum", type=int, default=30)
    parser.add_argument("--review-decisions", type=Path)
    args = parser.parse_args()

    frame = _load_frame()
    image_index = _extract_images(frame)
    if args.stage in {"official", "all-pre-review"}:
        prepare_official(frame, image_index)
    if args.stage in {"candidates", "all-pre-review"}:
        candidates = build_candidates(
            frame,
            image_index,
            per_stratum=args.candidates_per_stratum,
            seed=args.seed,
        )
        make_review_sheets(candidates)
        print(
            textwrap.dedent(
                f"""
                Wrote {len(candidates)} pending candidates.
                Review sheets: {SHEET_DIR}
                Record rejected IDs in a review-decisions JSON file. Finalization
                requires exactly 200 visually approved candidates.
                """
            ).strip()
        )
    if args.stage == "finalize":
        if args.review_decisions is None:
            parser.error("--review-decisions is required for finalize")
        finalize(args.review_decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
