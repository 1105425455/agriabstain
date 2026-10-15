#!/usr/bin/env python
# ruff: noqa: E501
"""Build the PlantWild out-of-distribution probe: an abstention-only stress test.

PlantWild replaces PlantVillage as the single changed variable, with every other construction
choice held fixed: 13 categories, 12 images per category, seed 42, and only OOD_absent and
OOD_confusion records, with no answerable branch.

The two record types follow the same definitions as the PlantVillage builder:
  - OOD_absent:    the query names a disease of another crop, reference abstain no_target
  - OOD_confusion: the query names another disease of the same crop, reference abstain
                   wrong_object

Constraints:
  - only the official PlantWild test split, i.e. split=2 in trainval.txt
  - only the 13 categories that map onto the naming used here with an exact semantic match
  - files whose name contains ` - Copy` are excluded outright
  - de-duplication by MD5 and by perceptual hash within the source, followed by cross
    de-duplication against PlantDoc by perceptual hash
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# --- Constants ---
# Inputs are external datasets and are not part of this archive. Set PLANTWILD_ZIP and
# PLANTDOC_DIR to local copies; OUTPUT_ROOT defaults to ./plantwild_build.
ZIP_PATH = Path(os.environ.get("PLANTWILD_ZIP", "plantwild.zip"))
PLANTDOC_DIR = Path(os.environ.get("PLANTDOC_DIR", "plantdoc"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "plantwild_build"))
OUTPUT_IMG_DIR = OUTPUT_ROOT / "images"
OUTPUT_JSONL = OUTPUT_ROOT / "ood_plantwild.jsonl"
OUTPUT_STATS = OUTPUT_ROOT / "build_stats.json"
OUTPUT_AUDIT = OUTPUT_ROOT / "dedup_audit.json"

PER_CLASS = 12
SEED = 42

SYSTEM_PROMPT = (
    "You are a vision-language grounding assistant. "
    "First reason briefly inside <think> </think> about whether the queried target is the object shown "
    "and whether it can be localized. Then give the final answer inside <answer> </answer> as a single JSON object. "
    'If the queried target is present and localizable, answer with '
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]}. '
    'Otherwise abstain with {"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image). "
    "Boxes use relative coordinates in [0,1000]. Output exactly: "
    "<think> your reasoning </think><answer> the JSON </answer>"
)

# PlantWild names mapped to the naming used here (PV13 form)
PW_TO_PV13: dict[str, str] = {
    "apple scab": "apple scab leaf",
    "apple rust": "apple rust leaf",
    "corn gray leaf spot": "corn gray leaf spot",
    "corn rust": "corn rust leaf",
    "corn northern leaf blight": "corn leaf blight",
    "potato late blight": "potato leaf late blight",
    "squash powdery mildew": "squash powdery mildew leaf",
    "tomato bacterial leaf spot": "tomato leaf bacterial spot",
    "tomato early blight": "tomato early blight leaf",
    "tomato late blight": "tomato leaf late blight",
    "tomato leaf mold": "tomato mold leaf",
    "tomato septoria leaf spot": "tomato septoria leaf spot",
    "tomato yellow leaf curl virus": "tomato leaf yellow virus",
}

# The 13 category names, in the naming used here
PV13_NAMES: set[str] = set(PW_TO_PV13.values())

# Grouped by crop, for the absent and confusion pairings
def _crop_of(label: str) -> str:
    return label.split()[0]

CROP_GROUPS: dict[str, list[str]] = defaultdict(list)
for pv_name in PV13_NAMES:
    CROP_GROUPS[_crop_of(pv_name)].append(pv_name)

# Only apple, corn and tomato carry several diseases, so only they support confusion pairs
CONFUSION_CROPS = {c for c, labs in CROP_GROUPS.items() if len(labs) >= 2}

# Manual quality review: the entries below were confirmed unusable and are excluded
# - corn rust / google_0181.jpg: a four-panel severity figure with text and a scale bar,
#   not a natural leaf photograph; apple scab / 71.jpg: a watermarked fruit photograph
MANUAL_EXCLUDE: set[str] = {
    "corn rust/google_0181.jpg",
    "apple scab/71.jpg",
}

# Categories eligible for confusion: one confusion record per base image
CONFUSION_CLASS_COUNT = sum(len(labs) for labs in CROP_GROUPS.values() if len(labs) >= 2)  # 11


# --- Perceptual hashing ---
def _dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Perceptual difference hash of an image, used for near-duplicate detection."""
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    bits = 0
    for y in range(hash_size):
        for x in range(hash_size):
            left = pixels[y * (hash_size + 1) + x]
            right = pixels[y * (hash_size + 1) + x + 1]
            if left < right:
                bits |= (1 << (y * hash_size + x))
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# --- Main routine ---
def main() -> None:
    for label, path, var in (
        ("PlantWild zip", ZIP_PATH, "PLANTWILD_ZIP"),
        ("PlantDoc directory", PLANTDOC_DIR, "PLANTDOC_DIR"),
    ):
        if not path.exists():
            sys.exit(f"{label} not found at {path}. Set {var} to a local copy.")

    rng = random.Random(SEED)
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_STATS.parent.mkdir(parents=True, exist_ok=True)

    audit: dict = {
        "internal_exact_dedup": {"excluded": [], "groups": []},
        "internal_perceptual_dedup": {"excluded": [], "near_duplicate_groups": []},
        "copy_files_excluded": [],
        "manual_quality_excluded": [],
        "plantdoc_cross_dedup": {"excluded": [], "suspicious_6_8": []},
        "plantdoc_images_loaded": 0,
        "candidates_internal_dedup": 0,
        "candidates_after_cross_dedup": {},
        "final_sampled": {},
    }

    # --- 1. Read trainval.txt and collect the split=2 candidates ---
    print("[step 1] 读取 PlantWild trainval.txt，筛选 split=2…", flush=True)
    z = zipfile.ZipFile(str(ZIP_PATH), "r")
    tv_lines = z.read("plantwild/trainval.txt").decode("utf-8").strip().split("\n")

    # Line format: classname/filename=label=split
    # Collect (zip_path, raw_bytes) grouped by category
    candidates: dict[str, list[dict]] = defaultdict(list)
    for line in tv_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("=")
        if len(parts) != 3:
            continue
        cls_file, _, split = parts
        if split.strip() != "2":
            continue
        cls = cls_file.split("/")[0]
        if cls not in PW_TO_PV13:
            continue
        fname = cls_file.split("/")[1]
        # Skip files marked as copies
        if " - Copy" in fname:
            audit["copy_files_excluded"].append(f"{cls}/{fname}")
            continue
        # Apply the manual quality-review exclusions listed above
        entry_key = f"{cls}/{fname}"
        if entry_key in MANUAL_EXCLUDE:
            audit.setdefault("manual_quality_excluded", []).append({
                "entry": entry_key,
                "cls": cls,
                "original_fname": fname,
                "reason": (
                    "论文4宫格严重度分级图，带文字和标尺，非自然叶片病害图"
                    if entry_key == "corn rust/google_0181.jpg"
                    else "苹果果实图，带 BIGSTOCK 水印，非叶片病害主场景"
                ),
            })
            continue
        zip_inner_path = f"plantwild/images/{cls}/{fname}"
        candidates[cls].append({
            "cls": cls,
            "fname": fname,
            "zip_inner_path": zip_inner_path,
        })

    print(f"   split=2 PV13 候选（去 Copy + 人工排除前）：{sum(len(v) for v in candidates.values())} 张", flush=True)
    print(f"   人工质量审查排除：{len(audit['manual_quality_excluded'])} 张", flush=True)
    for mqe in audit["manual_quality_excluded"]:
        print(f"     - {mqe['entry']}: {mqe['reason']}", flush=True)
    for cls in sorted(candidates):
        print(f"     {cls}: {len(candidates[cls])}", flush=True)

    # --- 2. Exact de-duplication by MD5, within each category ---
    print("\n[step 2] MD5 精确去重…", flush=True)
    md5_seen: dict[str, str] = {}  # md5 → (cls, fname)
    deduped: dict[str, list[dict]] = defaultdict(list)
    for cls in sorted(candidates):
        for cand in candidates[cls]:
            raw = z.read(cand["zip_inner_path"])
            h = _md5_bytes(raw)
            if h in md5_seen:
                prev_cls, prev_fname = md5_seen[h]
                audit["internal_exact_dedup"]["excluded"].append({
                    "kept": f"{prev_cls}/{prev_fname}",
                    "excluded": f"{cls}/{cand['fname']}",
                    "md5": h,
                })
                continue
            md5_seen[h] = (cls, cand["fname"])
            cand["bytes"] = raw
            cand["md5"] = h
            deduped[cls].append(cand)
    print(f"   MD5 去重后：{sum(len(v) for v in deduped.values())} 张", flush=True)

    # --- 3. Perceptual de-duplication by dHash, within a category, Hamming distance <= 5 ---
    print("\n[step 3] dHash 感知去重（同类内 Hamming ≤ 5）…", flush=True)
    phash_deduped: dict[str, list[dict]] = defaultdict(list)
    for cls in sorted(deduped):
        items = deduped[cls]
        # Sort by resolution first, so the higher-resolution copy is the one kept
        items_with_res = []
        for cand in items:
            im = Image.open(io.BytesIO(cand["bytes"]))
            w, h = im.size
            cand["dhash"] = _dhash(im)
            cand["resolution"] = (w, h)
            cand["resolution_product"] = w * h
            items_with_res.append(cand)
        items_with_res.sort(key=lambda x: x["resolution_product"], reverse=True)

        kept: list[dict] = []
        kept_dhashes: list[int] = []
        for cand in items_with_res:
            dup = False
            for existing_hash in kept_dhashes:
                if _hamming(cand["dhash"], existing_hash) <= 5:
                    dup = True
                    audit["internal_perceptual_dedup"]["excluded"].append({
                        "kept": f"{cls}/{kept[len(kept_dhashes) - len(kept_dhashes) + kept_dhashes.index(existing_hash)]['fname']}",
                        "excluded": f"{cls}/{cand['fname']}",
                        "hamming": _hamming(cand["dhash"], existing_hash),
                    })
                    break
            if not dup:
                kept.append(cand)
                kept_dhashes.append(cand["dhash"])
        phash_deduped[cls] = kept

    audit["candidates_internal_dedup"] = sum(len(v) for v in phash_deduped.values())
    print(f"   dHash 去重后：{audit['candidates_internal_dedup']} 张", flush=True)
    for cls in sorted(phash_deduped):
        print(f"     {cls}: {len(phash_deduped[cls])}", flush=True)

    # --- 4. Require at least 12 surviving images per category ---
    print("\n[step 4] 检查每类候选数 ≥ 12…", flush=True)
    for cls in sorted(phash_deduped):
        n = len(phash_deduped[cls])
        if n < PER_CLASS:
            print(f"   FAIL: {cls} 只剩 {n} 张，不足 {PER_CLASS}", flush=True)
            sys.exit(1)
        print(f"   {cls}: {n} 张 ✓", flush=True)

    # --- 5. Cross de-duplication against PlantDoc, excluding dHash distance <= 8 ---
    # Done before sampling: every candidate is checked, then the sample is drawn from survivors
    print("\n[step 5] 与 PlantDoc 交叉 dHash 去重（全量候选，先于抽样）…", flush=True)
    # Load every PlantDoc source image, across train, val and test
    plantdoc_images: list[Image.Image] = []
    for split_name in ["train", "val", "test"]:
        img_dir = PLANTDOC_DIR / split_name / "images"
        if not img_dir.exists():
            continue
        for fpath in sorted(img_dir.iterdir()):
            if fpath.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    im = Image.open(fpath).convert("RGB")
                    plantdoc_images.append(im)
                except Exception:
                    pass
    audit["plantdoc_images_loaded"] = len(plantdoc_images)
    print(f"   PlantDoc 源图加载：{len(plantdoc_images)} 张", flush=True)

    # Precompute the PlantDoc hashes
    plantdoc_dhashes = [_dhash(im) for im in plantdoc_images]
    del plantdoc_images  # free memory

    cross_deduped: dict[str, list[dict]] = defaultdict(list)
    for cls in sorted(phash_deduped):
        for cand in phash_deduped[cls]:
            im = Image.open(io.BytesIO(cand["bytes"]))
            cand_dh = _dhash(im)
            excluded = False
            min_hamming = 999
            for pd_hash in plantdoc_dhashes:
                h = _hamming(cand_dh, pd_hash)
                if h < min_hamming:
                    min_hamming = h
                if h <= 5:
                    excluded = True
                    audit["plantdoc_cross_dedup"]["excluded"].append({
                        "file": f"{cls}/{cand['fname']}",
                        "min_hamming": h,
                    })
                    break
            if not excluded and 6 <= min_hamming <= 8:
                audit["plantdoc_cross_dedup"]["suspicious_6_8"].append({
                    "file": f"{cls}/{cand['fname']}",
                    "min_hamming": min_hamming,
                })
                # Exclude by default, which is the conservative choice
                excluded = True
            if not excluded:
                cross_deduped[cls].append(cand)

    for cls in sorted(cross_deduped):
        audit["candidates_after_cross_dedup"][cls] = len(cross_deduped[cls])

    print(f"   与 PlantDoc 去重后：{sum(len(v) for v in cross_deduped.values())} 张", flush=True)
    for cls in sorted(cross_deduped):
        marker = " ✓" if len(cross_deduped[cls]) >= PER_CLASS else " ⚠"
        print(f"     {cls}: {len(cross_deduped[cls])} 张{marker}", flush=True)

    # Re-check that 12 images per category survive the cross de-duplication
    for cls in sorted(cross_deduped):
        n = len(cross_deduped[cls])
        if n < PER_CLASS:
            print(f"   FAIL: 与 PlantDoc 去重后 {cls} 只剩 {n} 张，不足 {PER_CLASS}", flush=True)
            sys.exit(1)

    # --- 6. Sample 12 images per category from the survivors, at seed 42 ---
    print(f"\n[step 6] seed={SEED} 从 PlantDoc 去重后幸存者中每类随机抽 {PER_CLASS} 张…", flush=True)
    sampled: dict[str, list[dict]] = {}
    for cls in sorted(cross_deduped):
        pool = list(cross_deduped[cls])
        rng.shuffle(pool)
        sampled[cls] = pool[:PER_CLASS]
        audit["final_sampled"][cls] = len(sampled[cls])
        print(f"   {cls}: 抽 {len(sampled[cls])} 张", flush=True)

    # --- 7. Write the images to the output directory ---
    print("\n[step 7] 保存图片…", flush=True)
    saved_images: dict[str, list[dict]] = defaultdict(list)
    for cls in sorted(sampled):
        for i, cand in enumerate(sampled[cls]):
            safe_cls = cls.replace(" ", "_")
            out_name = f"pw_{safe_cls}_{i}.jpg"
            out_path = OUTPUT_IMG_DIR / out_name
            im = Image.open(io.BytesIO(cand["bytes"])).convert("RGB")
            im.save(out_path, "JPEG", quality=92)
            saved_images[cls].append({
                "fname": out_name,
                "abs_path": str(out_path.resolve()),
                "true_label": PW_TO_PV13[cls],
                "wild_cls": cls,
                "original_fname": cand["fname"],
            })
        print(f"   {PW_TO_PV13[cls]}: {len(saved_images[cls])} 张", flush=True)

    # --- 8. Build the absent and confusion records ---
    print("\n[step 8] 构造 OOD 记录…", flush=True)
    records: list[dict] = []
    for cls in sorted(saved_images):
        true_label = PW_TO_PV13[cls]
        crop = _crop_of(true_label)
        for img_info in saved_images[cls]:
            base_sid = f"pw-{cls.replace(' ', '_')}-{img_info['fname'].replace('.jpg', '')}"

            # (a) absent: a disease of a different crop
            other_crops = [c for c in CROP_GROUPS if c != crop]
            if other_crops:
                oc = rng.choice(other_crops)
                absent_label = rng.choice(CROP_GROUPS[oc])
                records.append(_make_record(
                    f"{base_sid}-absent", img_info["abs_path"],
                    absent_label, "abstain", None, "no_target",
                    "OOD_absent", "absent",
                ))

            # (b) confusion: another disease of the same crop, where the crop has more than one
            siblings = [d for d in CROP_GROUPS[crop] if d != true_label]
            if siblings:
                conf_label = rng.choice(siblings)
                records.append(_make_record(
                    f"{base_sid}-conf", img_info["abs_path"],
                    conf_label, "abstain", None, "wrong_object",
                    "OOD_confusion", "confusion",
                ))

    rng.shuffle(records)

    # --- 9. Write the JSONL ---
    print(f"\n[step 9] 写 JSONL → {OUTPUT_JSONL}…", flush=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 10. Summary counts ---
    by_probe: dict[str, int] = defaultdict(int)
    by_relation: dict[str, int] = defaultdict(int)
    for r in records:
        m = r["metadata"]
        by_probe[m["probe"]] += 1
        by_relation[m["relation"]] += 1

    true_class_counts: dict[str, int] = Counter()
    for cls in saved_images:
        true_class_counts[PW_TO_PV13[cls]] = len(saved_images[cls])

    stats = {
        "total_records": len(records),
        "total_unique_images": sum(len(v) for v in saved_images.values()),
        "per_class": PER_CLASS,
        "seed": SEED,
        "by_probe": dict(by_probe),
        "by_relation": dict(by_relation),
        "true_class_counts": dict(true_class_counts),
        "manual_quality_excluded": len(audit["manual_quality_excluded"]),
        "expected_absent": 13 * PER_CLASS,
        "expected_confusion": CONFUSION_CLASS_COUNT * PER_CLASS,
        "expected_total": 13 * PER_CLASS + CONFUSION_CLASS_COUNT * PER_CLASS,
    }

    print("\n统计:", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)

    with OUTPUT_STATS.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # --- 11. Write the de-duplication audit ---
    with OUTPUT_AUDIT.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("\n✅ 完成！", flush=True)
    print(f"   JSONL: {OUTPUT_JSONL} ({len(records)} 条)", flush=True)
    print(f"   图片: {OUTPUT_IMG_DIR} ({sum(len(v) for v in saved_images.values())} 张)", flush=True)
    print(f"   统计: {OUTPUT_STATS}", flush=True)
    print(f"   审计: {OUTPUT_AUDIT}", flush=True)

    z.close()


def _make_record(
    sid: str, img_path: str, query_label: str,
    decision: str, label: str | None, reason: str | None,
    relation: str, probe: str,
) -> dict:
    if decision == "answer":
        ans = json.dumps({"decision": "answer", "label": label, "box": None},
                         ensure_ascii=False, separators=(",", ":"))
    else:
        ans = json.dumps({"decision": "abstain", "reason": reason},
                         ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>Locate the " + query_label + " in this image."},
        ],
        "images": [img_path],
        "solution": ans,
        "metadata": {
            "sample_id": sid,
            "dataset": "plantwild_ood",
            "split": "test",
            "probe": probe,
            "relation": relation,
            "decision": decision,
            "label": label,
            "box": None,
            "reason": reason,
            "target_visible_ratio": None,
            "achieved_visible_ratio": None,
        },
    }


if __name__ == "__main__":
    main()