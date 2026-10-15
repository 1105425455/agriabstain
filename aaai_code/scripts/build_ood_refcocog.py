#!/usr/bin/env python
"""General-domain out-of-distribution probe, mapping RefCOCOg into the benchmark schema.

The probe tests whether grounding learned on the agricultural benchmark still holds on unseen
general-domain referring expressions. RefCOCOg carries **real reference boxes** and therefore
complements the agricultural probes, where PlantWild is label-only: the general domain
carries the quantitative grounding signal and the agricultural domain the fine-grained
abstention signal.

RefCOCOg is natively a region-to-caption task and is used in reverse here:
  - answer branch: the referring expression becomes the query and the bounding box the
    reference, so IoU can be scored
  - absent branch: the query is replaced by a COCO category that does not appear in the image,
    drawn from categories seen during training so that abstention is not a capability
    limitation, with reference abstain and reason no_target

Box conversion: RefCOCOg uses absolute COCO [x, y, w, h], converted here to [x1, y1, x2, y2]
relative to [0, 1000], which requires the image size. The output follows the same schema as
the in-domain probes.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[2]
# Determine mode: package (aaai_code/benchmark) vs source (full research repo)
AAAI_CODE = Path(__file__).resolve().parents[1]  # aaai_code/
IS_PACKAGE = (AAAI_CODE / "benchmark").is_dir()


SYSTEM_PROMPT = (
    "You are AgriAbstain, a vision-language grounding assistant. "
    "First reason briefly inside <think> </think> about whether the queried target is the "
    "object shown and whether it can be localized. Then give the final answer inside "
    "<answer> </answer> as a single JSON object. "
    "If the queried target is present and localizable, answer with "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]}. '
    'Otherwise abstain with {"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, "
    "not the queried one. "
    "no_target: the queried target is not locatable (absent from the image). "
    "Boxes use relative coordinates in [0,1000]. Output exactly: "
    "<think> your reasoning </think><answer> the JSON </answer>"
)

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def _short_query(answer_list: list[str]) -> str | None:
    """Choose the shortest, most phrase-like referring expression as the query, avoiding long captions.

    """
    if not answer_list:
        return None
    cands = [a.strip().rstrip(".") for a in answer_list if a and a.strip()]
    if not cands:
        return None
    cands.sort(key=len)
    q = cands[0]
    q = re.sub(r"^(the|a|an)\s+", "", q, flags=re.I)
    return q if 0 < len(q) <= 60 else None


def _coco_to_rel_box(bbox: list[float], w: int, h: int) -> list[int] | None:
    """Convert absolute COCO [x, y, w, h] to [x1, y1, x2, y2] relative to [0, 1000]."""
    if not bbox or len(bbox) != 4 or w <= 0 or h <= 0:
        return None
    x, y, bw, bh = bbox
    x1 = max(0, min(1000, round(x / w * 1000)))
    y1 = max(0, min(1000, round(y / h * 1000)))
    x2 = max(0, min(1000, round((x + bw) / w * 1000)))
    y2 = max(0, min(1000, round((y + bh) / h * 1000)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _absent_class(query: str, rng: random.Random) -> str:
    """Choose a COCO category that the query does not mention, for the absent branch."""
    ql = query.lower()
    cands = [c for c in COCO_CLASSES if c not in ql]
    return rng.choice(cands)


def _record(
    sid,
    img_path,
    query,
    decision,
    box,
    reason,
    relation,
    probe,
    coco_image_id=None,
    refcocog_row_index=None,
    coco_file_name=None,
    question_id=None,
):
    if decision == "answer":
        ans = json.dumps(
            {"decision": "answer", "label": query, "box": box},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        label = query
    else:
        ans = json.dumps(
            {"decision": "abstain", "reason": reason}, ensure_ascii=False, separators=(",", ":")
        )
        label = None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>Locate the " + query + " in this image."},
        ],
        "images": [img_path],
        "solution": ans,
        "metadata": {
            "sample_id": sid,
            "dataset": "refcocog_ood",
            "split": "val",
            "probe": probe,
            "relation": relation,
            "decision": decision,
            "label": label,
            "box": box,
            "reason": reason,
            "target_visible_ratio": None,
            "achieved_visible_ratio": None,
            "coco_image_id": coco_image_id,
            "refcocog_row_index": refcocog_row_index,
            "coco_file_name": coco_file_name,
            "question_id": question_id,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    if IS_PACKAGE:
        default_parquet = str(
            AAAI_CODE / "benchmark/data/ood/refcocog/metadata/val-00000-of-00003.parquet"
        )
        default_img_dir = str(AAAI_CODE / "benchmark/data/ood/refcocog/images")
        default_output = str(AAAI_CODE / "benchmark/annotations/eval/ood_refcocog.jsonl")
        default_gate = str(AAAI_CODE / "benchmark/data/ood/refcocog/metadata/gate_refcocog.json")
    else:
        default_parquet = str(REPO / "data/ood/refcocog/data/val-00000-of-00003.parquet")
        default_img_dir = str(REPO / "data/ood/refcocog/images")
        default_output = str(REPO / "new_reason/outputs/eval_probes/ood_refcocog.jsonl")
        default_gate = "outputs/ood/cal/gate_refcocog.json"
    ap.add_argument(
        "--parquet",
        default=default_parquet,
    )
    ap.add_argument("--img-dir", default=default_img_dir)
    ap.add_argument(
        "--output",
        default=default_output,
        help="Output path for the final RefCOCOg probes",
    )
    ap.add_argument("--n-answer", type=int, default=400, help="Number of answer probes (with ground-truth boxes, scored by IoU)")
    ap.add_argument(
        "--gate",
        default=default_gate,
        help="iou_only gate JSON: build probes only on images the base model localizes correctly (IoU >= 0.5), so that OOD absent/insufficient abstentions are attributable to hallucination rather than to the base model failing to localize across domains",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img_dir = Path(args.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    kept_idx: set[int] | None = None
    if args.gate and Path(args.gate).exists():
        gate = json.loads(Path(args.gate).read_text())
        sids = gate.get("kept_sample_ids")
        if not sids:
            raise SystemExit(
                f"[gate] {args.gate} 无 kept_sample_ids（应为 iou_only 口径）。"
                f"请先 `compute_ood_gate.py --metric iou_only` 重算，或 --gate '' 关闭 gate。"
            )
        kept_idx = {int(m.group(1)) for sid in sids if (m := re.match(r"refg-(\d+)-", sid))}
        print(f"[gate] RefCOCOg 保留 {len(kept_idx)} 图（base IoU≥0.5）", flush=True)

    tbl = pq.read_table(args.parquet)
    n_rows = tbl.num_rows
    img_struct = tbl.column("image")
    answers = tbl.column("answer")
    bboxes = tbl.column("bbox")
    iscrowds = tbl.column("iscrowd")

    idxs = list(range(n_rows))
    rng.shuffle(idxs)

    records: list[dict] = []
    n_saved = 0
    for idx in idxs:
        if len([r for r in records if r["metadata"]["probe"] == "answerable"]) >= args.n_answer:
            break
        if kept_idx is not None and idx not in kept_idx:
            continue
        if iscrowds[idx].as_py():
            continue
        query = _short_query(answers[idx].as_py())
        if not query:
            continue
        b = img_struct[idx].as_py()["bytes"]
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        box = _coco_to_rel_box(bboxes[idx].as_py(), w, h)
        if box is None:
            continue
        fname = f"refg_{idx}.jpg"
        fpath = img_dir / fname
        if not fpath.exists():
            im.save(fpath, "JPEG", quality=92)
        n_saved += 1
        apath = str(fpath.resolve())
        sid = f"refg-{idx}"

        img_path_str: str = img_struct[idx].as_py()["path"]
        coco_fn = Path(img_path_str).name
        coco_id_match = re.match(r"COCO_(train|val)(2014|2017)_(\d+)\.jpg", coco_fn)
        if not coco_id_match:
            raise SystemExit(
                f"[ERROR] 第 {idx} 行 image.path 格式不符合预期: {coco_fn!r}。"
                f" 期望 COCO_(train|val)(2014|2017)_(数字).jpg。fail-fast。"
            )
        coco_image_id = int(coco_id_match.group(3))
        question_id_val: int = int(tbl.column("question_id")[idx].as_py())

        records.append(
            _record(
                f"{sid}-ans",
                apath,
                query,
                "answer",
                box,
                None,
                "OOD_answer",
                "answerable",
                coco_image_id=coco_image_id,
                refcocog_row_index=idx,
                coco_file_name=coco_fn,
                question_id=question_id_val,
            )
        )
        absent = _absent_class(query, rng)
        records.append(
            _record(
                f"{sid}-absent",
                apath,
                absent,
                "abstain",
                None,
                "no_target",
                "OOD_absent",
                "absent",
                coco_image_id=coco_image_id,
                refcocog_row_index=idx,
                coco_file_name=coco_fn,
                question_id=question_id_val,
            )
        )

    by_p: dict[str, list] = defaultdict(list)
    for r in records:
        by_p[r["metadata"]["probe"]].append(r)
    for p in by_p.values():
        rng.shuffle(p)
    n_absent = len(by_p["absent"])
    n_ans = min(len(by_p["answerable"]), n_absent)
    records = by_p["answerable"][:n_ans] + by_p["absent"][:n_absent]
    rng.shuffle(records)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_probe: dict[str, int] = defaultdict(int)
    by_dec: dict[str, int] = defaultdict(int)
    for r in records:
        by_probe[r["metadata"]["probe"]] += 1
        by_dec[r["metadata"]["decision"]] += 1
    print(f"OOD RefCOCOg：{len(records)} 条 → {out}（图 {n_saved} 张存 {img_dir}）", flush=True)
    print(f"  探针: {dict(by_probe)}（answerable 有真 box 评 IoU）", flush=True)
    print(f"  决策: {dict(by_dec)}", flush=True)


if __name__ == "__main__":
    main()
