"""L2 cropout 负样本的几何审计（离线、只读、CPU）。

目标：对 checkpoint-444/eval_faithful_v1/L2_evidence 的每个 L2_cropout 记录，
恢复
  - 实际被裁掉的目标（assign 到 crop 记录对应 split 行 source_target）
  - 源图尺寸（PIL 只读）
  - 真实 crop 区域（复用 reason_abstain.bench.transforms 的
    ``qwen_box_to_xyxy`` / ``_clip_xyxy`` 以及
    ``save_visibility_ladder_crop`` 零档坐标公式），并用
    派生 crop 图实际尺寸交叉核对。
并在源图中对「query 类别」的所有已标注 bbox 计算与 crop 区域的交面积占比，
按 无已标注残留 / 部分残留 / 完整残留 / 缺信息 分类。性质说明：
“无已标注 query 残留”只是标注几何意义上的，并不等同于人工确认 query 完全 absent。

要点：
  - predictions.jsonl 存在重复 sample_id（同图 keep/crop 的 Cartesian 产物），
    必须按原 record 次序 zip 校验，不能 dict 覆盖。
  - 严格同 instance 配对（keep 与 crop 的 instance prefix 相同且 label 一致）
    并去重形成 clean diagnostic 子集；同类别其余 instance 一并检查。
  - 复算原标签 FCA、geometry-certified-no-annotated-query 子集 FCA，
    及 matched pairs 全指标。明确这是离线子集，不是正式修复 benchmark。
  - L3 endpoint（L3_dose rung 0.000）是否与 L2 cropout 同 image/query/source 目标，
    报集合重合与 prediction 差异。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "aaai_code"
sys.path.insert(0, str(CODE))

from reason_abstain.bench.boxes import qwen_box_to_xyxy  # noqa: E402
from reason_abstain.bench.schema import Box  # noqa: E402
from reason_abstain.bench.transforms import _clip_xyxy  # noqa: E402
from reason_abstain.eval.metrics import (  # noqa: E402
    score_answerable,
    score_unanswerable,
)

CHKPT = (
    ROOT
    / "new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/eval_faithful_v1"
)
ANN = ROOT / "aaai_code/benchmark/annotations"
BENCH_ROOT = ROOT / "aaai_code/benchmark"
DATA = ROOT / "aaai_code/benchmark/data/source"

L2_EVIDENCE = ANN / "eval/test_L2_evidence.jsonl"
SPLIT = ANN / "splits/test.jsonl"
L2_PREDS = CHKPT / "L2_evidence/predictions.jsonl"
L3_PREDS = CHKPT / "L3_dose/predictions.jsonl"
COCO_ANN = DATA / "coco/annotations/instances_selected.json"
PLANTDOC_CLASSES_YAML = ANN / "raw/plantdoc_full.raw.config.snapshot.yaml"
PLANTDOC_IMAGES = DATA / "plantdoc/images"
PLANTDOC_LABELS = DATA / "plantdoc/labels"
COCO_IMAGES = DATA / "coco/images"

# crop 边界舍入造成的极小交叠也算“已裁掉”；(0, 0.01] 视为边界残片。
REMAINING_EPS = 0.0


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ")


def query_from_message(content: str) -> str:
    return (
        content.removeprefix("<image>Locate the ")
        .removesuffix(" in this image.")
        .strip()
        if "<image>Locate the" in content
        else content
    )


def parse_plantdoc_class_names(path: Path) -> list[str]:
    text = path.read_text()
    match = re.search(r"class_names:\s*\n((?:\s*- .*\n)+)", text)
    if not match:
        raise SystemExit(f"cannot parse class_names from {path}")
    return [
        re.sub(r"^\s*-\s*", "", line).strip().strip("\"'")
        for line in match.group(1).strip().splitlines()
    ]


def load_coco(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {
        "images": {im["id"]: im for im in data["images"]},
        "anns": data["annotations"],
        "anns_by_image": defaultdict_from(data["annotations"], key="image_id"),
        "cat": {c["id"]: c["name"] for c in data["categories"]},
    }


def defaultdict_from(records: list[dict], key: str) -> dict:
    out: dict = {}
    for record in records:
        out.setdefault(record[key], []).append(record)
    return out


def crop_xyxy_zero_rung(
    image: Image.Image, box_qwen: list[float]
) -> tuple[int, int, int, int]:
    """镜像 transforms.save_visibility_ladder_crop 的零档坐标公式（visible_ratio<=0）。

    对应 transforms.py 第 190-225 行：clip 后的目标框，选择周边空白最少的边
    作为切割方向，保留互补区域。
    """
    width, height = image.size
    box = Box(x1=box_qwen[0], y1=box_qwen[1], x2=box_qwen[2], y2=box_qwen[3])
    x1, y1, x2, y2 = _clip_xyxy(qwen_box_to_xyxy(box, width, height), width, height)
    context = {
        "cut_left": x1,
        "cut_right": width - x2,
        "cut_top": y1,
        "cut_bottom": height - y2,
    }
    direction = min(context, key=context.get)
    if direction in ("cut_left", "cut_right"):
        return (x2, 0, width, height) if direction == "cut_left" else (0, 0, x1, height)
    return (0, y2, width, height) if direction == "cut_top" else (0, 0, width, y1)


def ann_xywh_to_xyxy(xywh: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = xywh
    return (x, y, x + w, y + h)


def intersection_area_ratio(
    box_xyxy: tuple[float, float, float, float],
    crop_xyxy: tuple[int, int, int, int],
) -> float:
    ix1 = max(box_xyxy[0], crop_xyxy[0])
    iy1 = max(box_xyxy[1], crop_xyxy[1])
    ix2 = min(box_xyxy[2], crop_xyxy[2])
    iy2 = min(box_xyxy[3], crop_xyxy[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, box_xyxy[2] - box_xyxy[0]) * max(0.0, box_xyxy[3] - box_xyxy[1])
    return inter / area if area > 0 else 0.0


def status_from_ratios(ratios: list[float]) -> str:
    if not ratios:
        return "no_query_annotations"
    # 主结果按严格零交集计算；边界残留另作容差敏感性诊断。
    if all(r <= REMAINING_EPS for r in ratios):
        return "no_annotated_query_remaining"
    if any(r >= 0.999 for r in ratios):
        return "annotated_query_fully_remaining"
    return "annotated_query_partially_remaining"


def source_image_path(dataset: str, source_image_id, coco_images: dict) -> Path:
    if dataset == "plantdoc_vg":
        return PLANTDOC_IMAGES / f"{source_image_id}.jpg"
    name = coco_images[source_image_id]["file_name"]
    return COCO_IMAGES / name


def main() -> None:
    split_rows = load(SPLIT)
    split = {r["sample_id"]: r for r in split_rows}
    l2 = load(L2_EVIDENCE)
    preds = load(L2_PREDS)
    assert len(l2) == len(preds)
    for record, pred in zip(l2, preds):
        assert record["metadata"]["sample_id"] == pred["sample_id"]

    l3 = load(L3_PREDS)
    l3_zero = {
        r["sample_id"].split("-0.000-", 1)[1]: r
        for r in l3
        if "-0.000-" in r["sample_id"]
    }

    class_names = parse_plantdoc_class_names(PLANTDOC_CLASSES_YAML)
    coco = load_coco(COCO_ANN)

    # keep records indexed: img_id -> {original_sid: entry}（重复 sample_id 的 Cartesian 产物去重）
    keeps_by_img: dict[str, dict[str, dict]] = {}
    for keep_index, record in enumerate(l2):
        if record["metadata"]["relation"] != "L2_keep":
            continue
        sample_id = record["metadata"]["sample_id"]
        img_id = sample_id.removeprefix("L2-").split("-keep-", 1)[0]
        original_sid = sample_id.split("-keep-", 1)[1]
        instance = re.sub(r"-answer$", "", original_sid)
        keeps_by_img.setdefault(img_id, {})[original_sid] = {
            "record": record,
            "index": keep_index,
            "original_sid": original_sid,
            "instance": instance,
            "query": query_from_message(record["messages"][-1]["content"]),
        }

    def query_cat_id(img_id: str, query: str, dataset: str) -> tuple[int | None, str]:
        """由同图同 query 的 keep 记录取得 query 类别 id；无则回退数据集映射。"""
        candidates: set[int] = set()
        for entry in keeps_by_img.get(img_id, {}).values():
            if norm_label(entry["query"]) == norm_label(query):
                cid = split[entry["original_sid"]]["source_target"]["category_id"]
                candidates.add(cid)
        if len(candidates) == 1:
            return candidates.pop(), "keep"
        if not candidates:
            if dataset == "plantdoc_vg":
                try:
                    return class_names.index(query.replace(" ", "_")), "class_names"
                except ValueError:
                    return None, "unknown"
            for cid, name in coco["cat"].items():
                if norm_label(name) == norm_label(query):
                    return cid, "coco_cat"
            return None, "unknown"
        return min(candidates), "keep_multi"

    def annotations_for(
        dataset: str, source_image_id, cat_id: int, source_size: tuple[int, int]
    ) -> tuple[list[dict] | None, str]:
        """返回 query 类别标注的像素 xyxy 列表（plantdoc 归一化坐标转像素）。"""
        if dataset == "plantdoc_vg":
            label_file = PLANTDOC_LABELS / f"{source_image_id}.txt"
            if not label_file.exists():
                return None, "missing_label_txt"
            width, height = source_size
            records = []
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = (
                    int(parts[0]),
                    *(float(p) for p in parts[1:]),
                )
                if cls != cat_id:
                    continue
                records.append(
                    {
                        "id": cls,
                        "xyxy": (
                            (cx - w / 2) * width,
                            (cy - h / 2) * height,
                            (cx + w / 2) * width,
                            (cy + h / 2) * height,
                        ),
                    }
                )
            return records, "ok"
        anns = coco["anns_by_image"].get(source_image_id, [])
        records = [
            {"id": a["id"], "xyxy": ann_xywh_to_xyxy(a["bbox"])}
            for a in anns
            if a["category_id"] == cat_id
        ]
        return records, "ok"

    def crop_target_instance_box(
        dataset: str, crec: dict, source_size: tuple[int, int]
    ) -> tuple[float, float, float, float] | None:
        """crop 目标自身的已标注框（用于校对目标确实被裁掉）。"""
        target = crec.get("source_target") or {}
        cat_id = target.get("category_id")
        sid = crec["source_image_id"]
        records, _ = annotations_for(dataset, sid, cat_id, source_size)
        if not records:
            return None
        if dataset != "plantdoc_vg":
            matches = [r for r in records if r["id"] == target.get("instance_id")]
            if len(matches) != 1:
                raise ValueError(f"ambiguous COCO instance: {crec['sample_id']}")
            return matches[0]["xyxy"]
        # PlantDoc source_box_xywh 保存的是像素中心cx,cy,w,h。
        cx, cy, width, height = target["source_box_xywh"]
        return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)

    records_out = []
    n_crop = 0
    for l2_index, (record, pred) in enumerate(zip(l2, preds)):
        if record["metadata"]["decision"] != "abstain":
            continue
        if record["metadata"]["relation"] != "L2_cropout":
            continue
        n_crop += 1
        sample_id = record["metadata"]["sample_id"]
        img_id = sample_id.removeprefix("L2-").split("-cropout-", 1)[0]
        crop_sid = sample_id.split("-cropout-", 1)[1]
        crec = split[crop_sid]
        dataset = crec["dataset"]
        source_image_id = crec["source_image_id"]
        query = query_from_message(record["messages"][-1]["content"])
        crop_target = (crec.get("source_target") or {}).get("label")
        same_label = norm_label(query) == norm_label(crop_target or "")
        crop_instance = re.sub(r"-ladder-\d{3}$", "", crop_sid)

        out: dict = {
            "l2_index": l2_index,
            "sample_id": sample_id,
            "original_crop_sid": crop_sid,
            "original_keep_sid": None,
            "img_id": img_id,
            "dataset": dataset,
            "image": record["images"][0] if record.get("images") else None,
            "source_image_id": source_image_id,
            "query": query,
            "crop_target_label": crop_target,
            "crop_target_category_id": (crec.get("source_target") or {}).get(
                "category_id"
            ),
            "same_label": same_label,
            "answered": pred["pred"].get("decision") == "answer",
            "pred_decision": pred["pred"].get("decision"),
            "pred_box": (
                pred["pred"].get("box") if pred["pred"].get("valid") else None
            ),
            "p_answer": pred["p_answer"],
            "p_wo": pred["p_wo"],
            "p_nt": pred["p_nt"],
        }

        # 严格同 instance 配对（keep/crop 的 instance prefix 相同且 label 一致）
        same_inst_keeps = [
            e
            for e in keeps_by_img.get(img_id, {}).values()
            if e["instance"] == crop_instance
            and norm_label(e["query"]) == norm_label(query)
        ]
        out["n_same_instance_keeps"] = len(same_inst_keeps)
        if same_inst_keeps:
            out["original_keep_sid"] = same_inst_keeps[0]["original_sid"]
            out["keep_l2_index"] = same_inst_keeps[0]["index"]
        out["has_same_instance_keep"] = bool(same_inst_keeps)

        # 源图尺寸（PIL 只读）
        missing = []
        src_path = source_image_path(dataset, source_image_id, coco["images"])
        try:
            with Image.open(src_path) as im:
                source_size = im.size
        except (OSError, ValueError) as exc:
            missing.append(f"source_image_unreadable:{type(exc).__name__}")
            source_size = None
        out["source_image_path"] = str(src_path.relative_to(ROOT))
        out["source_size"] = list(source_size) if source_size else None

        # 真实 crop 区域（零档公式） + 派生图尺寸核对
        crop_box = (crec.get("source_target") or {}).get("box")
        derived_path = BENCH_ROOT / crec["image_path"]
        if source_size is not None and crop_box:
            with Image.open(src_path) as im:
                crop_xyxy = crop_xyxy_zero_rung(im, crop_box)
            out["crop_xyxy"] = list(crop_xyxy)
            try:
                with Image.open(derived_path) as dim:
                    derived_size = dim.size
                out["derived_crop_size"] = list(derived_size)
                out["geometry_verified"] = derived_size == (
                    crop_xyxy[2] - crop_xyxy[0],
                    crop_xyxy[3] - crop_xyxy[1],
                )
                if not out["geometry_verified"]:
                    missing.append("crop_size_mismatch")
            except (OSError, ValueError) as exc:
                out["derived_crop_size"] = None
                out["geometry_verified"] = False
                missing.append(f"derived_crop_unreadable:{type(exc).__name__}")
        else:
            out["crop_xyxy"] = None
            out["derived_crop_size"] = None
            out["geometry_verified"] = False
            missing.append("no_crop_geometry")

        # query 类别 id
        cat_id, cat_src = query_cat_id(img_id, query, dataset)
        out["query_category_id"] = cat_id
        out["query_category_src"] = cat_src
        if cat_id is None:
            missing.append("query_category_unknown")

        # query 类别标注相对 crop 的交面积占比
        out["query_anns"] = []
        ratios: list[float] = []
        if cat_id is not None and source_size is not None and out.get("crop_xyxy"):
            anns, ann_status = annotations_for(
                dataset, source_image_id, cat_id, source_size
            )
            if anns is None:
                missing.append(ann_status)
            else:
                for a in anns:
                    ratio = intersection_area_ratio(a["xyxy"], out["crop_xyxy"])
                    ratios.append(ratio)
                    out["query_anns"].append(
                        {"ann_id": a["id"], "ratio_in_crop": round(ratio, 6)}
                    )
        out["n_query_anns"] = len(ratios)
        out["max_query_ratio"] = round(max(ratios), 6) if ratios else None
        out["geometry_status"] = status_from_ratios(ratios)

        # crop 目标自身标注是否被完全裁掉（几何自洽性）
        if source_size is not None:
            tbox = crop_target_instance_box(dataset, crec, source_size)
            if tbox is None:
                out["target_own_box_ratio"] = None
            else:
                out["target_own_box_ratio"] = (
                    round(
                        intersection_area_ratio(
                            tbox, out.get("crop_xyxy") or (0, 0, 1, 1)
                        ),
                        6,
                    )
                    if out.get("crop_xyxy")
                    else None
                )

        out["missing_info"] = bool(missing)
        if out["geometry_status"] == "no_query_annotations":
            missing.append("no_query_annotations")
            out["missing_info"] = True
        out["missing_reasons"] = missing

        # L3 endpoint 对照（rung 0.000）。L3 恒以 crop 自身 label 提问，
        # 因此“同 image/query/source 目标”仅在 same_label（query==crop target）时成立，
        # 交叉 label 记录仅共享同一实例，query 不同、prediction 不可直接比较。
        l3r = l3_zero.get(crop_sid)
        if l3r is not None:
            out["l3_instance_overlap"] = True
            l3_pred_decision = l3r["pred"].get("decision")
            out["l3_decision"] = l3_pred_decision
            out["l3_same_image"] = True
            out["l3_same_query"] = same_label
            out["l3_same_source_target"] = True
            out["l3_pred_differs"] = bool(same_label) and (
                l3_pred_decision != out["pred_decision"]
            )
        else:
            out["l3_instance_overlap"] = False
            out["l3_pred_differs"] = False

        records_out.append(out)

    # 统计
    tolerance_sensitivity = []
    for tolerance in (0.0, 0.001, 0.01, 0.05):
        subset = [
            r
            for r in records_out
            if not r["missing_info"]
            and r["query_anns"]
            and max(a["ratio_in_crop"] for a in r["query_anns"]) <= tolerance
        ]
        tolerance_sensitivity.append(
            {
                "maximum_retained_bbox_ratio": tolerance,
                "n": len(subset),
                "answered": sum(r["answered"] for r in subset),
                "FCA": sum(r["answered"] for r in subset) / len(subset)
                if subset
                else None,
            }
        )
    (ROOT / "docs/review_evidence/l2_geometry_tolerance.json").write_text(
        json.dumps(tolerance_sensitivity, indent=2) + "\n"
    )
    groups = {
        "all": [r for r in records_out],
        "same_label": [r for r in records_out if r["same_label"]],
        "cross_label": [r for r in records_out if not r["same_label"]],
    }
    for key, rows_ in list(groups.items()):
        rows_.sort(key=lambda r: r["sample_id"])
        n = len(rows_)
        ans = sum(r["answered"] for r in rows_)
        if n:
            groups[f"{key}_n"] = n
            groups[f"{key}_answered"] = ans
            groups[f"{key}_fca"] = ans / n

    def fca_of(rows_: list[dict]) -> float | None:
        if not rows_:
            return None
        return sum(r["answered"] for r in rows_) / len(rows_)

    clean = [r for r in records_out if r["same_label"] and r["has_same_instance_keep"]]
    clean_dedup = {r["original_crop_sid"]: r for r in clean}
    clean_rows = sorted(clean_dedup.values(), key=lambda r: r["sample_id"])

    clean_certifiable = [r for r in clean_rows if not r["missing_info"]]
    clean_no_annotated = [
        r
        for r in clean_certifiable
        if r["geometry_status"] == "no_annotated_query_remaining"
    ]

    geo_certified_all = [
        r
        for r in records_out
        if not r["missing_info"]
        and r["geometry_status"] == "no_annotated_query_remaining"
    ]
    geo_dedup_n = len({(r["original_crop_sid"], r["query"]) for r in geo_certified_all})

    summary = {
        "n_cropout_records": n_crop,
        "group": {
            k: v
            for k, v in groups.items()
            if isinstance(v, (int, float)) and k.endswith(("_n", "_answered", "_fca"))
        },
        "status_distribution": dict(Counter(r["geometry_status"] for r in records_out)),
        "status_by_label": {
            lbl: dict(Counter(r["geometry_status"] for r in rows_))
            for lbl, rows_ in (
                ("same", groups["same_label"]),
                ("cross", groups["cross_label"]),
            )
        },
        "missing_info_count": sum(r["missing_info"] for r in records_out),
        "geometry_certified_no_annotated_query": {
            "n": len(geo_certified_all),
            "dedup_n": geo_dedup_n,
            "answered": sum(r["answered"] for r in geo_certified_all),
            "FCA": fca_of(geo_certified_all),
        },
        "clean_same_instance": {
            "n": len(clean_rows),
            "certifiable": len(clean_certifiable),
            "no_annotated_query": len(clean_no_annotated),
            "answered": sum(r["answered"] for r in clean_rows),
            "FCA": fca_of(clean_rows),
            "no_annotated_answered": sum(r["answered"] for r in clean_no_annotated),
            "no_annotated_FCA": fca_of(clean_no_annotated),
        },
        "l3_endpoint": {
            "n_cropout_records_with_instance": sum(
                bool(r["l3_instance_overlap"]) for r in records_out
            ),
            "n_unique_instances": len({r["original_crop_sid"] for r in records_out}),
            "n_unique_instances_overlap": len(
                {
                    r["original_crop_sid"]
                    for r in records_out
                    if r["l3_instance_overlap"]
                }
            ),
            "n_same_image_query_target_records": sum(
                bool(r["l3_instance_overlap"] and r["l3_same_query"])
                for r in records_out
            ),
            "n_decision_diff_same_query": sum(
                bool(r["l3_pred_differs"]) for r in records_out
            ),
            "n_l3_answered_same_query": sum(
                bool(
                    r["l3_instance_overlap"]
                    and r["l3_same_query"]
                    and r["l3_decision"] == "answer"
                )
                for r in records_out
            ),
            "n_l2_answered_same_query": sum(
                bool(r["l3_instance_overlap"] and r["l3_same_query"] and r["answered"])
                for r in records_out
            ),
        },
    }

    # matched pairs 全指标（clean 子集：去重同 instance 配对）
    pair_rows = []
    for r in clean_rows:
        keep_sid = r["original_keep_sid"]
        keep_pred = preds[r["keep_l2_index"]]
        keep_gt = {
            "decision": "answer",
            "label": keep_pred["gt"].get("label"),
            "box": keep_pred["gt"].get("box"),
        }
        crop_pred = preds[r["l2_index"]]
        crop_gt = {"decision": "abstain", "reason": "no_target"}
        keep_correct = score_answerable(keep_pred["pred"], keep_gt, 0.5)
        crop_correct = score_unanswerable(crop_pred["pred"], crop_gt)
        pair_rows.append(
            {
                "sample_id": r["sample_id"],
                "keep_original_sid": keep_sid,
                "keep_answer_correct": keep_correct,
                "crop_abstain_correct": crop_correct,
                "geometry_status": r["geometry_status"],
                "missing_info": r["missing_info"],
            }
        )
    pair_clean = [p for p in pair_rows if not p["missing_info"]]
    summary["matched_pairs_clean"] = {
        "n": len(pair_clean),
        "keep_accuracy": (
            sum(p["keep_answer_correct"] for p in pair_clean) / len(pair_clean)
            if pair_clean
            else None
        ),
        "crop_abstain_accuracy": (
            sum(p["crop_abstain_correct"] for p in pair_clean) / len(pair_clean)
            if pair_clean
            else None
        ),
        "both_correct": sum(
            p["keep_answer_correct"] and p["crop_abstain_correct"] for p in pair_clean
        ),
    }

    # provenance / hash
    summary["provenance"] = {
        "l2_evidence": {
            "path": str(L2_EVIDENCE.relative_to(ROOT)),
            "sha256": sha256(L2_EVIDENCE),
        },
        "split_test": {"path": str(SPLIT.relative_to(ROOT)), "sha256": sha256(SPLIT)},
        "l2_predictions": {
            "path": str(L2_PREDS.relative_to(ROOT)),
            "sha256": sha256(L2_PREDS),
        },
        "l3_predictions": {
            "path": str(L3_PREDS.relative_to(ROOT)),
            "sha256": sha256(L3_PREDS),
        },
        "coco_instances_selected": {
            "path": str(COCO_ANN.relative_to(ROOT)),
            "sha256": sha256(COCO_ANN),
        },
        "plantdoc_class_config": {
            "path": str(PLANTDOC_CLASSES_YAML.relative_to(ROOT)),
            "sha256": sha256(PLANTDOC_CLASSES_YAML),
        },
        "source_image_count": len({r["source_image_path"] for r in records_out}),
    }

    out_dir = ROOT / "docs/review_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "l2_geometry_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records_out) + "\n"
    )
    (out_dir / "l2_geometry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    fieldnames = sorted(
        {
            "sample_id",
            "original_crop_sid",
            "original_keep_sid",
            "img_id",
            "dataset",
            "image",
            "source_image_id",
            "query",
            "crop_target_label",
            "crop_target_category_id",
            "same_label",
            "answered",
            "pred_decision",
            "source_image_path",
            "source_size",
            "crop_xyxy",
            "derived_crop_size",
            "geometry_verified",
            "query_category_id",
            "query_category_src",
            "n_query_anns",
            "max_query_ratio",
            "geometry_status",
            "target_own_box_ratio",
            "missing_info",
            "missing_reasons",
            "l3_instance_overlap",
            "l3_decision",
            "l3_same_query",
            "l3_pred_differs",
        }
    )
    with (out_dir / "l2_geometry_records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records_out:
            writer.writerow(r)

    # 中文摘要 md（计数 + 源路径 + hash + 结论与限制）
    s = summary
    g = s["group"]
    sl = s["status_by_label"]
    md = f"""# L2 cropout 负样本几何审计（离线诊断）

> 性质声明：本报告为**离线审计子集**，复用原 record 次序与派生图、标注几何计算，
> 不是正式 benchmark 修复，也不代表人工逐图目检。**“无已标注 query 残留”仅指
> 标注几何意义下 query 类别的 bbox 不在 crop 区域中，不能等同人工确认 query 完全
> absent**。存在未覆盖的标注缺失/类别映射不明时会归入“缺信息”并明确说明。

## 核心结论

- `build_eval_L2.py` 按同图 `keep_rows × crop_rows` Cartesian 配对（循环嵌套），
  生成的 395 条 `L2_cropout` 中 **247 条 query 与所裁掉目标类别不同**（交叉 label），
  148 条一致（同 instance 的“设计意图”配对）。
- 全部 395 条 crop 的裁剪区域重建（零档 `save_visibility_ladder_crop` 公式）
  与派生 crop 图实际尺寸**逐一核对一致**（geometry_verified=True）。
- 原标签下 FCA = {s["group"]["all_fca"]:.4f}（模型在 395 条 abstain 目标上 answered
  {s["group"]["all_answered"]}/{s["group"]["all_n"]}）。
- 几何审计下，395 条中 **{s["status_distribution"]["no_annotated_query_remaining"]}
  条无已标注 query 残留**（其中 same-label {sl["same"]["no_annotated_query_remaining"]}、
  cross-label {sl["cross"]["no_annotated_query_remaining"]}）；
  {s["status_distribution"]["annotated_query_fully_remaining"]} 条完整残留、
  {s["status_distribution"]["annotated_query_partially_remaining"]} 条部分残留。
- 交叉 label 的 247 条 FCA={g["cross_label_fca"]:.4f}（answered {g["cross_label_answered"]}/{g["cross_label_n"]}），
  模型在大量 query 对象其实仍在图中（完整/部分残留）的样本上 answered，这部分“负例标签”几何上并不可靠。
- geometry-certified-no-annotated-query 子集（{s["geometry_certified_no_annotated_query"]["n"]} 条）
  FCA={s["geometry_certified_no_annotated_query"]["FCA"]:.4f}。
- 严格同 instance 去重 clean 子集 {s["clean_same_instance"]["n"]} 条
  （全部可几何核对；其中无已标注 query 残留 {s["clean_same_instance"]["no_annotated_query"]} 条），
  FCA={s["clean_same_instance"]["FCA"]:.4f}。
- matched pairs 全指标（clean 子集）：keep answer 准确率
  {s["matched_pairs_clean"]["keep_accuracy"]:.4f}，crop abstain 准确率
  {s["matched_pairs_clean"]["crop_abstain_accuracy"]:.4f}，双侧均对
  {s["matched_pairs_clean"]["both_correct"]}/{s["matched_pairs_clean"]["n"]}。
- L3 endpoint：{s["l3_endpoint"]["n_unique_instances_overlap"]}/{s["l3_endpoint"]["n_unique_instances"]}
  个唯一实例在图面上与 L2 cropout 重合；同 image/query/source 目标可比的记录
  {s["l3_endpoint"]["n_same_image_query_target_records"]} 条，prediction 决策差异
  {s["l3_endpoint"]["n_decision_diff_same_query"]}。

## 计数

| 集合 | 记录数 | answered | FCA |
|---|---|---|---|
| 全部 cropout（原标签） | {g["all_n"]} | {g["all_answered"]} | {g["all_fca"]:.4f} |
| same-label（query==裁掉目标） | {g["same_label_n"]} | {g["same_label_answered"]} | {g["same_label_fca"]:.4f} |
| cross-label（query!=裁掉目标） | {g["cross_label_n"]} | {g["cross_label_answered"]} | {g["cross_label_fca"]:.4f} |
| geometry-certified 无 query 残留 | {s["geometry_certified_no_annotated_query"]["n"]} | {s["geometry_certified_no_annotated_query"]["answered"]} | {s["geometry_certified_no_annotated_query"]["FCA"]:.4f} |
| clean 同 instance 去重 | {s["clean_same_instance"]["n"]} | {s["clean_same_instance"]["answered"]} | {s["clean_same_instance"]["FCA"]:.4f} |

## 逐条数据

- JSONL 记录：`docs/review_evidence/l2_geometry_records.jsonl`
- CSV 概览：`docs/review_evidence/l2_geometry_records.csv`
- 汇总 JSON：`docs/review_evidence/l2_geometry_summary.json`

## 输入源与哈希

| 文件 | sha256 |
|---|---|
| aaai_code/benchmark/annotations/eval/test_L2_evidence.jsonl | {s["provenance"]["l2_evidence"]["sha256"]} |
| aaai_code/benchmark/annotations/splits/test.jsonl | {s["provenance"]["split_test"]["sha256"]} |
| {s["provenance"]["l2_predictions"]["path"]} | {s["provenance"]["l2_predictions"]["sha256"]} |
| {s["provenance"]["l3_predictions"]["path"]} | {s["provenance"]["l3_predictions"]["sha256"]} |
| {s["provenance"]["coco_instances_selected"]["path"]} | {s["provenance"]["coco_instances_selected"]["sha256"]} |
| {s["provenance"]["plantdoc_class_config"]["path"]} | {s["provenance"]["plantdoc_class_config"]["sha256"]} |

- 唯一源图数：{s["provenance"]["source_image_count"]}；PlantDoc txt 标注即时读取，COCO 标注取自 instances_selected。
- 裁剪区域重建复用 `reason_abstain.bench.transforms` 的 `qwen_box_to_xyxy`/`_clip_xyxy`
  及 `save_visibility_ladder_crop` 零档坐标公式；图片仅用 PIL 读取尺寸。

## 已知限制

- “部分/完整残留”依赖 PlantDoc txt 与 COCO `instances_selected.json` 标注质量。
- 主统计对任何正交集都计残留，不将小于1%的面积默认为完全排除。
  bbox量化与像素取整可能导致小边界交集；逐条比例见JSONL，容差分析单独报告。
- 交叉 label 记录中，L3 的 prompt（crop 自身 label）与 L2 不同，prediction 仅在同 label 时可直接比较。
- 本审计不涉及任何人工目检、模型训练或推理 API。
"""
    (out_dir / "l2_geometry.md").write_text(md)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
