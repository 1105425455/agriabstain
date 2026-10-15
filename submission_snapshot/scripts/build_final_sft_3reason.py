#!/usr/bin/env python
"""Build the SFT training mixture for the three-state discriminative head.

Record format:
  - image: ["<absolute path>"]
  - system: DECISION_SYSTEM_PROMPT
  - decision_label: 1 (answer) / 0 (abstain), retained from the two-way format
  - reason_label: 0 (wrong_object) / 1 (no_target) / -1 (answer), retained
  - head_label: 0 (answer) / 1 (wrong_object) / 2 (no_target), the three-state field
  - conversations: [{"from": "human", "value": "<image>{query}"},
                    {"from": "gpt", "value": "<compact JSON>"}]

The assistant side takes one of three forms:
  - answer:                    {"decision":"answer","label":"X","box":[x1,y1,x2,y2]}
  - abstain, near-miss:        {"decision":"abstain","reason":"wrong_object"}
  - abstain, target absent:    {"decision":"abstain","reason":"no_target"}

head_label mapping:
  - answer                    -> 0
  - abstain + wrong_object     -> 1
  - abstain + no_target        -> 2

The reason is always read from the `reason` field of the source record and never assigned by
hand. Degradation samples, marked `dose_only`, never enter training: they are absent from
`train.jsonl` to begin with and are skipped again here.

Sources:
  - training: benchmark/annotations/splits/train.jsonl only
  - test conversion: benchmark/annotations/splits/test.jsonl, rewritten into evaluation format

Mixture, at seed 42:
  1. paired discrimination on one image: real disease answer plus near-miss abstain
     (wrong_object), covering every T6 variant
  2. paired evidence on one image: fully visible answer plus cropped-to-0% abstain
     (no_target)
  3. answer backgrounds: 100 each from T2, T3 and T7_include
  4. shortcut control: 220 T4_control records, where a degraded background must not trigger
     abstention
  5. additional no_target: 150 cross-crop T5 records and 150 occluded twins
  6. general coverage: 1000 COCO answer and 700 COCO no_target records

The resulting head_label distribution is {0: 2403, 1: 973, 2: 2481}; the composition is
reported in the paper.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IS_PACKAGE = (PROJECT_ROOT / "benchmark").is_dir()
REPO = PROJECT_ROOT
if IS_PACKAGE:
    DEFAULT_PREFIX = "benchmark/annotations"
else:
    # source mode fallback
    REPO = PROJECT_ROOT.parent
    DEFAULT_PREFIX = "new_reason/outputs"

VALID_REASONS = {"wrong_object", "no_target"}

DECISION_SYSTEM_PROMPT = (
    "You are a vision-language grounding assistant for high-stakes plant disease diagnosis. "
    "Locate the queried target. If the queried target is present and localizable, return "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]} with box in [0,1000] relative coords. '
    "Otherwise abstain: "
    '{"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image, or cropped out of view)."
)


def clean_label(r: dict) -> str:
    """Resolve the category name: prefer `label`, then `mismatch_label` for T6 and `absent_label`
    for T5, falling back to `source_target` or the query.
    """
    md = r.get("metadata", {})
    if md.get("mismatch_label"):
        return md["mismatch_label"]
    if md.get("absent_label"):
        return md["absent_label"]
    if r.get("label"):
        return r["label"]
    st = r.get("source_target") or {}
    if st.get("label"):
        return st["label"]
    s = (r.get("query", "") or "").strip()
    for pre in ["Locate the ", "Find the ", "Where is the ", "Where is "]:
        if s.startswith(pre):
            s = s[len(pre) :]
            break
    for suf in [" in this image.", " and return its bounding box.", " in the image?", "?", "."]:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s.strip()


def make_answer_record(r: dict) -> dict:
    """Answer record: decision_label=1, head_label=0, a box is required. The query keeps the
    phrasing of the source record.
    """
    box = r.get("box")
    assert box is not None and len(box) == 4, f"answer 必须有 box: {r.get('sample_id')}"
    label = r.get("label") or clean_label(r)
    aj = json.dumps(
        {"decision": "answer", "label": label, "box": box},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "image": [r["image_path"]],
        "system": DECISION_SYSTEM_PROMPT,
        "decision_label": 1,
        "reason_label": -1,
        "head_label": 0,
        "conversations": [
            {"from": "human", "value": f"<image>{r['query']}"},
            {"from": "gpt", "value": aj},
        ],
    }


def make_abstain_record(r: dict) -> dict:
    """Abstention record: decision_label=0, head_label 1 or 2, with the reason read from the source
    record rather than assigned here. The query keeps the phrasing of the source record.
    """
    reason = r.get("reason")
    assert reason in VALID_REASONS, f"abstain reason 非法({reason}): {r.get('sample_id')}"
    assert not r.get("metadata", {}).get("dose_only"), (
        f"dose_only 样本不该进训练: {r.get('sample_id')}"
    )
    aj = json.dumps(
        {"decision": "abstain", "reason": reason}, ensure_ascii=False, separators=(",", ":")
    )
    reason_label = 0 if reason == "wrong_object" else 1
    head_label = 1 if reason == "wrong_object" else 2
    return {
        "image": [r["image_path"]],
        "system": DECISION_SYSTEM_PROMPT,
        "decision_label": 0,
        "reason_label": reason_label,
        "head_label": head_label,
        "conversations": [
            {"from": "human", "value": f"<image>{r['query']}"},
            {"from": "gpt", "value": aj},
        ],
    }


def make_test_record(row: dict) -> dict:
    """Rewrite a `test.jsonl` record into the evaluation format: messages, images, solution, metadata.

    """
    md = row.get("metadata", {})
    decision = row.get("decision", "answer")
    label = row.get("label") or clean_label(row)
    box = row.get("box")
    reason = row.get("reason")

    messages = [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>{row['query']}"},
    ]
    solution = {
        "decision": decision,
        "label": label if decision == "answer" else None,
        "box": box if decision == "answer" else None,
        "reason": reason if decision == "abstain" else None,
    }
    test_metadata = {
        "decision": decision,
        "label": label if decision == "answer" else None,
        "box": box if decision == "answer" else None,
        "reason": reason if decision == "abstain" else None,
        "relation": md.get("relation"),
        "achieved_visible_ratio": md.get("achieved_visible_ratio"),
        "target_visible_ratio": md.get("target_visible_ratio"),
        "dose_only": md.get("dose_only", False),
        "sample_id": row.get("sample_id"),
    }
    return {
        "messages": messages,
        "images": [row["image_path"]],
        "solution": solution,
        "metadata": test_metadata,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"{DEFAULT_PREFIX}/splits/train.jsonl")
    ap.add_argument("--test", default=f"{DEFAULT_PREFIX}/splits/test.jsonl")
    ap.add_argument("--dev", default=f"{DEFAULT_PREFIX}/splits/dev.jsonl")
    output_dir = "benchmark/annotations/sft" if IS_PACKAGE else "new_reason/outputs/sft_3reason"
    ap.add_argument("--out-train", default=f"{output_dir}/sft_train.json")
    ap.add_argument("--out-test", default=f"{output_dir}/sft_test.jsonl")
    ap.add_argument("--out-dev", default=f"{output_dir}/sft_dev.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-t2", type=int, default=100)
    ap.add_argument("--n-t3", type=int, default=100)
    ap.add_argument("--n-t7inc", type=int, default=100)
    ap.add_argument("--n-t4control", type=int, default=220)
    ap.add_argument("--n-t5", type=int, default=1500)
    ap.add_argument("--n-occluded", type=int, default=200)
    ap.add_argument("--n-coco-ans", type=int, default=1100)
    ap.add_argument("--n-coco-abs", type=int, default=500)
    ap.add_argument(
        "--wo-imbalance-ratio",
        type=float,
        default=999,
        help="Cap on the wrong_object imbalance ratio (default 999 = no cap, keep all)",
    )
    ap.add_argument(
        "--n-evidence-crop0", type=int, default=669, help="Number of evidence samples cropped to 0%% visibility, keeping all (default 669)"
    )
    ap.add_argument(
        "--n-evidence-mid",
        type=int,
        default=300,
        help="Mid-range answer samples (achieved_visible_ratio 0.4-0.8, default 300)",
    )
    ap.add_argument(
        "--wo-oversample-target",
        type=int,
        default=1500,
        help="Target total count for wrong_object oversampling (default 1500)",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(line) for line in open(REPO / args.train)]
    test_rows = [json.loads(line) for line in open(REPO / args.test)]

    def is_coco(r):
        return "coco" in (r.get("dataset", "") or "").lower()

    def rel(r):
        return r.get("metadata", {}).get("relation")

    rows = [r for r in rows if not r.get("metadata", {}).get("dose_only")]

    out_train: list[dict] = []
    counts: dict = defaultdict(int)

    by_img = defaultdict(lambda: {"real": [], "t6": []})
    for r in rows:
        if is_coco(r):
            continue
        if rel(r) == "base" and r["decision"] == "answer" and r.get("box"):
            by_img[r["source_image_id"]]["real"].append(r)
        elif rel(r) == "T6":
            by_img[r["source_image_id"]]["t6"].append(r)
    paired = {img: d for img, d in by_img.items() if d["real"] and d["t6"]}
    print(f"① judge 配对图数: {len(paired)}", flush=True)

    crop_real_count = Counter()
    crop_t6_samples = defaultdict(list)

    for img, d in paired.items():
        real_r = d["real"][0]
        crop = clean_label(real_r).lower().split()[0]
        crop_real_count[crop] += 1
        for t6 in d["t6"]:
            crop_t6_samples[crop].append((img, t6))

    crop_wo_quota = {}
    for crop, real_cnt in crop_real_count.items():
        quota = int(real_cnt * args.wo_imbalance_ratio)
        crop_wo_quota[crop] = quota

    print(f"  作物分布: real={dict(crop_real_count)}, wo配额={crop_wo_quota}", flush=True)

    for _img, d in paired.items():
        real_r = d["real"][0]
        out_train.append(make_answer_record(real_r))
        counts["judge_real"] += 1

    crop_t6_indices = {crop: 0 for crop in crop_t6_samples}
    crop_t6_selected_count = Counter()

    for crop in crop_t6_samples:
        rng.shuffle(crop_t6_samples[crop])

    while True:
        advanced = False
        for crop in sorted(crop_t6_samples.keys()):
            if (
                crop_t6_indices[crop] < len(crop_t6_samples[crop])
                and crop_t6_selected_count[crop] < crop_wo_quota[crop]
            ):
                img, t6 = crop_t6_samples[crop][crop_t6_indices[crop]]
                crop_t6_indices[crop] += 1
                crop_t6_selected_count[crop] += 1
                out_train.append(make_abstain_record(t6))
                counts["judge_wrong_object"] += 1
                advanced = True
        if not advanced:
            break

    print(
        f"  wrong_object 实际选取: {dict(crop_t6_selected_count)} (配额={crop_wo_quota})",
        flush=True,
    )

    paired_img_ids = set(paired.keys())
    isolated_real = [
        r
        for r in rows
        if not is_coco(r)
        and rel(r) == "base"
        and r["decision"] == "answer"
        and r.get("box")
        and r["source_image_id"] not in paired_img_ids
    ]
    print(f"② judge_real_extra 孤立真病 answer 数: {len(isolated_real)}", flush=True)
    for r in isolated_real:
        out_train.append(make_answer_record(r))
        counts["judge_real_extra"] += 1

    by_img2 = defaultdict(lambda: {"vis": [], "crop0": []})
    for r in rows:
        if is_coco(r):
            continue
        if rel(r) == "base" and r["decision"] == "answer" and r.get("box"):
            by_img2[r["source_image_id"]]["vis"].append(r)
        elif (
            rel(r) == "T7_ladder"
            and r.get("metadata", {}).get("achieved_visible_ratio", 1) <= 0.001
        ):
            by_img2[r["source_image_id"]]["crop0"].append(r)
    ev_paired = {img: d for img, d in by_img2.items() if d["vis"] and d["crop0"]}
    print(f"② evidence 配对图数 (非COCO): {len(ev_paired)}", flush=True)
    for img, d in ev_paired.items():
        out_train.append(make_answer_record(d["vis"][0]))
        counts["evidence_visible"] += 1
        out_train.append(make_abstain_record(d["crop0"][0]))
        counts["evidence_no_target"] += 1

    all_crop0 = [
        r
        for r in rows
        if rel(r) == "T7_ladder" and r.get("metadata", {}).get("achieved_visible_ratio", 1) <= 0.001
    ]
    paired_crop0_ids = set()
    for img, d in ev_paired.items():
        for c in d["crop0"]:
            paired_crop0_ids.add(c["sample_id"])
    unpaired_crop0 = [r for r in all_crop0 if r["sample_id"] not in paired_crop0_ids]
    rng.shuffle(unpaired_crop0)
    n_remaining = args.n_evidence_crop0 - counts["evidence_no_target"]
    if n_remaining > 0:
        print(f"  A2: 添加 {n_remaining} 条未配对 crop0 样本", flush=True)
        for r in unpaired_crop0[:n_remaining]:
            out_train.append(make_abstain_record(r))
            counts["evidence_no_target_extra"] += 1

    mid_samples = [
        r
        for r in rows
        if not is_coco(r)
        and rel(r) == "T7_ladder"
        and 0.5 <= r.get("metadata", {}).get("achieved_visible_ratio", 0) <= 0.8
        and r["decision"] == "answer"
        and r.get("box")
    ]
    rng.shuffle(mid_samples)
    mid_selected = mid_samples[: args.n_evidence_mid]
    print(f"③.5 中间档 answer: 候选={len(mid_samples)}, 选取={len(mid_selected)}", flush=True)
    for r in mid_selected:
        out_train.append(make_answer_record(r))
        counts["evidence_mid_answer"] += 1

    for rname, n in [("T2", args.n_t2), ("T3", args.n_t3), ("T7_include", args.n_t7inc)]:
        subset = [
            r
            for r in rows
            if not is_coco(r) and rel(r) == rname and r["decision"] == "answer" and r.get("box")
        ]
        rng.shuffle(subset)
        for r in subset[:n]:
            out_train.append(make_answer_record(r))
            counts[rname] += 1

    t4c = [
        r
        for r in rows
        if not is_coco(r) and rel(r) == "T4_control" and r["decision"] == "answer" and r.get("box")
    ]
    rng.shuffle(t4c)
    for r in t4c[: args.n_t4control]:
        out_train.append(make_answer_record(r))
        counts["t4_control"] += 1

    def extract_crop_key(r):
        """Take the crop name of a T5 record as the stratification key."""
        meta = r.get("metadata", {})
        label = (
            meta.get("absent_label") or r.get("source_target", {}).get("label") or r.get("label")
        )
        if not label:
            return "unknown"
        return label.lower().split()[0] if label else "unknown"

    def stratified_sample_by_crop(t5_list, n, rng):
        """Sample evenly across crops, taking one record at a time from each non-empty bucket in turn.

        """
        buckets = defaultdict(list)
        for r in t5_list:
            key = extract_crop_key(r)
            buckets[key].append(r)
        for key in buckets:
            rng.shuffle(buckets[key])
        selected = []
        keys = sorted(buckets.keys())
        while len(selected) < n:
            advanced = False
            for key in keys:
                if buckets[key]:
                    selected.append(buckets[key].pop(0))
                    advanced = True
                    if len(selected) >= n:
                        break
            if not advanced:
                break
        return selected

    t5 = [r for r in rows if not is_coco(r) and rel(r) == "T5"]
    t5_selected = stratified_sample_by_crop(t5, args.n_t5, rng)
    for r in t5_selected:
        out_train.append(make_abstain_record(r))
        counts["t5_no_target"] += 1
    occ = [r for r in rows if not is_coco(r) and "occluded" in (rel(r) or "")]
    rng.shuffle(occ)
    for r in occ[: args.n_occluded]:
        out_train.append(make_abstain_record(r))
        counts["occluded_no_target"] += 1

    coco_ans = [r for r in rows if is_coco(r) and r["decision"] == "answer" and r.get("box")]
    rng.shuffle(coco_ans)
    for r in coco_ans[: args.n_coco_ans]:
        out_train.append(make_answer_record(r))
        counts["coco_ans"] += 1
    coco_abs = [r for r in rows if is_coco(r) and r["decision"] == "abstain"]
    rng.shuffle(coco_abs)
    for r in coco_abs[: args.n_coco_abs]:
        out_train.append(make_abstain_record(r))
        counts["coco_no_target"] += 1

    # ===== wrong_object oversample =====
    wo_samples = [r for r in out_train if r["head_label"] == 1]
    wo_count_before = len(wo_samples)
    n_wo_needed = max(0, args.wo_oversample_target - wo_count_before)

    if n_wo_needed > 0:
        oversample_rng = random.Random(args.seed + 1000)
        wo_pool = wo_samples * ((n_wo_needed // wo_count_before) + 1)
        oversample_rng.shuffle(wo_pool)
        wo_to_add = wo_pool[:n_wo_needed]
        out_train.extend(wo_to_add)
        print("\n=== wrong_object oversample ===", flush=True)
        print(f"  oversample前 wrong_object 数: {wo_count_before}", flush=True)
        print(f"  复制样本数: {len(wo_to_add)}", flush=True)
        print(f"  oversample后 wrong_object 数: {wo_count_before + len(wo_to_add)}", flush=True)
    else:
        print("\n=== wrong_object oversample ===", flush=True)
        print(
            f"  当前 wrong_object 数: {wo_count_before} >= 目标 {args.wo_oversample_target}, 无需复制",
            flush=True,
        )

    # ===== shuffle + dump =====
    rng.shuffle(out_train)
    outp_train = REPO / args.out_train
    outp_train.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_train, open(outp_train, "w"), ensure_ascii=False, indent=1)

    out_test = [make_test_record(r) for r in test_rows]
    outp_test = REPO / args.out_test
    with open(outp_test, "w") as f:
        for r in out_test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dev_rows = [json.loads(line) for line in open(REPO / args.dev)]
    dev_rows = [r for r in dev_rows if not r.get("metadata", {}).get("dose_only")]
    out_dev: list[dict] = []
    for r in dev_rows:
        if r["decision"] == "answer" and r.get("box"):
            out_dev.append(make_answer_record(r))
        elif r["decision"] == "abstain" and r.get("reason") in VALID_REASONS:
            out_dev.append(make_abstain_record(r))
    rng.shuffle(out_dev)
    outp_dev = REPO / args.out_dev
    json.dump(out_dev, open(outp_dev, "w"), ensure_ascii=False, indent=1)

    n_ans = sum(1 for r in out_train if r["decision_label"] == 1)
    n_abs = len(out_train) - n_ans
    judge = counts["judge_real"] + counts["judge_wrong_object"]
    same_img = judge + (counts["evidence_visible"] + counts["evidence_no_target"])
    coco = counts["coco_ans"] + counts["coco_no_target"]

    wo_by_crop = Counter()
    for r in out_train:
        if r["head_label"] == 1:  # wrong_object
            obj = json.loads(r["conversations"][1]["value"])
            crop = clean_label(r).lower().split()[0] if r.get("label") else "unknown"
            wo_by_crop[crop] += 1

    reason_dist = Counter()
    for r in out_train:
        obj = json.loads(r["conversations"][1]["value"])
        if obj["decision"] == "abstain":
            reason_dist[obj["reason"]] += 1

    reason_label_dist = Counter(r["reason_label"] for r in out_train)

    head_label_dist = Counter(r["head_label"] for r in out_train)

    print("\n=== new_reason 3reason SFT 审计 ===", flush=True)
    print(f"sft_train 总计: {len(out_train)} 条", flush=True)
    print(f"  成分: {dict(counts)}", flush=True)
    print(
        f"  answer(decision_label=1): {n_ans} ({n_ans / len(out_train):.0%}) / abstain(decision_label=0): {n_abs} ({n_abs / len(out_train):.0%})",
        flush=True,
    )
    print(f"  reason 分布: {dict(reason_dist)}", flush=True)
    print(f"  reason_label 分布: {dict(reason_label_dist)}", flush=True)
    print(f"  head_label 分布(三分类): {dict(head_label_dist)}", flush=True)
    print(f"  wrong_object 按作物: {dict(wo_by_crop)}", flush=True)
    print(
        f"  evidence 裁0%: {counts['evidence_no_target'] + counts.get('evidence_no_target_extra', 0)} (配对={counts['evidence_no_target']}, 额外={counts.get('evidence_no_target_extra', 0)})",
        flush=True,
    )
    print(f"  中间档 answer: {counts.get('evidence_mid_answer', 0)}", flush=True)
    print(
        f"  同图配对: judge {judge} + evidence(配对) {counts['evidence_visible'] + counts['evidence_no_target']} = {same_img} ({same_img / len(out_train):.0%}) [红线≥37%]",
        flush=True,
    )
    print(f"  COCO 占比: {coco} ({coco / len(out_train):.0%})", flush=True)
    print(f"  输出: {outp_train}", flush=True)
    print(f"\nsft_test 总计: {len(out_test)} 条 → {outp_test}", flush=True)
    print(f"sft_dev 总计: {len(out_dev)} 条 → {outp_dev} (训练时取~100条算eval loss)", flush=True)

    for r in out_train:
        assert isinstance(r["image"], list), "image 必须 list"
        assert r["decision_label"] in (0, 1)
        assert "reason_label" in r, "reason_label 字段缺失"
        assert "head_label" in r, "head_label 字段缺失"
        obj = json.loads(r["conversations"][1]["value"])
        if obj["decision"] == "answer":
            assert r["decision_label"] == 1
            assert r["reason_label"] == -1, "answer 记录 reason_label 必须 -1"
            assert r["head_label"] == 0, f"answer 记录 head_label 必须 0, 实际={r['head_label']}"
            assert len(obj["box"]) == 4 and all(0 <= v <= 1000 for v in obj["box"])
        else:
            assert r["decision_label"] == 0
            assert obj["reason"] in VALID_REASONS, f"非法 reason: {obj['reason']}"
            expected_reason_label = 0 if obj["reason"] == "wrong_object" else 1
            assert r["reason_label"] == expected_reason_label, (
                f"reason_label 不一致: reason={obj['reason']}, reason_label={r['reason_label']}, 期望={expected_reason_label}"
            )
            expected_head_label = 1 if obj["reason"] == "wrong_object" else 2
            assert r["head_label"] == expected_head_label, (
                f"head_label 不一致: reason={obj['reason']}, head_label={r['head_label']}, 期望={expected_head_label}"
            )
    print("\n✓ 所有样本格式+reason+decision_label+reason_label+head_label 验证通过", flush=True)

    print("\n=== 抽 3 条(answer/wrong_object/no_target 各1) ===", flush=True)
    for tag, pred in [
        ("answer", lambda o: o["decision"] == "answer"),
        ("wrong_object", lambda o: o.get("reason") == "wrong_object"),
        ("no_target", lambda o: o.get("reason") == "no_target"),
    ]:
        for r in out_train:
            obj = json.loads(r["conversations"][1]["value"])
            if pred(obj):
                print(
                    f"  [{tag}] decision_label={r['decision_label']} head_label={r['head_label']} | human: {r['conversations'][0]['value'][:55]}",
                    flush=True,
                )
                print(f"         gpt: {r['conversations'][1]['value']}", flush=True)
                break


if __name__ == "__main__":
    main()
