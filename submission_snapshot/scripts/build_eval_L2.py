#!/usr/bin/env python
"""L2 evidence probe: fully visible (keep) against cropped to 0% and unlocatable (cropout).

Built from the T7 ladder samples of `test.jsonl`:
- keep branch (answer): relation in {base, T1}, decision answer, and a box present, i.e. the
  target is fully visible
- crop branch (no_target): relation T7_ladder with `metadata.achieved_visible_ratio <= 0.001`,
  i.e. cropped to 0%
- the two branches are paired on the image and `metadata.relation` is set to `L2_keep` or
  `L2_cropout`
- each branch keeps its own `image_path`
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if (REPO / "benchmark").is_dir():
    DATA_ROOT = REPO / "benchmark/annotations"
else:
    DATA_ROOT = REPO / "outputs"

SYSTEM_PROMPT = (
    "You are a vision-language grounding assistant for high-stakes plant disease diagnosis. "
    "First reason briefly inside <think> </think>: describe the visible symptoms, then decide whether the "
    "queried target is the object shown and whether it can be localized. Then give the final answer inside "
    "<answer> </answer> as a single JSON object. If the queried target is present and localizable, answer "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]}. Otherwise abstain '
    '{"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image, or cropped out of view). "
    "Boxes use relative coordinates in [0,1000]. Output exactly: "
    "<think> your reasoning </think><answer> the JSON </answer>"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", choices=["test", "dev"], default="test", help="Data source: test (default) or dev"
    )
    args = ap.parse_args()

    split = args.split
    TEST = DATA_ROOT / f"splits/{split}.jsonl"
    OUT = DATA_ROOT / f"eval/{split}_L2_evidence.jsonl"

    by_img: dict[str, list[dict]] = defaultdict(list)
    with open(TEST) as f:
        for line in f:
            row = json.loads(line)
            sid = row.get("source_image_id")
            if sid:
                by_img[sid].append(row)

    pairs = []
    n_unmatched = 0
    for img_id, rows in by_img.items():
        keep_rows = [
            r
            for r in rows
            if r.get("metadata", {}).get("relation") == "base"
            and r.get("decision") == "answer"
            and r.get("box")
        ]
        crop_rows = [
            r
            for r in rows
            if r.get("metadata", {}).get("relation") == "T7_ladder"
            and r.get("metadata", {}).get("achieved_visible_ratio", 1) <= 0.001
        ]

        if not keep_rows or not crop_rows:
            n_unmatched += 1
            continue

        for keep_row in keep_rows:
            for crop_row in crop_rows:
                keep_img_path = keep_row["image_path"]
                crop_img_path = crop_row["image_path"]
                query = keep_row["label"]
                keep_box = keep_row.get("box")

                if not keep_box:
                    continue

                sol_keep = json.dumps(
                    {"decision": "answer", "label": keep_row["label"], "box": keep_box},
                    separators=(",", ":"),
                )
                rec_keep = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>Locate the {query} in this image."},
                    ],
                    "images": [keep_img_path],
                    "solution": sol_keep,
                    "metadata": {
                        "decision": "answer",
                        "label": keep_row["label"],
                        "box": keep_box,
                        "reason": None,
                        "relation": "L2_keep",
                        "achieved_visible_ratio": None,
                        "sample_id": f"L2-{img_id}-keep-{keep_row['sample_id']}",
                    },
                }
                pairs.append(rec_keep)

                sol_crop = json.dumps(
                    {"decision": "abstain", "reason": "no_target"}, separators=(",", ":")
                )
                rec_crop = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>Locate the {query} in this image."},
                    ],
                    "images": [crop_img_path],
                    "solution": sol_crop,
                    "metadata": {
                        "decision": "abstain",
                        "label": None,
                        "box": None,
                        "reason": "no_target",
                        "relation": "L2_cropout",
                        "achieved_visible_ratio": crop_row.get("metadata", {}).get(
                            "achieved_visible_ratio"
                        ),
                        "sample_id": f"L2-{img_id}-cropout-{crop_row['sample_id']}",
                    },
                }
                pairs.append(rec_crop)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_rel = defaultdict(int)
    by_dec = defaultdict(int)
    for r in pairs:
        by_rel[r["metadata"]["relation"]] += 1
        by_dec[r["metadata"]["decision"]] += 1

    n_pairs = len([r for r in pairs if r["metadata"]["relation"] == "L2_keep"])
    print(f"L2_evidence ({split}): {len(pairs)} 条 → {OUT}")
    print(f"  同图配对数: {n_pairs} 对 (keep+cropout)")
    print(f"  探针分布: {dict(by_rel)}")
    print(f"  决策分布: {dict(by_dec)}")
    print(f"  未匹配图(缺keep或crop腿): {n_unmatched}")


if __name__ == "__main__":
    main()
