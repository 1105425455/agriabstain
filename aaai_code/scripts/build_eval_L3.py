#!/usr/bin/env python
"""L3 dose probe: the full T7 visibility ladder at 1.0, 0.75, 0.5, 0.25 and 0.0.

The source is the test split, so no image was seen during training, and all five levels are
included:
  1) the T7_ladder records in `test.jsonl`, which are the 1.0 and 0.0 levels of test images
  2) the intermediate levels, 0.75, 0.5 and 0.25, drawn from `dose_pool.jsonl` for those same
     source images

Reference annotation is three-valued:
- achieved_visible_ratio >= 0.999  -> gt_decision="answer", with a box
- achieved_visible_ratio <= 0.001  -> gt_decision="abstain", reason no_target
- in between                        -> gt_decision="ambiguous", which is not scored as right
  or wrong and only contributes the abstention rate of the dose-response curve
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
    "You are AgriAbstain, a vision-language grounding assistant for high-stakes plant disease diagnosis. "
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


def bin_label(ratio: float) -> str:
    if ratio >= 0.999:
        return "1.0"
    elif ratio >= 0.74:
        return "0.75"
    elif ratio >= 0.49:
        return "0.5"
    elif ratio >= 0.24:
        return "0.25"
    elif ratio <= 0.001:
        return "0.0"
    else:
        return "0.25"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", choices=["test", "dev"], default="test", help="Data source: test (default) or dev"
    )
    args = ap.parse_args()

    split = args.split
    TEST = DATA_ROOT / f"splits/{split}.jsonl"
    DOSE_POOL = DATA_ROOT / "splits/dose_pool.jsonl"
    OUT = DATA_ROOT / f"eval/{split}_L3_dose.jsonl"

    split_img_ids = set()
    with open(TEST) as f:
        for line in f:
            row = json.loads(line)
            if row.get("metadata", {}).get("relation") == "T7_ladder":
                split_img_ids.add(row["source_image_id"])

    records = []

    with open(TEST) as f:
        for line in f:
            row = json.loads(line)
            if row.get("metadata", {}).get("relation") != "T7_ladder":
                continue

            avr = row.get("metadata", {}).get("achieved_visible_ratio")
            if avr is None:
                continue

            img_path = row["image_path"]
            query = (row.get("source_target") or {}).get("label") or row.get("label") or ""
            box = row.get("box")

            if avr >= 0.999:
                gt_decision = "answer"
                sol = json.dumps(
                    {"decision": "answer", "label": row["label"], "box": box}, separators=(",", ":")
                )
            elif avr <= 0.001:
                gt_decision = "abstain"
                sol = json.dumps(
                    {"decision": "abstain", "reason": "no_target"}, separators=(",", ":")
                )
            else:
                gt_decision = "ambiguous"
                sol = json.dumps({"decision": "ambiguous"}, separators=(",", ":"))

            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>Locate the {query} in this image."},
                ],
                "images": [img_path],
                "solution": sol,
                "metadata": {
                    "decision": gt_decision if gt_decision != "ambiguous" else None,
                    "label": row["label"] if gt_decision == "answer" else None,
                    "box": box if gt_decision == "answer" else None,
                    "reason": "no_target" if gt_decision == "abstain" else None,
                    "relation": "L3_ladder",
                    "achieved_visible_ratio": avr,
                    "visible_ratio_bin": bin_label(avr),
                    "gt_decision": gt_decision,
                    "sample_id": f"L3-{row['source_image_id']}-{avr:.3f}-{row['sample_id']}",
                },
            }
            records.append(rec)

    with open(DOSE_POOL) as f:
        for line in f:
            row = json.loads(line)
            if row.get("metadata", {}).get("relation") != "T7_ladder":
                continue
            if row["source_image_id"] not in split_img_ids:
                continue

            avr = row.get("metadata", {}).get("achieved_visible_ratio")
            if avr is None:
                continue

            img_path = row["image_path"]
            query = (row.get("source_target") or {}).get("label") or row.get("label") or ""
            box = row.get("box")

            if avr >= 0.999:
                gt_decision = "answer"
                sol = json.dumps(
                    {"decision": "answer", "label": row["label"], "box": box}, separators=(",", ":")
                )
            elif avr <= 0.001:
                gt_decision = "abstain"
                sol = json.dumps(
                    {"decision": "abstain", "reason": "no_target"}, separators=(",", ":")
                )
            else:
                gt_decision = "ambiguous"
                sol = json.dumps({"decision": "ambiguous"}, separators=(",", ":"))

            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>Locate the {query} in this image."},
                ],
                "images": [img_path],
                "solution": sol,
                "metadata": {
                    "decision": gt_decision if gt_decision != "ambiguous" else None,
                    "label": row["label"] if gt_decision == "answer" else None,
                    "box": box if gt_decision == "answer" else None,
                    "reason": "no_target" if gt_decision == "abstain" else None,
                    "relation": "L3_ladder",
                    "achieved_visible_ratio": avr,
                    "visible_ratio_bin": bin_label(avr),
                    "gt_decision": gt_decision,
                    "sample_id": f"L3-{row['source_image_id']}-{avr:.3f}-{row['sample_id']}",
                },
            }
            records.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_dec = defaultdict(int)
    by_bin = defaultdict(int)
    split_imgs = set()
    for r in records:
        gt = r["metadata"]["gt_decision"]
        by_dec[gt] += 1
        by_bin[r["metadata"]["visible_ratio_bin"]] += 1
        split_imgs.add(r["metadata"]["sample_id"].split("-")[1])

    print(f"L3_dose ({split}): {len(records)} 条 → {OUT}")
    print(f"  各档分布: {dict(by_bin)}")
    print(f"  gt_decision分布: {dict(by_dec)}")
    print(f"  {split}图数量: {len(split_imgs)} 张")
    print(
        f"  中间档(0.75/0.5/0.25)合计: {by_bin.get('0.75', 0) + by_bin.get('0.5', 0) + by_bin.get('0.25', 0)} 条"
    )


if __name__ == "__main__":
    main()
