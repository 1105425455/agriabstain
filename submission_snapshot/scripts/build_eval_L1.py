#!/usr/bin/env python
"""L1 discrimination probe: one image carrying both the real disease (answer) and a near-miss
category (abstain).

Built from `test.jsonl`:
- group by `source_image_id` and keep images that carry both a real disease, with relation in
  {base, T1} and decision answer, and a near-miss, with relation T6
- emit two records per image: the real branch, with answer and box, and the near-miss branch,
  with abstain and reason target_absent
- the answer branch must have a reference box, taken from the top-level `box` field of the
  test record and written into both `solution` and `metadata.box`
- `metadata.relation` is set to `L1_real` or `L1_mismatch`
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
    OUT = DATA_ROOT / f"eval/{split}_L1_discrim.jsonl"

    by_img: dict[str, list[dict]] = defaultdict(list)
    with open(TEST) as f:
        for line in f:
            row = json.loads(line)
            sid = row.get("source_image_id")
            if sid:
                by_img[sid].append(row)

    pairs = []
    n_skipped = 0
    for img_id, rows in by_img.items():
        real_rows = [
            r
            for r in rows
            if r.get("metadata", {}).get("relation") == "base" and r.get("decision") == "answer"
        ]
        t6_rows = [r for r in rows if r.get("metadata", {}).get("relation") == "T6"]

        if not real_rows or not t6_rows:
            n_skipped += 1
            continue

        for real_row in real_rows:
            for t6_row in t6_rows:
                img_path = real_row["image_path"]
                real_query = real_row["label"]
                t6_query = t6_row["metadata"].get("mismatch_label")
                real_box = real_row.get("box")

                if not real_box:
                    continue

                sol_real = json.dumps(
                    {"decision": "answer", "label": real_row["label"], "box": real_box},
                    separators=(",", ":"),
                )
                rec_real = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"<image>Locate the {real_query} in this image.",
                        },
                    ],
                    "images": [img_path],
                    "solution": sol_real,
                    "metadata": {
                        "decision": "answer",
                        "label": real_row["label"],
                        "box": real_box,
                        "reason": None,
                        "relation": "L1_real",
                        "achieved_visible_ratio": None,
                        "sample_id": f"L1-{img_id}-real-{real_row['sample_id']}",
                    },
                }
                pairs.append(rec_real)

                sol_t6 = json.dumps(
                    {"decision": "abstain", "reason": "wrong_object"}, separators=(",", ":")
                )
                rec_t6 = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>Locate the {t6_query} in this image."},
                    ],
                    "images": [img_path],
                    "solution": sol_t6,
                    "metadata": {
                        "decision": "abstain",
                        "label": None,
                        "box": None,
                        "reason": "wrong_object",
                        "relation": "L1_mismatch",
                        "achieved_visible_ratio": None,
                        "sample_id": f"L1-{img_id}-mismatch-{t6_row['sample_id']}",
                    },
                }
                pairs.append(rec_t6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for rec in pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_rel = defaultdict(int)
    by_dec = defaultdict(int)
    for r in pairs:
        by_rel[r["metadata"]["relation"]] += 1
        by_dec[r["metadata"]["decision"]] += 1

    n_pairs = len([r for r in pairs if r["metadata"]["relation"] == "L1_real"])
    print(f"L1_discrim ({split}): {len(pairs)} 条 → {OUT}")
    print(f"  同图配对数: {n_pairs} 对 (真病+近义病)")
    print(f"  探针分布: {dict(by_rel)}")
    print(f"  决策分布: {dict(by_dec)}")
    print(f"  跳过无配对图: {n_skipped}")


if __name__ == "__main__":
    main()
