"""经SFT记录及original_index精确恢复论文100条dev缓存位置。"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rebuild_paper_dev import build_paper_100_dev

ROOT = Path(__file__).resolve().parents[2]
selected, counts = build_paper_100_dev(ROOT / "new_reason/outputs/splits/dev.jsonl")
sft = json.loads(
    (ROOT / "aaai_code/benchmark/annotations/sft/sft_dev.json").read_text()
)
cache = np.load(
    ROOT / "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/repr/dev/dev.npz",
    allow_pickle=True,
)
lookup = defaultdict(list)
for i, original in enumerate(cache["original_index"]):
    row = sft[int(original)]
    assert Path(row["image"][0]).name == Path(str(cache["image"][i])).name
    assert row["head_label"] == cache["head_label"][i]
    key = (
        Path(row["image"][0]).name,
        row["conversations"][0]["value"].removeprefix("<image>"),
        row["head_label"],
    )
    lookup[key].append(i)
indices = []
for row in selected:
    label = (
        0
        if row["decision"] == "answer"
        else 1
        if row["reason"] == "wrong_object"
        else 2
    )
    key = (Path(row["image_path"]).name, row["query"], label)
    assert len(lookup[key]) == 1, (row["sample_id"], lookup[key])
    indices.append(lookup[key][0])
assert len(set(indices)) == 100
out = ROOT / "docs/review_evidence/frozen_and_prior/exact_dev100.json"
out.write_text(
    json.dumps(
        {
            "cache_indices": indices,
            "sample_ids": [r["sample_id"] for r in selected],
            "counts": dict(counts),
            "match": "original_index->sft_dev plus image basename/query/state, unique100/100",
        },
        indent=2,
    )
    + "\n"
)
print("100/100 unique matches")
