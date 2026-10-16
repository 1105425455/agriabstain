"""L2 与 comprehensive/L3 的离线构成核查。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    annotation = ROOT / "aaai_code/benchmark/annotations"
    l2 = load(annotation / "eval/test_L2_evidence.jsonl")
    split = load(annotation / "splits/test.jsonl")
    by_id = {r["sample_id"]: r for r in split}
    base = (
        ROOT
        / "new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/eval_faithful_v1"
    )
    preds = load(base / "L2_evidence/predictions.jsonl")
    assert len(l2) == len(preds)
    groups: dict[str, Counter] = defaultdict(Counter)
    cross_label = 0
    negatives = []
    for row, pred in zip(l2, preds):
        assert row["metadata"]["sample_id"] == pred["sample_id"]
        if row["metadata"]["decision"] != "abstain":
            continue
        sid = row["metadata"]["sample_id"].split("-cropout-", 1)[1]
        original = by_id[sid]
        query = (
            row["messages"][-1]["content"]
            .removeprefix("<image>Locate the ")
            .removesuffix(" in this image.")
        )
        target = original["source_target"]["label"]
        same = query.replace("_", " ").lower() == target.replace("_", " ").lower()
        cross_label += not same
        answered = pred["pred"].get("decision") == "answer"
        for key in (
            "all",
            f"domain:{original['dataset']}",
            f"query:{query}",
            f"target_label_matches_query:{same}",
        ):
            groups[key]["n"] += 1
            groups[key]["answered"] += answered
        negatives.append(
            {
                "sample_id": pred["sample_id"],
                "original_sample_id": sid,
                "query": query,
                "crop_target": target,
                "same_label": same,
                "answered": answered,
                "image": row["images"][0],
            }
        )
    comprehensive = load(
        ROOT
        / "new_reason/outputs/base_native_regate_20260720/regate/comprehensive/predictions.jsonl"
    )
    crop = {}
    for relation in ("T2_occluded", "T3_occluded", "base_occluded", "T7_ladder"):
        rs = [
            r
            for r in comprehensive
            if r["relation"] == relation and r["gt"]["decision"] == "abstain"
        ]
        crop[relation] = {
            "n": len(rs),
            "answered": sum(r["p_answer"] >= max(r["p_wo"], r["p_nt"]) for r in rs),
        }
    l3 = load(base / "L3_dose/predictions.jsonl")
    rs = [r for r in l3 if r["gt"].get("decision") == "abstain"]
    result = {
        "paired_probability_analysis": json.loads(
            (
                ROOT
                / "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/evidence/L2/test/analysis.json"
            ).read_text()
        )["L2"],
        "l2_n": len(l2),
        "l2_unsupported": len(negatives),
        "cross_target_label_count": cross_label,
        "l2_groups": {k: dict(v) for k, v in groups.items()},
        "comprehensive_unsupported": crop,
        "l3_endpoint": {
            "n": len(rs),
            "answered": sum(r["pred"].get("decision") == "answer" for r in rs),
        },
        "l2_negative_records": negatives,
    }
    p = ROOT / "docs/review_evidence/l2_breakdown.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "l2_negative_records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
