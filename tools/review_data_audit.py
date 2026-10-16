"""从已保存标注和 GPT 结果生成审稿证据，不训练、不推理。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/review_evidence"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sft_path = ROOT / "aaai_code/benchmark/annotations/sft/sft_train.json"
    rows = json.loads(sft_path.read_text())
    unique = Counter(json.dumps(row, sort_keys=True) for row in rows)
    state_counts = Counter(str(row["head_label"]) for row in rows)
    unique_states = Counter(str(json.loads(key)["head_label"]) for key in unique)
    query_counts: dict[str, Counter] = defaultdict(Counter)
    domain_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        state = str(row["head_label"])
        text = row["conversations"][0]["value"].removeprefix("<image>")
        for prefix, suffix in (
            ("Locate the ", " in this image."),
            ("Find the ", " and return its bounding box."),
            ("Where is the ", " in the image?"),
        ):
            if text.startswith(prefix) and text.endswith(suffix):
                text = text[len(prefix) : -len(suffix)]
                break
        query_counts[text][state] += 1
        image = row["image"][0]
        domain = "coco" if "/coco/" in image else "plantdoc"
        domain_counts[domain][state] += 1
    gpt = {}
    base = ROOT / "aaai_code/experiments/gpt56_sol_full/outputs/decision_prompt_v1"
    for directory in sorted(base.iterdir()):
        config_path = directory / "api_outputs.run_config.json"
        prediction_path = directory / "analysis/predictions.jsonl"
        if not config_path.exists() or not prediction_path.exists():
            continue
        config = json.loads(config_path.read_text())
        predictions = read_jsonl(prediction_path)
        invalid = [p for p in predictions if not p["pred"].get("valid", False)]
        gpt[directory.name] = {
            "config": {
                k: config.get(k)
                for k in (
                    "model_requested",
                    "temperature",
                    "image_detail",
                    "stream",
                    "requested_records",
                    "completed_records",
                    "missing_records",
                    "prompt_policy",
                )
            },
            "response_models": dict(Counter(p.get("api_model") for p in predictions)),
            "n": len(predictions),
            "invalid": len(invalid),
            "invalid_rate": len(invalid) / len(predictions),
            "invalid_by_gt": dict(Counter(p["gt"].get("decision") for p in invalid)),
            "config_path": str(config_path.relative_to(ROOT)),
            "predictions_path": str(prediction_path.relative_to(ROOT)),
            "predictions_sha256": hashlib.sha256(
                prediction_path.read_bytes()
            ).hexdigest(),
        }
    result = {
        "wrong_object_multiplicities": dict(
            Counter(
                count
                for key, count in unique.items()
                if json.loads(key)["head_label"] == 1
            )
        ),
        "sft_path": str(sft_path.relative_to(ROOT)),
        "sft_sha256": hashlib.sha256(sft_path.read_bytes()).hexdigest(),
        "n": len(rows),
        "unique_records": len(unique),
        "state_counts": dict(state_counts),
        "unique_state_counts": dict(unique_states),
        "domain_by_state": {k: dict(v) for k, v in domain_counts.items()},
        "query_category_by_state": {
            k: dict(v) for k, v in sorted(query_counts.items())
        },
        "gpt": gpt,
    }
    (OUT / "data_and_gpt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 训练 query 类别组成（按实际训练实例计，含重复）",
        "",
        "| query 类别 | answer | wrong_object | no_target |",
        "|---|---:|---:|---:|",
    ]
    for key, counts in sorted(query_counts.items()):
        lines.append(f"| {key} | {counts['0']} | {counts['1']} | {counts['2']} |")
    (OUT / "category_composition.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "n",
                    "unique_records",
                    "state_counts",
                    "unique_state_counts",
                    "domain_by_state",
                )
            },
            ensure_ascii=False,
        )
    )
    print(json.dumps(gpt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
