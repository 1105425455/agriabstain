from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are AgriAbstain, a vision-language grounding assistant. "
    "Return only valid JSON. If visual evidence is sufficient, answer with "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]}. '
    "If evidence is insufficient, return "
    '{"decision":"abstain","reason":"target_absent|visual_evidence_insufficient"}. '
    "Boxes use relative coordinates in [0,1000]."
)


def sample_to_qwen_sft_record(sample: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_output(sample)
    return {
        "id": sample["sample_id"],
        "images": [sample["image_path"]],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": sample["query"]},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(expected, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "metadata": {
            "dataset": sample["dataset"],
            "split": sample["split"],
            "group_id": sample["group_id"],
            "sample_type": sample["sample_type"],
            "transform": sample.get("transform", "none"),
        },
    }


def _expected_output(sample: dict[str, Any]) -> dict[str, Any]:
    if sample["decision"] == "answer":
        return {
            "decision": "answer",
            "label": sample["label"],
            "box": sample["box"],
        }
    return {
        "decision": "abstain",
        "reason": sample["reason"],
    }
