#!/usr/bin/env python3
"""Evaluate teacher-forced LM loss for grouping-ablation checkpoint selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead
from transformers import AutoProcessor


def build_input_ids_labels(
    row: dict[str, Any], processor: Any
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    conversations = row["conversations"]
    human_value = next(item["value"] for item in conversations if item["from"] == "human")
    assistant_value = next(item["value"] for item in conversations if item["from"] == "gpt")
    image_value = row["image"]
    image_path = image_value[0] if isinstance(image_value, list) else image_value
    user_text = human_value.replace("<image>", "").strip()

    full_messages = [
        {"role": "system", "content": row.get("system", "")},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": assistant_value},
    ]
    prompt_messages = full_messages[:2]
    image_inputs, _ = process_vision_info(full_messages)
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    labels = full_inputs["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
    return full_inputs, labels


@torch.no_grad()
def evaluate(
    *,
    base: Path,
    adapter: Path,
    data_path: Path,
    limit: int,
    max_pixels: int,
) -> float:
    records = json.loads(data_path.read_text(encoding="utf-8"))[:limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(
        base,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )
    model = Qwen3VLWithTriHead.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(model, adapter).to(device)
    model.eval()

    total_loss = 0.0
    evaluated = 0
    started_at = time.time()
    for index, row in enumerate(records, start=1):
        full_inputs, labels = build_input_ids_labels(row, processor)
        if int((labels != -100).sum()) == 0:
            raise ValueError(f"record {index - 1} has no assistant labels")
        batch = {key: value.to(device) for key, value in full_inputs.items()}
        batch["labels"] = labels.to(device)
        total_loss += float(model(**batch).loss.item())
        evaluated += 1
        if index % 10 == 0:
            elapsed = time.time() - started_at
            print(
                f"[dev_loss] {index}/{len(records)} "
                f"rate={index / elapsed:.2f} samples/s avg={total_loss / evaluated:.6f}",
                flush=True,
            )
    if evaluated != len(records):
        raise RuntimeError(f"evaluated {evaluated}/{len(records)} records")
    return total_loss / evaluated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-pixels", type=int, default=262144)
    args = parser.parse_args()

    loss = evaluate(
        base=args.base,
        adapter=args.adapter,
        data_path=args.data,
        limit=args.limit,
        max_pixels=args.max_pixels,
    )
    print(f"DEV_LOSS={loss:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
