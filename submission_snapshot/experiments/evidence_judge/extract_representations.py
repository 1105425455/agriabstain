#!/usr/bin/env python3
"""Extract four layer-27 representations from the frozen backbone for a like-for-like comparison.

All four are read from the same forward pass and are 4096-dimensional:
  1. last_token         hidden state at the generation-prompt last token
  2. visual_mean        mean over the hidden states of the visual span
  3. visual_max         per-dimension max over the hidden states of the visual span
  4. visual_query_attn  visual hidden states weighted by softmax of their dot product with
                        the last-token hidden state, scaled by 1/sqrt(d)

Accepted inputs:
  - SFT JSON arrays with `image`, `conversations` and `head_label`
  - evaluation JSONL with `images` and `messages`, where the label is derived from the
    decision and reason in `metadata`

Usage:
  python experiments/evidence_judge/extract_representations.py \
      --base /path/to/Qwen3-VL-8B-Instruct \
      --data benchmark/annotations/sft/sft_train.json --output-dir ./outputs \
      --split-name train --limit 10

  torchrun --nproc_per_node=4 experiments/evidence_judge/extract_representations.py \
      --base ... --data ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

_lr = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
if "CUDA_VISIBLE_DEVICES" not in os.environ and _lr >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_lr)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent  # experiments/evidence_judge/
_EXPERIMENTS = _THIS_DIR.parent  # experiments/
_AAAI_CODE = _EXPERIMENTS.parent  # aaai_code/
_BENCHMARK = _AAAI_CODE / "benchmark"
_TRAINING = _AAAI_CODE / "training"
sys.path.insert(0, str(_TRAINING))

REPR_LAYER = 27
VISUAL_TOKEN_ID = 151655  # Qwen3-VL visual placeholder token
MAX_PIXELS = 262144
MERGE_SIZE = 2  # Qwen3-VL default merge size


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    print(f"[extract_repr r{rank}] {msg}", flush=True)


def resolve_image_path(rel_path: str, derived_root: str | None = None) -> str:
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    # Derived images: map data/derived/... to derived_root / suffix
    if rel_path.startswith(os.path.join("data", "derived") + os.sep) or rel_path.startswith(
        "data/derived/"
    ):
        if derived_root is None:
            derived_root = str(_BENCHMARK / "data" / "derived")
        suffix = rel_path[len("data/derived/") :]
        resolved = str(Path(derived_root) / suffix)
        return resolved
    # Source images: benchmark root
    return str(_BENCHMARK / rel_path)


def build_messages(system: str, user_text: str, image_path: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def parse_sft_row(row: dict, derived_root: str | None = None) -> tuple[str, str, int, str]:
    image_path = row["image"][0] if isinstance(row.get("image"), list) else row["image"]
    human = None
    for conv in row.get("conversations", []):
        if conv.get("from") == "human":
            human = conv.get("value", "")
            break
    if human is None:
        raise ValueError("no human conversation found")
    user_text = human.replace("<image>", "").strip()
    label = int(row["head_label"])
    system = row.get("system", "")
    resolved_img = resolve_image_path(image_path, derived_root=derived_root)
    return system, user_text, label, resolved_img


def parse_eval_row(row: dict, derived_root: str | None = None) -> tuple[str, str, int, str, str]:
    messages = row.get("messages", [])
    system = ""
    user_text = ""
    for msg in messages:
        if msg.get("role") == "system":
            system = msg.get("content", "")
        elif msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        user_text = part.get("text", "")
            else:
                user_text = content

    images = row.get("images", [])
    image_path = resolve_image_path(images[0], derived_root=derived_root) if images else ""

    meta = row.get("metadata", {})
    decision = meta.get("decision", "")
    reason = meta.get("reason")
    if decision == "answer":
        label = 0
    elif decision == "abstain" and reason == "wrong_object":
        label = 1
    elif decision == "abstain" and reason == "no_target":
        label = 2
    else:
        raise ValueError(f"no head_label from decision={decision}, reason={reason}")

    sample_id = meta.get("sample_id", "")
    return system, user_text, label, image_path, sample_id


def compute_expected_visual_tokens(
    image_grid_thw: torch.Tensor, merge_size: int = MERGE_SIZE
) -> int:
    """Expected number of visual tokens, from `image_grid_thw` and the merge size.

    Summed over image grids as `t * (h // merge) * (w // merge)`, without rounding up.
    `image_grid_thw` is [N, 3] with columns [temporal, height_grid, width_grid].
    """
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return 0
    total = 0
    for i in range(image_grid_thw.shape[0]):
        t, h_grid, w_grid = image_grid_thw[i].tolist()
        # floor division, not ceil
        n_patches_h = h_grid // merge_size
        n_patches_w = w_grid // merge_size
        total += t * n_patches_h * n_patches_w
    return total


def _extract_single(
    model,
    processor,
    device: torch.device,
    index: int,
    system: str,
    user_text: str,
    image_path: str,
    label: int,
    sample_id: str,
) -> dict:
    """Single-sample wrapper around _extract_batch for batch_size=1 compatibility.

    Returns same dict keys as _extract_batch but with batch dimension = 1.
    """
    return _extract_batch(
        model,
        processor,
        device,
        [index],
        [system],
        [user_text],
        [image_path],
        [label],
        [sample_id],
    )


def _extract_batch(
    model,
    processor,
    device: torch.device,
    batch_indices: list[int],
    batch_systems: list[str],
    batch_user_texts: list[str],
    batch_image_paths: list[str],
    batch_labels: list[int],
    batch_sids: list[str],
) -> dict:
    """Batch forward for representation extraction.

    Processes all samples in a single forward pass and returns extracted features.
    Supports left/right padding via attention_mask.

    Returns dict with keys:
        last_token: np.ndarray [B, 4096]
        visual_mean: np.ndarray [B, 4096]
        visual_max: np.ndarray [B, 4096]
        visual_query_attn: np.ndarray [B, 4096]
        head_label: list[int]
        sample_id: list[str]
        image_path: list[str]
        original_index: list[int]
        visual_token_count: list[int]
    """
    from qwen_vl_utils import process_vision_info  # noqa: E402

    B = len(batch_indices)
    all_texts: list[str] = []
    all_images: list[Image.Image] = []

    for system, user_text, image_path in zip(batch_systems, batch_user_texts, batch_image_paths):
        msgs = build_messages(system, user_text, image_path)
        text = processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_texts.append(text)
        image_inputs, _ = process_vision_info(msgs)
        all_images.append(image_inputs[0])

    inputs = processor(
        text=all_texts,
        images=[img for img in all_images],
        padding=True,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]  # [B, seq_len]
    attention_mask = inputs.get("attention_mask")  # [B, seq_len]
    image_grid_thw = inputs.get("image_grid_thw")  # [N_grids, 3]; N_grids == B for 1 img/sample
    pixel_values = inputs.get("pixel_values")

    if image_grid_thw is not None:
        n_expected_grid_rows = B
        if image_grid_thw.shape[0] != n_expected_grid_rows:
            raise RuntimeError(
                f"image_grid_thw rows ({image_grid_thw.shape[0]}) != "
                f"batch size ({n_expected_grid_rows}) — expected one grid row per sample"
            )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=None,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) <= REPR_LAYER:
        raise RuntimeError(f"hidden_states[{REPR_LAYER}] 不可用")

    hs_layer = hidden_states[REPR_LAYER]  # [B, seq_len, 4096]

    last_token_list: list[np.ndarray] = []
    visual_mean_list: list[np.ndarray] = []
    visual_max_list: list[np.ndarray] = []
    visual_attn_list: list[np.ndarray] = []
    visual_token_counts: list[int] = []

    for j in range(B):
        hs = hs_layer[j]  # [seq_len_j, 4096] after padding removal? No, still padded.
        mask_j = attention_mask[j] if attention_mask is not None else None
        ids_j = input_ids[j]

        # ── last-token index: last non-zero position in attention_mask ──
        # This handles both left-padding (non-zero at end) and right-padding (non-zero at start).
        # In left-padding: mask = [0,0,1,1,1,...], last_idx = last 1
        # In right-padding: mask = [1,1,1,0,0,...], last_idx = last 1
        if mask_j is not None:
            nonzero_positions = torch.where(mask_j != 0)[0]
            if nonzero_positions.numel() == 0:
                last_token_idx = hs.shape[0] - 1
            else:
                last_token_idx = int(nonzero_positions[-1])
        else:
            last_token_idx = hs.shape[0] - 1
        last_token_idx = max(0, min(last_token_idx, hs.shape[0] - 1))

        # ── visual tokens: positions where input_ids == VISUAL_TOKEN_ID ──
        visual_mask = ids_j == VISUAL_TOKEN_ID
        visual_indices = torch.where(visual_mask)[0]

        # image_grid_thw for this sample: row j corresponds to sample j
        sample_grid_thw = image_grid_thw[j : j + 1] if image_grid_thw is not None else None

        expected_visual_tokens = compute_expected_visual_tokens(
            sample_grid_thw, merge_size=MERGE_SIZE
        )
        actual_visual_tokens = len(visual_indices)
        if actual_visual_tokens != expected_visual_tokens:
            raise RuntimeError(
                f"样本 {batch_indices[j]} visual token 数不匹配: "
                f"预期={expected_visual_tokens}, 实际={actual_visual_tokens}. "
                f"image_grid_thw={sample_grid_thw}, input_ids={input_ids.shape}"
            )
        if actual_visual_tokens == 0:
            raise RuntimeError(f"样本 {batch_indices[j]} 无 visual token 在 input_ids 中")

        visual_hs = hs[visual_indices]

        last_token_feat = hs[last_token_idx].float().cpu().numpy()
        visual_mean_feat = visual_hs.float().mean(dim=0).cpu().numpy()
        visual_max_feat = visual_hs.float().max(dim=0).values.cpu().numpy()

        d = hs.shape[-1]
        last_h = hs[last_token_idx].float()
        visual_h = visual_hs.float()
        scores = (visual_h @ last_h) / (d**0.5)
        attn_weights = torch.softmax(scores, dim=0)
        visual_attn_feat = (attn_weights[:, None] * visual_h).sum(dim=0).cpu().numpy()

        last_token_list.append(last_token_feat.astype(np.float32))
        visual_mean_list.append(visual_mean_feat.astype(np.float32))
        visual_max_list.append(visual_max_feat.astype(np.float32))
        visual_attn_list.append(visual_attn_feat.astype(np.float32))
        visual_token_counts.append(int(actual_visual_tokens))

    return {
        "last_token": np.stack(last_token_list, axis=0),
        "visual_mean": np.stack(visual_mean_list, axis=0),
        "visual_max": np.stack(visual_max_list, axis=0),
        "visual_query_attn": np.stack(visual_attn_list, axis=0),
        "head_label": batch_labels,
        "sample_id": batch_sids,
        "image_path": batch_image_paths,
        "original_index": batch_indices,
        "visual_token_count": visual_token_counts,
    }


def run_fingerprint(args) -> str:
    """Fingerprint of a run: mtime and hash of the base and adapter files, data size and mtime,
    world size, layer index and pixel budget.
    """  # noqa: E501
    raw_parts = []
    raw_parts.append(f"base={args.base}")
    for p in [args.base, args.adapter] if args.adapter else [args.base]:
        if p and os.path.exists(p):
            # Use modification time and a quick hash of directory listing
            mtime = os.path.getmtime(p)
            raw_parts.append(f"mtime({p})={mtime}")
            # For directory models, hash key constituent files
            pobj = Path(p)
            if pobj.is_dir():
                key_files = (
                    sorted(pobj.rglob("config.json")) + sorted(pobj.rglob("*.safetensors"))[:5]
                )
                for kf in key_files:
                    if kf.exists():
                        raw_parts.append(f"kf({kf.name})={os.path.getmtime(kf)}")

    data_path = str(Path(args.data).resolve())
    if os.path.exists(args.data):
        raw_parts.append(f"data_mtime={os.path.getmtime(args.data)}")
        raw_parts.append(f"data_size={os.path.getsize(args.data)}")
    raw_parts.append(f"data={data_path}")
    raw_parts.append(f"split={args.split_name}")
    raw_parts.append(f"world={args.world}")
    raw_parts.append(f"layer={REPR_LAYER}")
    raw_parts.append(f"maxpixels={MAX_PIXELS}")
    raw_parts.append(f"batch_size={args.batch_size}")
    # derived root in fingerprint
    derived_root = getattr(args, "derived_root", None)
    if derived_root:
        raw_parts.append(f"derived_root={derived_root}")
    raw = "|".join(raw_parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def merge_shards(
    output_dir: Path,
    split_name: str,
    fp: str,
    world: int,
    expected_indices: list[int],
    timeout: int = 3600,
) -> dict | None:
    """Rank 0 waits for every shard and merges them.

    Args:
        output_dir: output directory
        split_name: name of the split
        fp: fingerprint
        world: number of shards
        timeout: seconds to wait

    Returns:
        the merged dict on success, None on failure
    """
    log(f"rank0 等待 {world} 个分片完成...")
    deadline = time.time() + timeout
    all_present = False
    while time.time() < deadline:
        all_present = True
        for r in range(world):
            done = output_dir / f"shard_rank{r}of{world}_{fp}.done"
            if not done.exists():
                all_present = False
                break
        if all_present:
            break
        time.sleep(2)

    if not all_present:
        log("FATAL: 超时，未收到全部分片")
        return None

    shard_arrays: list[dict[str, np.ndarray]] = []
    for r in range(world):
        sp = output_dir / f"shard_rank{r}of{world}_{fp}.npz"
        with np.load(sp, allow_pickle=True) as data:
            arrays = {
                "last_token": data["last_token"].astype(np.float32),
                "visual_mean": data["visual_mean"].astype(np.float32),
                "visual_max": data["visual_max"].astype(np.float32),
                "visual_query_attn": data["visual_query_attn"].astype(np.float32),
                "head_label": data["head_label"].copy(),
                "sample_id": data["sample_id"].copy(),
                "image": data["image"].copy(),
                "original_index": data["original_index"].copy(),
                "visual_token_count": data["visual_token_count"].copy(),
            }
        n = len(arrays["original_index"])
        for key, val in arrays.items():
            if len(val) != n:
                log(f"FATAL: shard r{r} key {key} length {len(val)} != {n}")
                return None
        shard_arrays.append(arrays)

    concat: dict[str, np.ndarray] = {}
    for key in shard_arrays[0]:
        concat[key] = np.concatenate([sa[key] for sa in shard_arrays], axis=0)

    all_orig = concat["original_index"].tolist()
    unique_orig = sorted(set(all_orig))
    has_dup = len(all_orig) != len(unique_orig)
    expected_index_set = set(expected_indices)
    actual_indices = set(all_orig)
    missing = expected_index_set - actual_indices
    unexpected = actual_indices - expected_index_set

    if has_dup or missing or unexpected:
        dup_str = f"重复={len(all_orig) - len(unique_orig)}" if has_dup else ""
        miss_str = f"缺失={sorted(missing)[:20]}" if missing else ""
        extra_str = f"意外索引={sorted(unexpected)[:20]}" if unexpected else ""
        log(f"FATAL: 分片完整性错误: {dup_str} {miss_str} {extra_str}")
        return None

    sort_idx = np.argsort(concat["original_index"])
    merged: dict[str, np.ndarray] = {}
    for key in shard_arrays[0]:
        merged[key] = concat[key][sort_idx]

    merged["n_selected"] = len(expected_indices)
    merged["n_merged"] = len(concat["original_index"])

    return merged


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract four layer-27 representations from the frozen backbone for a fair comparison",
    )
    ap.add_argument("--base", required=True, help="Base model path")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter path (diagnostic only)")
    ap.add_argument("--data", required=True, help="Input data path (SFT JSON or eval JSONL)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--split-name", required=True, help="Split name (e.g. train, dev, test)")
    ap.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process")
    ap.add_argument("--rank", type=int, default=None, help="Manually specify the rank")
    ap.add_argument("--world", type=int, default=None, help="Manually specify the world size")
    ap.add_argument("--dry-run", action="store_true", help="Do not load the model; only print information")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for model forward (default: 1, must be positive integer)",
    )
    ap.add_argument(
        "--derived-root",
        default=None,
        help="Root directory of external derived images (default: benchmark/data/derived). Applies only to relative paths carrying the data/derived/ prefix; paths such as data/source/ are still resolved from the benchmark root.",
    )
    args = ap.parse_args()

    if args.batch_size < 1:
        log(f"FATAL: --batch-size must be positive, got {args.batch_size}")
        return 1

    derived_root = args.derived_root
    if derived_root is not None:
        derived_root = str(Path(derived_root).resolve())
    else:
        derived_root = str((_BENCHMARK / "data" / "derived").resolve())

    if not os.path.isdir(derived_root):
        log(f"WARNING: derived_root 目录不存在: {derived_root} (derived images 引用将失败)")

    log(f"derived_root: {derived_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.derived_root = derived_root

    if args.rank is not None:
        rank = args.rank
    else:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if args.world is not None:
        world = args.world
    else:
        world = int(os.environ.get("WORLD_SIZE", 1))
    args.world = world

    fp = run_fingerprint(args)
    log(f"fingerprint={fp}, rank={rank}, world={world}")

    data_path = Path(args.data)
    if not data_path.exists():
        log(f"FATAL: data 不存在: {data_path}")
        return 1

    is_jsonl = data_path.suffix == ".jsonl"
    rows: list[dict] = []
    if is_jsonl:
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(data_path) as f:
            rows = json.load(f)
    n_total = len(rows)
    log(f"加载 {data_path.name}: {n_total} 条 (jsonl={is_jsonl})")

    if args.limit is not None and args.limit > 0 and args.limit < n_total:
        selected_indices = list(range(args.limit))
        log(f"limit={args.limit}，取前 {len(selected_indices)} 条")
    else:
        selected_indices = list(range(n_total))

    mine_indices = [i for i in selected_indices if i % world == rank]
    log(f"分片: rank={rank}, world={world}, 本片索引数={len(mine_indices)}")

    manifest = {
        "base": args.base,
        "adapter": args.adapter,
        "data": str(data_path.resolve()),
        "split_name": args.split_name,
        "fingerprint": fp,
        "rank": rank,
        "world": world,
        "layer": REPR_LAYER,
        "max_pixels": MAX_PIXELS,
        "n_total": n_total,
        "n_selected": len(selected_indices),
        "n_mine": len(mine_indices),
        "is_adapter_diagnostic": args.adapter is not None,
        "derived_root": args.derived_root,
        "batch_size": args.batch_size,
    }

    # dry-run
    if args.dry_run:
        log("--dry-run 模式，不加载模型")
        log(f"  base: {args.base}")
        log(f"  adapter: {args.adapter}")
        log(f"  data: {data_path}")
        log(f"  split: {args.split_name}")
        log(f"  output_dir: {output_dir}")
        log(f"  derived_root: {args.derived_root}")
        log(f"  fingerprint: {fp}")
        log(f"  manifest: {json.dumps(manifest, indent=2)}")
        hl_counter: dict = {}
        for i in mine_indices:
            row = rows[i]
            if is_jsonl:
                meta = row.get("metadata", {})
                decision = meta.get("decision", "")
                reason = meta.get("reason")
                if decision == "answer":
                    hl = 0
                elif decision == "abstain" and reason == "wrong_object":
                    hl = 1
                elif decision == "abstain" and reason == "no_target":
                    hl = 2
                else:
                    hl = -1
            else:
                hl = row.get("head_label", -1)
            hl_counter[hl] = hl_counter.get(hl, 0) + 1
        log(f"  本片 head_label 分布: {dict(sorted(hl_counter.items()))}")
        return 0

    from transformers import AutoProcessor  # noqa: E402
    from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead  # noqa: E402

    log(f"加载 processor: {args.base}")
    processor = AutoProcessor.from_pretrained(
        args.base,
        trust_remote_code=True,
        max_pixels=MAX_PIXELS,
    )

    log(f"加载 model: {args.base}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLWithTriHead.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    if args.adapter:
        from peft import PeftModel  # noqa: E402

        log(f"加载 adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        manifest["adapter"] = args.adapter

    last_token_list: list[np.ndarray] = []
    visual_mean_list: list[np.ndarray] = []
    visual_max_list: list[np.ndarray] = []
    visual_attn_list: list[np.ndarray] = []
    head_labels: list[int] = []
    sample_ids: list[str] = []
    images_list: list[str] = []
    original_indices: list[int] = []
    visual_token_counts: list[int] = []

    t0 = time.time()
    n_failed = 0

    # Main loop: process in chunks of batch_size
    batch_size = args.batch_size
    chunk_indices = [
        mine_indices[i : i + batch_size] for i in range(0, len(mine_indices), batch_size)
    ]

    for chunk_i, chunk in enumerate(chunk_indices):
        # Parse rows for this chunk
        batch_systems: list[str] = []
        batch_user_texts: list[str] = []
        batch_image_paths: list[str] = []
        batch_labels: list[int] = []
        batch_sids: list[str] = []
        valid_chunk: list[int] = []
        for i in chunk:
            row = rows[i]
            try:
                if is_jsonl:
                    system, user_text, label, image_path, sample_id = parse_eval_row(
                        row, derived_root=derived_root
                    )
                else:
                    system, user_text, label, image_path = parse_sft_row(
                        row, derived_root=derived_root
                    )
                    sample_id = str(i)
            except Exception as e:
                log(f"样本 {i} 解析异常: {type(e).__name__}: {e}")
                n_failed += 1
                continue
            batch_systems.append(system)
            batch_user_texts.append(user_text)
            batch_image_paths.append(image_path)
            batch_labels.append(label)
            batch_sids.append(sample_id)
            valid_chunk.append(i)

        if not valid_chunk:
            continue

        try:
            if batch_size == 1:
                # Use _extract_single wrapper for batch_size=1 compatibility
                result = _extract_single(
                    model,
                    processor,
                    device,
                    valid_chunk[0],
                    batch_systems[0],
                    batch_user_texts[0],
                    batch_image_paths[0],
                    batch_labels[0],
                    batch_sids[0],
                )
            else:
                result = _extract_batch(
                    model,
                    processor,
                    device,
                    valid_chunk,
                    batch_systems,
                    batch_user_texts,
                    batch_image_paths,
                    batch_labels,
                    batch_sids,
                )
        except Exception as e:
            log(f"batch {chunk_i} 异常 (原始索引 {valid_chunk}): {type(e).__name__}: {e}")
            log(traceback.format_exc())
            n_failed += len(valid_chunk)
            continue

        # ── shape assertion: each chunk must have [B, 4096] with consistent B ──
        _B = result["last_token"].shape[0]
        assert result["last_token"].shape == (_B, 4096), (
            f"chunk {chunk_i}: last_token shape {result['last_token'].shape} != ({_B}, 4096)"
        )
        assert result["visual_mean"].shape == (_B, 4096), (
            f"chunk {chunk_i}: visual_mean shape {result['visual_mean'].shape} != ({_B}, 4096)"
        )
        assert result["visual_max"].shape == (_B, 4096), (
            f"chunk {chunk_i}: visual_max shape {result['visual_max'].shape} != ({_B}, 4096)"
        )
        assert result["visual_query_attn"].shape == (_B, 4096), (
            f"chunk {chunk_i}: visual_query_attn shape {result['visual_query_attn'].shape} != ({_B}, 4096)"
        )

        last_token_list.append(result["last_token"])
        visual_mean_list.append(result["visual_mean"])
        visual_max_list.append(result["visual_max"])
        visual_attn_list.append(result["visual_query_attn"])
        head_labels.extend(result["head_label"])
        sample_ids.extend(result["sample_id"])
        images_list.extend(result["image_path"])
        original_indices.extend(result["original_index"])
        visual_token_counts.extend(result["visual_token_count"])

        if ((chunk_i + 1) * batch_size) % 50 == 0 or (chunk_i + 1) == len(chunk_indices):
            elapsed = time.time() - t0
            done = min((chunk_i + 1) * batch_size, len(mine_indices))
            rate = done / elapsed
            log(f"进度 {done}/{len(mine_indices)} ({elapsed:.1f}s, {rate:.2f} it/s)")

    dt = time.time() - t0
    n_ok = len(original_indices)
    log(f"本片完成: {n_ok} 条成功 / {len(mine_indices)} 总, 失败={n_failed}, {dt:.1f}s")

    if n_failed > 0:
        log(f"FATAL: {n_failed} 条处理失败，不允许不完整输出")
        return 1

    shard_tag = f"shard_rank{rank}of{world}_{fp}"
    shard_path = output_dir / f"{shard_tag}.npz"

    if n_ok == 0:
        np.savez_compressed(
            str(shard_path),
            last_token=np.empty((0, 4096), dtype=np.float32),
            visual_mean=np.empty((0, 4096), dtype=np.float32),
            visual_max=np.empty((0, 4096), dtype=np.float32),
            visual_query_attn=np.empty((0, 4096), dtype=np.float32),
            head_label=np.empty((0,), dtype=np.int64),
            sample_id=np.empty((0,), dtype=object),
            image=np.empty((0,), dtype=object),
            original_index=np.empty((0,), dtype=np.int64),
            visual_token_count=np.empty((0,), dtype=np.int64),
            manifest_json=json.dumps(manifest),
        )
    else:
        np.savez_compressed(
            str(shard_path),
            last_token=np.concatenate(last_token_list, axis=0).astype(np.float32),
            visual_mean=np.concatenate(visual_mean_list, axis=0).astype(np.float32),
            visual_max=np.concatenate(visual_max_list, axis=0).astype(np.float32),
            visual_query_attn=np.concatenate(visual_attn_list, axis=0).astype(np.float32),
            head_label=np.array(head_labels, dtype=np.int64),
            sample_id=np.array(sample_ids, dtype=object),
            image=np.array(images_list, dtype=object),
            original_index=np.array(original_indices, dtype=np.int64),
            visual_token_count=np.array(visual_token_counts, dtype=np.int64),
            manifest_json=json.dumps(manifest),
        )

    done_path = output_dir / f"{shard_tag}.done"
    done_path.write_text("ok")
    log(f"保存分片: {shard_path} ({n_ok} 条)")

    if rank == 0:
        merged = merge_shards(
            output_dir,
            args.split_name,
            fp,
            world,
            selected_indices,
        )

        if merged is None:
            log("FATAL: 合并失败，不产出不完整文件")
            return 1

        manifest["n_merged"] = merged["n_merged"]
        manifest["n_selected"] = merged["n_selected"]
        # visual token count stats
        vtc = merged.get("visual_token_count")
        if vtc is not None and len(vtc) > 0:
            manifest["visual_token_count_stats"] = {
                "mean": float(vtc.mean()),
                "min": int(vtc.min()),
                "max": int(vtc.max()),
                "median": int(np.median(vtc)),
            }

        out_path = output_dir / f"{args.split_name}.npz"
        np.savez_compressed(
            str(out_path),
            last_token=merged["last_token"],
            visual_mean=merged["visual_mean"],
            visual_max=merged["visual_max"],
            visual_query_attn=merged["visual_query_attn"],
            head_label=merged["head_label"],
            sample_id=merged["sample_id"],
            image=merged["image"],
            original_index=merged["original_index"],
            visual_token_count=merged["visual_token_count"],
            manifest_json=json.dumps(manifest),
        )
        log(f"保存合并输出: {out_path} ({merged['last_token'].shape[0]} 条)")

        for r in range(world):
            sp = output_dir / f"shard_rank{r}of{world}_{fp}.npz"
            dp = output_dir / f"shard_rank{r}of{world}_{fp}.done"
            sp.unlink(missing_ok=True)
            dp.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
