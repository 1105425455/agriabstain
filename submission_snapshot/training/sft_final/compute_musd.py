#!/usr/bin/env python3
"""Estimate the frozen standardisation statistics used by the discriminative head.

Reads the layer-27 last-token representation of the training split with the base model and
accumulates per-dimension float32 mu and sd. The output dict {mu, sd, source, n} is written
to --output and is consumed by `DiscrimHead.load_musd`.

Usage:
  # small-sample check
  python training/sft_final/compute_musd.py --limit 100

  # full split
  torchrun --nproc_per_node=8 training/sft_final/compute_musd.py

  # dry run
  python training/sft_final/compute_musd.py --dry-run --limit 10
"""

from __future__ import annotations

import argparse
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

_THIS_DIR = Path(__file__).resolve().parent  # training/sft_final/
_TRAINING = _THIS_DIR.parent  # training/
_AAAI_CODE = _TRAINING.parent  # aaai_code/
_BENCHMARK = _AAAI_CODE / "benchmark"

sys.path.insert(0, str(_TRAINING))

_DEFAULT_SFT_TRAIN = _BENCHMARK / "annotations" / "sft" / "sft_train.json"

REPR_LAYER_HS_IDX = 27  # hidden_states[27]

DEFAULT_SEED = 42


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    print(f"[compute_musd r{rank}] {msg}", flush=True)


def build_messages(system: str, user_text: str, image_path: str) -> list[dict]:
    """Build the messages list."""
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


def resolve_image_path(rel_path: str, benchmark_root: Path) -> str:
    """Resolve a relative image path from `sft_train.json` against the package root.

    The `image` field holds paths such as `data/source/coco/images/xxx.jpg`, relative to the
    directory that contains this package.
    """
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str(benchmark_root / rel_path)


def extract_sft_train_fields(row: dict) -> tuple[str, str, int]:
    """Read `user_text`, `image_path` and `head_label` (0/1/2) from an `sft_train.json` record."""
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
    return user_text, image_path, label


def get_hidden_feature(
    model,
    processor,
    device: torch.device,
    messages: list[dict],
) -> np.ndarray:
    """Read the layer-27 last-token representation, returned as float32 numpy [4096].

    Identical to `calibrate_head.py`:
    - text = apply_chat_template(add_generation_prompt=True)
    - process_vision_info -> inputs
    - forward(output_hidden_states=True)
    - hidden_states[27][0, last_non_pad_idx].float().cpu().numpy()
    """
    from qwen_vl_utils import process_vision_info  # noqa: E402

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            labels=None,
            output_hidden_states=True,
        )
    hs = outputs.hidden_states[REPR_LAYER_HS_IDX][0]  # [S, 4096]
    S = hs.shape[0]
    idx = int(inputs["attention_mask"][0].sum()) - 1
    idx = max(0, min(idx, S - 1))
    feat = hs[idx].float().cpu().numpy()  # [4096]
    return feat


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract layer-27 representations with the base model and compute float32 mu/sd statistics",
    )
    ap.add_argument(
        "--base",
        default=os.environ.get("BASE_MODEL", ""),
        help="Base model path (may also be set via the BASE_MODEL env var)",
    )
    ap.add_argument(
        "--sft-train",
        default=str(_DEFAULT_SFT_TRAIN),
        help=f"Path to sft_train.json (default: {_DEFAULT_SFT_TRAIN})",
    )
    ap.add_argument(
        "--output",
        default=os.environ.get("MUSD_OUTPUT", ""),
        help="Output .pt path (may also be set via the MUSD_OUTPUT env var)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    ap.add_argument(
        "--strategy",
        choices=["deterministic", "uniform"],
        default="deterministic",
        help="Sampling strategy: deterministic = first N records, uniform = uniform random (default: deterministic)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Number of samples (default: 2000)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not load the model; only print data information",
    )
    ap.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Manually specify the rank (overrides the torchrun environment variable)",
    )
    ap.add_argument(
        "--world",
        type=int,
        default=None,
        help="Manually specify the world size (overrides the torchrun environment variable)",
    )
    args = ap.parse_args()

    base = args.base
    if not base:
        ap.error("需要 --base 或 BASE_MODEL 环境变量")

    output = args.output
    if not output:
        ap.error("需要 --output 或 MUSD_OUTPUT 环境变量")

    if args.rank is not None:
        rank = args.rank
    else:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if args.world is not None:
        world = args.world
    else:
        world = int(os.environ.get("WORLD_SIZE", 1))

    sft_path = Path(args.sft_train)
    if not sft_path.exists():
        log(f"FATAL: sft_train 不存在: {sft_path}")
        return 1

    with open(sft_path) as f:
        sft_data: list[dict] = json.load(f)
    n_total = len(sft_data)
    log(f"加载 sft_train.json: {n_total} 条")

    limit = args.limit
    if limit is not None and limit > 0 and limit < n_total:
        if args.strategy == "deterministic":
            indices = list(range(limit))
            log(f"策略=deterministic: 取前 {limit} 条 (seed={args.seed} 无关)")
        elif args.strategy == "uniform":
            rng = np.random.default_rng(args.seed)
            indices = sorted(rng.choice(n_total, size=limit, replace=False).tolist())
            log(f"策略=uniform: 均匀采样 {limit} 条 (seed={args.seed})")
        else:
            indices = list(range(min(limit, n_total)))
    else:
        indices = list(range(n_total))
        limit = n_total

    log(f"采样后总索引数: {len(indices)}")

    # ── dry-run ──
    if args.dry_run:
        log("--dry-run 模式，不加载模型")
        log(f"  base: {base}")
        log(f"  sft_train: {sft_path}")
        log(f"  output: {output}")
        log(f"  strategy: {args.strategy}, seed: {args.seed}, limit: {limit}")
        log(f"  world: {world}, rank: {rank}")
        log("  mu/sd 将保存为 dict(mu=float32[4096], sd=float32[4096], source=str, n=int)")
        log("  输出与 DiscrimHead.load_musd 兼容")
        hl_counter = {}
        for i in indices:
            hl = sft_data[i].get("head_label", -1)
            hl_counter[hl] = hl_counter.get(hl, 0) + 1
        log(f"  采样 head_label 分布: {dict(sorted(hl_counter.items()))}")
        return 0

    mine_indices = [i for i in indices if i % world == rank]
    log(f"分片: rank={rank}, world={world}, 本片索引数={len(mine_indices)}")

    from transformers import AutoProcessor  # noqa: E402
    from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead  # noqa: E402

    log(f"加载 processor: {base}")
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)

    log(f"加载 model: {base}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLWithTriHead.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    feats_list: list[np.ndarray] = []
    t0 = time.time()

    for batch_i, idx in enumerate(mine_indices):
        row = sft_data[idx]
        try:
            ut, ip, lab = extract_sft_train_fields(row)
            system = row.get("system", "")
            abs_ip = resolve_image_path(ip, _BENCHMARK)
            msgs = build_messages(system, ut, abs_ip)
            feat = get_hidden_feature(model, processor, device, msgs)
            feats_list.append(feat)
        except Exception as e:
            log(f"样本 {idx} 异常: {type(e).__name__}: {e}")
            log(traceback.format_exc().split("\n")[-2])
            continue

        if (batch_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (batch_i + 1) / elapsed
            log(f"进度 {batch_i + 1}/{len(mine_indices)} ({elapsed:.1f}s, {rate:.2f} it/s)")

    dt = time.time() - t0
    log(f"本片完成: {len(feats_list)} 条, {dt:.1f}s")

    if not feats_list:
        log("WARNING: 本片无有效特征")
        out_dir = Path(output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        shard_path = out_dir / f"musd_shard.rank{rank}of{world}.npz"
        np.savez(str(shard_path), feats=np.empty((0, 4096), dtype=np.float32))
        (out_dir / f"musd_shard.rank{rank}of{world}.done").write_text("ok")
        return 0

    all_feats = np.stack(feats_list, axis=0).astype(np.float32)  # [N_shard, 4096]
    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"musd_shard.rank{rank}of{world}.npz"
    np.savez(str(shard_path), feats=all_feats)
    (out_dir / f"musd_shard.rank{rank}of{world}.done").write_text("ok")
    log(f"保存分片: {shard_path} (shape={all_feats.shape})")

    if rank == 0:
        log(f"rank0 等待 {world} 个分片完成...")
        deadline = time.time() + 3600
        while time.time() < deadline:
            all_present = True
            for r in range(world):
                done_path = out_dir / f"musd_shard.rank{r}of{world}.done"
                if not done_path.exists():
                    all_present = False
                    break
            if all_present:
                break
            time.sleep(2)

        if not all_present:
            log("FATAL: 超时，未收到全部分片")
            return 1

        all_rank_feats: list[np.ndarray] = []
        for r in range(world):
            sp = out_dir / f"musd_shard.rank{r}of{world}.npz"
            data = np.load(sp)
            f = data["feats"]
            if f.shape[0] > 0:
                all_rank_feats.append(f)
            log(f"  rank{r}: {f.shape[0]} 条")

        if not all_rank_feats:
            log("FATAL: 所有分片均无有效特征")
            return 1

        full_feats = np.concatenate(all_rank_feats, axis=0).astype(np.float32)
        log(f"合并完成: {full_feats.shape}")

        mu = full_feats.mean(axis=0).astype(np.float32)  # [4096]
        sd = full_feats.std(axis=0).astype(np.float32)  # [4096]

        sd = np.clip(sd, 1e-6, None)

        log(f"mu shape: {mu.shape}, sd shape: {sd.shape}")
        log(f"mu range: [{mu.min():.4f}, {mu.max():.4f}]")
        log(f"sd range: [{sd.min():.4f}, {sd.max():.4f}]")

        save_dict = {
            "mu": torch.from_numpy(mu),
            "sd": torch.from_numpy(sd),
            "source": str(sft_path.resolve()),
            "n": full_feats.shape[0],
        }
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, str(output_path))
        log(f"已保存: {output_path}")

        for r in range(world):
            sp = out_dir / f"musd_shard.rank{r}of{world}.npz"
            dp = out_dir / f"musd_shard.rank{r}of{world}.done"
            sp.unlink(missing_ok=True)
            dp.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
