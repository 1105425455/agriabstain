#!/usr/bin/env python3
"""Precompute the reference head logits, z_ref, for every training record.

Runs the base model with the adapter over the training split and caches `head_logits` before
the softmax, so the paired objective can anchor against them without a second model in
memory. Every path is a command-line argument or an environment variable.

The dict written to --output, or to ZREF_CACHE_PATH, holds:
  {
    "zref":            tensor [N, 3] float32, head_logits before the softmax
    "source_image_id": list[str]
    "head_label":      list[int]
    "manifest":        dict with the base, the adapter, the record count and a fingerprint
  }

Usage:
  python training/margin_kl/dump_zref.py --dry-run --limit 10

  python training/margin_kl/dump_zref.py --base /path/to/Qwen3-VL-8B-Instruct \
    --adapter /path/to/checkpoint-444 --limit 100

  torchrun --nproc_per_node=8 training/margin_kl/dump_zref.py

  # merge existing shards only
  python training/margin_kl/dump_zref.py --merge-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent  # training/margin_kl/
_TRAINING = _THIS_DIR.parent  # training/
_AAAI_CODE = _TRAINING.parent  # aaai_code/
_BENCHMARK = _AAAI_CODE / "benchmark"

sys.path.insert(0, str(_TRAINING))

_DEFAULT_DATA = _BENCHMARK / "annotations" / "sft" / "sft_train_with_source.json"

MAX_PIXELS = 262144

REPR_LAYER_HS_IDX = 27

EXPECTED_N_SAMPLES = 7072

CHUNK_SIZE = 8 * 1024 * 1024


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    print(f"[dump_zref r{rank}] {msg}", flush=True)


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_base_identity(base_path: str | Path) -> dict:
    p = Path(base_path)
    if not p.is_dir():
        return {
            "config_sha256": file_sha256(base_path) if p.is_file() else None,
            "index_sha256": None,
            "shard_files": [],
            "n_shards": 0,
        }
    config_path = p / "config.json"
    index_path = p / "model.safetensors.index.json"
    config_sha = file_sha256(config_path) if config_path.exists() else None
    index_sha = file_sha256(index_path) if index_path.exists() else None
    shard_files = []
    for child in sorted(p.glob("*.safetensors")):
        st = child.stat()
        shard_files.append({"name": child.name, "size": st.st_size, "mtime": int(st.st_mtime)})
    return {
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "shard_files": shard_files,
        "n_shards": len(shard_files),
    }


def adapter_model_sha(adapter_dir: str | Path) -> str | None:
    path = Path(adapter_dir) / "adapter_model.safetensors"
    if path.exists():
        return file_sha256(path)
    return None


def resolve_image_path(rel_path: str, benchmark_root: Path) -> str:
    """Resolve a relative image path from the training records against the package root."""
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str(benchmark_root / rel_path)


def build_messages(record: dict, benchmark_root: Path) -> list[dict]:
    """Construct messages in sft_train format。"""
    system = record.get("system", "")
    conversations = record.get("conversations", [])
    human_text = ""
    for conv in conversations:
        if conv.get("from") == "human":
            human_text = conv.get("value", "").replace("<image>", "").strip()
            break
    raw_image = (
        record["image"][0] if isinstance(record.get("image"), list) else record.get("image", "")
    )
    image_path = resolve_image_path(raw_image, benchmark_root)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": human_text},
            ],
        },
    ]


def load_model_and_processor(
    base: str,
    adapter: str,
    device: torch.device,
) -> tuple:
    """Load the three-state model together with the LoRA adapter and the calibrated head.

    A checkpoint whose head norm differs from the expected value is reported rather than treated
    as fatal.
    """
    from peft import PeftModel  # noqa: E402
    from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead  # noqa: E402
    from transformers import AutoProcessor  # noqa: E402

    log("加载 processor")
    processor = AutoProcessor.from_pretrained(
        base,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )

    log("加载 base model (Qwen3VLWithTriHead)")
    model = Qwen3VLWithTriHead.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)

    log(f"加载 LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter).to(device)
    model.eval()

    calib_path = Path(adapter) / "discrim_head_calibrated.pt"
    if not calib_path.exists():
        log(f"WARNING: 校准头文件不存在: {calib_path}，跳过加载 calibrated head")
        return model, processor, None

    log(f"加载校准头: {calib_path}")
    calib = torch.load(str(calib_path), map_location="cpu", weights_only=False)

    m = model
    for _ in range(6):
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        elif hasattr(m, "module"):
            m = m.module
        else:
            break

    if not hasattr(m, "discrim_head"):
        log("WARNING: 穿包装后找不到 discrim_head 模块")
        return model, processor, None

    discrim_head_module = m.discrim_head
    if hasattr(discrim_head_module, "modules_to_save"):
        active_adapters = getattr(discrim_head_module, "active_adapters", None)
        if active_adapters:
            discrim_head_module = discrim_head_module.modules_to_save[active_adapters[0]]

    hs = calib["head_state"]
    clf_state = {k.replace("classifier.", ""): v for k, v in hs.items()}
    discrim_head_module.classifier.load_state_dict(clf_state)
    discrim_head_module.load_musd(calib["mu"], calib["sd"])

    with torch.inference_mode():
        w0_norm = discrim_head_module.classifier[0].weight.norm().item()
        sd_min_val = float(discrim_head_module.sd.min())
        sd_max_val = float(discrim_head_module.sd.max())

    log(f"自检: classifier[0].weight norm = {w0_norm:.4f}")
    log(f"自检: sd_range = [{sd_min_val:.3f}, {sd_max_val:.3f}]")

    return model, processor, discrim_head_module


def compute_zref(
    model,
    processor,
    discrim_head_module,
    record: dict,
    device: torch.device,
    benchmark_root: Path,
) -> np.ndarray | None:
    """Compute z_ref for one record, the three head logits before the softmax."""
    from qwen_vl_utils import process_vision_info  # noqa: E402

    try:
        messages = build_messages(record, benchmark_root)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hs_tuple = outputs.hidden_states
        if hs_tuple is None or len(hs_tuple) <= REPR_LAYER_HS_IDX:
            log(f"WARNING: hidden_states 无效（len={len(hs_tuple) if hs_tuple else None}）")
            return None

        hs = hs_tuple[REPR_LAYER_HS_IDX]

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            last_idx = int(attention_mask[0].sum()) - 1
            last_idx = max(0, min(last_idx, hs.shape[1] - 1))
        else:
            last_idx = hs.shape[1] - 1

        head_hidden = hs[0:1, last_idx, :]

        if discrim_head_module is not None:
            out_dtype = discrim_head_module.classifier[0].weight.dtype
            x = (
                head_hidden.float() - discrim_head_module.mu.float()
            ) / discrim_head_module.sd.float()
            head_logits = discrim_head_module.classifier(x.to(out_dtype))
            return head_logits[0].detach().cpu().float().numpy().astype(np.float32)
        else:
            log("WARNING: 无 discrim_head_module，跳过 zref 计算")
            return None

    except Exception as e:
        log(f"compute_zref 异常: {type(e).__name__}: {e}")
        log(traceback.format_exc().split("\n")[-2])
        return None


def compute_manifest(base: str, adapter: str, data_path: str) -> dict:
    manifest = {
        "base": base,
        "adapter": adapter,
        "adapter_model_sha256": adapter_model_sha(adapter),
        "input": data_path,
        "input_sha256": file_sha256(data_path) if Path(data_path).exists() else None,
        "max_pixels": MAX_PIXELS,
        "repr_layer_hs_idx": REPR_LAYER_HS_IDX,
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(__file__),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": None,
        "zref_shape": None,
        "base_identity": compute_base_identity(base),
    }
    return manifest


def detect_world_from_files(out_dir: Path) -> int | None:
    worlds: set[int] = set()
    for p in out_dir.glob("zref_shard.rank*of*.pt"):
        try:
            name = p.stem
            parts = name.split("of")
            if len(parts) >= 2:
                worlds.add(int(parts[-1]))
        except (ValueError, IndexError):
            continue
    if len(worlds) == 1:
        return worlds.pop()
    return None


def merge_and_write(args: argparse.Namespace) -> int:
    """Merge every shard into `zref_cache.pt`."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = args.world
    if world is None:
        world = args.num_shards
    if world is None:
        world = detect_world_from_files(out_dir)
    if world is None:
        log("FATAL: 无法确定 world size，指定 --world 或 --num-shards")
        return 1

    log(f"合并 world={world}")

    all_indices: list[int] = []
    all_zref: list[np.ndarray] = []
    all_sid: list[str] = []
    all_hl: list[int] = []

    for r in range(world):
        sp = out_dir / f"zref_shard.rank{r}of{world}.pt"
        if not sp.exists():
            log(f"WARNING: 分片缺失: {sp}")
            continue

        data = torch.load(str(sp), map_location="cpu", weights_only=False)
        all_indices.extend(data["indices"])
        all_zref.append(data["zref"])
        all_sid.extend(data["source_image_id"])
        all_hl.extend(data["head_label"])

    if not all_indices:
        log("FATAL: 无有效分片")
        return 1

    sort_order = np.argsort(all_indices)
    zref_full = np.concatenate(all_zref, axis=0)[sort_order].astype(np.float32)
    sid_full = [all_sid[i] for i in sort_order]
    hl_full = [all_hl[i] for i in sort_order]

    n = len(zref_full)
    log(f"合并完成: {n} 条, zref shape={zref_full.shape}")

    if not args.limit and n != EXPECTED_N_SAMPLES:
        log(f"WARNING: 样本数 {n} != 期望 {EXPECTED_N_SAMPLES}")

    manifest = compute_manifest(args.base, args.adapter, args.sft_train)
    manifest["n_samples"] = n
    manifest["zref_shape"] = list(zref_full.shape)

    cache = {
        "zref": torch.from_numpy(zref_full),
        "source_image_id": sid_full,
        "head_label": hl_full,
        "manifest": manifest,
    }

    output_path = Path(args.output) if args.output else out_dir / "zref_cache.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(output_path))
    log(f"保存 zref 缓存: {output_path}")

    loaded = torch.load(str(output_path), map_location="cpu", weights_only=False)
    log(f"验证: zref shape={loaded['zref'].shape}, dtype={loaded['zref'].dtype}")
    log(f"验证: head_label 分布={dict(sorted(Counter(loaded['head_label']).items()))}")

    return 0


def run_dump(args: argparse.Namespace) -> int:
    # ── Device ──
    if args.shard_id is not None and args.num_shards is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    with open(args.sft_train) as f:
        sft_data: list[dict] = json.load(f)
    n_total = len(sft_data)
    log(f"加载 sft_train_with_source.json: {n_total} 条")

    target_indices = list(range(n_total))
    if args.limit is not None and args.limit > 0:
        target_indices = target_indices[: args.limit]
        log(f"--limit={args.limit}，待处理 {len(target_indices)} 条")

    if args.rank is not None and args.world is not None:
        rank, world = args.rank, args.world
    elif args.shard_id is not None and args.num_shards is not None:
        rank, world = args.shard_id, args.num_shards
    else:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        world = int(os.environ.get("WORLD_SIZE", 1))

    all_indices = target_indices
    mine_indices = [i for i in all_indices if i % world == rank]
    log(f"分片: rank={rank}, world={world}, 本片索引数={len(mine_indices)}")

    model, processor, discrim_head_module = load_model_and_processor(
        args.base,
        args.adapter,
        device,
    )

    t0 = time.time()
    zref_list: list[np.ndarray] = []
    sid_list: list[str] = []
    hl_list: list[int] = []

    benchmark_root = _BENCHMARK

    for batch_i, idx in enumerate(mine_indices):
        record = sft_data[idx]
        zref = compute_zref(model, processor, discrim_head_module, record, device, benchmark_root)

        if zref is None:
            log(f"FATAL: index={idx} 计算 z_ref 失败")
            return 1

        if zref.shape != (3,):
            log(f"FATAL: index={idx} zref shape={zref.shape} != (3,)")
            return 1

        zref_list.append(zref)
        sid_list.append(str(record.get("source_image_id", "")))
        hl_list.append(record.get("head_label", -1))

        if (batch_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (batch_i + 1) / elapsed
            log(f"进度 {batch_i + 1}/{len(mine_indices)} ({elapsed:.1f}s, {rate:.2f} it/s)")

    dt = time.time() - t0
    log(f"本片计算完成: {len(zref_list)} 条, {dt:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_path = out_dir / f"zref_shard.rank{rank}of{world}.pt"
    shard_data = {
        "indices": mine_indices,
        "zref": np.stack(zref_list, axis=0).astype(np.float32),
        "source_image_id": sid_list,
        "head_label": hl_list,
    }
    torch.save(shard_data, shard_path)
    log(f"保存分片: {shard_path} (shape={shard_data['zref'].shape})")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline precomputation of the head_logits (z_ref) cache over train samples under base+adapter",
    )
    ap.add_argument(
        "--base",
        default=os.environ.get("BASE_MODEL", ""),
        help="Base model path (may also be set via the BASE_MODEL env var)",
    )
    ap.add_argument(
        "--adapter",
        default=os.environ.get("ZREF_ADAPTER", ""),
        help="LoRA adapter directory (checkpoint-N; may also be set via the ZREF_ADAPTER env var)",
    )
    ap.add_argument(
        "--sft-train",
        default=str(_DEFAULT_DATA),
        help=f"Path to sft_train_with_source.json (default: {_DEFAULT_DATA})",
    )
    ap.add_argument(
        "--output",
        default=os.environ.get("ZREF_CACHE_PATH", ""),
        help="Output .pt path (may also be set via the ZREF_CACHE_PATH env var)",
    )
    ap.add_argument("--out-dir", default="", help="Staging directory for shards (default: the directory containing output)")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--world", type=int, default=None)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--merge-only", action="store_true", help="Only merge existing shards")
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: use only the first N samples")
    ap.add_argument("--dry-run", action="store_true", help="Do not load the model; only print data information")
    args = ap.parse_args()

    if not args.base:
        ap.error("需要 --base 或 BASE_MODEL 环境变量")
    if not args.adapter:
        ap.error("需要 --adapter 或 ZREF_ADAPTER 环境变量")
    if not args.output:
        ap.error("需要 --output 或 ZREF_CACHE_PATH 环境变量")

    if not args.out_dir:
        args.out_dir = str(Path(args.output).parent)

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", 1))

    if args.merge_only:
        if rank == 0:
            return merge_and_write(args)
        return 0

    with open(args.sft_train) as f:
        sft_data = json.load(f)

    if args.dry_run:
        n_total = len(sft_data)
        hl_counter = Counter(d["head_label"] for d in sft_data)
        sid_unique = len(set(str(d["source_image_id"]) for d in sft_data))

        log("--dry-run 模式，不加载模型")
        log(f"  base: {args.base}")
        log(f"  adapter: {args.adapter}")
        log(f"  input: {args.sft_train}")
        log(f"  output: {args.output}")
        log(f"  out_dir: {args.out_dir}")
        log(f"  总样本数: {n_total}")
        log(
            f"  head_label 分布: 0(answer)={hl_counter[0]}, 1(wrong_object)={hl_counter[1]}, "
            f"2(no_target)={hl_counter[2]}"
        )
        log(f"  source_image_id 唯一数: {sid_unique}")
        log(f"  key 方案: 行 index (0~{n_total - 1})，天然唯一 ✓")
        if args.limit:
            subset = sft_data[: args.limit]
            hl_sub = Counter(d["head_label"] for d in subset)
            log(f"  --limit={args.limit} 子集 head_label: {dict(sorted(hl_sub.items()))}")
        log(f"  z_ref shape: [{n_total}, 3] float32")
        log(
            "  缓存格式: dict(zref=torch.Tensor[N,3], source_image_id=list, "
            "head_label=list, manifest=dict)"
        )
        log("  manifest 含 base/adapter 身份校验指纹")
        return 0

    ret = run_dump(args)
    if ret != 0:
        return ret

    if args.shard_id is None and args.num_shards is None:
        if rank == 0:
            merge_world = args.world if args.world is not None else world
            if merge_world > 1:
                log(f"rank0 等待 {merge_world} 个分片完成...")
                deadline = time.time() + 1800
                while time.time() < deadline:
                    all_present = True
                    for r in range(merge_world):
                        sp = Path(args.out_dir) / f"zref_shard.rank{r}of{merge_world}.pt"
                        if not sp.exists():
                            all_present = False
                            break
                    if all_present:
                        log(f"全部 {merge_world} 个分片已就绪，开始合并")
                        break
                    time.sleep(2)
                if not all_present:
                    log("FATAL: 超时，未收到全部分片")
                    return 1
            ret = merge_and_write(args)

    return ret


if __name__ == "__main__":
    raise SystemExit(main())
