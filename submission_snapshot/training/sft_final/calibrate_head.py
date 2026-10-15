#!/usr/bin/env python3
"""Stage-2 head calibration: fit the three-way head on frozen representations.

Stage 1 trains the backbone and the head jointly. This script then freezes the backbone,
re-fits the discriminative head to convergence on the fixed layer-27 representations, and
writes the result back into the checkpoint as `discrim_head`. Inference remains a single
forward pass; the head is a fully trained read-out over representations that were shaped
during Stage 1. The two stages are described in the Method section of the paper.

Label convention: 0 = answer, 1 = wrong_object, 2 = no_target.

Usage:
  # mode=extract: dump representations, one shard per rank
  torchrun --nproc_per_node=4 training/sft_final/calibrate_head.py --mode extract

  # mode=train: fit the head on a single device and write it back to the checkpoint
  python training/sft_final/calibrate_head.py --mode train

  # explicit paths
  python training/sft_final/calibrate_head.py --mode extract \
      --base /path/to/Qwen3-VL-8B-Instruct \
      --adapter /path/to/checkpoint-444 \
      --sft-train benchmark/annotations/sft/sft_train.json \
      --dev benchmark/annotations/splits/dev.jsonl \
      --log-dir /path/to/logs
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import copy
from pathlib import Path

_lr = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
if "CUDA_VISIBLE_DEVICES" not in os.environ and _lr >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_lr)
    print(f"[calibrate_head pre-import] LOCAL_RANK={_lr} → CUDA_VISIBLE_DEVICES={_lr}", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

try:
    from sklearn.metrics import classification_report
except ImportError:
    pass

_THIS_DIR = Path(__file__).resolve().parent
_AAAI_CODE = _THIS_DIR.parent.parent  # training/sft_final/../.. = aaai_code
_TRAINING = _THIS_DIR.parent  # training
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

_DEFAULT_BENCHMARK = _AAAI_CODE / "benchmark"
_DEFAULT_SFT_TRAIN = _DEFAULT_BENCHMARK / "annotations" / "sft" / "sft_train.json"
_DEFAULT_DEV = _DEFAULT_BENCHMARK / "annotations" / "splits" / "dev.jsonl"

DECISION_SYSTEM_PROMPT = (
    "You are a vision-language grounding assistant for high-stakes plant disease diagnosis. "
    "Locate the queried target. If the queried target is present and localizable, return "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]} with box in [0,1000] relative coords. '
    "Otherwise abstain: "
    '{"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image, or cropped out of view)."
)

ASSISTANT_TOKEN = 77091
IM_END = 151645
REPR_LAYER = 27


def log(msg: str) -> None:
    print(f"[calibrate_head r{_lr}] {msg}", flush=True)


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
    image = Path(image_path)
    if not image.is_absolute():
        image = _DEFAULT_BENCHMARK / image
    return user_text, str(image), label


def extract_dev_fields(row: dict) -> tuple[str, str, int]:
    """Read `user_text`, `image_path` and `head_label` from a `dev.jsonl` record.

    `head_label` is derived from decision and reason:
    - answer                   -> 0
    - abstain + wrong_object    -> 1
    - abstain + no_target       -> 2
    """
    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = _DEFAULT_BENCHMARK / image_path
    user_text = row.get("query", "").replace("<image>", "").strip()
    decision = row.get("decision")
    reason = row.get("reason")

    if decision == "answer":
        label = 0
    elif decision == "abstain" and reason == "wrong_object":
        label = 1
    elif decision == "abstain" and reason == "no_target":
        label = 2
    else:
        raise ValueError(f"无法从 decision={decision}, reason={reason} 派生 head_label")

    return user_text, str(image_path), label


def build_messages(system: str, user_text: str, image_path: str):
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


def get_hidden_feature(model, processor, device, messages, add_generation_prompt: bool):
    """Read the layer-27 last-token representation under the inference convention.

    The steps below must stay identical to the inference path, otherwise the head is calibrated
    on a different distribution than it sees at test time:
    - text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    - image_inputs, _ = process_vision_info(messages)
    - inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    - outputs = model(..., output_hidden_states=True, labels=None)
    - hs = outputs.hidden_states[27][0]                 [S, 4096]
    - idx = int(attention_mask[0].sum()) - 1
    - feat = hs[idx].float().cpu().numpy()              [4096], float() is required as the
      hidden states are bf16
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(
        device
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            labels=None,
            output_hidden_states=True,
        )
    hs = outputs.hidden_states[REPR_LAYER][0]  # [S, 4096]
    S = hs.shape[0]
    idx = int(inputs["attention_mask"][0].sum()) - 1
    idx = max(0, min(idx, S - 1))
    feat = hs[idx].float().cpu().numpy()  # [4096]
    return feat


def mode_extract(args) -> int:
    """Dump representations under torchrun; `world_size` is whatever torchrun provides."""
    from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead
    from transformers import AutoProcessor
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    log(f"加载 base + adapter，device={device}")
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    model = Qwen3VLWithTriHead.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()

    log(f"加载 sft_train: {args.sft_train}")
    with open(args.sft_train) as f:
        sft_rows = json.load(f)
    log(f"sft_train 共 {len(sft_rows)} 条")

    log(f"加载 dev: {args.dev}")
    dev_rows = []
    with open(args.dev) as f:
        for line in f:
            line = line.strip()
            if line:
                dev_rows.append(json.loads(line))

    import random

    _rng = random.Random(42)
    dev_by_class = {0: [], 1: [], 2: []}
    for r in dev_rows:
        decision = r.get("decision")
        reason = r.get("reason")
        if decision == "answer":
            cls = 0
        elif decision == "abstain" and reason == "wrong_object":
            cls = 1
        elif decision == "abstain" and reason == "no_target":
            cls = 2
        else:
            continue
        dev_by_class[cls].append(r)

    per_class = args.dev_max // 3
    picked = []
    for cls in [0, 1, 2]:
        _rng.shuffle(dev_by_class[cls])
        picked.extend(dev_by_class[cls][:per_class])
    if len(picked) < args.dev_max:
        remaining = []
        for cls in [0, 1, 2]:
            remaining.extend(dev_by_class[cls][per_class:])
        _rng.shuffle(remaining)
        picked.extend(remaining[: args.dev_max - len(picked)])
    _rng.shuffle(picked)
    dev_rows = picked
    log(
        f"dev 三层分层采样 {len(dev_rows)} 条 (answer={sum(1 for r in dev_rows if r.get('decision') == 'answer')} / wrong_object={sum(1 for r in dev_rows if r.get('decision') == 'abstain' and r.get('reason') == 'wrong_object')} / no_target={sum(1 for r in dev_rows if r.get('decision') == 'abstain' and r.get('reason') == 'no_target')})"
    )

    sft_mine = sft_rows[rank::world]
    dev_mine = dev_rows[rank::world]
    log(f"rank{rank}: sft_train={len(sft_mine)}, dev={len(dev_mine)}")

    sft_feats, sft_labels = [], []
    for i, row in enumerate(sft_mine):
        try:
            ut, ip, lab = extract_sft_train_fields(row)
            system = row.get("system", DECISION_SYSTEM_PROMPT)
            msgs = build_messages(system, ut, ip)
            feat = get_hidden_feature(model, processor, device, msgs, add_generation_prompt=True)
            sft_feats.append(feat)
            sft_labels.append(lab)
        except Exception as e:
            log(f"sft_train 样本{i} 异常: {type(e).__name__}: {e}")
            log(traceback.format_exc().split("\n")[-2])
            continue

    dev_feats, dev_labels = [], []
    for i, row in enumerate(dev_mine):
        try:
            ut, ip, lab = extract_dev_fields(row)
            msgs = build_messages(DECISION_SYSTEM_PROMPT, ut, ip)
            feat = get_hidden_feature(model, processor, device, msgs, add_generation_prompt=True)
            dev_feats.append(feat)
            dev_labels.append(lab)
        except Exception as e:
            log(f"dev 样本{i} 异常: {type(e).__name__}: {e}")
            log(traceback.format_exc().split("\n")[-2])
            continue

    sft_shard_path = args.log_dir / f"calib_train_rank{rank}.npz"
    dev_shard_path = args.log_dir / f"calib_dev_rank{rank}.npz"

    if len(sft_feats) > 0:
        np.savez(
            str(sft_shard_path),
            feats=np.array(sft_feats, dtype=np.float32),
            labels=np.array(sft_labels, dtype=np.int64),
        )
        (args.log_dir / f"calib_train_rank{rank}.done").write_text("ok")
        log(f"写 sft shard {sft_shard_path} ({len(sft_feats)} 条)")
    else:
        log("sft_train 无有效样本，写空 shard")
        np.savez(
            str(sft_shard_path),
            feats=np.empty((0, 4096), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
        )
        (args.log_dir / f"calib_train_rank{rank}.done").write_text("ok")

    if len(dev_feats) > 0:
        np.savez(
            str(dev_shard_path),
            feats=np.array(dev_feats, dtype=np.float32),
            labels=np.array(dev_labels, dtype=np.int64),
        )
        (args.log_dir / f"calib_dev_rank{rank}.done").write_text("ok")
        log(f"写 dev shard {dev_shard_path} ({len(dev_feats)} 条)")
    else:
        log("dev 无有效样本，写空 shard")
        np.savez(
            str(dev_shard_path),
            feats=np.empty((0, 4096), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
        )
        (args.log_dir / f"calib_dev_rank{rank}.done").write_text("ok")

    if rank == 0:
        for _ in range(180):
            if all(
                (args.log_dir / f"calib_train_rank{r}.done").exists()
                and (args.log_dir / f"calib_dev_rank{r}.done").exists()
                for r in range(world)
            ):
                break
            time.sleep(2)
        total_train = sum(
            _safe_load_npz(args.log_dir / f"calib_train_rank{r}.npz")["feats"].shape[0]
            for r in range(world)
        )
        total_dev = sum(
            _safe_load_npz(args.log_dir / f"calib_dev_rank{r}.npz")["feats"].shape[0]
            for r in range(world)
        )
        if total_train == 0:
            log("ERROR: sft_train 总有效样本为 0")
            return 1
        if total_dev == 0:
            log("ERROR: dev 总有效样本为 0")
            return 1
        log(f"extract 完成: sft_train={total_train}, dev={total_dev}")
        _write_world_metadata(args, world, total_train, total_dev)
    return 0


def _write_world_metadata(args, world: int, total_train: int, total_dev: int) -> None:
    """Rank 0 writes `world_size.json` once every shard is complete."""
    meta = {
        "world_size": world,
        "total_train": total_train,
        "total_dev": total_dev,
        "adapter": args.adapter,
    }
    meta_path = args.log_dir / "world_size.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log(
        f"已写 world_size.json: world={world}, train={total_train}, dev={total_dev}, adapter={args.adapter}"
    )


def _read_world_metadata(args) -> dict:
    """Read `world_size.json` for mode=train and fail fast when it is missing."""
    meta_path = args.log_dir / "world_size.json"
    if not meta_path.exists():
        log("FATAL: world_size.json 不存在，请先运行 --mode extract")
        raise SystemExit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    meta_adapter = meta.get("adapter", "")
    if meta_adapter != args.adapter:
        log(
            f"FATAL: world_size.json 中的 adapter ({meta_adapter}) 与当前 adapter ({args.adapter}) 不一致，"
            f"拒绝跨 checkpoint 缓存。请先运行 --mode extract 校准当前 checkpoint"
        )
        raise SystemExit(1)
    return meta


def _safe_load_npz(path, retries=3):
    """Load an npz defensively, retrying on truncated-archive errors."""
    import time

    for attempt in range(retries):
        try:
            return np.load(path)
        except Exception as e:
            if attempt == retries - 1:
                raise
            log(f"_safe_load_npz 第{attempt + 1}次重试: {type(e).__name__}: {e}")
            time.sleep(1)


def mcc_from_preds(preds, labels) -> float:
    """Multi-class MCC from predictions and labels.

    Uses the contingency-matrix formulation, matching the scikit-learn definition.
    """
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(labels, preds))


def mode_train(args) -> int:
    """Fit the head on one device and write it back into the checkpoint."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = _read_world_metadata(args)
    n_shards = meta["world_size"]
    log(
        f"world_size.json: n_shards={n_shards}, total_train={meta['total_train']}, total_dev={meta['total_dev']}, adapter={meta['adapter']}"
    )

    for r in range(n_shards):
        train_npz = args.log_dir / f"calib_train_rank{r}.npz"
        train_done = args.log_dir / f"calib_train_rank{r}.done"
        dev_npz = args.log_dir / f"calib_dev_rank{r}.npz"
        dev_done = args.log_dir / f"calib_dev_rank{r}.done"
        if not train_npz.exists():
            log(f"FATAL: train shard rank{r} npz 不存在: {train_npz}")
            return 1
        if not train_done.exists():
            log(f"FATAL: train shard rank{r} .done 不存在: {train_done}")
            return 1
        if not dev_npz.exists():
            log(f"FATAL: dev shard rank{r} npz 不存在: {dev_npz}")
            return 1
        if not dev_done.exists():
            log(f"FATAL: dev shard rank{r} .done 不存在: {dev_done}")
            return 1

    log("合并 shard")
    train_feats, train_labels = [], []
    dev_feats, dev_labels = [], []
    for r in range(n_shards):
        t = _safe_load_npz(args.log_dir / f"calib_train_rank{r}.npz")
        train_feats.append(t["feats"])
        train_labels.append(t["labels"])
        t = _safe_load_npz(args.log_dir / f"calib_dev_rank{r}.npz")
        dev_feats.append(t["feats"])
        dev_labels.append(t["labels"])
    X_train = np.concatenate(train_feats).astype(np.float32)
    y_train = np.concatenate(train_labels).astype(np.int64)
    X_dev = np.concatenate(dev_feats).astype(np.float32)
    y_dev = np.concatenate(dev_labels).astype(np.int64)
    log(f"train: {X_train.shape}, pos={int((y_train == 1).sum())} neg={int((y_train == 0).sum())}")
    log(f"dev:   {X_dev.shape},  pos={int((y_dev == 1).sum())} neg={int((y_dev == 0).sum())}")

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-6
    log(f"mu: {mu.shape}, sd: {sd.shape}")

    X_train_std = (X_train - mu) / sd
    X_dev_std = (X_dev - mu) / sd

    log("训练 MLP(4096→512→256→3) 三分类")
    X_train_t = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_dev_t = torch.tensor(X_dev_std, dtype=torch.float32, device=device)
    y_dev_t = torch.tensor(y_dev, dtype=torch.long, device=device)

    mlp = nn.Sequential(
        nn.Linear(4096, 512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 3),
    ).to(device)

    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()

    best_dev_loss = float("inf")
    best_state = None
    patience = 0
    patience_limit = 5
    eval_every = 50

    mlp.train()
    for epoch in range(1000):
        opt.zero_grad()
        logits = mlp(X_train_t)
        loss = lossf(logits, y_train_t)
        loss.backward()
        opt.step()

        if (epoch + 1) % eval_every == 0:
            mlp.eval()
            with torch.inference_mode():
                dev_logits_eval = mlp(X_dev_t)
                dev_loss = lossf(dev_logits_eval, y_dev_t).item()
                dev_preds_eval = dev_logits_eval.argmax(-1).cpu().numpy()
            cur_dev_mcc = mcc_from_preds(dev_preds_eval, y_dev)
            per_class_acc = {}
            for cls_id in range(3):
                cls_mask = y_dev == cls_id
                if cls_mask.any():
                    cls_acc = (dev_preds_eval[cls_mask] == y_dev[cls_mask]).mean()
                    cls_name = ["answer", "wrong_object", "no_target"][cls_id]
                    per_class_acc[cls_name] = f"{cls_acc:.2%}"
            log(
                f"  epoch {epoch + 1}/1000, train_loss={loss.item():.4f}, dev_loss={dev_loss:.4f}, dev_MCC={cur_dev_mcc:.4f}, per_class_acc={per_class_acc}"
            )
            mlp.train()

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_state = copy.deepcopy(mlp.state_dict())
                patience = 0
                log(f"    → 新最佳 dev loss: {best_dev_loss:.4f} (epoch {epoch + 1})")
            else:
                patience += 1
                log(f"    → patience {patience}/{patience_limit}")
                if patience >= patience_limit:
                    log(f"  早停于 epoch {epoch + 1}，最佳 dev loss: {best_dev_loss:.4f}")
                    break

    if best_state is not None:
        mlp.load_state_dict(best_state)
        log(f"已恢复最佳权重（最佳 dev loss: {best_dev_loss:.4f}）")

    mlp.eval()
    with torch.inference_mode():
        train_logits = mlp(X_train_t)
        train_preds = train_logits.argmax(-1).cpu().numpy()
        dev_logits = mlp(X_dev_t)
        dev_preds = dev_logits.argmax(-1).cpu().numpy()

    train_mcc = mcc_from_preds(train_preds, y_train)
    dev_mcc = mcc_from_preds(dev_preds, y_dev)
    log(f"train MCC (multi-class): {train_mcc:.4f}")
    log(f"dev   MCC (multi-class): {dev_mcc:.4f}")

    cls_names = ["answer(0)", "wrong_object(1)", "no_target(2)"]
    try:
        report = classification_report(y_dev, dev_preds, target_names=cls_names, digits=4)
        log(f"\n=== dev classification_report ===\n{report}")
    except Exception as e:
        log(f"classification_report 失败: {e}")

    for cls_id in range(3):
        cls_mask = y_dev == cls_id
        if cls_mask.any():
            cls_acc = (dev_preds[cls_mask] == y_dev[cls_mask]).mean()
            cls_name = cls_names[cls_id]
            log(f"  {cls_name} acc: {cls_acc:.4f} ({cls_mask.sum()} 样本)")

    log("写 discrim_head_calibrated.pt")
    classifier_state = {}
    for name, mod in mlp.named_modules():
        if isinstance(mod, nn.Linear):
            classifier_state[name] = mod.state_dict()

    head_state = {}
    for name, state in classifier_state.items():
        for k, v in state.items():
            head_state[f"classifier.{name}.{k}"] = v

    out_pt = Path(args.adapter) / "discrim_head_calibrated.pt"
    save_dict = {
        "head_state": head_state,
        "mu": torch.tensor(mu, dtype=torch.float32),
        "sd": torch.tensor(sd, dtype=torch.float32),
    }
    torch.save(save_dict, str(out_pt))
    log(f"已保存: {out_pt}")
    log(f"head_state keys: {list(head_state.keys())}")
    log(f"mu: {mu.shape}, sd: {sd.shape}")

    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Discriminative head calibration script (3reason three-class version)")
    parser.add_argument(
        "--mode",
        choices=["extract", "train"],
        required=True,
        help="extract: multi-GPU torchrun representation extraction; train: single-GPU head training written back into the checkpoint",
    )
    parser.add_argument(
        "--base",
        default=os.environ.get("BASE_MODEL", ""),
        help="Base model path (may also be set via BASE_MODEL)",
    )
    parser.add_argument(
        "--adapter",
        default=os.environ.get("CALIB_ADAPTER", ""),
        help="LoRA adapter directory (checkpoint-N; defaults to the CALIB_ADAPTER env var)",
    )
    parser.add_argument(
        "--sft-train",
        default=str(_DEFAULT_SFT_TRAIN),
        help=f"Path to sft_train.json (default: {_DEFAULT_SFT_TRAIN})",
    )
    parser.add_argument(
        "--dev", default=str(_DEFAULT_DEV), help=f"Path to dev.jsonl (default: {_DEFAULT_DEV})"
    )
    parser.add_argument(
        "--dev-max", type=int, default=100, help="Maximum number of stratified dev samples (default: 100)"
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("CALIB_LOG_DIR", str(_THIS_DIR / "outputs" / "calib_logs")),
        help="Log/cache directory (defaults to the CALIB_LOG_DIR env var, otherwise training/sft_final/outputs/calib_logs)",
    )
    args = parser.parse_args()

    if not args.base:
        parser.error("需要 --base 或 BASE_MODEL 环境变量")

    args.log_dir = Path(args.log_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "extract":
        if not args.adapter:
            parser.error("--mode extract 需要 --adapter 或 CALIB_ADAPTER 环境变量")
        return mode_extract(args)
    elif args.mode == "train":
        if not args.adapter:
            parser.error("--mode train 需要 --adapter 或 CALIB_ADAPTER 环境变量")
        return mode_train(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
