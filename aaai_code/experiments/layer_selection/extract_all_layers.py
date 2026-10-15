#!/usr/bin/env python3
"""extract_all_layers.py — CLI to extract 36-layer hidden states from selection records.

Extracts generation-prompt last-token hidden_states[1:] (excluding embedding)
for each sample in the selection records.  Outputs train/holdout NPZ files and
a manifest JSON.

Usage:
    python -m experiments.layer_selection.extract_all_layers \
        --base /path/to/base \
        --adapter /path/to/checkpoint-444 \
        --sft-train /path/to/sft_train.json \
        --selection-records /path/to/selection_records.jsonl \
        --split-assignment /path/to/split_assignment.json \
        --output-dir /path/to/output
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

import numpy as np
import torch

# ── REPO SETUP ──────────────────────────────────────────────────────────────
AAAI_CODE = Path(__file__).resolve().parents[2]
TRAINING_ROOT = AAAI_CODE / "training"
sys.path.insert(0, str(TRAINING_ROOT))

# ── Constants ───────────────────────────────────────────────────────────────
EXPECTED_N_LAYERS = 36
EXPECTED_HS_TUPLE_LEN = 37  # [0]=embedding, [1..36]=layer1..36
MAX_PIXELS = 262144
CHUNK_SIZE = 8 * 1024 * 1024


# ── Helpers ─────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    print(f"[extract_all_layers r{rank}] {msg}", flush=True)


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
        return {"config_sha256": file_sha256(base_path), "n_shards": 0}
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
    return file_sha256(path) if path.exists() else None


def build_messages(record: dict) -> list[dict]:
    """Construct messages in sft_train format (same as extract_hidden.py)."""
    system = record.get("system", "")
    conversations = record.get("conversations", [])
    human_text = ""
    for conv in conversations:
        if conv.get("from") == "human":
            human_text = conv.get("value", "").replace("<image>", "").strip()
            break
    image_path = (
        record["image"][0] if isinstance(record.get("image"), list) else record.get("image", "")
    )
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


# ── Fingerprint ─────────────────────────────────────────────────────────────


def compute_fingerprint(args: argparse.Namespace) -> dict:
    return {
        "input_sha256": file_sha256(args.sft_train),
        "base_identity": compute_base_identity(args.base),
        "adapter_model_sha256": adapter_model_sha(args.adapter),
        "script_sha256": file_sha256(__file__),
        "max_pixels": MAX_PIXELS,
        "selection_records_sha256": file_sha256(args.selection_records),
        "split_assignment_sha256": file_sha256(args.split_assignment),
        "expected_n_layers": EXPECTED_N_LAYERS,
    }


def fingerprint_path(out_dir: Path) -> Path:
    return out_dir / ".fingerprint_extract.json"


def check_fingerprint(args: argparse.Namespace) -> bool:
    fp_path = fingerprint_path(Path(args.output_dir))
    if not fp_path.exists():
        return True
    try:
        existing = json.loads(fp_path.read_text())
    except (json.JSONDecodeError, OSError):
        return True
    current = compute_fingerprint(args)
    for key in [
        "input_sha256",
        "base_identity",
        "adapter_model_sha256",
        "script_sha256",
        "max_pixels",
        "selection_records_sha256",
        "split_assignment_sha256",
    ]:
        if existing.get(key) != current.get(key):
            return False
    return True


def write_fingerprint(args: argparse.Namespace) -> None:
    fp = compute_fingerprint(args)
    fp_path = fingerprint_path(Path(args.output_dir))
    tmp = fp_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fp, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(fp_path)


# ── Model loading ──────────────────────────────────────────────────────────


def load_model_and_processor(base: str, adapter: str, device: torch.device):
    """Load base model + LoRA adapter, return (model, processor)."""
    # Lazy imports so the module can be imported without heavy deps.
    from peft import PeftModel  # noqa: E402
    from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead  # noqa: E402
    from transformers import AutoProcessor  # noqa: E402

    log("Loading processor")
    processor = AutoProcessor.from_pretrained(
        base,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )

    log("Loading base model (Qwen3VLWithTriHead)")
    model = Qwen3VLWithTriHead.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)

    log(f"Loading LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter).to(device)
    model.eval()
    return model, processor


# ── Extraction ──────────────────────────────────────────────────────────────


def extract_all_layers_hidden(
    model,
    processor,
    record: dict,
    device: torch.device,
) -> np.ndarray | None:
    """Extract all 36 layers' generation-prompt last-token hidden states.

    Returns shape (n_layers, 4096) float32, or None on failure.
    """
    from qwen_vl_utils import process_vision_info  # noqa: E402

    try:
        messages = build_messages(record)
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
        if hs_tuple is None:
            log("WARNING: hidden_states is None")
            return None

        n_hs = len(hs_tuple)
        if n_hs != EXPECTED_HS_TUPLE_LEN:
            raise ValueError(f"expected {EXPECTED_HS_TUPLE_LEN} hidden states, got {n_hs}")
        # Last-token index
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            last_idx = int(attention_mask[0].sum()) - 1
            last_idx = max(0, min(last_idx, hs_tuple[1].shape[1] - 1))
        else:
            last_idx = hs_tuple[1].shape[1] - 1

        layer_hiddens = []
        for i in range(1, n_hs):  # skip embedding at index 0
            layer_hs = hs_tuple[i]
            hidden = layer_hs[0, last_idx, :].cpu().float().numpy()
            layer_hiddens.append(hidden)

        all_hidden = np.stack(layer_hiddens, axis=0)
        return all_hidden

    except Exception as e:
        log(f"extract_all_layers_hidden error: {type(e).__name__}: {e}")
        log(traceback.format_exc().split("\n")[-2])
        return None


# ── Main extraction ─────────────────────────────────────────────────────────


def run_extraction(args: argparse.Namespace) -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    # Load data
    with open(args.selection_records) as f:
        records = [json.loads(line) for line in f if line.strip()]
    log(f"selection records: {len(records)}")

    with open(args.split_assignment) as f:
        split_assignment: dict[str, str] = json.load(f)

    train_sids = {sid for sid, sp in split_assignment.items() if sp == "train"}
    holdout_sids = {sid for sid, sp in split_assignment.items() if sp == "holdout"}
    train_records = [s for s in records if s["source_image_id"] in train_sids]
    holdout_records = [s for s in records if s["source_image_id"] in holdout_sids]
    log(f"train={len(train_records)}, holdout={len(holdout_records)}")

    if args.split == "train":
        target = train_records
    elif args.split == "holdout":
        target = holdout_records
    else:
        target = train_records + holdout_records

    if args.limit is not None and args.limit > 0:
        target = target[: args.limit]
        log(f"--limit={args.limit}, target={len(target)}")

    if args.dry_run:
        log(
            f"--dry-run mode: train={len(train_records)}, "
            f"holdout={len(holdout_records)}, target={len(target)}"
        )
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not check_fingerprint(args):
        if args.reset:
            for p in out_dir.glob("hidden_*.npz"):
                p.unlink()
            for p in out_dir.glob(".fingerprint*"):
                p.unlink()
        else:
            log("FATAL: fingerprint mismatch. Use --reset to rebuild.")
            return 1
    write_fingerprint(args)

    # Load model

    model, processor = load_model_and_processor(args.base, args.adapter, device)

    # Load sft_train for message construction
    with open(args.sft_train) as f:
        sft_train_data = json.load(f)
    log(f"Loaded sft_train.json: {len(sft_train_data)} records")

    # Extract
    train_hidden: dict[str, np.ndarray] = {}
    holdout_hidden: dict[str, np.ndarray] = {}
    t0 = time.time()

    for batch_i, rec in enumerate(target):
        orig_idx = rec["original_idx"]
        sid = rec["source_image_id"]
        sft_rec = sft_train_data[orig_idx]

        hidden = extract_all_layers_hidden(model, processor, sft_rec, device)
        if hidden is None:
            log(f"FATAL: original_idx={orig_idx} extraction failed")
            return 1

        key = f"hidden_{orig_idx}"
        target_dict = train_hidden if sid in train_sids else holdout_hidden
        target_dict[key] = hidden

        if (batch_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (batch_i + 1) / elapsed
            log(f"progress {batch_i + 1}/{len(target)} ({elapsed:.1f}s, {rate:.2f} it/s)")

    dt = time.time() - t0
    log(f"Extraction done: {len(target)} samples in {dt:.1f}s")

    # Write NPZ files
    def write_split_npz(data_dict: dict, path: Path) -> None:
        sorted_keys = sorted(data_dict.keys(), key=lambda k: int(k.split("_")[1]))
        hiddens = []
        orig_idxs = []
        head_labels = []
        roles = []
        source_image_ids = []
        cluster_ids = []
        difficulty_bins = []

        # Build a lookup from record
        rec_by_idx = {r["original_idx"]: r for r in records}

        for k in sorted_keys:
            hiddens.append(data_dict[k])
            idx = int(k.split("_")[1])
            orig_idxs.append(idx)
            r = rec_by_idx.get(idx, {})
            head_labels.append(r.get("head_label", -1))
            roles.append(r.get("role", "unknown"))
            source_image_ids.append(r.get("source_image_id", ""))
            cluster_ids.append(r.get("cluster_id", ""))
            difficulty_bins.append(r.get("difficulty_bin", None))

        stacked = np.stack(hiddens, axis=0).astype(np.float32)
        n_layers = stacked.shape[1]
        log(f"  hidden shape: {stacked.shape}")

        role_map = {"wo_core": 0, "answer_anchor": 1, "nt_control": 2}
        diff_map = {"stubborn": 0, "moderate": 1, "borderline": 2, "keep_correct": 3}

        out = {
            "original_idx": np.array(orig_idxs, dtype=np.int64),
            "head_label": np.array(head_labels, dtype=np.int64),
            "hidden": stacked,
            "n_layers": np.array([n_layers], dtype=np.int64),
            "hs_indices": np.arange(1, n_layers + 1, dtype=np.int64),
            "layer_index_note": (
                "npz array index L = hidden_states[L+1] = transformer layer L+1; "
                "baseline discrim head uses transformer layer 27 = npz index 26"
            ),
            "role": np.array([role_map.get(r, -1) for r in roles], dtype=np.int64),
            "difficulty_bin": np.array(
                [diff_map.get(d, -1) if d is not None else -1 for d in difficulty_bins],
                dtype=np.int64,
            ),
            "source_image_id": np.array(source_image_ids, dtype=object),
            "cluster_id": np.array(cluster_ids, dtype=object),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        log(f"Saved {path} ({len(orig_idxs)} samples, {n_layers} layers)")

    train_path = out_dir / "hidden_train.npz"
    holdout_path = out_dir / "hidden_holdout.npz"
    if train_hidden:
        write_split_npz(train_hidden, train_path)
    if holdout_hidden:
        write_split_npz(holdout_hidden, holdout_path)

    # Manifest
    manifest = {
        "base": args.base,
        "adapter": args.adapter,
        "input": args.sft_train,
        "input_sha256": file_sha256(args.sft_train),
        "selection_records": args.selection_records,
        "selection_records_sha256": file_sha256(args.selection_records),
        "split_assignment": args.split_assignment,
        "split_assignment_sha256": file_sha256(args.split_assignment),
        "max_pixels": MAX_PIXELS,
        "n_layers": EXPECTED_N_LAYERS,
        "n_train": len(train_hidden),
        "n_holdout": len(holdout_hidden),
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(__file__),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path = out_dir / "manifest_extract.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    log(f"Manifest: {manifest_path}")
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract all 36 layers hidden states for layer diagnostic",
    )
    ap.add_argument("--base", required=True, help="Base model path")
    ap.add_argument("--adapter", required=True, help="LoRA adapter (checkpoint-444) path")
    ap.add_argument("--sft-train", required=True, help="sft_train.json path")
    ap.add_argument(
        "--selection-records", required=True, help="Selection records JSONL (rl_samples equivalent)"
    )
    ap.add_argument(
        "--split-assignment",
        required=True,
        help="split_assignment.json (maps source_image_id -> train/holdout)",
    )
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument(
        "--split",
        choices=["train", "holdout", "all"],
        default="all",
        help="Which split to extract (default: all)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N samples (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="Don't load model, just print counts")
    ap.add_argument("--reset", action="store_true", help="Force clear old output and re-extract")
    args = ap.parse_args()

    if args.dry_run:
        with open(args.selection_records) as f:
            records = [json.loads(line) for line in f if line.strip()]
        with open(args.split_assignment) as f:
            sa: dict[str, str] = json.load(f)
        train_sids = {s for s, sp in sa.items() if sp == "train"}
        holdout_sids = {s for s, sp in sa.items() if sp == "holdout"}
        n_train = sum(1 for r in records if r["source_image_id"] in train_sids)
        n_holdout = sum(1 for r in records if r["source_image_id"] in holdout_sids)
        log(f"--dry-run: train={n_train}, holdout={n_holdout}")
        return 0

    return run_extraction(args)


if __name__ == "__main__":
    raise SystemExit(main())
