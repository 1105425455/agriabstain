#!/usr/bin/env python
"""Extract head probabilities (p_answer, p_wo, p_nt) for L2/L3 evidence data.

Head-only forward (no text generation, no GT fallback).  Supports counterfactual
variants: query rewrites via GROUNDING_TEMPLATES, and photometric image transforms
(brightness/contrast).  Torchrun-compatible with shard-based resume and fingerprint.

Usage:
    python experiments/evidence_judge/extract_head_probabilities.py \\
        --base <qwen3vl> --adapter <lora> --calibrated-head <calib.pt> \\
        --data annotations/eval/test_L2_evidence.jsonl \\
        --output outputs/evidence_prob/L2

    torchrun --nproc_per_node=2 ... \\
        --data ... --output ... --limit 100
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
from typing import Any

# ── path setup ──────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_AAAI_CODE = (_THIS_DIR / ".." / "..").resolve()  # experiments/evidence_judge/../../ = aaai_code
_TRAINING = _AAAI_CODE / "training"
sys.path.insert(0, str(_AAAI_CODE))
sys.path.insert(0, str(_TRAINING))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageEnhance  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from reason_abstain.bench.templates import GROUNDING_TEMPLATES  # noqa: E402

# ── constants ───────────────────────────────────────────────────────────────

DECISION_SYSTEM_PROMPT = (
    "You are AgriAbstain, a vision-language grounding assistant for high-stakes plant disease diagnosis. "
    "Locate the queried target. If the queried target is present and localizable, return "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]} with box in [0,1000] relative coords. '
    "Otherwise abstain: "
    '{"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image, or cropped out of view)."
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _positive_int(v: str) -> int:
    """Argparse type: positive integer."""
    iv = int(v)
    if iv < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {v!r}")
    return iv


def _derived_root_from_args(args: argparse.Namespace) -> Path:
    """return resolved derived root, falling back to package default."""
    dr = getattr(args, "derived_root", None)
    if dr is not None:
        return Path(dr).resolve()
    return (_AAAI_CODE / "benchmark" / "data" / "derived").resolve()


def rank_world() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world, local_rank


def log(rank: int, msg: str) -> None:
    print(f"[extract_prob r{rank}] {msg}", flush=True)


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def shard_path(out_dir: Path, rank: int, world: int) -> Path:
    return out_dir / "shards" / f"probs.rank{rank}of{world}.jsonl"


def _file_sig(p: str | None) -> str:
    """Return 'path@size_mtime' for fingerprinting."""
    if not p:
        return "NONE"
    pp = Path(p)
    try:
        st = pp.stat()
        return f"{pp}@{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        return f"{pp}@-1_-1"


def fingerprint_from_args(args: argparse.Namespace, data_path: Path) -> str:
    # File size + mtime for critical artifacts
    adapter_sig = _file_sig(str(Path(args.adapter) / "adapter_model.safetensors"))
    calib_sig = _file_sig(args.calibrated_head)
    try:
        data_st = data_path.stat()
        data_sig = f"{data_path}@{data_st.st_size}_{int(data_st.st_mtime)}"
    except OSError:
        data_sig = f"{data_path}@-1_-1"
    base_sig = _file_sig(args.base)
    parts = [
        f"base={base_sig}",
        f"adapter_model={adapter_sig}",
        f"calibrated_head={calib_sig}",
        f"data={data_sig}",
        f"world={int(os.environ.get('WORLD_SIZE', 1))}",
        f"limit={args.limit}",
        f"max_pixels={args.max_pixels}",
        f"variant_set={args.variant_set}",
        f"derived_root={_derived_root_from_args(args)}",
        f"batch_size={args.batch_size}",
    ]
    return " | ".join(parts)


def stale_shards_check(out_dir: Path, fingerprint: str, rank: int) -> None:
    fp_file = out_dir / "shards" / ".fingerprint"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    old = fp_file.read_text().strip() if fp_file.exists() else None
    if old is not None and old != fingerprint:
        removed = 0
        for sh in shard_dir.glob("probs.rank*.jsonl"):
            sh.unlink()
            removed += 1
        log(rank, f"fingerprint changed -> cleared {removed} stale shard(s)")
        log(rank, f"  old: {old}")
        log(rank, f"  new: {fingerprint}")
    fp_file.write_text(fingerprint)


def load_done_ids(path: Path) -> set[tuple[int, str]]:
    """Return set of (original_index, variant_id) tuples from existing shard."""
    done: set[tuple[int, str]] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                oi = rec.get("original_index")
                vid = rec.get("variant_id", "")
                if oi is not None:
                    done.add((int(oi), str(vid)))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return done


def resolve_image_path(image_rel: str, derived_root: str | None = None) -> str:
    """Resolve relative image path.

    - data/derived/... → mapped to derived_root / suffix
    - data/source/... etc → benchmark root
    - absolute paths → unchanged
    """
    p = Path(image_rel)
    if p.is_absolute():
        return str(p)
    if image_rel.startswith("data/derived/"):
        if derived_root is None:
            derived_root = str(_AAAI_CODE / "benchmark" / "data" / "derived")
        suffix = image_rel[len("data/derived/") :]
        return str(Path(derived_root) / suffix)
    return str(_AAAI_CODE / "benchmark" / image_rel)


def _find_user_text(messages: list[dict]) -> str:
    """Find the user message content (string only, not list)."""
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, str):
                return content
            # If it's a list, find the text part
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part["text"]
    return ""


def parse_target_label_from_messages(messages: list[dict]) -> str:
    """Parse target label from the user message that uses GROUNDING_TEMPLATES.

    These templates are:
        "Locate the {label} in this image."
        "Find the {label} and return its bounding box."
        "Where is the {label} in the image?"

    Only accepts string content (not list-based multimodal).  Raises ValueError
    on failure.
    """
    user_text = _find_user_text(messages)
    if not user_text:
        raise ValueError("No user message found")

    # Strip <image> tag if present
    clean = user_text.replace("<image>", "").strip()

    # Try each template pattern — only accept exact template match
    for tmpl in GROUNDING_TEMPLATES:
        parts_before = tmpl.split("{label}")
        if len(parts_before) != 2:
            continue
        prefix, suffix = parts_before
        if clean.startswith(prefix) and clean.endswith(suffix):
            label = clean[len(prefix) : len(clean) - len(suffix)]
            if label:
                return label.strip()

    raise ValueError(
        f"Cannot parse target label from user message: {clean!r}. "
        f"Expected one of templates: {GROUNDING_TEMPLATES}"
    )


def _required_image_variant_ids(variant_set: str) -> list[str]:
    """Return fixed image variant IDs; query IDs depend on the original template."""
    ids = ["original"]
    if variant_set in ("image", "all"):
        for factor in (0.9, 1.1):
            ids.append(f"image_brightness_{factor}")
        for factor in (0.9, 1.1):
            ids.append(f"image_contrast_{factor}")
    return ids


# ── variant generation ──────────────────────────────────────────────────────


def _image_photometric_variants(pil_img: Image.Image) -> dict[str, Image.Image]:
    """Generate semantic-preserving photometric variants.

    Returns dict mapping variant_id -> PIL Image.  Only photometric (brightness,
    contrast); no geometry/flip transforms.
    """
    variants: dict[str, Image.Image] = {}
    for factor in (0.9, 1.1):
        enhancer = ImageEnhance.Brightness(pil_img)
        variants[f"brightness_{factor}"] = enhancer.enhance(factor)
    for factor in (0.9, 1.1):
        enhancer = ImageEnhance.Contrast(pil_img)
        variants[f"contrast_{factor}"] = enhancer.enhance(factor)
    return variants


def _build_messages_for_variant(system: str, image_path: str, query_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query_text},
            ],
        },
    ]


def _build_variants(
    row: dict,
    variant_set: str,
    tmp_dir: Path | None,
    derived_root: str | None = None,
) -> list[dict]:
    """Build variant specs for a single data row.

    Each spec is a dict with:
        variant_type: "original" | "query" | "image"
        variant_id: str
        messages: list[dict] (ready for processor)
        pil_image_override: Path | None (for image variants, path to temp file)

    original+query max 3 total, all max 7.
    """
    results: list[dict] = []

    system = DECISION_SYSTEM_PROMPT
    image_path = resolve_image_path(row["images"][0], derived_root=derived_root)
    target_label = parse_target_label_from_messages(row["messages"])
    original_query = _find_user_text(row["messages"]).replace("<image>", "").strip()

    # ── original ────────────────────────────────────────────────────────
    results.append(
        {
            "variant_type": "original",
            "variant_id": "original",
            "messages": _build_messages_for_variant(system, image_path, original_query),
            "pil_image_override": None,
        }
    )

    # ── query variants ──────────────────────────────────────────────────
    if variant_set in ("query", "all"):
        seen_queries: set[str] = {original_query}
        for tid, tmpl in enumerate(GROUNDING_TEMPLATES):
            q = tmpl.format(label=target_label)
            if q in seen_queries:
                continue
            seen_queries.add(q)
            results.append(
                {
                    "variant_type": "query",
                    "variant_id": f"query_{tid}",
                    "messages": _build_messages_for_variant(system, image_path, q),
                    "pil_image_override": None,
                }
            )

    # ── image variants ──────────────────────────────────────────────────
    if variant_set in ("image", "all"):
        pil_img = Image.open(image_path).convert("RGB")
        photo_variants = _image_photometric_variants(pil_img)
        for vid, var_img in photo_variants.items():
            if tmp_dir is not None:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                ext = Path(image_path).suffix or ".jpg"
                # Deterministic filename: original_index is not available here,
                # but we use variant_id which is unique per sample.
                # Caller will pass original_index; we store it in the spec.
                tmp_path = tmp_dir / f"tmp_{vid}{ext}"
                var_img.save(tmp_path)
                tmp_img_path = str(tmp_path)
            else:
                tmp_img_path = None

            var_messages = _build_messages_for_variant(
                system, tmp_img_path or image_path, original_query
            )
            results.append(
                {
                    "variant_type": "image",
                    "variant_id": f"image_{vid}",
                    "messages": var_messages,
                    "pil_image_override": tmp_img_path,
                }
            )
        pil_img.close()

    return results


# ── model loading ───────────────────────────────────────────────────────────


def load_model_and_processor(args: argparse.Namespace, device: str, rank: int):
    """Load Qwen3VLWithTriHead + adapter + calibrated head.

    Mirrors eval_qwen3vl_3reason.py alignment.
    """
    from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead
    from peft import PeftModel
    from safetensors.torch import load_file as sf_load

    processor = AutoProcessor.from_pretrained(
        args.base, max_pixels=args.max_pixels, trust_remote_code=True
    )

    # 1. Load base model
    model = Qwen3VLWithTriHead.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map=None,
    ).to(device)

    # 2. Load LoRA adapter
    if args.adapter:
        log(rank, f"loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.to(device)

    # 3. Load discrim_head from adapter safetensors (same as eval script)
    adapter_path = Path(args.adapter)
    adapter_safetensors = adapter_path / "adapter_model.safetensors"
    if adapter_safetensors.exists():
        log(rank, f"loading discrim_head from adapter: {adapter_safetensors}")
        adapter_state = sf_load(str(adapter_safetensors))
        discrim_state = {}
        for k, v in adapter_state.items():
            if k.startswith("base_model.model.discrim_head."):
                short_k = k.replace("base_model.model.discrim_head.", "")
                discrim_state[short_k] = v
        if discrim_state:
            m = model
            for _ in range(6):
                if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
                    m = m.base_model.model
                elif hasattr(m, "module"):
                    m = m.module
                else:
                    break
            if hasattr(m, "discrim_head"):
                dh = m.discrim_head
                missing, unexpected = dh.load_state_dict(discrim_state, strict=False)
                if missing:
                    log(rank, f"[WARN] discrim_head missing keys: {missing[:5]}")
                if unexpected:
                    log(rank, f"[WARN] discrim_head unexpected keys: {unexpected[:5]}")
                with torch.inference_mode():
                    w0 = dh.classifier[0].weight
                    log(
                        rank,
                        f"discrim_head loaded, classifier[0].weight norm={w0.norm().item():.4f}",
                    )

    # 4. Apply calibrated head (overrides discrim_head)
    if args.calibrated_head:
        calib_path = args.calibrated_head
        if os.path.exists(calib_path):
            log(rank, f"loading calibrated head: {calib_path}")
            calib = torch.load(calib_path, map_location="cpu", weights_only=False)
            m = model
            for _ in range(6):
                if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
                    m = m.base_model.model
                elif hasattr(m, "module"):
                    m = m.module
                else:
                    break
            discrim_head_module = m.discrim_head if hasattr(m, "discrim_head") else None
            if discrim_head_module is not None and hasattr(discrim_head_module, "modules_to_save"):
                active_adapters = getattr(discrim_head_module, "active_adapters", None)
                if active_adapters:
                    inner = discrim_head_module.modules_to_save[active_adapters[0]]
                    discrim_head_module = inner
            if discrim_head_module is not None:
                hs = calib["head_state"]
                clf_state = {k.replace("classifier.", ""): v for k, v in hs.items()}
                discrim_head_module.classifier.load_state_dict(clf_state)
                discrim_head_module.load_musd(calib["mu"], calib["sd"])
                with torch.inference_mode():
                    w0 = discrim_head_module.classifier[0].weight
                    sd_min = float(discrim_head_module.sd.min())
                    sd_max = float(discrim_head_module.sd.max())
                log(
                    rank,
                    f"calibrated head loaded, classifier[0].weight norm={w0.norm().item():.4f}, "
                    f"sd_range=[{sd_min:.3f},{sd_max:.3f}]",
                )

    model.eval()
    return model, processor


# ── head forward ────────────────────────────────────────────────────────────


def _head_forward_batch(
    model,
    processor,
    variant_specs: list[dict],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Batch head-only forward pass.

    Processes multiple variant specs in a single forward pass.
    Returns list of result dicts (one per variant), in the same order as variant_specs.
    Each result: {p_answer, p_wo, p_nt, predicted_class, decision}.
    No text generation.  No GT fallback.
    """
    from qwen_vl_utils import process_vision_info

    batch_texts: list[str] = []
    batch_images: list[Image.Image] = []

    for vs in variant_specs:
        messages = vs["messages"]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_texts.append(text)
        image_inputs, video_inputs = process_vision_info(messages)
        # Each variant produces exactly one image (or none)
        if image_inputs:
            batch_images.append(image_inputs[0])
        else:
            # Create a dummy 1x1 image if no images (shouldn't happen for evidence data)
            batch_images.append(Image.new("RGB", (1, 1)))

    # Processor batch: text=list[str], images=flattened (one per sample)
    inputs = processor(
        text=batch_texts,
        images=[img for img in batch_images],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    # Extract head_logits — should be [B, 3]
    head_logits = None
    if isinstance(outputs, dict):
        head_logits = outputs.get("head_logits")
    else:
        head_logits = getattr(outputs, "head_logits", None)

    if head_logits is None:
        m = model
        for _ in range(6):
            if getattr(m, "last_head_logits", None) is not None:
                break
            if hasattr(m, "module"):
                m = m.module
            elif hasattr(m, "base_model") and hasattr(m.base_model, "model"):
                m = m.base_model.model
            else:
                break
        head_logits = getattr(m, "last_head_logits", None)

    B = len(variant_specs)

    results: list[dict[str, Any]] = []
    if head_logits is None:
        for _ in range(B):
            results.append(
                {
                    "p_answer": None,
                    "p_wo": None,
                    "p_nt": None,
                    "predicted_class": None,
                    "decision": None,
                    "error": "no head_logits",
                }
            )
        return results

    # head_logits should be [B, 3]
    if head_logits.shape[0] != B:
        raise RuntimeError(
            f"head_logits batch dimension ({head_logits.shape[0]}) does not match "
            f"requested batch size ({B})"
        )

    probs = F.softmax(head_logits.float(), dim=-1)  # [B, 3]

    for i in range(B):
        p_answer = probs[i, 0].item()
        p_wo = probs[i, 1].item()
        p_nt = probs[i, 2].item()
        pred_class = int(head_logits[i].argmax(dim=-1).item())
        decision = "abstain" if pred_class >= 1 else "answer"
        results.append(
            {
                "p_answer": p_answer,
                "p_wo": p_wo,
                "p_nt": p_nt,
                "predicted_class": pred_class,
                "decision": decision,
            }
        )

    return results


def _head_forward_single(
    model,
    processor,
    messages: list[dict],
    device: torch.device,
) -> dict[str, Any]:
    """Single head-only forward pass (fallback for error isolation).

    Returns {p_answer, p_wo, p_nt, predicted_class, decision}.
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    head_logits = None
    if isinstance(outputs, dict):
        head_logits = outputs.get("head_logits")
    else:
        head_logits = getattr(outputs, "head_logits", None)

    if head_logits is None:
        m = model
        for _ in range(6):
            if getattr(m, "last_head_logits", None) is not None:
                break
            if hasattr(m, "module"):
                m = m.module
            elif hasattr(m, "base_model") and hasattr(m.base_model, "model"):
                m = m.base_model.model
            else:
                break
        head_logits = getattr(m, "last_head_logits", None)

    if head_logits is None:
        return {
            "p_answer": None,
            "p_wo": None,
            "p_nt": None,
            "predicted_class": None,
            "decision": None,
            "error": "no head_logits",
        }

    probs = F.softmax(head_logits.float(), dim=-1)
    return {
        "p_answer": probs[0, 0].item(),
        "p_wo": probs[0, 1].item(),
        "p_nt": probs[0, 2].item(),
        "predicted_class": int(head_logits.argmax(dim=-1).item()),
        "decision": "abstain" if int(head_logits.argmax(dim=-1).item()) >= 1 else "answer",
    }


# ── main extraction ─────────────────────────────────────────────────────────


def run_extraction(args: argparse.Namespace) -> None:
    rank, world, local_rank = rank_world()
    device = (
        f"cuda:{local_rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

    data_path = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── fingerprint ─────────────────────────────────────────────────────
    fp = fingerprint_from_args(args, data_path)
    derived_root = str(_derived_root_from_args(args))
    log(rank, f"derived_root: {derived_root}")
    # Per-rank tmp dir: fingerprint-hash + rank to avoid cross-rank conflicts
    fp_hash = hashlib.sha256(fp.encode()).hexdigest()[:12]
    needs_tmp = args.variant_set in ("image", "all")
    tmp_root = out_dir.parent / ".tmp_extract_probs"
    # Per-rank subdirectory
    tmp_dir = (tmp_root / f"{fp_hash}_r{rank}") if needs_tmp else None

    if rank == 0:
        stale_shards_check(out_dir, fp, rank)
    if world > 1:
        import torch.distributed as dist

        if dist.is_available() and not dist.is_initialized():
            try:
                dist.init_process_group(backend="gloo")
            except Exception:
                pass
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    sp = shard_path(out_dir, rank, world)
    sp.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(data_path, args.limit)
    indexed_rows = list(enumerate(rows))
    mine = [(oi, r) for oi, r in indexed_rows if oi % world == rank]

    done = load_done_ids(sp)

    # Build variant specs for our shard
    todo = []
    for oi, row in mine:
        variant_specs = _build_variants(row, args.variant_set, tmp_dir, derived_root=derived_root)
        # Fill in deterministic temp filename using original_index
        for vs in variant_specs:
            if vs["pil_image_override"] is not None:
                old = vs["pil_image_override"]
                p = Path(old)
                new_name = f"oi{oi}_{vs['variant_id']}{p.suffix}"
                new_path = p.parent / new_name
                if p.exists():
                    p.rename(new_path)
                vs["pil_image_override"] = str(new_path)
                # Update messages image path
                for msg in vs["messages"]:
                    if msg["role"] == "user":
                        for part in msg["content"]:
                            if isinstance(part, dict) and part.get("type") == "image":
                                part["image"] = str(new_path)
        for vs in variant_specs:
            key = (oi, vs["variant_id"])
            if key not in done:
                todo.append((oi, vs))

    log(rank, f"total={len(rows)} mine={len(mine)} done={len(done)} todo={len(todo)} -> {sp}")

    if not todo:
        log(rank, "nothing to do")
        sp.touch(exist_ok=True)
        return

    if args.dry_run:
        log(rank, f"[dry-run] would process {len(todo)} variants")
        return

    # Load model
    model, processor = load_model_and_processor(args, device, rank)
    model.eval()

    error_count = 0
    t0 = time.time()
    batch_size = args.batch_size
    # Group todo into batches
    batch_todo: list[list[tuple[int, dict]]] = []
    for i in range(0, len(todo), batch_size):
        batch_todo.append(todo[i : i + batch_size])

    with sp.open("a") as pf:
        for batch_idx, batch in enumerate(batch_todo):
            batch_variant_specs = [vs for _, vs in batch]

            try:
                # Try batch forward
                batch_results = _head_forward_batch(
                    model, processor, batch_variant_specs, torch.device(device)
                )
            except Exception as exc:
                msg = f"batch {batch_idx} forward error: {exc}"
                log(rank, msg)
                if not args.allow_errors:
                    raise
                # Fall back to per-variant single forward to isolate failures
                batch_results = []
                for oi, vs in batch:
                    try:
                        single_result = _head_forward_single(
                            model, processor, vs["messages"], torch.device(device)
                        )
                        batch_results.append(single_result)
                    except Exception as exc2:
                        log(rank, f"ERROR original_index={oi} variant={vs['variant_id']}: {exc2}")
                        error_count += 1
                        batch_results.append(
                            {
                                "p_answer": None,
                                "p_wo": None,
                                "p_nt": None,
                                "predicted_class": None,
                                "decision": None,
                                "error": traceback.format_exc(),
                            }
                        )

            for (oi, vs), result in zip(batch, batch_results):
                row = rows[oi]
                md = row.get("metadata") or {}
                record = {
                    "sample_id": md.get("sample_id"),
                    "relation": md.get("relation"),
                    "image": row.get("images", [None])[0],
                    "target_label": parse_target_label_from_messages(row["messages"]),
                    "achieved_visible_ratio": md.get("achieved_visible_ratio"),
                    "visible_ratio_bin": md.get("visible_ratio_bin"),
                    "original_gt_decision": md.get("gt_decision") or md.get("decision"),
                    "variant_type": vs["variant_type"],
                    "variant_id": vs["variant_id"],
                    "p_answer": result.get("p_answer"),
                    "p_wo": result.get("p_wo"),
                    "p_nt": result.get("p_nt"),
                    "predicted_class": result.get("predicted_class"),
                    "decision": result.get("decision"),
                    "rank": rank,
                    "original_index": oi,
                }
                pf.write(json.dumps(record, ensure_ascii=False) + "\n")
                pf.flush()

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                done = min((batch_idx + 1) * batch_size, len(todo))
                log(rank, f"  {done}/{len(todo)} ({done / elapsed:.2f} it/s)")

    dt = time.time() - t0
    log(rank, f"done {len(todo)} variants in {dt:.1f}s -> {sp} (errors={error_count})")


# ── merge ───────────────────────────────────────────────────────────────────


def merge_shards(args: argparse.Namespace) -> dict:
    """Merge all rank shards into a single sorted JSONL.

    Returns manifest dict with metadata.  Validates all shards exist,
    checks variant completeness per original_index, and detects duplicates.
    """
    out_dir = Path(args.output)
    shard_dir = out_dir / "shards"
    world = int(os.environ.get("WORLD_SIZE", 1))

    # ── validate all shards exist ───────────────────────────────────────
    missing_shards = []
    for r in range(world):
        fp = shard_dir / f"probs.rank{r}of{world}.jsonl"
        if not fp.exists():
            missing_shards.append(str(fp))
    if missing_shards:
        raise RuntimeError(f"Missing shard(s): {missing_shards}")

    # ── load all records ────────────────────────────────────────────────
    all_recs: list[dict] = []
    for r in range(world):
        fp = shard_dir / f"probs.rank{r}of{world}.jsonl"
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # ── stable sort ─────────────────────────────────────────────────────
    all_recs.sort(key=lambda x: (x.get("original_index", -1), x.get("variant_id", "")))

    # ── validate variant completeness per original_index ────────────────
    required_image_ids = _required_image_variant_ids(args.variant_set)
    by_oi: dict[int, list[dict]] = {}
    for rec in all_recs:
        oi = rec.get("original_index")
        if oi is not None:
            by_oi.setdefault(oi, []).append(rec)

    n_rows = len(by_oi)
    completeness_errors: list[str] = []
    duplicate_errors: list[str] = []
    for oi, recs in by_oi.items():
        found_ids = [r.get("variant_id", "") for r in recs]
        # Check duplicates
        seen: set[str] = set()
        for vid in found_ids:
            if vid in seen:
                duplicate_errors.append(f"oi={oi} duplicate variant_id={vid}")
            seen.add(vid)
        # Check fixed original/image variants.
        for eid in required_image_ids:
            if eid not in found_ids:
                completeness_errors.append(f"oi={oi} missing variant_id={eid}")
        if args.variant_set in ("query", "all"):
            query_ids = {vid for vid in found_ids if vid.startswith("query_")}
            expected_query_count = len(GROUNDING_TEMPLATES) - 1
            if len(query_ids) != expected_query_count:
                completeness_errors.append(
                    f"oi={oi} expected {expected_query_count} query variants, "
                    f"found={sorted(query_ids)}"
                )

    expected_rows = len(load_jsonl(Path(args.data), args.limit))
    missing_rows = sorted(set(range(expected_rows)) - set(by_oi))
    unexpected_rows = sorted(set(by_oi) - set(range(expected_rows)))
    if missing_rows or unexpected_rows:
        raise RuntimeError(
            "Row completeness error: "
            f"missing={missing_rows[:20]}, unexpected={unexpected_rows[:20]}"
        )

    if completeness_errors:
        raise RuntimeError(
            f"Variant completeness errors ({len(completeness_errors)}):\n"
            + "\n".join(completeness_errors[:20])
        )
    if duplicate_errors:
        raise RuntimeError(
            f"Duplicate variant errors ({len(duplicate_errors)}):\n"
            + "\n".join(duplicate_errors[:20])
        )

    # ── write merged ────────────────────────────────────────────────────
    merged_path = out_dir / "head_probabilities.jsonl"
    with merged_path.open("w") as f:
        for rec in all_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── count errors in output ──────────────────────────────────────────
    error_count = sum(1 for r in all_recs if r.get("p_answer") is None)

    # ── build manifest ──────────────────────────────────────────────────
    manifest = {
        "fingerprint": fingerprint_from_args(args, Path(args.data)),
        "n_rows": n_rows,
        "n_variants": len(all_recs),
        "error_count": error_count,
        "model_paths": {
            "base": args.base,
            "adapter": args.adapter,
            "calibrated_head": args.calibrated_head,
        },
        "data_path": args.data,
        "variant_set": args.variant_set,
        "batch_size": args.batch_size,
        "world_size": world,
        "output": str(merged_path),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    log(0, f"merged {len(all_recs)} records ({n_rows} rows) -> {merged_path}")
    if error_count:
        log(0, f"  WARNING: {error_count} records have null p_answer (errors)")
    return manifest


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract head probabilities (p_answer/p_wo/p_nt) for L2/L3 evidence data"
    )
    ap.add_argument("--base", required=True, help="Qwen3-VL base model path")
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory")
    ap.add_argument("--calibrated-head", required=True, help="Calibrated head .pt path")
    ap.add_argument("--data", required=True, help="Input JSONL (L2 or L3)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of input rows")
    ap.add_argument("--max-pixels", type=int, default=262144, help="Max pixels for processor")
    ap.add_argument(
        "--variant-set",
        choices=["none", "query", "image", "all"],
        default="none",
        help="Variant set to generate (default: none = original only)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Dry run: load data but not model")
    ap.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Batch size for model forward (default: 1, must be positive integer)",
    )
    ap.add_argument(
        "--allow-errors",
        action="store_true",
        help="Continue on per-variant errors (write null records). Default: fail-fast.",
    )
    ap.add_argument(
        "--derived-root",
        default=None,
        help="Root directory of external derived images (default: benchmark/data/derived). Applies only to relative paths carrying the data/derived/ prefix.",
    )
    args = ap.parse_args()

    rank, world, local_rank = rank_world()

    if args.dry_run:
        if rank == 0:
            data_path = Path(args.data)
            rows = load_jsonl(data_path, args.limit)
            log(
                rank,
                f"[dry-run] would process {len(rows)} rows with variant_set={args.variant_set}",
            )
            derived_root = _derived_root_from_args(args)
            log(rank, f"  derived_root={derived_root}")
        return 0

    run_extraction(args)

    if world > 1:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # Cleanup per-rank tmp dirs: only rank 0 cleans the root tmp
    if rank == 0:
        import shutil

        tmp_root = Path(args.output).parent / ".tmp_extract_probs"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
            log(0, f"cleaned up tmp root: {tmp_root}")

    if rank == 0:
        merge_shards(args)
        log(rank, "all done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
