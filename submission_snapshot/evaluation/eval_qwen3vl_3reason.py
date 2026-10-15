#!/usr/bin/env python
"""Inference and scoring for the three-way discriminative head, shared by the dev gate and the test probes.

Loads the base model plus the SFT LoRA adapter, runs the probe JSONL, extracts the answer
JSON, and scores with `reason_abstain.eval.metrics` so that every configuration is measured
under the same definitions.

Backends:
  --backend vllm (default): each rank pins one device, starts a vLLM engine and generates the
    whole shard in one batch. Requires vllm >= 0.11 for Qwen3-VL support.
  --backend hf: HF `model.generate`, one sample at a time. Slower, used when the installed
    vLLM does not support the model.

Under torchrun the data is sharded by `index % world_size`. Each rank appends to its own
shard and can resume; rank 0 merges the shards once all ranks finish and then computes the
metrics. A single process runs the whole probe and scores it directly.

Outputs:
  <out>/shards/preds.rank{r}of{w}.jsonl   per-rank predictions, kept as the merge source
  <out>/predictions.jsonl                 merged per-sample records (gt, pred, raw)
  <out>/metrics.json                      format and decision gates, all metrics, MTS
  <out>/summary.txt                       human-readable summary

Usage:
  torchrun --nproc_per_node=2 evaluation/eval_qwen3vl_3reason.py \
      --data benchmark/annotations/splits/test.jsonl \
      --base <qwen3vl> --adapter <lora_ckpt> --out outputs/evals/test --tag test

  # single process: drop the torchrun prefix
  # re-score existing shards without re-running inference: add --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TP2_WORLD_SIZE = 2

if "--backend" in sys.argv:
    _bi = sys.argv.index("--backend")
    _backend = sys.argv[_bi + 1] if _bi + 1 < len(sys.argv) else ""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        _lr = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        _ws = int(os.environ.get("WORLD_SIZE", 1))
        if _backend == "vllm" and _ws > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(_lr)
            print(
                f"[eval pre-import] vllm 多卡：LOCAL_RANK={_lr} → CUDA_VISIBLE_DEVICES={_lr}",
                flush=True,
            )
        elif _backend == "hf" and _ws == TP2_WORLD_SIZE:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{_lr * 2},{_lr * 2 + 1}"
            print(
                f"[eval pre-import] hf TP=2(2进程×2卡)：LOCAL_RANK={_lr} → CUDA_VISIBLE_DEVICES={_lr * 2},{_lr * 2 + 1}",
                flush=True,
            )

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
_AAAI_CODE = _THIS_DIR.parent  # evaluation/.. = aaai_code
_TRAINING = _AAAI_CODE / "training"
sys.path.insert(0, str(_AAAI_CODE))
sys.path.insert(0, str(_TRAINING))

from reason_abstain.eval.metrics import (  # noqa: E402
    aggregate_metrics,
    box_iou,
    metamorphic_test_satisfaction,
)
from reason_abstain.eval.parser import normalize_label, parse_structured_output  # noqa: E402


REPR_LAYER_IDX = 26

DUAL_HEAD_CONF_THRESHOLD = 0.6


class _ReprHead(nn.Module):
    """Read-out head used by the `--repr-head` and `--std-head` paths: 4096 -> 512 -> 256 -> 2."""

    def __init__(self, hidden_size: int = 4096, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def rank_world() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world, local_rank


def log(rank: int, msg: str) -> None:
    print(f"[eval r{rank}] {msg}", flush=True)


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


PRESENT_SYSTEM_PROMPT = (
    "You are a visual grounding assistant for plant disease diagnosis. "
    "Locate the queried target. If it is present and localizable, return "
    '{"present":true,"label":"...","box":[x1,y1,x2,y2]} with box in [0,1000] relative coords. '
    "If it is not present, or its evidence is insufficient to localize, return "
    '{"present":false,"reason":"target_absent|visual_evidence_insufficient"}. '
    "target_absent: the queried target is not in this image (absent, or the visible disease "
    "of the same crop is a different one). "
    "visual_evidence_insufficient: the target may be there but is occluded/degraded/too small."
)

DECISION_SYSTEM_PROMPT = (
    "You are a vision-language grounding assistant for high-stakes plant disease diagnosis. "
    "Locate the queried target. If the queried target is present and localizable, return "
    '{"decision":"answer","label":"...","box":[x1,y1,x2,y2]} with box in [0,1000] relative coords. '
    "Otherwise abstain: "
    '{"decision":"abstain","reason":"wrong_object|no_target"}. '
    "wrong_object: the image shows a visually similar but different object/disease, not the queried one. "
    "no_target: the queried target is not locatable (absent from the image, or cropped out of view)."
)


def build_inputs(row: dict, system_override: str | None = None) -> tuple[str, str, str | None]:
    system = ""
    user_text = ""
    for m in row["messages"]:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            user_text = m["content"].replace("<image>", "").strip()
    image = row["images"][0] if row.get("images") else None
    if image is not None and not Path(image).is_absolute():
        image = str(_AAAI_CODE / "benchmark" / image)
    if system_override is not None:
        if system_override == "present":
            system = PRESENT_SYSTEM_PROMPT
        elif system_override == "decision":
            system = DECISION_SYSTEM_PROMPT
        else:
            system = system_override
    return system, user_text, image


def _valid_candidate_box(box: object) -> bool:
    """Strict box validation, equivalent to `parser._valid_box`, so that `box_iou` never sees
    malformed input.

    Requires a 4-element list of numbers with 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.
    """
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) for v in box):
        return False
    x1, y1, x2, y2 = [float(v) for v in box]
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def _greedy_generate_answer_candidate(
    model,
    processor,
    row: dict,
    device: torch.device,
    system_override: str | None = None,
    max_new_tokens: int = 384,
) -> dict:
    """Run one greedy generation and parse only the label and the box, ignoring the decision.

    Generation parameters are identical to the answer branch of `--faithful-dual-output`:
    do_sample=False, temperature=None, top_p=None, top_k=None.

    Returns:
        {"raw": str, "label": str|None, "box": list|None, "valid": bool}
    """
    messages = _build_messages(row, system_override=system_override)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    import json as _json

    _json_str = None
    try:
        from reason_abstain.eval.parser import _extract_json_object

        _json_str = _extract_json_object(raw)
    except Exception:
        pass
    gen_label = None
    gen_box = None
    if _json_str:
        try:
            _raw_obj = _json.loads(_json_str)
            gen_label = _raw_obj.get("label")
            gen_box = _raw_obj.get("box")
        except Exception:
            pass

    if not _valid_candidate_box(gen_box):
        gen_box = None
    if not isinstance(gen_label, str):
        gen_label = None

    return {
        "raw": raw,
        "label": gen_label,
        "box": gen_box,
        "valid": gen_label is not None or gen_box is not None,
    }


def _score_answer_candidate(
    candidate: dict,
    gt: dict,
    iou_threshold: float,
) -> dict:
    """Score an answer candidate and return a fixed schema.

    Args:
        candidate: the value returned by `_greedy_generate_answer_candidate`
        gt: the value returned by `gt_from_row`
        iou_threshold: the same threshold used by the reported metrics

    Returns:
        {"valid": bool, "label": str|None, "box": list|None, "raw": str,
         "iou": float|None, "label_correct": bool|None, "grounding_correct": bool|None}

    Meaning of `valid`:
      - ground truth is supported and a ground-truth box exists: a valid label and a valid box
        are both required
      - ground truth is supported with no box, as in the label-only PlantWild probe: a valid
        label is required
      - ground truth is unsupported: `valid` only records whether the candidate is structurally
        usable as an answer, while `iou`, `label_correct` and `grounding_correct` stay None
    """
    gt_supported = gt.get("decision") == "answer"
    gt_label = gt.get("label")
    gt_box = gt.get("box")

    c_label = candidate.get("label")
    c_box = candidate.get("box")
    c_box_valid = _valid_candidate_box(c_box)

    if gt_supported:
        if gt_box is not None:
            _valid = isinstance(c_label, str) and c_box_valid
        else:
            _valid = isinstance(c_label, str)
    else:
        _valid = isinstance(c_label, str) and c_box_valid

    result = {
        "valid": _valid,
        "label": c_label,
        "box": c_box if c_box_valid else None,
        "raw": candidate.get("raw", ""),
        "iou": None,
        "label_correct": None,
        "grounding_correct": None,
    }
    if not gt_supported:
        return result

    # label_correct
    label_correct = (
        normalize_label(c_label) == normalize_label(gt_label)
        if isinstance(c_label, str) and isinstance(gt_label, str)
        else False
    )
    result["label_correct"] = label_correct

    if c_box_valid and gt_box is not None:
        try:
            result["iou"] = box_iou(c_box, gt_box)
        except Exception:
            result["iou"] = None

    if gt_box is None:
        result["grounding_correct"] = label_correct
    else:
        if not c_box_valid:
            result["grounding_correct"] = False
        else:
            result["grounding_correct"] = label_correct and box_iou(c_box, gt_box) >= iou_threshold

    return result


def gt_from_row(row: dict) -> dict:
    md = row.get("metadata") or {}
    return {
        "decision": md.get("decision"),
        "label": md.get("label"),
        "box": md.get("box"),
        "reason": md.get("reason"),
        "relation": md.get("relation"),
        "sample_id": md.get("sample_id"),
        "metadata": {
            "relation": md.get("relation"),
            "achieved_visible_ratio": md.get("achieved_visible_ratio"),
            "target_visible_ratio": md.get("target_visible_ratio"),
            "sample_id": md.get("sample_id"),
        },
    }


def image_key_from_sample_id(sample_id: str | None) -> str | None:
    """Recover the image identity from an L1 `sample_id`, for the within-pair analysis.

    Identifiers look like `L1-<imgname>.rf.<hash>-real-...` or `L1-<imgname>.rf.<hash>-mismatch-...`;
    the segment between `L1-` and `-real-` or `-mismatch-` is a stable key for one image.
    Returns None for anything that is not in the L1 format, since the analysis only applies to
    the L1 discrimination pairs.
    """
    if not sample_id or not sample_id.startswith("L1-"):
        return None
    s = sample_id[3:]
    for marker in ("-real-", "-mismatch-"):
        if marker in s:
            return s.split(marker)[0]
    return None


def shard_path(out_dir: Path, rank: int, world: int) -> Path:
    return out_dir / "shards" / f"preds.rank{rank}of{world}.jsonl"


def eval_fingerprint(args: argparse.Namespace, data_path: Path | None = None) -> str:
    """Fingerprint of an evaluation run: base model, adapter, data, world size, limit and backend,
    each with path and mtime.

    Any change to the model or to the data changes the fingerprint, which marks existing shards
    as stale. Resuming compares fingerprints rather than sample ids alone, so predictions from a
    different model or a different probe are never reused.

    For multi-probe runs, `data_path` is the path of the current probe, keeping the fingerprints
    of different probes independent.
    """

    def path_sig(p: str | None) -> str:
        if not p:
            return "NONE"
        pp = Path(p)
        try:
            return f"{pp}@{int(pp.stat().st_mtime)}"
        except OSError:
            return f"{pp}@-1"

    dp = data_path or Path(args.data)
    try:
        data_mtime = int(dp.stat().st_mtime)
    except OSError:
        data_mtime = -1
    parts = [
        f"base={path_sig(getattr(args, 'base', None))}",
        f"adapter={path_sig(args.adapter)}",
        f"data={dp}",
        f"data_mtime={data_mtime}",
        f"world={int(os.environ.get('WORLD_SIZE', 1))}",
        f"limit={args.limit}",
        f"backend={args.backend}",
        f"calibrated_head={path_sig(getattr(args, 'calibrated_head', None))}",
        f"system_override={getattr(args, 'system_override', None)}",
        f"dual_head={getattr(args, 'dual_head', False)}",
        f"faithful_dual_output={getattr(args, 'faithful_dual_output', False)}",
        f"dump_all_answer_candidates={getattr(args, 'dump_all_answer_candidates', False)}",
    ]
    return " | ".join(parts)


def stale_shards_check(out_dir: Path, fingerprint: str, rank: int) -> None:
    """Rank 0 compares fingerprints and clears the whole `shards/` directory on a mismatch, then
    writes the new fingerprint.

    Only rank 0 deletes; the other ranks wait on a barrier and then read their own shard, which
    by that point has either been cleared or kept.
    """
    fp_file = out_dir / "shards" / ".fingerprint"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    old = fp_file.read_text().strip() if fp_file.exists() else None
    if old is not None and old != fingerprint:
        removed = 0
        for sh in shard_dir.glob("preds.rank*.jsonl"):
            sh.unlink()
            removed += 1
        log(rank, f"⚠️ 指纹变化(模型/数据已更新) → 清空 {removed} 个过期 shard")
        log(rank, f"   旧: {old}")
        log(rank, f"   新: {fingerprint}")
    fp_file.write_text(fingerprint)


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def decision_correct(pred: dict, gt: dict) -> bool:
    """A decision counts as correct when the decision matches and, for answers, the label matches,
    or, for abstentions, the reason matches.
    """
    if not pred.get("valid") or pred.get("decision") != gt.get("decision"):
        return False
    if gt.get("decision") == "answer":
        return normalize_label(pred.get("label")) == normalize_label(gt.get("label"))
    if gt.get("decision") == "abstain":
        return pred.get("reason") == gt.get("reason")
    return False


def _build_messages(row: dict, system_override: str | None = None) -> list[dict]:
    system, user_text, image = build_inputs(row, system_override=system_override)
    image_content: dict = {"type": "image", "image": image}
    vision_max_pixels = row.get("metadata", {}).get("vision_max_pixels")
    if vision_max_pixels is not None:
        if not isinstance(vision_max_pixels, int) or vision_max_pixels <= 0:
            raise ValueError(
                f"metadata.vision_max_pixels must be a positive integer, got "
                f"{vision_max_pixels!r}"
            )
        image_content["max_pixels"] = vision_max_pixels
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _load_repr_head(path: str) -> _ReprHead:
    """Load `repr_head.pt`, an OrderedDict of `classifier.{0,3,6}.{weight,bias}`."""
    head = _ReprHead(hidden_size=4096, num_classes=2)
    state_dict = torch.load(path, map_location="cpu")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def _load_std_head(path: str) -> tuple:
    """Load `std_head.pt`, an MLP head together with its standardisation statistics.

    Returns:
        (head: _ReprHead, mu: torch.Tensor, sd: torch.Tensor)
    """
    ckpt = torch.load(path, map_location="cpu")
    head_state = ckpt["head_state"]
    mu = ckpt["mu"]
    sd = ckpt["sd"]

    head = _ReprHead(hidden_size=4096, num_classes=2)
    head.load_state_dict(head_state)
    head.eval()

    return head, mu, sd


def _extract_dual_head_decision(
    model, processor, row: dict, device: torch.device, system_override: str | None = None
) -> dict:
    """Read the decision of the three-way head from a single forward pass.

    Steps:
      1. build the inputs
      2. call `model(**inputs)`, which returns `head_logits` of shape [B, 3]
      3. marginalise: compare p(answer) against p(wrong_object) + p(no_target), so an argmax of
         1 or 2 maps to abstain and an argmax of 0 maps to answer
      4. read the reason as the argmax over the two abstention classes

    Notes:
      - label convention: 0 = answer, 1 = wrong_object, 2 = no_target
      - the standardisation statistics live inside the model, so no external head file is needed
      - one forward pass yields the language-model output and the discriminative logits together
      - abstaining does not require a second generation pass to recover the reason

    Returns:
        {"decision": "answer"|"abstain", "reason": "wrong_object"|"no_target"|None,
         "valid": True, "confidence": float}
    """
    messages = _build_messages(row, system_override=system_override)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        # The gate only consumes hidden_states at the pre-response token. Restricting
        # the unused LM logits to the final token avoids allocating an S×V tensor for
        # long high-resolution prompts without changing hidden states or gate outputs.
        outputs = model(**inputs, logits_to_keep=1)

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
        return {"decision": None, "reason": None, "valid": False, "error": "no head_logits"}

    import torch.nn.functional as F

    probs = F.softmax(head_logits.float(), dim=-1)  # [B, 3]

    p_answer = probs[0, 0].item()  # P(answer)
    p_wo = probs[0, 1].item()  # P(wrong_object)
    p_nt = probs[0, 2].item()  # P(no_target)
    p_abstain = p_wo + p_nt  # P(abstain) = P(wo) + P(nt)

    pred_class = int(head_logits.argmax(dim=-1).item())
    decision = "abstain" if pred_class >= 1 else "answer"

    reason = None
    if decision == "abstain":
        if p_wo > p_nt:
            reason = "wrong_object"
        else:
            reason = "no_target"

    conf = p_answer if decision == "answer" else p_abstain

    head_hidden_out = None
    if isinstance(outputs, dict):
        head_hidden_out = outputs.get("head_hidden")
    else:
        head_hidden_out = getattr(outputs, "head_hidden", None)
    if head_hidden_out is None:
        m = model
        for _ in range(6):
            if getattr(m, "last_head_hidden", None) is not None:
                break
            if hasattr(m, "module"):
                m = m.module
            elif hasattr(m, "base_model") and hasattr(m.base_model, "model"):
                m = m.base_model.model
            else:
                break
        head_hidden_out = getattr(m, "last_head_hidden", None)

    if head_hidden_out is not None:
        head_hidden_out = head_hidden_out[0].to(torch.float16).cpu().tolist()  # float16 list

    return {
        "decision": decision,
        "reason": reason,
        "valid": True,
        "confidence": conf,
        "p_answer": p_answer,
        "p_wo": p_wo,
        "p_nt": p_nt,
        "head_hidden": head_hidden_out,
    }


def _find_decoder_layers(model):
    """Locate the text-decoder layers through the peft and deepspeed wrappers."""
    m = model
    for _ in range(6):
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        elif hasattr(m, "module"):
            m = m.module
        else:
            break
    node = m
    for attr in ("model", "language_model", "layers"):
        if hasattr(node, attr):
            node = getattr(node, attr)
        else:
            node = None
            break
    return node


def _register_repr_hook(model, cache: dict) -> None:
    """Register a forward hook on decoder layer index 26 and cache the layer-27 hidden state."""
    layers = _find_decoder_layers(model)
    if layers is None or len(layers) <= REPR_LAYER_IDX:
        raise RuntimeError(
            f"[repr] 找不到 decoder layers[{REPR_LAYER_IDX}]，hook 挂载失败。"
            f"layers={type(layers)} len={len(layers) if layers is not None else None}"
        )
    target = layers[REPR_LAYER_IDX]

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        cache["h"] = hs

    target.register_forward_hook(hook)


def _extract_repr_decision(
    model,
    processor,
    row: dict,
    repr_head: _ReprHead,
    device: torch.device,
    hook_cache: dict,
    std_head_params: tuple = None,
) -> dict:
    """Decide with the read-out head: forward, take the layer-27 representation at the
    generation-prompt position, pass it through the head.

    The read-out position must match the one used when the head was fitted:
    - training: the index just before the first non-masked label, i.e. before the assistant
      content
    - inference: the final token after `add_generation_prompt`, which is the same position

    Under `--std-head` the feature is standardised as (feat - mu) / sd in float32 before the
    head, matching how the head was fitted.

    Args:
        std_head_params: the (mu, sd) pair, required with --std-head

    Returns:
        {"decision": "answer"|"abstain", "valid": True}
    """
    messages = _build_messages(row)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    hook_cache.pop("h", None)

    with torch.inference_mode():
        _ = model(**inputs)

    hs = hook_cache.get("h")  # [B, S, 4096]
    if hs is None:
        return {"decision": None, "valid": False, "error": "hook 未捕获 hidden"}

    input_len = inputs.input_ids.shape[1]
    if inputs.attention_mask is not None:
        last_idx = int(inputs.attention_mask.sum(dim=1)[0]) - 1
    else:
        last_idx = input_len - 1
    last_idx = max(0, min(last_idx, hs.size(1) - 1))

    feat = hs[0, last_idx, :]  # [4096]

    if std_head_params is not None:
        mu, sd = std_head_params
        feat = feat.to(torch.float32)
        feat = (feat - mu.to(device)) / sd.to(device)

    with torch.inference_mode():
        logits = repr_head(feat.unsqueeze(0).to(torch.float32))  # [1, 2]
        pred_class = logits.argmax(dim=-1).item()

    decision = "answer" if pred_class == 1 else "abstain"
    return {"decision": decision, "valid": True}


def _write_record(
    pf,
    row: dict,
    raw: str,
    pred_override: dict | None = None,
    head_probs: dict | None = None,
    head_hidden: list | None = None,
    answer_candidate: dict | None = None,
) -> None:
    """Write one prediction record.

    Args:
        raw: the text the model actually generated, stored verbatim for inspection
        pred_override: in the head-based modes, the prediction assembled from the head decision,
            including the label, box and reason fallbacks. When given it is used as `pred`
            directly instead of re-parsing `raw`, so a head decision is never overwritten by the
            parse of the generated text, while `raw` still holds the model output. When absent,
            `pred` is `parse_structured_output(raw)`, as in the pure language-model path.
        head_probs: the head probabilities p_answer, p_wo and p_nt, available in the head-based
            modes from `_extract_dual_head_decision`
        answer_candidate: written only under --dump-all-answer-candidates
    """
    gt = gt_from_row(row)
    pred = pred_override if pred_override is not None else parse_structured_output(raw)
    record = {
        "sample_id": gt.get("sample_id"),
        "relation": gt.get("relation"),
        "gt": {
            **{k: gt.get(k) for k in ("decision", "label", "box", "reason")},
            "metadata": gt.get("metadata"),
        },
        "pred": pred,
        "raw": raw,
    }
    if head_probs is not None:
        record["p_answer"] = head_probs.get("p_answer")
        record["p_wo"] = head_probs.get("p_wo")
        record["p_nt"] = head_probs.get("p_nt")
    if head_hidden is not None:
        record["head_hidden"] = head_hidden
        _md = row.get("metadata") or {}
        record["head_label"] = _md.get("head_label")
        _imgs = row.get("images") or []
        record["image"] = _imgs[0] if _imgs else None
    if answer_candidate is not None:
        record["answer_candidate"] = answer_candidate
    pf.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_one_probe_hf(
    args,
    rank: int,
    world: int,
    local_rank: int,
    data_path: Path,
    out_dir: Path,
    model,
    processor,
    probe_idx: int,
    repr_head=None,
    std_head_params=None,
    system_override: str | None = None,
    guide_prefix: str | None = None,
    no_guide: bool = False,
    dual_head: bool = False,
) -> None:
    """Run one probe on the HF backend, reusing an already loaded model and processor.

    Decision paths:
      --repr-head: decide with the read-out head instead of parsing generated tokens
      --std-head: as above, with the feature standardised before the head
      --dual-head: decide with the head that is part of the model, in a single forward pass

    Data loading, sharding, metric computation and output are shared by all paths.

    Args:
        std_head_params: the (mu, sd) pair, required with --std-head
        dual_head: True under --dual-head
    """
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    sp = shard_path(out_dir, rank, world)

    rows = load_jsonl(data_path, args.limit)
    mine = [r for i, r in enumerate(rows) if i % world == rank]
    done = load_done_ids(sp)
    todo = [r for r in mine if (r.get("metadata") or {}).get("sample_id") not in done]
    log(
        rank,
        f"[hf probe#{probe_idx}] total={len(rows)} mine={len(mine)} done={len(done)} todo={len(todo)} -> {sp}",
    )
    if not todo:
        log(rank, f"probe#{probe_idx} nothing to do")
        return

    hook_cache = {}
    if repr_head is not None:
        if std_head_params is not None:
            repr_head = repr_head.to(device=model.device, dtype=torch.float32)
        else:
            repr_head = repr_head.to(device=model.device, dtype=torch.bfloat16)
        _register_repr_hook(model, hook_cache)
        mode_desc = "标准化表征头" if std_head_params is not None else "表征头"
        log(
            rank,
            f"[repr-head] 已注册 hook 到 decoder layers[{REPR_LAYER_IDX}]，decision 走{mode_desc}",
        )

    answer_stats = {
        "first_ok": 0,
        "retry_ok": 0,
        "still_placeholder": 0,
        "conf_fallback_abstain": 0,
    }

    t0 = time.time()
    device = model.device
    with sp.open("a", encoding="utf-8") as pf, torch.inference_mode():
        for i, row in enumerate(todo):
            if dual_head:
                faithful_dual_output = getattr(args, "faithful_dual_output", False)
                if faithful_dual_output:
                    head_result = _extract_dual_head_decision(
                        model, processor, row, device, system_override=system_override
                    )
                    decision = head_result["decision"]
                    head_probs = {
                        "p_answer": head_result.get("p_answer"),
                        "p_wo": head_result.get("p_wo"),
                        "p_nt": head_result.get("p_nt"),
                    }
                    head_hidden_dump = (
                        head_result.get("head_hidden")
                        if getattr(args, "dump_hidden", False)
                        else None
                    )

                    dump_candidates = getattr(args, "dump_all_answer_candidates", False)
                    answer_candidate_record = None
                    if dump_candidates:
                        _candidate = _greedy_generate_answer_candidate(
                            model,
                            processor,
                            row,
                            device,
                            system_override=system_override,
                            max_new_tokens=args.max_new_tokens,
                        )
                        _gt = gt_from_row(row)
                        answer_candidate_record = _score_answer_candidate(
                            _candidate,
                            _gt,
                            iou_threshold=args.iou_threshold,
                        )

                    if decision == "abstain":
                        reason = head_result.get("reason")
                        p_wo = head_result.get("p_wo", 0.0)
                        p_nt = head_result.get("p_nt", 0.0)
                        which_wo_nt = "wrong_object" if p_wo > p_nt else "no_target"
                        raw = f"[faithful-dual-output: head abstain ({which_wo_nt})]"
                        pred = {"valid": True, "decision": "abstain", "reason": reason}
                        _write_record(
                            pf,
                            row,
                            raw,
                            pred_override=pred,
                            head_probs=head_probs,
                            head_hidden=head_hidden_dump,
                            answer_candidate=answer_candidate_record,
                        )

                    elif decision == "answer":
                        if not dump_candidates or answer_candidate_record is None:
                            _candidate = _greedy_generate_answer_candidate(
                                model,
                                processor,
                                row,
                                device,
                                system_override=system_override,
                                max_new_tokens=args.max_new_tokens,
                            )
                            if dump_candidates and answer_candidate_record is None:
                                _gt = gt_from_row(row)
                                answer_candidate_record = _score_answer_candidate(
                                    _candidate,
                                    _gt,
                                    iou_threshold=args.iou_threshold,
                                )
                        raw = _candidate["raw"]
                        pred = {
                            "valid": True,
                            "decision": "answer",
                            "label": _candidate["label"],
                            "box": _candidate["box"],
                        }
                        _write_record(
                            pf,
                            row,
                            raw,
                            pred_override=pred,
                            head_probs=head_probs,
                            head_hidden=head_hidden_dump,
                            answer_candidate=answer_candidate_record,
                        )

                    else:
                        pred = {"valid": False, "error": "head_no_decision"}
                        _write_record(
                            pf,
                            row,
                            "",
                            pred_override=pred,
                            head_probs=head_probs,
                            head_hidden=head_hidden_dump,
                            answer_candidate=answer_candidate_record,
                        )

                else:
                    head_result = _extract_dual_head_decision(
                        model, processor, row, device, system_override=system_override
                    )
                    decision = head_result["decision"]
                    head_probs = {
                        "p_answer": head_result.get("p_answer"),
                        "p_wo": head_result.get("p_wo"),
                        "p_nt": head_result.get("p_nt"),
                    }
                    head_hidden_dump = (
                        head_result.get("head_hidden")
                        if getattr(args, "dump_hidden", False)
                        else None
                    )

                    if decision == "abstain":
                        reason = head_result.get("reason")

                        messages = _build_messages(row, system_override=system_override)
                        text = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        from qwen_vl_utils import process_vision_info

                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to(device)
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                        )
                        trimmed = generated[:, inputs.input_ids.shape[1] :]
                        raw = processor.batch_decode(
                            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]

                        parsed = parse_structured_output(raw)
                        gen_reason = parsed.get("reason")
                        valid_reasons = ("wrong_object", "no_target")
                        final_reason = gen_reason if gen_reason in valid_reasons else reason

                        pred = {"valid": True, "decision": "abstain", "reason": final_reason}
                        head_probs = {
                            "p_answer": head_result.get("p_answer"),
                            "p_wo": head_result.get("p_wo"),
                            "p_nt": head_result.get("p_nt"),
                        }
                        _write_record(
                            pf,
                            row,
                            raw,
                            pred_override=pred,
                            head_probs=head_probs,
                            head_hidden=head_hidden_dump,
                        )
                    elif decision == "answer":

                        messages = _build_messages(row, system_override=system_override)
                        text = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        from qwen_vl_utils import process_vision_info

                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to(device)
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                        )
                        trimmed = generated[:, inputs.input_ids.shape[1] :]
                        raw = processor.batch_decode(
                            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]

                        parsed = parse_structured_output(raw)
                        gen_box = parsed.get("box")
                        gen_label = parsed.get("label")

                        conf_threshold = getattr(
                            args, "head_conf_threshold", DUAL_HEAD_CONF_THRESHOLD
                        )
                        head_conf = head_result.get("confidence", 1.0)
                        if head_conf < conf_threshold:
                            parsed_decision = parsed.get("decision")
                            parsed_reason = parsed.get("reason")
                            valid_reasons = ("wrong_object", "no_target")
                            if parsed_decision == "abstain" and parsed_reason in valid_reasons:
                                answer_stats["conf_fallback_abstain"] += 1
                                pred = {
                                    "valid": True,
                                    "decision": "abstain",
                                    "reason": parsed_reason,
                                }
                                head_probs = {
                                    "p_answer": head_result.get("p_answer"),
                                    "p_wo": head_result.get("p_wo"),
                                    "p_nt": head_result.get("p_nt"),
                                }
                                _write_record(
                                    pf,
                                    row,
                                    raw,
                                    pred_override=pred,
                                    head_probs=head_probs,
                                    head_hidden=head_hidden_dump,
                                )
                                continue

                        has_valid_box = gen_box and isinstance(gen_box, list) and len(gen_box) == 4

                        if has_valid_box:
                            answer_stats["first_ok"] += 1
                            pred = {
                                "valid": True,
                                "decision": "answer",
                                "label": gen_label
                                if gen_label
                                else ((row.get("gt", {}) or {}).get("label", "object")),
                                "box": gen_box,
                            }
                            model_raw = raw
                        else:
                            guided_messages = _build_messages(row, system_override=system_override)
                            if guide_prefix:
                                prefix = guide_prefix
                            elif no_guide:
                                prefix = None
                            else:
                                prefix = '{"decision":"answer",'

                            if prefix is None:
                                answer_stats["still_placeholder"] += 1
                                pred = {
                                    "valid": True,
                                    "decision": "answer",
                                    "label": (row.get("gt", {}) or {}).get("label", "object"),
                                    "box": [0, 0, 1000, 1000],
                                }
                                _write_record(
                                    pf,
                                    row,
                                    raw,
                                    pred_override=pred,
                                    head_probs=head_probs,
                                    head_hidden=head_hidden_dump,
                                )
                                continue

                            guided_text = processor.apply_chat_template(
                                guided_messages, tokenize=False, add_generation_prompt=True
                            )
                            guided_text_with_prefix = guided_text + prefix

                            guided_inputs = processor(
                                text=[guided_text_with_prefix],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt",
                            ).to(device)
                            guided_generated = model.generate(
                                **guided_inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                temperature=None,
                                top_p=None,
                                top_k=None,
                            )
                            guided_trimmed = guided_generated[:, guided_inputs.input_ids.shape[1] :]
                            guided_raw = processor.batch_decode(
                                guided_trimmed,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )[0]
                            guided_full_raw = prefix + guided_raw
                            model_raw = f"[first]{raw}\n[guided]{guided_full_raw}"

                            guided_parsed = parse_structured_output(guided_full_raw)
                            guided_box = guided_parsed.get("box")
                            guided_label = guided_parsed.get("label")

                            has_guided_box = (
                                guided_box and isinstance(guided_box, list) and len(guided_box) == 4
                            )

                            if has_guided_box:
                                answer_stats["retry_ok"] += 1
                                pred = {
                                    "valid": True,
                                    "decision": "answer",
                                    "label": guided_label
                                    if guided_label
                                    else ((row.get("gt", {}) or {}).get("label", "object")),
                                    "box": guided_box,
                                }
                            else:
                                answer_stats["still_placeholder"] += 1
                                pred = {
                                    "valid": True,
                                    "decision": "answer",
                                    "label": (row.get("gt", {}) or {}).get("label", "object"),
                                    "box": [0, 0, 1000, 1000],
                                }

                        _write_record(
                            pf,
                            row,
                            model_raw,
                            pred_override=pred,
                            head_probs=head_probs,
                            head_hidden=head_hidden_dump,
                        )
                    else:
                        pred = {"valid": False, "error": "headno_decision"}
                        _write_record(
                            pf,
                            row,
                            "",
                            pred_override=pred,
                            head_probs=head_probs,
                            head_hidden=head_hidden_dump,
                        )
            elif repr_head is not None:
                head_result = _extract_repr_decision(
                    model,
                    processor,
                    row,
                    repr_head,
                    device,
                    hook_cache,
                    std_head_params=std_head_params,
                )
                decision = head_result["decision"]

                if decision == "abstain":
                    pred = {"valid": True, "decision": "abstain", "reason": "no_target"}
                    _write_record(
                        pf, row, "[repr-head abstain: no LM generation]", pred_override=pred
                    )
                elif decision == "answer":

                    messages = _build_messages(row)
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    from qwen_vl_utils import process_vision_info

                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )
                    trimmed = generated[:, inputs.input_ids.shape[1] :]
                    raw = processor.batch_decode(
                        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                    parsed = parse_structured_output(raw)
                    gen_box = parsed.get("box")
                    gen_label = parsed.get("label")

                    has_valid_box = gen_box and isinstance(gen_box, list) and len(gen_box) == 4

                    if has_valid_box:
                        answer_stats["first_ok"] += 1
                        pred = {
                            "valid": True,
                            "decision": "answer",
                            "label": gen_label
                            if gen_label
                            else ((row.get("gt", {}) or {}).get("label", "object")),
                            "box": gen_box,
                        }
                        model_raw = raw
                    else:
                        guided_messages = _build_messages(row, system_override=system_override)
                        if guide_prefix:
                            prefix = guide_prefix
                        elif no_guide:
                            prefix = None
                        else:
                            prefix = '{"decision":"answer",'

                        if prefix is None:
                            answer_stats["still_placeholder"] += 1
                            pred = {
                                "valid": True,
                                "decision": "answer",
                                "label": (row.get("gt", {}) or {}).get("label", "object"),
                                "box": [0, 0, 1000, 1000],
                            }
                            _write_record(pf, row, raw, pred_override=pred)
                            continue

                        guided_text = processor.apply_chat_template(
                            guided_messages, tokenize=False, add_generation_prompt=True
                        )
                        guided_text_with_prefix = guided_text + prefix

                        guided_inputs = processor(
                            text=[guided_text_with_prefix],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to(device)
                        guided_generated = model.generate(
                            **guided_inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                        )
                        guided_trimmed = guided_generated[:, guided_inputs.input_ids.shape[1] :]
                        guided_raw = processor.batch_decode(
                            guided_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                        guided_full_raw = prefix + guided_raw
                        model_raw = f"[first]{raw}\n[guided]{guided_full_raw}"

                        guided_parsed = parse_structured_output(guided_full_raw)
                        guided_box = guided_parsed.get("box")
                        guided_label = guided_parsed.get("label")

                        has_guided_box = (
                            guided_box and isinstance(guided_box, list) and len(guided_box) == 4
                        )

                        if has_guided_box:
                            answer_stats["retry_ok"] += 1
                            pred = {
                                "valid": True,
                                "decision": "answer",
                                "label": guided_label
                                if guided_label
                                else ((row.get("gt", {}) or {}).get("label", "object")),
                                "box": guided_box,
                            }
                        else:
                            answer_stats["still_placeholder"] += 1
                            pred = {
                                "valid": True,
                                "decision": "answer",
                                "label": (row.get("gt", {}) or {}).get("label", "object"),
                                "box": [0, 0, 1000, 1000],
                            }

                    _write_record(pf, row, model_raw, pred_override=pred)
                else:
                    pred = {"valid": False, "error": "head_no_decision"}
                    _write_record(pf, row, "", pred_override=pred)
            else:
                messages = _build_messages(row, system_override=system_override)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
                trimmed = generated[:, inputs.input_ids.shape[1] :]
                raw = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                _write_record(pf, row, raw)

            pf.flush()
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                log(rank, f"probe#{probe_idx} {i + 1}/{len(todo)} ({(i + 1) / elapsed:.2f} it/s)")
    log(rank, f"probe#{probe_idx} shard done in {time.time() - t0:.1f}s -> {sp}")

    if repr_head is not None:
        total_answer = sum(answer_stats.values())
        if total_answer > 0:
            log(
                rank,
                f"[answer stats] first_ok={answer_stats['first_ok']} ({answer_stats['first_ok'] / total_answer:.1%}) "
                f"retry_ok={answer_stats['retry_ok']} ({answer_stats['retry_ok'] / total_answer:.1%}) "
                f"still_placeholder={answer_stats['still_placeholder']} ({answer_stats['still_placeholder'] / total_answer:.1%})",
            )


def run_inference(args: argparse.Namespace, rank: int, world: int, local_rank: int) -> None:
    """Run this rank's shard and append to its shard file, so the run can resume.

    With several probes, the HF backend loads the model once and loops over the probes, while
    the vLLM backend starts one engine per probe.
    """
    data_list = getattr(args, "_data_list", [args.data])
    out_list = getattr(args, "_out_list", [args.out])
    tag_list = getattr(args, "_tag_list", [args.tag])

    if args.backend == "hf":
        device = None
        hf_model_sharded = (
            world == TP2_WORLD_SIZE or os.environ.get("HF_MODEL_SHARDING") == "1"
        )
        if torch.cuda.is_available():
            if hf_model_sharded:
                device = "cuda"
            else:
                dev_idx = local_rank % torch.cuda.device_count()
                torch.cuda.set_device(dev_idx)
                device = f"cuda:{dev_idx}"
        log(rank, f"device={device} loading base: {args.base}")

        if getattr(args, "dual_head", False):
            log(rank, "[dual-head] 使用 Qwen3VLWithTriHead 模型类")
            from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead
            from transformers import AutoProcessor
            from peft import PeftModel

            processor = AutoProcessor.from_pretrained(
                args.base, max_pixels=args.max_pixels, trust_remote_code=True
            )
            model = Qwen3VLWithTriHead.from_pretrained(
                args.base,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map="balanced" if hf_model_sharded else None,
            )
            if device and not hf_model_sharded:
                model = model.to(device)
            if args.adapter:
                log(rank, f"loading LoRA adapter: {args.adapter}")
                model = PeftModel.from_pretrained(model, args.adapter)
                if device and not hf_model_sharded:
                    model = model.to(device)
            from safetensors.torch import load_file as sf_load

            adapter_path = Path(args.adapter)
            adapter_safetensors = adapter_path / "adapter_model.safetensors"
            if adapter_safetensors.exists():
                log(rank, f"[dual-head] 从 adapter 加载 discrim_head: {adapter_safetensors}")
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
                        discrim_head_module = m.discrim_head
                        missing, unexpected = discrim_head_module.load_state_dict(
                            discrim_state, strict=False
                        )
                        if missing:
                            log(rank, f"[WARN] discrim_head 缺失 keys: {missing[:5]}")
                        if unexpected:
                            log(rank, f"[WARN] discrim_head 多余 keys: {unexpected[:5]}")
                        with torch.inference_mode():
                            w0 = discrim_head_module.classifier[0].weight
                            norm = w0.norm().item()
                            log(
                                rank,
                                f"[dual-head] discrim_head 已加载，classifier[0].weight 范数={norm:.4f}",
                            )
                            try:
                                sd_min = float(discrim_head_module.sd.min())
                                sd_max = float(discrim_head_module.sd.max())
                            except Exception:
                                sd_min = sd_max = 1.0
                            if abs(sd_min - 1.0) < 1e-6 and abs(sd_max - 1.0) < 1e-6:
                                musd_path = os.environ.get("DISCRIM_MUSD_PATH", "")
                                if os.path.exists(musd_path):
                                    _musd = torch.load(musd_path, map_location="cpu")
                                    discrim_head_module.load_musd(_musd["mu"], _musd["sd"])
                                    log(rank, f"[dual-head][兜底] sd 全为1，已从 {musd_path} 回退")
                                else:
                                    log(
                                        rank,
                                        "[dual-head][FATAL] sd 全为1 且 DISCRIM_MUSD_PATH 未设置或不存在，标准化失效！",
                                    )
                            else:
                                log(
                                    rank,
                                    f"[dual-head] mu/sd 已加载: sd_range=[{sd_min:.3f},{sd_max:.3f}]",
                                )
                    else:
                        log(rank, "[WARN] 穿包装后找不到 discrim_head 模块")
                else:
                    log(
                        rank,
                        "[WARN] adapter_model.safetensors 里没有 base_model.model.discrim_head.* 的 key",
                    )
            else:
                log(rank, f"[WARN] {adapter_safetensors} 不存在，discrim_head 未加载")

            calib_path = getattr(args, "calibrated_head", None)
            if calib_path and not os.path.isfile(calib_path):
                raise FileNotFoundError(f"--calibrated-head 文件不存在: {calib_path}")
            if calib_path:
                log(rank, f"[dual-head] 用校准头覆盖端到端头: {calib_path}")
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
                if discrim_head_module is not None and hasattr(
                    discrim_head_module, "modules_to_save"
                ):
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
                        norm = w0.norm().item()
                        sd_min = float(discrim_head_module.sd.min())
                        sd_max = float(discrim_head_module.sd.max())
                    log(
                        rank,
                        f"[dual-head] 校准头已加载, classifier[0].weight 范数={norm:.4f}, sd_range=[{sd_min:.3f},{sd_max:.3f}]",
                    )
                else:
                    raise RuntimeError("校准头覆盖失败：找不到 discrim_head_module")
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            processor = AutoProcessor.from_pretrained(
                args.base, max_pixels=args.max_pixels, trust_remote_code=True
            )
            model = AutoModelForImageTextToText.from_pretrained(
                args.base,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map="balanced" if hf_model_sharded else None,
            )
            if device and not hf_model_sharded:
                model = model.to(device)
            if args.adapter:
                from peft import PeftModel

                log(rank, f"loading LoRA adapter: {args.adapter}")
                model = PeftModel.from_pretrained(model, args.adapter)
                if device and not hf_model_sharded:
                    model = model.to(device)

        model.eval()

        repr_head = None
        std_head_params = None  # (mu, sd) for std-head mode

        if getattr(args, "std_head", None):
            log(rank, f"loading std head (with normalization): {args.std_head}")
            repr_head, mu, sd = _load_std_head(args.std_head)
            repr_head.eval()
            std_head_params = (mu, sd)
            log(rank, f"  std_head: mu={mu.shape}, sd={sd.shape}, head loaded on cpu")
        elif getattr(args, "repr_head", None):
            log(rank, f"loading repr head (no normalization): {args.repr_head}")
            repr_head = _load_repr_head(args.repr_head)
            repr_head.eval()

        mode_desc = (
            "[std-head mode]" if std_head_params else ("[repr-head mode]" if repr_head else "")
        )
        dual_head_mode = getattr(args, "dual_head", False)
        if dual_head_mode:
            mode_desc = "[dual-head mode]"
            log(rank, "dual-head 原生模型：一次 forward 拿双路输出（端到端）")
        log(rank, f"model loaded on {device}, running {len(data_list)} probe(s) {mode_desc}")

        for idx, (data_p, out_d, tag_t) in enumerate(zip(data_list, out_list, tag_list)):
            log(
                rank, f"=== probe {idx + 1}/{len(data_list)}: {data_p} -> {out_d} (tag={tag_t}) ==="
            )
            Path(out_d).mkdir(parents=True, exist_ok=True)
            _run_one_probe_hf(
                args,
                rank,
                world,
                local_rank,
                data_path=Path(data_p),
                out_dir=Path(out_d),
                model=model,
                processor=processor,
                probe_idx=idx,
                repr_head=repr_head,
                std_head_params=std_head_params,
                system_override=getattr(args, "system_override", None),
                guide_prefix=getattr(args, "guide_prefix", None),
                no_guide=getattr(args, "no_guide", False),
                dual_head=dual_head_mode,
            )
    else:
        if len(data_list) > 1:
            log(rank, "⚠️ vllm 后端当前只支持单探针，多探针将只跑第一个（建议用 --backend hf）")
        args._data_list = [data_list[0]]
        args._out_list = [out_list[0]]
        args._tag_list = [tag_list[0]]
        out_dir = Path(args._out_list[0])
        (out_dir / "shards").mkdir(parents=True, exist_ok=True)
        sp = shard_path(out_dir, rank, world)

        rows = load_jsonl(Path(args._data_list[0]), args.limit)
        mine = [r for i, r in enumerate(rows) if i % world == rank]
        done = load_done_ids(sp)
        todo = [r for r in mine if (r.get("metadata") or {}).get("sample_id") not in done]
        log(
            rank,
            f"[{args.backend}] total={len(rows)} mine={len(mine)} done={len(done)} todo={len(todo)} -> {sp}",
        )
        if not todo:
            log(rank, "nothing to do")
            return

        _infer_vllm(args, rank, local_rank, todo, sp)


def _infer_vllm(args, rank: int, local_rank: int, todo: list[dict], sp: Path) -> None:
    """Offline batched inference with vLLM: one generate call for the whole shard.

    Requires vllm >= 0.11 for Qwen3-VL. Each rank pins a single device through
    `CUDA_VISIBLE_DEVICES` and starts a single-device engine; the LoRA adapter is attached with
    `LoRARequest` under `enable_lora=True`.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank % max(torch.cuda.device_count(), 1))
        vis = os.environ["CUDA_VISIBLE_DEVICES"]
    log(rank, f"vllm CUDA_VISIBLE_DEVICES={vis} loading base: {args.base}")

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    processor = AutoProcessor.from_pretrained(
        args.base, max_pixels=args.max_pixels, trust_remote_code=True
    )

    llm_kwargs = dict(
        model=args.base,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1, "video": 0},
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
    )
    lora_req = None
    if args.adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
        lora_req = LoRARequest("sft", 1, args.adapter)
        log(rank, f"vllm LoRA adapter: {args.adapter}")
    llm = LLM(**llm_kwargs)

    sp_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    requests = []
    for row in todo:
        messages = _build_messages(row)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm: dict = {}
        if image_inputs:
            mm["image"] = image_inputs
        requests.append({"prompt": text, "multi_modal_data": mm})

    t0 = time.time()
    outputs = llm.generate(
        requests,
        sampling_params=sp_params,
        lora_request=lora_req,
    )
    with sp.open("a", encoding="utf-8") as pf:
        for row, out in zip(todo, outputs):
            raw = out.outputs[0].text
            _write_record(pf, row, raw)
        pf.flush()
    dt = time.time() - t0
    log(rank, f"shard done in {dt:.1f}s ({len(todo) / dt:.2f} it/s) -> {sp}")


def merge_and_score(args: argparse.Namespace) -> dict:
    """Merge the shards into `predictions.jsonl`, compute the gates and all metrics, and write them out.

    """
    out_dir = Path(args.out)
    shard_dir = out_dir / "shards"
    world = int(os.environ.get("WORLD_SIZE", 1))
    per_rank: dict[int, list[dict]] = {}
    for r in range(world):
        fp = shard_dir / f"preds.rank{r}of{world}.jsonl"
        recs: list[dict] = []
        if fp.exists():
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        per_rank[r] = recs

    total = sum(len(v) for v in per_rank.values())
    records: list[dict] = []
    cursor = {r: 0 for r in range(world)}
    for i in range(total):
        r = i % world
        if cursor[r] < len(per_rank[r]):
            records.append(per_rank[r][cursor[r]])
            cursor[r] += 1
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as pf:
        for rec in records:
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pairs: list[tuple[dict, dict]] = []
    n_valid = 0
    n_dec_ok = 0
    n_ambiguous_skipped = 0
    for rec in records:
        pred = rec["pred"]
        g = rec["gt"]
        if g.get("decision") is None:
            n_ambiguous_skipped += 1
            continue
        gt_md = g.get("metadata") or {"relation": rec.get("relation")}
        gt = {**g, "relation": rec.get("relation") or gt_md.get("relation"), "metadata": gt_md}
        pairs.append((pred, gt))
        if pred.get("valid"):
            n_valid += 1
            if decision_correct(pred, gt):
                n_dec_ok += 1

    n = len(records)
    fmt_rate = n_valid / n if n else 0.0
    dec_acc = n_dec_ok / n if n else 0.0

    metrics = aggregate_metrics(pairs, iou_threshold=args.iou_threshold)
    mts = metamorphic_test_satisfaction(pairs, iou_threshold=args.iou_threshold)

    within_image = None
    by_img: dict[str, list[tuple]] = {}
    for pred, gt in pairs:
        ik = image_key_from_sample_id((gt.get("metadata") or {}).get("sample_id"))
        if ik is None:
            continue
        by_img.setdefault(ik, []).append((pred, gt))
    if by_img:
        paired_pairs: list[tuple] = []
        n_paired_imgs = 0
        for ik, ps in by_img.items():
            golds = {(g.get("decision")) for _, g in ps}
            if "answer" in golds and "abstain" in golds:
                paired_pairs.extend(ps)
                n_paired_imgs += 1
        if paired_pairs:
            wm = aggregate_metrics(paired_pairs, iou_threshold=args.iou_threshold)
            within_image = {
                "n_paired_images": n_paired_imgs,
                "n_samples": len(paired_pairs),
                "MCC": wm["safeguard"]["MCC"],
                "AAC": wm["safeguard"]["AAC"],
                "FCA": wm["headline"]["FCA"],
                "confusion_matrix": wm["confusion_matrix"],
            }

    result = {
        "tag": args.tag,
        "data": str(args.data),
        "adapter": args.adapter,
        "n_samples": n,
        "n_ambiguous_skipped": n_ambiguous_skipped,
        "gate": {
            "format_valid_rate": round(fmt_rate, 4),
            "decision_accuracy": round(dec_acc, 4),
            "format_valid_min": args.gate_fmt_min,
            "decision_acc_min": args.gate_dec_min,
            "passed": fmt_rate >= args.gate_fmt_min and dec_acc >= args.gate_dec_min,
        },
        "metrics": metrics,
        "mts": mts,
        "within_image": within_image,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    g = result["gate"]
    h = metrics["headline"]
    s = metrics["safeguard"]
    gr = metrics["grounding_reason"]
    lines = [
        f"=== ReGate eval [{args.tag}] ===",
        f"data: {args.data}",
        f"adapter: {args.adapter}",
        f"n={n}",
        "",
        "--- Gate (format and decision precheck) ---",
        f"format_valid_rate = {g['format_valid_rate']:.4f}  (>= {g['format_valid_min']} ?)",
        f"decision_accuracy = {g['decision_accuracy']:.4f}  (>= {g['decision_acc_min']} ?)",
        f"GATE PASSED = {g['passed']}",
        "",
        "--- Headline ---",
        f"FCA (lower better)       = {h['FCA']:.4f}",
        f"HOR_FCA (lower better)   = {h['HOR_FCA']:.4f}  "
        f"(n={h['HOR_count']['high_occlusion_unanswerable']})",
        f"MTS_micro                = {mts['MTS_micro']:.4f}",
        f"MTS_macro                = {mts['MTS_macro']:.4f}",
        "",
        "--- Safeguard ---",
        f"A_FU (lower better)      = {s['A_FU']:.4f}",
        f"U_Abs (higher better)    = {s['U_Abs']:.4f}",
        f"AAC (higher better)      = {s['AAC']:.4f}",
        f"MCC                      = {s['MCC']:.4f}",
        "",
        "--- Grounding / Reason ---",
        f"E2E_mIoU (higher better) = {gr['E2E_mIoU']:.4f}",
        f"E2E_mIoU_c               = {gr['E2E_mIoU_c']:.4f}",
        f"A_Ground (alias)         = {gr['A_Ground']:.4f}",
        f"A_Ground_c (alias)       = {gr['A_Ground_c']:.4f}",
        f"U_Reason (higher better) = {gr['U_Reason']:.4f}",
        f"U_Reason_c               = {gr['U_Reason_c']:.4f}",
        "",
        f"invalid_rate             = {metrics['invalid_rate']:.4f}",
        f"confusion_matrix         = {metrics['confusion_matrix']}",
        "",
        "--- MTS per relation ---",
    ]
    for rel, val in mts["per_relation"].items():
        cnt = mts["per_relation_counts"][rel]
        lines.append(f"  {rel:16s} {val:.4f}  ({cnt['pass']}/{cnt['total']})")
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"[eval] written: {out_dir}/metrics.json , predictions.jsonl , summary.txt", flush=True)
    return result


def _split_csv_arg(value: str, name: str) -> list[str]:
    """Split a comma-separated multi-probe argument, returning a non-empty list."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"--{name} 不能为空")
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        required=True,
        help="Data path(s); a single probe or several comma-separated probes, matched one-to-one in length with --out/--tag",
    )
    ap.add_argument("--base", help="Base model path (required for inference; may be omitted with --merge-only)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory (None = bare base model)")
    ap.add_argument(
        "--out", required=True, help="Output directory(ies); a single directory or several comma-separated ones, matched in length with --data/--tag"
    )
    ap.add_argument(
        "--tag", default="eval", help="Tag(s); a single tag or several comma-separated tags, matched in length with --data/--out"
    )
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N records (gate/debug)")
    ap.add_argument(
        "--gate", action="store_true", help="Gate mode: write a pass/fail marker to disk (does not abort the job)"
    )
    ap.add_argument("--merge-only", action="store_true", help="Only merge existing shards and compute metrics; skip inference")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--gate-fmt-min", type=float, default=0.95)
    ap.add_argument("--gate-dec-min", type=float, default=0.80)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90, help="vLLM GPU memory utilization cap")
    ap.add_argument("--max-model-len", type=int, default=8192, help="vLLM maximum context length")
    ap.add_argument("--max-lora-rank", type=int, default=64, help="vLLM maximum LoRA rank")
    ap.add_argument(
        "--enforce-eager", action="store_true", help="Disable vLLM CUDA graphs (faster startup and lower memory, slightly slower inference)"
    )
    ap.add_argument(
        "--max-num-seqs", type=int, default=64, help="vLLM maximum number of concurrent sequences (guards against sampler warmup OOM)"
    )
    ap.add_argument(
        "--repr-head", default=None, help="Representation head path (repr_head.pt; head-decision mode, no standardization)"
    )
    ap.add_argument(
        "--std-head",
        default=None,
        help="Standardized representation head path (std_head.pt, carries mu/sd standardization parameters; recommended)",
    )
    ap.add_argument(
        "--dual-head",
        action="store_true",
        help="Native dual-head model mode: obtain the LM and discriminative head outputs in a single forward pass",
    )
    ap.add_argument(
        "--head-conf-threshold", type=float, default=0.6, help="Confidence threshold for the dual-head discriminative head"
    )
    ap.add_argument(
        "--faithful-dual-output",
        action="store_true",
        help="Use only together with --dual-head. Strictly faithful evaluation mode",
    )
    ap.add_argument(
        "--dump-all-answer-candidates",
        action="store_true",
        help="Use only together with --dual-head --faithful-dual-output. For every sample (regardless of the head decision), additionally record one greedy-generate candidate label/box together with its scoring into the answer_candidate field (label/box/iou/label_correct/grounding_correct). This keeps a usable candidate answer available when the threshold is lowered. When the flag is off, records carry no answer_candidate field and no extra generate call is made.",
    )
    ap.add_argument(
        "--dump-hidden",
        action="store_true",
        help="Pre-flight dump mode: attach head_hidden[4096] to each record",
    )
    ap.add_argument(
        "--calibrated-head", default=None, help="Calibrated head path (discrim_head_calibrated.pt)"
    )
    ap.add_argument("--system-override", default=None, help="Override for the system prompt")
    ap.add_argument("--guide-prefix", default=None, help="Prefix used for guided regeneration")
    ap.add_argument(
        "--no-guide", action="store_true", help="Disable guided regeneration; if the first pass yields no valid box, emit a placeholder directly"
    )
    args = ap.parse_args()

    if args.faithful_dual_output and not args.dual_head:
        ap.error("--faithful-dual-output 仅在 --dual-head 时有效")

    if getattr(args, "dump_all_answer_candidates", False) and not (
        args.dual_head and args.faithful_dual_output
    ):
        ap.error("--dump-all-answer-candidates 仅在 --dual-head --faithful-dual-output 时有效")

    data_list = _split_csv_arg(args.data, "data")
    out_list = _split_csv_arg(args.out, "out")
    tag_list = _split_csv_arg(args.tag, "tag")
    if len(data_list) != len(out_list) or len(data_list) != len(tag_list):
        ap.error(
            f"--data/--out/--tag 必须等长: data={len(data_list)} out={len(out_list)} tag={len(tag_list)}"
        )
    args._data_list = data_list
    args._out_list = out_list
    args._tag_list = tag_list

    rank, world, local_rank = rank_world()

    data_list = args._data_list
    out_list = args._out_list
    tag_list = args._tag_list

    if args.merge_only:
        if rank == 0:
            for idx, (data_p, out_d, tag_t) in enumerate(zip(data_list, out_list, tag_list)):
                log(rank, f"=== merge-only probe {idx + 1}/{len(data_list)}: {tag_t} ===")
                args.data = data_p
                args.out = out_d
                args.tag = tag_t
                merge_and_score(args)
        return 0

    if not args.base:
        ap.error("--base is required for inference (omit only with --merge-only)")

    if world > 1:
        import torch.distributed as dist

        if dist.is_available() and not dist.is_initialized():
            try:
                dist.init_process_group(backend="gloo")
            except Exception:
                pass

    for idx, (data_p, out_d, tag_t) in enumerate(zip(data_list, out_list, tag_list)):
        Path(out_d).mkdir(parents=True, exist_ok=True)
        fingerprint = eval_fingerprint(args, data_path=Path(data_p))
        if rank == 0:
            stale_shards_check(Path(out_d), fingerprint, rank)
        if world > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

    run_inference(args, rank, world, local_rank)

    if world > 1:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        elif dist.is_available():
            try:
                dist.init_process_group(backend="gloo")
                dist.barrier()
            except Exception:
                log(rank, "⚠️ 无法 barrier，rank0 直接合并（可能漏未完成 shard）")

    if rank == 0:
        results = []
        for idx, (data_p, out_d, tag_t) in enumerate(zip(data_list, out_list, tag_list)):
            log(rank, f"=== merge probe {idx + 1}/{len(data_list)}: {tag_t} ===")
            args.data = data_p
            args.out = out_d
            args.tag = tag_t
            result = merge_and_score(args)
            results.append(result)
            if args.gate and not result["gate"]["passed"]:
                log(0, f"⚠️ GATE NOT PASSED [{tag_t}] — 仅警示，不阻断")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
