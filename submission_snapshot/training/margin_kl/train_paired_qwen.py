#!/usr/bin/env python3
"""train_paired_qwen.py — Paired retrain entry point (aaai_code edition).

The local ``training/qwenvl`` package provides the shared model and data code.

Usage:
    export SFT_INIT_CKPT=/path/to/checkpoint-444
    export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
    export DISCRIM_MUSD_PATH=/path/to/discrim_musd.pt
    export ZREF_CACHE_PATH=/path/to/zref_cache.pt
    export OUTPUT_DIR=/path/to/output
    torchrun --nproc_per_node=8 \\
        training/margin_kl/train_paired_qwen.py \\
        --deepspeed training/deepspeed/zero2.json \\
        --model_name_or_path "$BASE_MODEL" \\
        ...
"""

import os
import re
from pathlib import Path

import torch
import transformers


from qwenvl.train.train_qwen import *  # noqa: F403, F401
from qwenvl.train.train_qwen import (
    local_rank,
    rank0_print,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    Qwen3VLWithTriHead,
    AutoProcessor,
    make_supervised_data_module,
    safe_save_model_for_hf_trainer,
    set_model,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    replace_qwen2_vl_attention_class,
)

from paired_trainer import PairedTriHeadTrainer
from paired_dataset import PairedDatasetWrapper, PairedDataCollator
from safetensors.torch import load_file


class GuardrailCallback(transformers.TrainerCallback):
    """Print diagnostics from rank 0 every twenty steps."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 20 == 0 and state.global_step > 0:
            trainer = kwargs.get("trainer", None)
            loss_info = ""
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        loss_info = f" loss={entry['loss']:.4f}"
                        break
            msg = f"[Guardrail] step={state.global_step}{loss_info}"
            if trainer is not None and hasattr(trainer, "zref"):
                msg += f" zref_N={trainer.zref.shape[0]}"
            if local_rank == 0 or local_rank is None:
                print(msg, flush=True)


def train_paired(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    sft_init_ckpt = os.environ.get("SFT_INIT_CKPT", "")
    if not sft_init_ckpt:
        raise RuntimeError(
            "SFT_INIT_CKPT 环境变量必须指定（指向 checkpoint-444 目录，"
            "含 adapter_model.safetensors）"
        )

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if (
        "qwen3" in model_args.model_name_or_path.lower()
        and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower()
    ):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        rank0_print("[paired][3reason] 使用 Qwen3VLWithTriHead 加载模型（三分类）")
        model = Qwen3VLWithTriHead.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(
        f"the initlized model is {model_args.model_name_or_path} "
        f"the class is {model.__class__.__name__}"
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, TaskType

        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules="all-linear",
            modules_to_save=["discrim_head"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)
        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    rank0_print(f"[续训] 加载 adapter 权重: {sft_init_ckpt}/adapter_model.safetensors")
    adapter_sd = load_file(f"{sft_init_ckpt}/adapter_model.safetensors")
    load_result = set_peft_model_state_dict(model, adapter_sd, adapter_name="default")
    rank0_print(
        f"[续训] adapter 加载 missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
    )
    if load_result.unexpected_keys:
        rank0_print(f"[续训] unexpected 前5: {load_result.unexpected_keys[:5]}")
    assert len(load_result.unexpected_keys) == 0, (
        f"[续训] 有 {len(load_result.unexpected_keys)} 个 LoRA 权重未映射成功(原bug)! "
        f"unexpected前5: {load_result.unexpected_keys[:5]}"
    )


    def _extract_layer_num(name):
        m = re.search(r"language_model\.layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            ln = _extract_layer_num(name)
            if ln is not None and 20 <= ln <= 26:
                p.requires_grad = True
                n_train += 1
            else:
                p.requires_grad = False
                n_freeze += 1
        if "visual" in name or "vision" in name or "merger" in name:
            p.requires_grad = False
    rank0_print(f"[窄带LoRA] 可训lora={n_train} 冻结lora={n_freeze}")

    base_for_attach = model.base_model.model if hasattr(model, "base_model") else model
    for p in base_for_attach.discrim_head.parameters():
        p.requires_grad = True
    rank0_print("[paired][3reason] discrim_head 参数已激活: requires_grad=True")

    _dh = base_for_attach.discrim_head
    if hasattr(_dh, "modules_to_save") and hasattr(_dh, "active_adapters"):
        _inner = _dh.modules_to_save[_dh.active_adapters[0]]
    else:
        _inner = _dh
    calib_path = f"{sft_init_ckpt}/discrim_head_calibrated.pt"
    calib_sd = torch.load(calib_path, map_location="cpu")
    _inner.load_musd(calib_sd["mu"], calib_sd["sd"])
    rank0_print(
        f"[paired][3reason] 已从calibrated.pt载入mu/sd(与z_ref同源) "
        f"sd_range=[{float(_inner.sd.min()):.3f},{float(_inner.sd.max()):.3f}]"
    )
    clf_state = {k.replace("classifier.", ""): v for k, v in calib_sd["head_state"].items()}
    _inner.classifier.load_state_dict(clf_state)
    rank0_print(f"[续训] head classifier 权重加载完成（{len(clf_state)} 个 tensor）")

    data_module = make_supervised_data_module(processor, data_args=data_args)
    data_module["train_dataset"] = PairedDatasetWrapper(data_module["train_dataset"])
    data_module["data_collator"] = PairedDataCollator(tokenizer=tokenizer)

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  TriHeadTrainer → PairedTriHeadTrainer                          ║
    # ╚══════════════════════════════════════════════════════════════════╝
    trainer = PairedTriHeadTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        head_lambda=getattr(training_args, "head_lambda", 0.4),
        head_warmup_steps=getattr(training_args, "head_warmup_steps", 0),
        **data_module,
    )

    trainer.add_callback(GuardrailCallback())

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train_paired(attn_implementation="sdpa")
