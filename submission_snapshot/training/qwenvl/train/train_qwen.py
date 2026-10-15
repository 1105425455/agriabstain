# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import torch.nn as nn
import transformers
from pathlib import Path

from qwenvl.train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.aux_head import VisualAuxHead, AuxHeadTrainer
from qwenvl.train.repr_head import ReprHead, ReprHeadTrainer
from qwenvl.model.qwen3vl_dual_head import Qwen3VLWithDualHead
from qwenvl.model.qwen3vl_dual_head_3reason import Qwen3VLWithTriHead
from qwenvl.model.qwen3vl_dual_head_2reason import Qwen3VLWithDualHead2Reason
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class DualHeadTrainer(Trainer):
    """Trainer for the earlier two-way head; not used by the reported configuration.

    Joint objective: language-model SFT loss plus a cross-entropy head loss on
    `decision_labels`, weighted by lambda with a short head warm-up. The head is a base-model
    module under LoRA, so it is added to the optimizer explicitly and given a larger learning
    rate than the adapter.

    See `TriHeadTrainer` for the configuration that produces the reported numbers.
    """

    def __init__(self, *args, head_lambda: float = 0.3, head_warmup_steps: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_lambda = head_lambda
        self.head_warmup_steps = head_warmup_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        decision_labels = inputs.pop("decision_labels", None)
        reason_labels = inputs.pop("reason_labels", None)

        outputs = model(**inputs)

        NO_TARGET_W = float(os.environ.get("NO_TARGET_LOSS_WEIGHT", 1.0))
        WRONG_OBJECT_W = float(os.environ.get("WRONG_OBJECT_LOSS_WEIGHT", 1.0))
        ANSWER_W = 1.0

        labels = inputs.get("labels") if isinstance(inputs, dict) else None
        if labels is None:
            labels = getattr(outputs, "labels", None)

        if not hasattr(outputs, "logits") or labels is None:
            main_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        else:
            logits = outputs.logits  # [B, S, V]
            shift_logits = logits[:, :-1, :].contiguous()  # [B, S-1, V]
            shift_labels = labels[:, 1:].contiguous()  # [B, S-1]

            B, S_minus_1, V = shift_logits.shape
            # per-token CE [B*(S-1)]
            per_token_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                reduction="none",
            )  # [B*(S-1)]
            per_token_loss = per_token_loss.view(B, S_minus_1)  # [B, S-1]

            mask = (shift_labels != -100).float()  # [B, S-1]
            per_token_loss = per_token_loss * mask

            if reason_labels is not None:
                # reason_label: -1=answer/none→1.0, 0=wrong_object→WRONG_OBJECT_W, 1=no_target→NO_TARGET_W
                sample_weight = torch.where(
                    reason_labels == 1,
                    torch.tensor(
                        NO_TARGET_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                    ),
                    torch.where(
                        reason_labels == 0,
                        torch.tensor(
                            WRONG_OBJECT_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                        torch.tensor(
                            ANSWER_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                    ),
                )
            elif decision_labels is not None:
                if not getattr(self, "_reason_fallback_warned", False):
                    import os as _os

                    if int(_os.environ.get("LOCAL_RANK", 0)) == 0:
                        print(
                            "[compute_loss][WARN] reason_labels 缺失，回退到 decision_labels 加权(abstain=1.5)",
                            flush=True,
                        )
                    self._reason_fallback_warned = True
                sample_weight = torch.where(
                    decision_labels == 0,
                    torch.tensor(1.5, device=per_token_loss.device, dtype=per_token_loss.dtype),
                    torch.tensor(1.0, device=per_token_loss.device, dtype=per_token_loss.dtype),
                )
            else:
                sample_weight = torch.ones(
                    B, device=per_token_loss.device, dtype=per_token_loss.dtype
                )

            weight_matrix = sample_weight.unsqueeze(1)  # [B, 1] → broadcast to [B, S-1]

            _num = (per_token_loss.float() * weight_matrix.float() * mask.float()).sum()
            _den = mask.float().sum().clamp_min(1.0)
            main_loss = (_num / _den).to(logits.dtype)

        head_logits = (
            outputs.get("head_logits")
            if isinstance(outputs, dict)
            else getattr(outputs, "head_logits", None)
        )
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

        head_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        if head_logits is not None and decision_labels is not None:
            valid_mask = decision_labels >= 0
            if valid_mask.any():
                valid_logits = head_logits[valid_mask]  # [n_valid, 2]
                valid_labels = decision_labels[valid_mask].long()  # [n_valid]
                head_loss = nn.functional.cross_entropy(valid_logits, valid_labels)

        lam = self.head_lambda
        if self.state.global_step < self.head_warmup_steps:
            lam = self.head_lambda * (self.state.global_step / max(self.head_warmup_steps, 1))

        loss = main_loss + lam * head_loss

        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "head_loss": float(head_loss.detach()),
                "main_loss": float(main_loss.detach()),
                "head_lambda": lam,
            }
            if reason_labels is not None:
                _psl = (per_token_loss * mask).sum(1) / mask.sum(1).clamp_min(
                    1.0
                )
                _log_extra = {}
                for _name, _rid in [
                    ("loss_answer", -1),
                    ("loss_wrong_object", 0),
                    ("loss_no_target", 1),
                ]:
                    _m = reason_labels == _rid
                    if _m.any():
                        _log_extra[_name] = float(_psl[_m].mean().detach())
                log_dict.update(_log_extra)
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Add `discrim_head` to the optimizer explicitly, since LoRA freezes the backbone."""
        if self.optimizer is not None:
            return self.optimizer

        super().create_optimizer()

        discrim_params = (
            list(self.model.discrim_head.parameters())
            if hasattr(self.model, "discrim_head")
            else []
        )
        if discrim_params:
            existing = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            missing = [p for p in discrim_params if id(p) not in existing]
            if missing:
                discrim_lr = self.args.learning_rate * 10
                self.optimizer.add_param_group({"params": missing, "lr": discrim_lr})

        return self.optimizer


class TriHeadTrainer(Trainer):
    """Trainer for the three-way discriminative head; this is the reported configuration.

    Joint objective:
      - language-model SFT loss
      - cross-entropy head loss on `head_labels` (0 = answer, 1 = wrong_object, 2 = no_target)
      - the two terms are combined with a configurable lambda after a short head warm-up

    The head is a base-model module under LoRA, so it is added to the optimizer explicitly and
    trained with a larger learning rate than the adapter. A class weight compensates for the
    label imbalance of the training mixture; the mixture and the weight are reported in the
    technical supplement.

    Selected by `USE_TRI_HEAD=true` together with `--use_dual_head True`.
    """

    def __init__(self, *args, head_lambda: float = 0.3, head_warmup_steps: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_lambda = head_lambda
        self.head_warmup_steps = head_warmup_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        head_labels = inputs.pop("head_labels", None)
        reason_labels = inputs.pop("reason_labels", None)

        outputs = model(**inputs)

        NO_TARGET_W = float(os.environ.get("NO_TARGET_LOSS_WEIGHT", 1.0))
        WRONG_OBJECT_W = float(os.environ.get("WRONG_OBJECT_LOSS_WEIGHT", 1.0))
        ANSWER_W = 1.0

        labels = inputs.get("labels") if isinstance(inputs, dict) else None
        if labels is None:
            labels = getattr(outputs, "labels", None)

        if not hasattr(outputs, "logits") or labels is None:
            main_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        else:
            logits = outputs.logits  # [B, S, V]
            shift_logits = logits[:, :-1, :].contiguous()  # [B, S-1, V]
            shift_labels = labels[:, 1:].contiguous()  # [B, S-1]

            B, S_minus_1, V = shift_logits.shape
            # per-token CE [B*(S-1)]
            per_token_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                reduction="none",
            )  # [B*(S-1)]
            per_token_loss = per_token_loss.view(B, S_minus_1)  # [B, S-1]

            mask = (shift_labels != -100).float()  # [B, S-1]
            per_token_loss = per_token_loss * mask

            if reason_labels is not None:
                # reason_label: -1=answer/none→1.0, 0=wrong_object→WRONG_OBJECT_W, 1=no_target→NO_TARGET_W
                sample_weight = torch.where(
                    reason_labels == 1,
                    torch.tensor(
                        NO_TARGET_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                    ),
                    torch.where(
                        reason_labels == 0,
                        torch.tensor(
                            WRONG_OBJECT_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                        torch.tensor(
                            ANSWER_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                    ),
                )
            elif head_labels is not None:
                sample_weight = torch.where(
                    head_labels == 2,  # no_target
                    torch.tensor(
                        NO_TARGET_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                    ),
                    torch.where(
                        head_labels == 1,  # wrong_object
                        torch.tensor(
                            WRONG_OBJECT_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                        torch.tensor(
                            ANSWER_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                    ),
                )
            else:
                sample_weight = torch.ones(
                    B, device=per_token_loss.device, dtype=per_token_loss.dtype
                )

            weight_matrix = sample_weight.unsqueeze(1)  # [B, 1] → broadcast to [B, S-1]

            _num = (per_token_loss.float() * weight_matrix.float() * mask.float()).sum()
            _den = mask.float().sum().clamp_min(1.0)
            main_loss = (_num / _den).to(logits.dtype)

        head_logits = (
            outputs.get("head_logits")
            if isinstance(outputs, dict)
            else getattr(outputs, "head_logits", None)
        )
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

        head_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        if head_logits is not None and head_labels is not None:
            valid_mask = head_labels >= 0
            if valid_mask.any():
                valid_logits = head_logits[valid_mask]  # [n_valid, 3]
                valid_labels = head_labels[valid_mask].long()  # [n_valid]

                wo_head_weight = float(os.environ.get("WO_HEAD_WEIGHT", 1.5))
                class_weights = torch.tensor(
                    [1.0, wo_head_weight, 1.0], device=valid_logits.device, dtype=valid_logits.dtype
                )
                weights_per_sample = class_weights[valid_labels]

                per_sample_loss = nn.functional.cross_entropy(
                    valid_logits, valid_labels, reduction="none"
                )
                head_loss = (per_sample_loss * weights_per_sample).mean()

        lam = self.head_lambda
        if self.state.global_step < self.head_warmup_steps:
            lam = self.head_lambda * (self.state.global_step / max(self.head_warmup_steps, 1))

        loss = main_loss + lam * head_loss

        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "head_loss": float(head_loss.detach()),
                "main_loss": float(main_loss.detach()),
                "head_lambda": lam,
            }
            if reason_labels is not None:
                _psl = (per_token_loss * mask).sum(1) / mask.sum(1).clamp_min(
                    1.0
                )
                _log_extra = {}
                for _name, _rid in [
                    ("loss_answer", -1),
                    ("loss_wrong_object", 0),
                    ("loss_no_target", 1),
                ]:
                    _m = reason_labels == _rid
                    if _m.any():
                        _log_extra[_name] = float(_psl[_m].mean().detach())
                log_dict.update(_log_extra)
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Add `discrim_head` to the optimizer explicitly, since LoRA freezes the backbone."""
        if self.optimizer is not None:
            return self.optimizer

        super().create_optimizer()

        discrim_params = (
            list(self.model.discrim_head.parameters())
            if hasattr(self.model, "discrim_head")
            else []
        )
        if discrim_params:
            existing = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            missing = [p for p in discrim_params if id(p) not in existing]
            if missing:
                discrim_lr = self.args.learning_rate * 10
                self.optimizer.add_param_group({"params": missing, "lr": discrim_lr})

        return self.optimizer


class BiHeadTrainer(Trainer):
    """Trainer for the two-way collapse of the head; used only by the ablation on reason arity.

    Differs from `TriHeadTrainer` in the head output width and in how the labels are consumed:
    the three-way labels are mapped to binary (answer versus abstain) while the per-sample head
    weight still follows the original reason, so the two arms differ only in output arity. The
    language-model loss is unchanged.

    The head cross-entropy uses `reduction='none'` and is averaged after multiplying by the
    per-sample weight.
    """

    def __init__(self, *args, head_lambda: float = 0.4, head_warmup_steps: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_lambda = head_lambda
        self.head_warmup_steps = head_warmup_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        head_labels = inputs.pop("head_labels", None)
        reason_labels = inputs.pop("reason_labels", None)

        outputs = model(**inputs)

        NO_TARGET_W = float(os.environ.get("NO_TARGET_LOSS_WEIGHT", 1.0))
        WRONG_OBJECT_W = float(os.environ.get("WRONG_OBJECT_LOSS_WEIGHT", 1.0))
        ANSWER_W = 1.0

        labels = inputs.get("labels") if isinstance(inputs, dict) else None
        if labels is None:
            labels = getattr(outputs, "labels", None)

        if not hasattr(outputs, "logits") or labels is None:
            main_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        else:
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            B, S_minus_1, V = shift_logits.shape
            per_token_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                reduction="none",
            )
            per_token_loss = per_token_loss.view(B, S_minus_1)

            mask = (shift_labels != -100).float()
            per_token_loss = per_token_loss * mask

            if reason_labels is not None:
                sample_weight = torch.where(
                    reason_labels == 1,
                    torch.tensor(
                        NO_TARGET_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                    ),
                    torch.where(
                        reason_labels == 0,
                        torch.tensor(
                            WRONG_OBJECT_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                        torch.tensor(
                            ANSWER_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                    ),
                )
            elif head_labels is not None:
                sample_weight = torch.where(
                    head_labels == 2,
                    torch.tensor(
                        NO_TARGET_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                    ),
                    torch.where(
                        head_labels == 1,
                        torch.tensor(
                            WRONG_OBJECT_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                        torch.tensor(
                            ANSWER_W, device=per_token_loss.device, dtype=per_token_loss.dtype
                        ),
                    ),
                )
            else:
                sample_weight = torch.ones(
                    B, device=per_token_loss.device, dtype=per_token_loss.dtype
                )

            weight_matrix = sample_weight.unsqueeze(1)
            _num = (per_token_loss.float() * weight_matrix.float() * mask.float()).sum()
            _den = mask.float().sum().clamp_min(1.0)
            main_loss = (_num / _den).to(logits.dtype)

        head_logits = (
            outputs.get("head_logits")
            if isinstance(outputs, dict)
            else getattr(outputs, "head_logits", None)
        )
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

        head_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        if head_logits is not None and head_labels is not None:
            valid_mask = head_labels >= 0
            if valid_mask.any():
                valid_logits = head_logits[valid_mask]  # [n_valid, 2]
                valid_labels_raw = head_labels[valid_mask].long()

                binary_labels = torch.where(
                    valid_labels_raw == 0,
                    torch.tensor(0, device=valid_labels_raw.device, dtype=valid_labels_raw.dtype),
                    torch.tensor(1, device=valid_labels_raw.device, dtype=valid_labels_raw.dtype),
                )

                wo_head_weight = float(os.environ.get("WO_HEAD_WEIGHT", 1.5))
                sample_weights = torch.where(
                    valid_labels_raw == 1,
                    torch.tensor(
                        wo_head_weight, device=valid_labels_raw.device, dtype=valid_logits.dtype
                    ),
                    torch.tensor(1.0, device=valid_labels_raw.device, dtype=valid_logits.dtype),
                )

                per_sample_loss = nn.functional.cross_entropy(
                    valid_logits, binary_labels, reduction="none"
                )
                head_loss = (per_sample_loss * sample_weights).mean()

        # λ warmup
        lam = self.head_lambda
        if self.state.global_step < self.head_warmup_steps:
            lam = self.head_lambda * (self.state.global_step / max(self.head_warmup_steps, 1))

        loss = main_loss + lam * head_loss

        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "head_loss": float(head_loss.detach()),
                "main_loss": float(main_loss.detach()),
                "head_lambda": lam,
            }
            if reason_labels is not None:
                _psl = (per_token_loss * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                _log_extra = {}
                for _name, _rid in [
                    ("loss_answer", -1),
                    ("loss_wrong_object", 0),
                    ("loss_no_target", 1),
                ]:
                    _m = reason_labels == _rid
                    if _m.any():
                        _log_extra[_name] = float(_psl[_m].mean().detach())
                log_dict.update(_log_extra)
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Add `discrim_head` to the optimizer explicitly, since LoRA freezes the backbone."""
        if self.optimizer is not None:
            return self.optimizer

        super().create_optimizer()

        discrim_params = (
            list(self.model.discrim_head.parameters())
            if hasattr(self.model, "discrim_head")
            else []
        )
        if discrim_params:
            existing = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            missing = [p for p in discrim_params if id(p) not in existing]
            if missing:
                discrim_lr = self.args.learning_rate * 10
                self.optimizer.add_param_group({"params": missing, "lr": discrim_lr})

        return self.optimizer


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        if isinstance(trainer, ReprHeadTrainer) and trainer.repr_head is not None:
            import os

            head_path = os.path.join(output_dir, "repr_head.pt")
            torch.save(trainer.repr_head.state_dict(), head_path)
            print(f"[repr] 表征头权重已保存到 {head_path}", flush=True)
        if isinstance(trainer, DualHeadTrainer) and hasattr(trainer.model, "discrim_head"):
            import os

            trainer.model.save_dual_head(output_dir)
        elif isinstance(trainer, TriHeadTrainer) and hasattr(trainer.model, "discrim_head"):
            import os

            trainer.model.save_dual_head(output_dir)
        elif isinstance(trainer, BiHeadTrainer) and hasattr(trainer.model, "discrim_head"):
            import os

            trainer.model.save_dual_head(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    if isinstance(trainer, ReprHeadTrainer) and trainer.repr_head is not None:
        import os

        head_path = os.path.join(output_dir, "repr_head.pt")
        torch.save(trainer.repr_head.state_dict(), head_path)
        print(f"[repr] 表征头权重已保存到 {head_path}", flush=True)

    if isinstance(trainer, DualHeadTrainer) and hasattr(trainer.model, "discrim_head"):
        import os

        trainer.model.save_dual_head(output_dir)
    elif isinstance(trainer, TriHeadTrainer) and hasattr(trainer.model, "discrim_head"):
        import os

        trainer.model.save_dual_head(output_dir)
    elif isinstance(trainer, BiHeadTrainer) and hasattr(trainer.model, "discrim_head"):
        import os

        trainer.model.save_dual_head(output_dir)


class StopAfterStepCallback(transformers.TrainerCallback):
    """Save and stop at a chosen `global_step`.

    Controlled by the `STOP_AFTER_STEP` environment variable and disabled when it is unset or
    non-positive. It does not set `max_steps` and therefore leaves the scheduler untouched.
    """

    def on_step_end(self, args, state, control, **kwargs):
        target = int(os.environ.get("STOP_AFTER_STEP", "0"))
        if target <= 0:
            return control
        if state.global_step >= target:
            control.should_save = True
            control.should_training_stop = True
            if state.is_world_process_zero:
                print(
                    f"[StopAfterStepCallback] 达到目标步数 "
                    f"global_step={state.global_step} >= {target}，"
                    f"触发保存并停止训练",
                    flush=True,
                )
        return control


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if (
        "qwen3" in model_args.model_name_or_path.lower()
        and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower()
    ):
        use_dual_head = getattr(training_args, "use_dual_head", False)
        if use_dual_head:
            rank0_print("[dual-head][WARN] Qwen3VL-MoE 不支持 Qwen3VLWithDualHead，回退到标准模型")
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        use_dual_head = getattr(training_args, "use_dual_head", False)
        use_tri_head = os.environ.get("USE_TRI_HEAD", "False").lower() == "true"
        use_2reason_head = os.environ.get("USE_2REASON_HEAD", "False").lower() == "true"

        if use_tri_head:
            rank0_print("[dual-head][3reason] 使用 Qwen3VLWithTriHead 加载模型（三分类）")
            model = Qwen3VLWithTriHead.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        elif use_2reason_head:
            rank0_print("[dual-head][2reason] 使用 Qwen3VLWithDualHead2Reason 加载模型（二分类）")
            model = Qwen3VLWithDualHead2Reason.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        elif use_dual_head:
            rank0_print("[dual-head] 使用 Qwen3VLWithDualHead 加载模型")
            model = Qwen3VLWithDualHead.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
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
        f"the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}"
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

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

    use_dual_head = getattr(training_args, "use_dual_head", False)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType

        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules="all-linear",
            modules_to_save=["discrim_head"]
            if (use_dual_head or use_tri_head or use_2reason_head)
            else None,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(processor, data_args=data_args)

    use_aux = getattr(training_args, "use_aux_head", False)
    use_repr = getattr(training_args, "use_repr_head", False)
    use_tri_head = os.environ.get("USE_TRI_HEAD", "False").lower() == "true"
    use_2reason_head = os.environ.get("USE_2REASON_HEAD", "False").lower() == "true"

    if use_tri_head:
        rank0_print(
            f"[dual-head][3reason] 三分类判别头启用: 一次 forward 拿双路输出, lambda={getattr(training_args, 'head_lambda', 0.3)}"
        )
        base_for_attach = model.base_model.model if hasattr(model, "base_model") else model
        for p in base_for_attach.discrim_head.parameters():
            p.requires_grad = True
        rank0_print("[dual-head][3reason] discrim_head 参数已激活: requires_grad=True")
        musd_path = os.environ.get("DISCRIM_MUSD_PATH", None)
        if musd_path is None:
            raise ValueError(
                "[dual-head][3reason] 环境变量 DISCRIM_MUSD_PATH 未设置。判别头必须用固定 mu/sd 标准化"
                "（无标准化 MCC 会从 0.99 塌到 0.34），请设置 DISCRIM_MUSD_PATH 指向 discrim_musd.pt。"
            )
        if not os.path.exists(musd_path):
            raise FileNotFoundError(
                f"[dual-head][3reason] mu/sd 文件不存在: {musd_path}。判别头必须用固定 mu/sd 标准化"
                "（无标准化 MCC 会从 0.99 塌到 0.34），请先生成 discrim_musd.pt。"
            )
        _musd = torch.load(musd_path, map_location="cpu")
        _dh = base_for_attach.discrim_head
        if hasattr(_dh, "modules_to_save") and hasattr(_dh, "active_adapters"):
            _inner = _dh.modules_to_save[_dh.active_adapters[0]]
        else:
            _inner = _dh
        _inner.load_musd(_musd["mu"], _musd["sd"])
        rank0_print(
            f"[dual-head][3reason] 已载入 mu/sd 标准化统计量: {musd_path} "
            f"(source={_musd.get('source', 'n/a')}, n={_musd.get('n', 'n/a')}, "
            f"sd_range=[{float(_inner.sd.min()):.3f},{float(_inner.sd.max()):.3f}])"
        )
        trainer = TriHeadTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            head_lambda=getattr(training_args, "head_lambda", 0.3),
            head_warmup_steps=getattr(training_args, "head_warmup_steps", 10),
            **data_module,
        )
    elif use_2reason_head:
        rank0_print(
            f"[dual-head][2reason] 二分类判别头启用: 一次 forward 拿双路输出, lambda={getattr(training_args, 'head_lambda', 0.4)}"
        )
        base_for_attach = model.base_model.model if hasattr(model, "base_model") else model
        for p in base_for_attach.discrim_head.parameters():
            p.requires_grad = True
        rank0_print("[dual-head][2reason] discrim_head 参数已激活: requires_grad=True")
        musd_path = os.environ.get("DISCRIM_MUSD_PATH", None)
        if musd_path is None:
            raise ValueError(
                "[dual-head][2reason] 环境变量 DISCRIM_MUSD_PATH 未设置。判别头必须用固定 mu/sd 标准化"
                "（无标准化 MCC 会从 0.99 塌到 0.34），请设置 DISCRIM_MUSD_PATH 指向 discrim_musd.pt。"
            )
        if not os.path.exists(musd_path):
            raise FileNotFoundError(
                f"[dual-head][2reason] mu/sd 文件不存在: {musd_path}。判别头必须用固定 mu/sd 标准化"
                "（无标准化 MCC 会从 0.99 塌到 0.34），请先生成 discrim_musd.pt。"
            )
        _musd = torch.load(musd_path, map_location="cpu")
        _dh = base_for_attach.discrim_head
        if hasattr(_dh, "modules_to_save") and hasattr(_dh, "active_adapters"):
            _inner = _dh.modules_to_save[_dh.active_adapters[0]]
        else:
            _inner = _dh
        _inner.load_musd(_musd["mu"], _musd["sd"])
        rank0_print(
            f"[dual-head][2reason] 已载入 mu/sd 标准化统计量: {musd_path} "
            f"(source={_musd.get('source', 'n/a')}, n={_musd.get('n', 'n/a')}, "
            f"sd_range=[{float(_inner.sd.min()):.3f},{float(_inner.sd.max()):.3f}])"
        )
        trainer = BiHeadTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            head_lambda=getattr(training_args, "head_lambda", 0.4),
            head_warmup_steps=getattr(training_args, "head_warmup_steps", 10),
            **data_module,
        )
    elif use_dual_head:
        rank0_print(
            f"[dual-head] Dual-head 原生模型启用: 一次 forward 拿双路输出, lambda={getattr(training_args, 'head_lambda', 0.3)}"
        )
        base_for_attach = model.base_model.model if hasattr(model, "base_model") else model
        for p in base_for_attach.discrim_head.parameters():
            p.requires_grad = True
        rank0_print("[dual-head] discrim_head 参数已激活: requires_grad=True")
        musd_path = os.environ.get("DISCRIM_MUSD_PATH", None)
        if musd_path is None:
            raise ValueError(
                "[dual-head] 环境变量 DISCRIM_MUSD_PATH 未设置。判别头必须用固定 mu/sd 标准化"
                "（无标准化 MCC 会从 0.99 塌到 0.34），请设置 DISCRIM_MUSD_PATH 指向 discrim_musd.pt。"
            )
        if not os.path.exists(musd_path):
            raise FileNotFoundError(
                f"[dual-head] mu/sd 文件不存在: {musd_path}。判别头必须用固定 mu/sd 标准化"
                "（无标准化 MCC 会从 0.99 塌到 0.34），请先生成 discrim_musd.pt。"
            )
        _musd = torch.load(musd_path, map_location="cpu")
        _dh = base_for_attach.discrim_head
        if hasattr(_dh, "modules_to_save") and hasattr(_dh, "active_adapters"):
            _inner = _dh.modules_to_save[_dh.active_adapters[0]]
        else:
            _inner = _dh
        _inner.load_musd(_musd["mu"], _musd["sd"])
        rank0_print(
            f"[dual-head] 已载入 mu/sd 标准化统计量: {musd_path} "
            f"(source={_musd.get('source', 'n/a')}, n={_musd.get('n', 'n/a')}, "
            f"sd_range=[{float(_inner.sd.min()):.3f},{float(_inner.sd.max()):.3f}])"
        )
        trainer = DualHeadTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            head_lambda=getattr(training_args, "head_lambda", 0.3),
            head_warmup_steps=getattr(training_args, "head_warmup_steps", 10),
            **data_module,
        )
    elif use_repr:
        hidden_size = model.config.text_config.hidden_size  # 4096
        repr_head = ReprHead(hidden_size=hidden_size).to(
            device=model.device, dtype=(torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        for p in repr_head.parameters():
            p.requires_grad = True
        base_for_attach = model.base_model.model if hasattr(model, "base_model") else model
        base_for_attach.repr_head = repr_head
        rank0_print(
            f"[repr] 表征头主判别器启用: layer27 generation-prompt 位置 → CE头, lambda={getattr(training_args, 'head_lambda', 1.0)}"
        )
        trainer = ReprHeadTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            repr_head=repr_head,
            head_lambda=getattr(training_args, "head_lambda", 1.0),
            head_warmup_steps=getattr(training_args, "head_warmup_steps", 10),
            **data_module,
        )
    elif use_aux:
        hidden_size = model.config.text_config.hidden_size  # 4096
        aux_head = VisualAuxHead(hidden_size=hidden_size).to(
            device=model.device, dtype=(torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        for p in aux_head.parameters():
            p.requires_grad = True
        base_for_attach = model.base_model.model if hasattr(model, "base_model") else model
        base_for_attach.aux_head = aux_head
        rank0_print(
            f"[aux] 联合表征辅助头启用(轻量监控): layer27 last-token → CE头(label_smoothing0.1), lambda={getattr(training_args, 'aux_lambda', 0.5)}"
        )
        trainer = AuxHeadTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            aux_head=aux_head,
            aux_lambda=getattr(training_args, "aux_lambda", 0.5),
            aux_warmup_steps=getattr(training_args, "aux_warmup_steps", 100),
            **data_module,
        )
    else:
        trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )

    _stop_after = os.environ.get("STOP_AFTER_STEP", "0")
    if _stop_after.isdigit() and int(_stop_after) > 0:
        rank0_print(f"[StopAfterStepCallback] 已启用: global_step >= {_stop_after} 时保存并停止")
        trainer.add_callback(StopAfterStepCallback())

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="sdpa")
