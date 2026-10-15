from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Trainer

IMAGE_TOKEN_ID = 151655  # Qwen3-VL image token
AUX_LAYER_IDX = 26


class VisualAuxHead(nn.Module):
    """Auxiliary probe over the joint representation; not used by the reported configuration.

    A 3-layer MLP (4096 -> 512 -> 256 -> 2) trained with label-smoothed cross-entropy as a
    lightweight monitor of representation separability, at a small loss weight. It reads the
    joint image-and-query representation at the last token rather than a pooled visual
    representation.

    The reported discriminative head is defined in
    `training/qwenvl/model/qwen3vl_dual_head_3reason.py`.
    """

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
        """Return classification logits [B, num_classes]."""
        return self.classifier(x)


class AuxHeadTrainer(Trainer):
    """Adds the auxiliary probe above to the standard trainer; not used by the reported configuration.

    A forward hook caches the selected decoder layer, and `compute_loss` reads that cache,
    passes the representation through the probe and adds the auxiliary term. Each sample must
    carry `aux_label` (1 = answer, 0 = abstain) so that the collator can forward it.
    """

    def __init__(
        self,
        *args,
        aux_head: nn.Module = None,
        aux_lambda: float = 0.05,
        aux_warmup_steps: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aux_head = aux_head
        self.aux_lambda = aux_lambda
        self.aux_warmup_steps = aux_warmup_steps
        self._aux_hidden_cache = {}
        self._hook_registered = False

    def _find_decoder_layers(self, m):
        """Locate the text-decoder layers through the surrounding wrappers."""
        for _ in range(6):
            if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
                m = m.base_model.model
            elif hasattr(m, "module"):
                m = m.module
            else:
                break
        # Qwen3VLForConditionalGeneration.model.language_model.layers
        node = m
        for attr in ("model", "language_model", "layers"):
            if hasattr(node, attr):
                node = getattr(node, attr)
            else:
                node = None
                break
        return node

    def _register_aux_hook(self, model):
        """Register a forward hook on the selected decoder layer and cache its [B, S, 4096] output.

        """
        layers = self._find_decoder_layers(model)
        if layers is None or len(layers) <= AUX_LAYER_IDX:
            raise RuntimeError(
                f"[aux] 找不到 decoder layers[{AUX_LAYER_IDX}]，hook 挂载失败。"
                f"layers={type(layers)} len={len(layers) if layers is not None else None}"
            )
        target = layers[AUX_LAYER_IDX]

        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            self._aux_hidden_cache["h"] = hs

        target.register_forward_hook(hook)
        self._hook_registered = True
        print(f"[aux] forward hook 已挂到 decoder layers[{AUX_LAYER_IDX}]（第27层）", flush=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.aux_head is not None and not self._hook_registered:
            self._register_aux_hook(model)
        aux_labels = inputs.pop("aux_labels", None)
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        self._aux_hidden_cache.pop("h", None)
        outputs = model(**inputs)
        main_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        aux_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        if (
            self.aux_head is not None
            and aux_labels is not None
            and "h" in self._aux_hidden_cache
            and input_ids is not None
        ):
            hs = self._aux_hidden_cache["h"]  # [B, S, 4096]
            if self.state.global_step < 3 and not hs.requires_grad:
                print(
                    f"[aux][WARN] step{self.state.global_step}: 第27层hidden requires_grad=False! "
                    f"辅助loss反传不到LoRA，辅助头将失效。检查gradient_checkpointing/hook。",
                    flush=True,
                )
            if attention_mask is not None:
                last_pos = attention_mask.long().sum(dim=1) - 1  # [B]
            else:
                last_pos = torch.full(
                    (hs.size(0),), hs.size(1) - 1, device=hs.device, dtype=torch.long
                )
            feats, valid_labels = [], []
            for i in range(hs.size(0)):
                if int(aux_labels[i]) < 0:
                    continue
                p = int(last_pos[i].clamp(min=0, max=hs.size(1) - 1))
                feats.append(hs[i, p, :])
                valid_labels.append(int(aux_labels[i]))
            if feats:
                feats = torch.stack(feats).to(dtype=self.aux_head.classifier[0].weight.dtype)
                valid_labels_tensor = torch.tensor(
                    valid_labels, device=feats.device, dtype=torch.long
                )

                logits = self.aux_head(feats)  # [n, 2]
                aux_loss = nn.functional.cross_entropy(
                    logits, valid_labels_tensor, label_smoothing=0.1
                )

        lam = self.aux_lambda
        if self.state.global_step < self.aux_warmup_steps:
            lam = self.aux_lambda * (self.state.global_step / max(self.aux_warmup_steps, 1))

        loss = main_loss + lam * aux_loss

        if self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "aux_loss": float(aux_loss.detach()),
                    "main_loss": float(main_loss.detach()),
                    "aux_lambda": lam,
                }
            )

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Add the probe parameters to the optimizer as an explicit parameter group.

        The HF trainer collects `requires_grad` parameters from `self.model`. Under LoRA the
        backbone is frozen and only the adapter is trainable, so the externally attached MLP is
        registered here rather than relying on attribute discovery.
        """
        if self.optimizer is not None:
            return self.optimizer
        super().create_optimizer()
        aux_params = list(self.aux_head.parameters()) if self.aux_head is not None else []
        if aux_params:
            existing = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            missing = [p for p in aux_params if id(p) not in existing]
            if missing:
                aux_lr = self.args.learning_rate * 10
                self.optimizer.add_param_group({"params": missing, "lr": aux_lr})
        return self.optimizer
