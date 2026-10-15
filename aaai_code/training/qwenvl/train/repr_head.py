from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Trainer

IMAGE_TOKEN_ID = 151655  # Qwen3-VL image token
REPR_LAYER_IDX = 26
IGNORE_INDEX = -100


class ReprHead(nn.Module):
    """Earlier read-out head over the selected decoder layer; not used by the reported configuration.

    Takes the representation at the generation-prompt position and classifies it with an MLP
    (4096 -> 512 -> 256 -> 2). The read-out position is the index just before the first
    non-masked label during training, which is the same position as the final token after
    `add_generation_prompt` at inference time.

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


class ReprHeadTrainer(Trainer):
    """Trainer for the head above; not used by the reported configuration.

    A forward hook caches the selected decoder layer, and `compute_loss` reads that cache,
    takes the representation at the position described in `ReprHead`, passes it through the head
    and adds the head loss. Each sample must carry `decision_label` (1 = answer, 0 = abstain)
    so that the collator can forward it.
    """

    def __init__(
        self,
        *args,
        repr_head: nn.Module = None,
        head_lambda: float = 1.0,
        head_warmup_steps: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.repr_head = repr_head
        self.head_lambda = head_lambda
        self.head_warmup_steps = head_warmup_steps
        self._repr_hidden_cache = {}
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

    def _register_repr_hook(self, model):
        """Register a forward hook on the selected decoder layer and cache its [B, S, 4096] output.

        """
        layers = self._find_decoder_layers(model)
        if layers is None or len(layers) <= REPR_LAYER_IDX:
            raise RuntimeError(
                f"[repr] 找不到 decoder layers[{REPR_LAYER_IDX}]，hook 挂载失败。"
                f"layers={type(layers)} len={len(layers) if layers is not None else None}"
            )
        target = layers[REPR_LAYER_IDX]

        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            self._repr_hidden_cache["h"] = hs

        target.register_forward_hook(hook)
        self._hook_registered = True
        print(f"[repr] forward hook 已挂到 decoder layers[{REPR_LAYER_IDX}]（第27层）", flush=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.repr_head is not None and not self._hook_registered:
            self._register_repr_hook(model)

        decision_labels = inputs.pop("decision_labels", None)
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        self._repr_hidden_cache.pop("h", None)

        outputs = model(**inputs)
        main_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        head_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
        if (
            self.repr_head is not None
            and decision_labels is not None
            and "h" in self._repr_hidden_cache
            and input_ids is not None
            and labels is not None
        ):
            hs = self._repr_hidden_cache["h"]  # [B, S, 4096]

            if self.state.global_step < 3 and not hs.requires_grad:
                print(
                    f"[repr][WARN] step{self.state.global_step}: 第27层hidden requires_grad=False! "
                    f"head loss 反传不到 LoRA，表征头将失效。检查 gradient_checkpointing/hook。",
                    flush=True,
                )

            feats, valid_labels = [], []
            for i in range(hs.size(0)):
                if int(decision_labels[i]) < 0:
                    continue

                label_row = labels[i]  # [S]
                non_ignore_positions = torch.where(label_row != IGNORE_INDEX)[0]

                if len(non_ignore_positions) == 0:
                    continue

                assistant_first_idx = int(non_ignore_positions[0])

                repr_idx = assistant_first_idx - 1
                if repr_idx < 0:
                    continue

                repr_idx = max(0, min(repr_idx, hs.size(1) - 1))

                feats.append(hs[i, repr_idx, :])
                valid_labels.append(int(decision_labels[i]))

            if feats:
                feats = torch.stack(feats).to(dtype=self.repr_head.classifier[0].weight.dtype)
                valid_labels_tensor = torch.tensor(
                    valid_labels, device=feats.device, dtype=torch.long
                )

                logits = self.repr_head(feats)  # [n, 2]
                head_loss = nn.functional.cross_entropy(logits, valid_labels_tensor)

        lam = self.head_lambda
        if self.state.global_step < self.head_warmup_steps:
            lam = self.head_lambda * (self.state.global_step / max(self.head_warmup_steps, 1))

        loss = main_loss + lam * head_loss

        if self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "head_loss": float(head_loss.detach()),
                    "main_loss": float(main_loss.detach()),
                    "head_lambda": lam,
                }
            )

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Add the head parameters to the optimizer as an explicit parameter group.

        The HF trainer collects `requires_grad` parameters from `self.model`. Under LoRA the
        backbone is frozen and only the adapter is trainable, so the externally attached MLP is
        registered here rather than relying on attribute discovery.
        """
        if self.optimizer is not None:
            return self.optimizer
        super().create_optimizer()
        repr_params = list(self.repr_head.parameters()) if self.repr_head is not None else []
        if repr_params:
            existing = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            missing = [p for p in repr_params if id(p) not in existing]
            if missing:
                repr_lr = self.args.learning_rate * 10
                self.optimizer.add_param_group({"params": missing, "lr": repr_lr})
        return self.optimizer
