from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration
from typing import Optional

REPR_LAYER_HS_IDX = 27


class DiscrimHead(nn.Module):
    """Discriminative head: frozen per-dimension standardisation followed by a 3-layer MLP.

    Architecture:
      standardize: (x - mu) / sd        mu, sd are frozen buffers and receive no gradient
      Linear(4096 -> 512) -> GELU -> Dropout(0.1)
      Linear(512 -> 256)  -> GELU -> Dropout(0.1)
      Linear(256 -> 3)                 0 = answer, 1 = wrong_object, 2 = no_target

    mu and sd are estimated once on the training split with the base backbone and then kept
    fixed, so inference stays a single forward pass. They play the role of the running
    statistics of a frozen BatchNorm and are serialised together with the head through
    `modules_to_save`. The choice of representation layer and the effect of standardisation
    are reported in the technical supplement.
    """

    def __init__(self, hidden_size: int = 4096, num_classes: int = 3):
        super().__init__()

        self.register_buffer("mu", torch.zeros(hidden_size))
        self.register_buffer("sd", torch.ones(hidden_size))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def load_musd(self, mu: torch.Tensor, sd: torch.Tensor) -> None:
        """Load the frozen standardisation statistics; call once before training."""
        with torch.no_grad():
            self.mu.copy_(mu.to(self.mu.dtype))
            self.sd.copy_(sd.to(self.sd.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise with (x - mu) / sd, then classify with the MLP.

        Standardisation runs in float32 because mu spans a wide range, then the result is cast
        back to the dtype of the classifier weights.

        Args:
            x: [B, hidden_size] representation at the generation-prompt last token

        Returns:
            logits: [B, num_classes] with 0 = answer, 1 = wrong_object, 2 = no_target
        """
        out_dtype = self.classifier[0].weight.dtype
        x = (x.float() - self.mu.float()) / self.sd.float()
        return self.classifier(x.to(out_dtype))


class Qwen3VLWithTriHead(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with an attached three-way discriminative head.

    Extends Qwen3VLForConditionalGeneration with:
      1. `discrim_head`: the DiscrimHead defined above (num_classes=3)
      2. `output_hidden_states=True` to read the selected decoder layer
      3. `head_logits` and `head_hidden` fields on the forward output

    One `model(**inputs)` call therefore yields both the language-model output and the
    discriminative logits, with no forward hook involved.

    Label convention: 0 = answer, 1 = wrong_object, 2 = no_target.

    Under LoRA the head is a base-model module rather than an adapter module, so it must be
    saved and restored explicitly; see `save_dual_head` / `load_dual_head` below.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.text_config.hidden_size  # 4096
        self.discrim_head = DiscrimHead(hidden_size=hidden_size, num_classes=3)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Emit the language-model output and the discriminative logits in a single forward pass.

        Steps:
          1. call `super().forward()` with `output_hidden_states=True`
          2. read `outputs.hidden_states[REPR_LAYER_HS_IDX]`
          3. take the representation at the generation-prompt last token
          4. pass it through `discrim_head` to obtain `head_logits`
          5. attach `head_logits` and `head_hidden` to the returned object

        The read-out position must match between training and inference: during training it is the
        index just before the first non-masked label, i.e. immediately before the assistant
        content; at inference it is the final token after `add_generation_prompt`. Both refer to
        the same position.

        Returns:
            outputs with `head_logits` [B, 3] and `head_hidden` [B, 4096]
        """
        kwargs.pop("output_hidden_states", None)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            labels=labels,
            return_dict=True if return_dict is None else return_dict,
            output_hidden_states=True,
            **kwargs,
        )

        hs = None
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            if len(outputs.hidden_states) > REPR_LAYER_HS_IDX:
                hs = outputs.hidden_states[REPR_LAYER_HS_IDX]  # [B, S, 4096]

        import os

        if hs is None:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(
                    f"[dual-head][ERROR] hidden_states[{REPR_LAYER_HS_IDX}] 为 None，head_logits 将无法产出！",
                    flush=True,
                )

        head_logits = None
        head_hidden = None

        if hs is not None and input_ids is not None:
            B, S, _ = hs.shape
            repr_indices = []

            if labels is not None:
                for i in range(B):
                    label_row = labels[i]
                    non_ignore_positions = torch.where(label_row != -100)[0]
                    if len(non_ignore_positions) > 0:
                        assistant_first_idx = int(non_ignore_positions[0])
                        repr_idx = assistant_first_idx - 1
                        repr_idx = max(0, min(repr_idx, S - 1))
                        repr_indices.append(repr_idx)
                    else:
                        if attention_mask is not None:
                            last_idx = int(attention_mask[i].sum()) - 1
                            repr_indices.append(max(0, min(last_idx, S - 1)))
                        else:
                            repr_indices.append(S - 1)
            else:
                if attention_mask is not None:
                    for i in range(B):
                        last_idx = int(attention_mask[i].sum()) - 1
                        repr_indices.append(max(0, min(last_idx, S - 1)))
                else:
                    repr_indices = [S - 1] * B

            batch_indices = torch.arange(B, device=hs.device)
            repr_idx_tensor = torch.tensor(repr_indices, device=hs.device)
            head_hidden = hs[batch_indices, repr_idx_tensor, :]  # [B, 4096]

            head_logits = self.discrim_head(head_hidden)  # [B, 3]

            if not getattr(self, "_head_logits_logged", False):
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(
                        f"[dual-head] head_logits shape={head_logits.shape} 已产出（三分类）",
                        flush=True,
                    )
                self._head_logits_logged = True

        if head_logits is not None:
            outputs.head_logits = head_logits  # [B, 3]
            self.last_head_logits = head_logits
        if head_hidden is not None:
            outputs.head_hidden = head_hidden  # [B, 4096]
            self.last_head_hidden = head_hidden

        return outputs

    def save_dual_head(self, save_dir: str):
        """Save the `discrim_head` parameters.

        Under LoRA the head belongs to the base model, so `PeftModel.save_pretrained` does not
        cover it and this method has to be called separately.

        Args:
            save_dir: destination directory
        """
        import os
        import torch

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "discrim_head.pt")
        torch.save(self.discrim_head.state_dict(), save_path)
        print(f"[dual-head] discrim_head 已保存到 {save_path}", flush=True)

    def load_dual_head(self, load_dir: str):
        """Load the `discrim_head` parameters.

        Call after `PeftModel.from_pretrained`, for the reason given in `save_dual_head`.

        Args:
            load_dir: source directory
        """
        import os
        import torch

        load_path = os.path.join(load_dir, "discrim_head.pt")
        if not os.path.exists(load_path):
            print(f"[dual-head][WARN] {load_path} 不存在，跳过加载", flush=True)
            return

        state_dict = torch.load(load_path, map_location="cpu")
        self.discrim_head.load_state_dict(state_dict)
        print(f"[dual-head] discrim_head 已从 {load_path} 加载", flush=True)
