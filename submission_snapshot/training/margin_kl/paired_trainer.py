"""Trainer for the margin-and-anchor variant of the head objective.

Extends `TriHeadTrainer` and overrides `compute_loss` with three changes:
  A. a directional margin that replaces the head cross-entropy on wrong_object samples
  B. a KL anchor on the answer and no_target samples only, not on wrong_object
  C. a per-batch consistency check between `head_label` and the cached reference logits

It also overrides `_get_train_sampler` to return `PatternedDistributedSampler`.

ZREF_CACHE_PATH has no default and must be set explicitly. `TriHeadTrainer` is imported
lazily inside the function so that importing this module does not depend on `sys.path`
already containing the training package; callers are responsible for PYTHONPATH.
"""

import os

import torch
import torch.nn.functional as F


PAIR_MARGIN = float(os.environ.get("PAIR_MARGIN", "0.5"))
PAIR_BETA = float(os.environ.get("PAIR_BETA", "1.0"))
HEAD_LAMBDA = float(os.environ.get("PAIR_HEAD_LAMBDA", "0.4"))
W_WO = float(os.environ.get("PAIR_W_WO", "1.5"))
KL_TEMPERATURE = float(os.environ.get("PAIR_KL_T", "2.0"))


class PairedTriHeadTrainer:
    """Three-state head trainer with a directional margin and a KL anchor.

    Inherits from `TriHeadTrainer`, imported lazily, and overrides `compute_loss`:
      A. directional margin: for wrong_object samples a margin cross-entropy pushes z[1] above
         z[0] by at least the margin
      B. KL anchor: samples with label 0 or 2 add KL(z_ref || z_cur), wrong_object samples do not
      C. reference lookup: every batch checks that `head_label` agrees with the cached entry

    Callers must ensure PYTHONPATH covers the training package, otherwise `__init__` raises.
    ZREF_CACHE_PATH is required and has no default.
    """

    def __new__(cls, *args, **kwargs):
        """Import the parent class lazily and build the real trainer instance."""
        from qwenvl.train.train_qwen import TriHeadTrainer

        class _RealPairedTriHeadTrainer(TriHeadTrainer):
            """The implementation behind `PairedTriHeadTrainer`."""

            def __init__(
                self, *args, head_lambda: float = HEAD_LAMBDA, head_warmup_steps: int = 0, **kwargs
            ):
                super().__init__(
                    *args, head_lambda=head_lambda, head_warmup_steps=head_warmup_steps, **kwargs
                )

                zref_path = os.environ.get("ZREF_CACHE_PATH", "")
                if not zref_path:
                    raise RuntimeError("ZREF_CACHE_PATH 环境变量必须指定（指向 zref_cache.pt）")
                if not os.path.exists(zref_path):
                    raise FileNotFoundError(
                        f"zref_cache not found at {zref_path}. "
                        "Run dump_zref.py first to generate it."
                    )
                cache = torch.load(zref_path, map_location="cpu", weights_only=False)
                self.zref: torch.Tensor = cache["zref"]  # [N, 3]
                self.zref_head_labels: list = cache["head_label"]  # list[int], len=N
                self.zref_source_image_ids: list = cache.get("source_image_id", [])
                self._zref_buffer = self.zref.clone()
                print(
                    f"[PairedTriHeadTrainer] 加载 zref_cache: {zref_path}, "
                    f"shape={self.zref.shape}, N={len(self.zref_head_labels)}"
                )

            def _get_train_sampler(self, train_dataset=None):
                from paired_sampler import PatternedDistributedSampler

                if train_dataset is None:
                    train_dataset = self.train_dataset

                ds = train_dataset
                if hasattr(ds, "dataset"):
                    ds = ds.dataset
                if hasattr(ds, "list_data_dict"):
                    head_labels = [r.get("head_label", -1) for r in ds.list_data_dict]
                else:
                    head_labels = []
                    for i in range(len(ds)):
                        sample = ds[i]
                        hl = sample.get("head_label", -1) if isinstance(sample, dict) else -1
                        head_labels.append(hl)

                return PatternedDistributedSampler(
                    head_labels=head_labels,
                    num_replicas=self.args.world_size if hasattr(self.args, "world_size") else 1,
                    rank=self.args.process_index if hasattr(self.args, "process_index") else 0,
                    seed=getattr(self.args, "seed", 42),
                    drop_last=True,
                )

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                head_labels = inputs.pop("head_labels", None)
                reason_labels = inputs.pop("reason_labels", None)
                sample_indices = inputs.pop("sample_indices", None)

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

                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, V),
                        shift_labels.view(-1),
                        reduction="none",
                    ).view(B, S_minus_1)

                    mask = (shift_labels != -100).float()
                    per_token_loss = per_token_loss * mask

                    if reason_labels is not None:
                        sample_weight = torch.where(
                            reason_labels == 1,
                            torch.tensor(
                                NO_TARGET_W,
                                device=per_token_loss.device,
                                dtype=per_token_loss.dtype,
                            ),
                            torch.where(
                                reason_labels == 0,
                                torch.tensor(
                                    WRONG_OBJECT_W,
                                    device=per_token_loss.device,
                                    dtype=per_token_loss.dtype,
                                ),
                                torch.tensor(
                                    ANSWER_W,
                                    device=per_token_loss.device,
                                    dtype=per_token_loss.dtype,
                                ),
                            ),
                        )
                    elif head_labels is not None:
                        sample_weight = torch.where(
                            head_labels == 2,
                            torch.tensor(
                                NO_TARGET_W,
                                device=per_token_loss.device,
                                dtype=per_token_loss.dtype,
                            ),
                            torch.where(
                                head_labels == 1,
                                torch.tensor(
                                    WRONG_OBJECT_W,
                                    device=per_token_loss.device,
                                    dtype=per_token_loss.dtype,
                                ),
                                torch.tensor(
                                    ANSWER_W,
                                    device=per_token_loss.device,
                                    dtype=per_token_loss.dtype,
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

                if sample_indices is not None and head_labels is not None:
                    self._assert_zref_alignment(sample_indices, head_labels)

                head_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
                kl_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
                margin = PAIR_MARGIN
                w_wo = W_WO

                if head_logits is not None and head_labels is not None:
                    valid_mask = head_labels >= 0
                    if valid_mask.any():
                        valid_logits = head_logits[valid_mask]  # [n_valid, 3]
                        valid_labels = head_labels[valid_mask].long()  # [n_valid]

                        w_wo = W_WO

                        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction="none")

                        is_wo = valid_labels == 1

                        if is_wo.any():
                            z_wo = valid_logits[is_wo]  # [n_wo, 3]
                            z_margin = z_wo.clone()
                            z_margin[:, 0] = z_margin[:, 0] + margin
                            margin_ce = w_wo * (-z_wo[:, 1] + torch.logsumexp(z_margin, dim=1))
                            ce_loss = torch.where(is_wo, margin_ce, ce_loss)

                        head_loss = ce_loss.mean()

                        if sample_indices is not None:
                            batch_indices = sample_indices[valid_mask]  # [n_valid]
                            z_ref = self._zref_buffer.to(valid_logits.device)[batch_indices]

                            kl_mask = valid_labels != 1
                            if kl_mask.any():
                                z_cur_kl = valid_logits[kl_mask].float()
                                z_ref_kl = z_ref[kl_mask].float()

                                T = KL_TEMPERATURE
                                p_ref = F.softmax(z_ref_kl / T, dim=1)
                                p_cur = F.softmax(z_cur_kl / T, dim=1)

                                kl_per_sample = (
                                    p_ref * (torch.log(p_ref + 1e-10) - torch.log(p_cur + 1e-10))
                                ).sum(dim=1)
                                kl_loss = (T * T) * kl_per_sample.mean()

                        head_loss = head_loss + PAIR_BETA * kl_loss

                lam = self.head_lambda
                if self.state.global_step < self.head_warmup_steps:
                    lam = self.head_lambda * (
                        self.state.global_step / max(self.head_warmup_steps, 1)
                    )

                loss = main_loss + lam * head_loss

                if self.state.global_step % self.args.logging_steps == 0:
                    log_dict = {
                        "head_loss": float(head_loss.detach()),
                        "main_loss": float(main_loss.detach()),
                        "kl_loss": float(kl_loss.detach()),
                        "head_lambda": lam,
                        "pair_margin": margin,
                        "pair_beta": PAIR_BETA,
                    }
                    self.log(log_dict)

                return (loss, outputs) if return_outputs else loss

            def _assert_zref_alignment(self, sample_indices, head_labels):
                n_check = min(len(sample_indices), 8)

                for i in range(n_check):
                    idx = int(sample_indices[i].item())
                    hl_batch = int(head_labels[i].item())
                    hl_cache = self.zref_head_labels[idx]

                    if hl_batch != hl_cache:
                        raise AssertionError(
                            f"z_ref 对齐灾难！样本 index={idx} 的 head_label 不一致："
                            f"batch={hl_batch} (={['answer', 'WO', 'nt'][hl_batch] if 0 <= hl_batch <= 2 else '?'}), "
                            f"zref_cache={hl_cache} (={['answer', 'WO', 'nt'][hl_cache] if 0 <= hl_cache <= 2 else '?'}). "
                            f"请检查 sft_train_with_source.json 与 zref_cache 是否对应同一份数据。"
                        )

        return _RealPairedTriHeadTrainer(*args, **kwargs)
