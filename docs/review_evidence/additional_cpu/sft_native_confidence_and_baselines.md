## SFT native 自身 confidence 可用性 与 相近 baseline protocol 核查

### 4a. SFT native dev/test 预测是否含自身 confidence（logprob/entropy 等）

| 文件 | 顶层字段 | pred 字段 | 检出 confidence 字段 |
|---|---|---|---|
| test comprehensive (`native/sft/.../predictions.jsonl`, 2646 行) | gt, pred, raw, relation, sample_id | box, decision, label, valid | 无 |
| dev (`matched_retention_dev.../dev/native_sft/...`, 2408 行) | gt, pred, raw, relation, sample_id | box, decision, label, valid | 无 |

eval 脚本为 greedy generation（`do_sample=False`），未保存 `output_scores`/`return_dict_in_generate`，因此**没有**逐 token logprob 或任何 per-sample 自身置信度。没有真正自身信号就不能离线 dev-fit 阈值曲线；不能用 ReGate head 概率冒充 SFT native 自身 confidence。

### 4b. 现有 groundingme/openref 输出是否即 CounterfactualREC / GREC / RECANTFormer？

不是。现有 `aaai_code/experiments/groundingme_external/` 与 `openref_external/` 是**外部 benchmark-derived probe**：

- GroundingME：HF 数据集 `lirang04/GroundingME`（CVPR 2026，arXiv 2512.17495），`prepare_groundingme.py` 将其映射到本文三态接口；运行的是本论文自己的 base/SFT/ReGate（run_config 中 `base_model=Qwen3-VL-8B-Instruct`，`adapter=checkpoint-444`）。
- OpenRef：`prepare_openref.py` 从 OpenRef val.json 派生 single-pair 诊断（500 pairs），同样是本文 base/SFT/ReGate 的输出。

这些输出不含任何 CounterfactualREC / GREC / RECANTFormer 模型结果，也不是这些方法的同 protocol 实现。因此 review 中“上述方法的同 protocol 比较尚未补充”的表述仍然成立。完整证据：`docs/review_evidence/additional_cpu/sft_native_confidence_and_baselines.json`（含 `prep_docstring` 与 `run_config`）。
