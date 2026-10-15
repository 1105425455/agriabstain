# Evidence Judgement — 表征公平比较与证据归因分析

本目录实现 **Experiment A（表征公平比较）** 与 **Experiment B（证据归因分析）**，
用于评估 frozen backbone 下不同 visual representation 对 head-level 分类性能的影响，
以及 calibrated head 在 L2/L3 证据探针上的 counterfactual、dose-response 和 consistency 分析。

> **方法学边界（重要）**
>
> - **主结果仅基于 frozen base backbone**（`--base` 路径指向的 Qwen3-VL-8B-Instruct），
>   不包含任何 task-specific LoRA adapter 的影响。
> - **Adapter 结果标记为 diagnostic**（`--adapter` 参数），因为 checkpoint-444 的
>   last-token head 在 SFT 训练过程中直接受 head loss 优化，其特征空间已被塑形，
>   不能视为公平的 representation 比较。
> - 本实验**不能 claim 解决 hallucination、实现 direct grounding 或验证 causal mechanism**。
>   它仅提供一种 probing methodology：在固定 backbone 上，通过 lightweight probe 比较
>   不同视觉聚合方式的信息保留能力，以及通过 head 输出分析模型对输入扰动的敏感性。
> - L2/L3 分析基于 calibrated head 的原始概率，不涉及 text generation 或 GT fallback。

---

## 目录

- [Experiment A：表征公平比较](#experiment-a表征公平比较)
  - [四种 representation 精确定义](#四种-representation-精确定义)
  - [数据与命令](#数据与命令)
  - [输出 schema](#输出-schema)
- [Experiment B：证据归因分析](#experiment-b证据归因分析)
  - [L2 证据探针（keep vs cropout）](#l2-证据探针keep-vs-cropout)
  - [L3 剂量探针（dose-response）](#l3-剂量探针dose-response)
  - [Input Equivariance Consistency](#input-equivariance-consistency)
  - [数据与命令](#数据与命令-1)
  - [输出 schema](#输出-schema-1)
- [运行示例](#运行示例)
- [依赖](#依赖)
- [资源需求](#资源需求)
- [Claim 口径](#claim-口径)
- [失败策略](#失败策略)

---

## Experiment A：表征公平比较

### 四种 representation 精确定义

所有四种方法从 **同一单次 forward** 提取 `hidden_states[27]`（layer 27，4096 维），
确保比较基于相同输入、相同计算图、相同 MLP 层。每种方法独立训练 probe（独立 mu/sd 统计）。

| 编号 | 方法 | 公式 | 说明 |
|------|------|------|------|
| 1 | `last_token` | `h[last_token_idx]` | generation-prompt 的 last token hidden state。attention mask 有效时取 `sum(attention_mask) - 1`，否则取 `S - 1`。 |
| 2 | `visual_mean` | `mean(visual_hs, dim=0)` | visual span 内所有 token 的 hidden 逐维均值。 |
| 3 | `visual_max` | `max(visual_hs, dim=0).values` | visual span 内所有 token 的 hidden 逐维最大值。 |
| 4 | `visual_query_attn` | `softmax((visual_hs @ h_last) / sqrt(d)) @ visual_hs` | visual hidden 与 last-token hidden 的点积除以 sqrt(d) 后 softmax 加权。d = 4096，scores 形状 `[n_visual]`。 |

**视觉 token 校验**：visual span 定义为 `input_ids == VISUAL_TOKEN_ID`（151655）的位置。
其数量通过 `compute_expected_visual_tokens` 严格校验：

```python
def compute_expected_visual_tokens(image_grid_thw, merge_size=2):
    total = 0
    for t, h_grid, w_grid in image_grid_thw:
        total += t * (h_grid // merge_size) * (w_grid // merge_size)
    return total
```

公式使用 **floor division**（`//`），不使用 `ceil`。
`image_grid_thw` 形状为 `[N, 3]`，列顺序为 `[temporal, height_grid, width_grid]`。
对多 grid 场景（视频或多图），各 grid 的 token 数累加。

**公平性保证**：
- 同一次 forward 提取所有四种表示
- 同一 MLP 架构 4096→512→256→3 (GELU+Dropout0.1)
- 同一 Adam 超参 (lr=1e-3, wd=1e-3)
- 每方法**独立计算**逐维 mu/sd（从 train split 计算，clamp sd ≥ 1e-6），绝不复用 last-token 统计
- 按 dev loss early stop（patience=5），test 仅最终评估一次
- 5 seeds (13, 42, 123, 2024, 7) paired bootstrap（visual_method vs last_token 的 MCC delta）
- 校验三个 split 的 manifest 口径一致（base/adapter/is_adapter_diagnostic）

### 数据与命令

数据文件（均位于 `benchmark/annotations/sft/`）：

| 文件 | 说明 |
|------|------|
| `sft_train.json` | 7,072 条，三分类 head_label 0/1/2 |
| `sft_dev.json` | 2,408 条 |
| `sft_test.jsonl` | 2,646 条，JSONL 格式 |

> **derived images 说明**：SFT 数据中的 `image` 字段可能引用 `data/derived/` 路径下的
> 派生图片。这些图片**不随发布包分发**，必须先按顶层 README 第 8 节重建，
> 或通过 `--derived-root`（CLI）或 `DERIVED_ROOT`（launcher env）指向外部重建目录。
> 默认是 `benchmark/data/derived`（发布包中为空）。
> ```bash
> # 重建命令（从 aaai_code/ 根运行）
> python scripts/build_benchmark.py --dataset plantdoc --config configs/plantdoc_full.yaml --output /tmp/plantdoc_rebuild.jsonl
> python scripts/build_benchmark.py --dataset coco --config configs/coco_full.yaml --output /tmp/coco_rebuild.jsonl
> # 使用外部 derived 目录
> export DERIVED_ROOT=/path/to/transformed_images
> bash experiments/evidence_judge/run_representation.sh all
> ```
>
> **路径映射规则**：`--derived-root` 仅影响 `data/derived/` 前缀的相对路径（映射为 `{derived_root}/{suffix}`）；
> `data/source/` 等仍从 `benchmark/` 根解析；绝对路径原样保留。
>
> **batch_size 说明**：CLI 默认 `--batch-size 1`（单样本兼容），launcher 默认 `BATCH_SIZE=3`（2 卡 H100 推荐值）。
> 如果显存不足（如 OOM 或 CUDA OOM），请减小 `BATCH_SIZE`（如设为 1 或 2），
> 可通过 `export BATCH_SIZE=1` 或命令行 `--batch-size 1` 设置。

所有特征/结果写入 `$OUTPUT_ROOT` 外部目录，不写入代码目录。

#### 1. 提取表征

```bash
# 单卡调试（batch_size=1，CLI 默认值）
python experiments/evidence_judge/extract_representations.py \
    --base $BASE --data benchmark/annotations/sft/sft_train.json \
    --output-dir $OUTPUT_ROOT/repr/train --split-name train --limit 10

# 三卡 torchrun 三个 split，batch_size=3（launcher 默认值）
torchrun --nproc_per_node=4 experiments/evidence_judge/extract_representations.py \
    --base $BASE --data benchmark/annotations/sft/sft_train.json \
    --output-dir $OUTPUT_ROOT/repr/train --split-name train

# 也可通过 env 控制 batch_size（launcher 默认 BATCH_SIZE=3）
export BATCH_SIZE=3
bash run_representation.sh extract

torchrun --nproc_per_node=4 experiments/evidence_judge/extract_representations.py \
    --base $BASE --data benchmark/annotations/sft/sft_dev.json \
    --output-dir $OUTPUT_ROOT/repr/dev --split-name dev

torchrun --nproc_per_node=4 experiments/evidence_judge/extract_representations.py \
    --base $BASE --data benchmark/annotations/sft/sft_test.jsonl \
    --output-dir $OUTPUT_ROOT/repr/test --split-name test
```

#### 2. 训练 probe 并比较

```bash
python experiments/evidence_judge/train_probes.py \
    --train $OUTPUT_ROOT/repr/train/train.npz \
    --dev $OUTPUT_ROOT/repr/dev/dev.npz \
    --test $OUTPUT_ROOT/repr/test/test.npz \
    --output $OUTPUT_ROOT/repr/probe_results.json
```

### 输出 schema

**提取表征输出**（`{split_name}.npz`）：

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `last_token` | `[N, 4096]` | float32 | 四种表征之一 |
| `visual_mean` | `[N, 4096]` | float32 | 四种表征之一 |
| `visual_max` | `[N, 4096]` | float32 | 四种表征之一 |
| `visual_query_attn` | `[N, 4096]` | float32 | 四种表征之一 |
| `head_label` | `[N]` | int64 | 0=answer, 1=wrong_object, 2=no_target |
| `sample_id` | `[N]` | object | 样本标识 |
| `image` | `[N]` | object | 图片路径 |
| `original_index` | `[N]` | int64 | 原始数据行索引 |
| `visual_token_count` | `[N]` | int64 | 每样本 visual token 数 |
| `manifest_json` | str | — | JSON 序列化的 manifest |

**probe 训练输出 JSON**（`probe_results.json`）：

| 顶层字段 | 说明 |
|----------|------|
| `input_hash` | 输入文件 hash |
| `input_manifest` | 三个 split 的 manifest |
| `parameters` | 训练超参 |
| `param_counts` | 每方法参数量 |
| `claim_scope` | `"frozen_base_representation_probe"` 或 `"adapter_diagnostic"` |
| `results` | 每方法的结果 dict，含 `seeds` 数组和 `aggregate` |
| `results.{method}.seeds[].test_preds` | test 预测类别列表（用于可核查） |
| `results.{method}.bootstrap_vs_last_token_summary` | 跨 seed 的 bootstrap 汇总 |

---

## Experiment B：证据归因分析

使用 calibrated head（checkpoint-444 的 SFT adapter + 校准后的 `discrim_head`）在 L2/L3 数据上
提取 head-only 概率（p_answer, p_wo, p_nt）。**不进行 text generation，不依赖 GT fallback。**

### L2 证据探针（keep vs cropout）

**数据**：`{dev,test}_L2_evidence.jsonl`，每对样本共享同一 group key（sample_id 中 `-keep-` 或 `-cropout-` 前后缀）。Group key = `{image_id}__{source_sample_id}`，其中 source_sample_id 完整保留（例如 `plantdoc-train-<hash>-0` 或 `coco-train-<coco_id>-<ann_id>`），确保同一图片上不同 annotation ID 的对象**不合并**为同一组。

**Pairing key**：L2 分析使用 `pairing_key = stable_group_key + normalized_target_label`（`source_instance+target_label`），确保同一 source instance 下不同 target_label 的 cropout cross-product 记录不会错误合并。`target_label` 经 `strip().casefold()` 规范化。缺少 target_label 的记录计为 parse failure。

**分析**：
- 对每个 pairing key，`variant_id == "original"` 的记录中，keep 的 p_answer 减去 cropout 的 p_answer
- `delta_p_answer = p_answer_keep - p_answer_cropout`
- 正确方向定义为 `delta_p_answer > 0`（keep 应高于 cropout）
- 聚合统计：mean delta、median、correct direction ratio、bootstrap 95% CI
- 去重：同一 (sample_id, relation, image, target_label, achieved_visible_ratio, original_gt_decision) 且概率一致的记录自动折叠，报告 `deduplicated_records`；概率冲突或同 relation 下存在不同 semantic identity 则整组标记为 `conflicting_duplicate_groups` 并计入 `invalid_groups`

### L3 剂量探针（dose-response）

**数据**：`{dev,test}_L3_dose.jsonl`，每个样本有 `achieved_visible_ratio`（0.0, 0.25, 0.5, 0.75, 1.0）。

**Group key**：`L3-{image_id}__{source_sample_id}`，其中 source_sample_id 完整保留（含 COCO annotation ID），确保同一图片上不同对象的多档剂量不合并。

**分析**：
- 按 `stable_group_key` 分组（sample_id 中 `-<ratio>-plantdoc-...` 前缀）
- 每组内按 `achieved_visible_ratio` 聚合 p_answer 均值
- **Spearman rho** 和 **Kendall tau**（组内单调性）
- **相邻非递减违例率**（`adjacent_violation_rate`：相邻 ratio 间 p_answer 下降的比例）
- **Endpoint delta**：**仅当组内同时存在精确 ratio 0.0 和 1.0 时**计算，不伪造 endpoint
- 跨组 bootstrap CI、shuffled-ratio baseline

#### T7 训练与评估口径

T7 在 SFT 与 L3 evaluation 中承担不同职责。SFT 的离散三状态监督将 achieved
visibility `>=0.5` 的 hard-label records 作为 supported，将 `0.0` 作为
target unavailable；`0.25` 仅用于 graded analysis。L3 hard fixed-point metrics 采用更保守的
endpoint protocol，只使用语义明确的 `1.0` 与 `0.0`。`0.75/0.5/0.25` 保留为连续响应分析，
用于计算 `p_answer` 随 achieved visibility 的 Spearman rho、Kendall tau 和 adjacent
violation rate。该协议将训练监督阈值与测试 oracle 分开，避免通过复现训练阈值形成循环评估。

`analyze_evidence.py` 只负责全剂量连续响应统计。Hard endpoint subset 由
`filter_l3_endpoints.py` 显式生成，再交给标准 fixed-GT evaluator：

```bash
python -m experiments.evidence_judge.filter_l3_endpoints \
  --input benchmark/annotations/eval/test_L3_dose.jsonl \
  --output /path/to/test_L3_endpoints.jsonl \
  --manifest /path/to/test_L3_endpoints.manifest.json
```

### Input Equivariance Consistency

**定义**：consistency 是指对于 T1 query paraphrase 与 brightness/contrast 输入变换，
模型输出的 p_answer 应保持近似不变（等变性一致性）。

> **不是**相同 greedy 输入重复运行的 sampling stability，也不是 inference-time sampling 的稳定性。
> 而是：输入经过语义等价变换（query 措辞变化、光照/对比度微调）后，head 输出概率的一致性。

**variant 类型**：
- **Query variants**（`variant_set=query` 或 `all`）：使用 `GROUNDING_TEMPLATES` 中的其他模板重写 query。
  共 3 个模板，原始 query 匹配其一，因此最多 2 个额外 query variant。
  Template: `"Locate the {label} in this image."`, `"Find the {label} and return its bounding box."`, `"Where is the {label} in the image?"`
- **Image variants**（`variant_set=image` 或 `all`）：brightness 0.9/1.1、contrast 0.9/1.1，共 4 个。
  几何变换（flip/crop）不属于输入等变一致性范畴。

**指标**：p_answer std、range、mean abs delta from original、argmax decision flip rate、
all_equivalents_std（含 original）。按 variant_type 和 relation 分别聚合。

### 数据与命令

```bash
# 1. 提取 L2 head probabilities（可指定 --batch-size，CLI 默认 1，launcher 默认 3）
torchrun --nproc_per_node=4 experiments/evidence_judge/extract_head_probabilities.py \
    --base $BASE --adapter $ADAPTER --calibrated-head $CALIBRATED_HEAD \
    --data benchmark/annotations/eval/dev_L2_evidence.jsonl \
    --output $OUTPUT_ROOT/evidence/L2/dev --variant-set none --batch-size 3

# 2. 提取 L3 head probabilities
torchrun --nproc_per_node=4 experiments/evidence_judge/extract_head_probabilities.py \
    --base $BASE --adapter $ADAPTER --calibrated-head $CALIBRATED_HEAD \
    --data benchmark/annotations/eval/test_L3_dose.jsonl \
    --output $OUTPUT_ROOT/evidence/L3/test --variant-set none --batch-size 3

# 3. 提取 L2/L3 + variants（query + image 全量）
torchrun --nproc_per_node=4 experiments/evidence_judge/extract_head_probabilities.py \
    --base $BASE --adapter $ADAPTER --calibrated-head $CALIBRATED_HEAD \
    --data benchmark/annotations/eval/test_L2_evidence.jsonl \
    --output $OUTPUT_ROOT/evidence/L2/test_variants --variant-set all --batch-size 3

# 4. 分析 L2/L3/consistency
python experiments/evidence_judge/analyze_evidence.py \
    --input $OUTPUT_ROOT/evidence/L2/dev/head_probabilities.jsonl \
    --output $OUTPUT_ROOT/evidence/L2/dev/analysis.json

python experiments/evidence_judge/analyze_evidence.py \
    --input $OUTPUT_ROOT/evidence/L3/test/head_probabilities.jsonl \
    --output $OUTPUT_ROOT/evidence/L3/test/analysis.json

python experiments/evidence_judge/analyze_evidence.py \
    --input $OUTPUT_ROOT/evidence/L2/test_variants/head_probabilities.jsonl \
    --output $OUTPUT_ROOT/evidence/L2/test_variants/analysis.json
```

### 输出 schema

**提取概率输出**（`head_probabilities.jsonl`，每行一条 record）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sample_id` | str | 样本标识 |
| `relation` | str | L2_keep / L2_cropout / L3_ladder 等 |
| `image` | str | 图片路径 |
| `target_label` | str | 解析出的目标标签 |
| `achieved_visible_ratio` | float | L3 的可见性比例 |
| `visible_ratio_bin` | str | 离散化 bin |
| `original_gt_decision` | str | 原始 GT decision |
| `variant_type` | str | `original` / `query` / `image` |
| `variant_id` | str | 变体标识 |
| `p_answer` | float | answer 概率 |
| `p_wo` | float | wrong_object 概率 |
| `p_nt` | float | no_target 概率 |
| `predicted_class` | int | argmax 预测类别 |
| `decision` | str | `answer` / `abstain` |
| `original_index` | int | 原始数据行索引 |

**manifest.json**（合并输出目录）：

| 字段 | 说明 |
|------|------|
| `fingerprint` | 运行指纹 |
| `n_rows` | 原始行数 |
| `n_variants` | 总 variant 记录数 |
| `error_count` | 含 null p_answer 的记录数 |
| `model_paths` | base/adapter/calibrated_head 路径 |
| `variant_set` | 使用的 variant_set |

**分析报告 JSON**：顶层包含 `L2`、`L3`、`input_equivariance_consistency` 三个 key，
各包含对应的聚合统计和 bootstrap CI。

L2 分析输出新增字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `deduplicated_records` | int | 语义去重折叠的记录数 |
| `groups_with_duplicates` | int | 存在 byte-identical 重复（已折叠）的组数 |
| `conflicting_duplicate_groups` | int | 因概率冲突或不同 semantic identity 而无效的组数 |
| `invalid_groups` | int | 总无效组数（含 incomplete 和 conflict） |
| `n_total_groups` | int | 原始 normalized group 总数 |
| `pairing_key` | str | 固定为 `"source_instance+target_label"`，标识 grouping 策略 |

---

## 运行示例

所有命令从任意目录运行（launcher 内部通过 `SCRIPT_DIR/../..` 推导 `AAAI_CODE` 绝对路径，不依赖调用 cwd）。**默认阶段为 `check`**，仅检查参数而不启动 GPU 流程；显式指定 `all` 才运行完整流程。Launcher 会验证 `$OUTPUT_ROOT` 在 `AAAI_CODE` 外部、所有 model/data 路径存在。

### 完整三 split 表征提取 + probe 训练

```bash
# ── 安全：默认仅 check ──
export BASE=/path/to/Qwen3-VL-8B-Instruct
export OUTPUT_ROOT=/scratch/evidence_judge
# 执行后将仅检查参数，不启动 GPU
bash experiments/evidence_judge/run_representation.sh

# ── 显式 all 才实际运行 ──
export BASE=/path/to/Qwen3-VL-8B-Instruct
export OUTPUT_ROOT=/scratch/evidence_judge
# 可选：指向外部 derived images
export DERIVED_ROOT=/scratch/transformed_images
# 可选：控制 batch_size（CLI 默认 1，launcher 默认 3，2 卡 H100 推荐值）
export BATCH_SIZE=3
bash experiments/evidence_judge/run_representation.sh all
```

### 完整证据分析

```bash
export BASE=/path/to/Qwen3-VL-8B-Instruct
export ADAPTER=/path/to/checkpoint-444
export CALIBRATED_HEAD=/path/to/checkpoint-444/discrim_head_calibrated.pt
export OUTPUT_ROOT=/scratch/evidence_judge

# L2 — dev
torchrun --nproc_per_node=4 experiments/evidence_judge/extract_head_probabilities.py \
    --base $BASE --adapter $ADAPTER --calibrated-head $CALIBRATED_HEAD \
    --data benchmark/annotations/eval/dev_L2_evidence.jsonl \
    --output $OUTPUT_ROOT/evidence/L2/dev --variant-set none --batch-size 3

# L3 — test
torchrun --nproc_per_node=4 experiments/evidence_judge/extract_head_probabilities.py \
    --base $BASE --adapter $ADAPTER --calibrated-head $CALIBRATED_HEAD \
    --data benchmark/annotations/eval/test_L3_dose.jsonl \
    --output $OUTPUT_ROOT/evidence/L3/test --variant-set none --batch-size 3

# Consistency — L2 test with all variants
torchrun --nproc_per_node=4 experiments/evidence_judge/extract_head_probabilities.py \
    --base $BASE --adapter $ADAPTER --calibrated-head $CALIBRATED_HEAD \
    --data benchmark/annotations/eval/test_L2_evidence.jsonl \
    --output $OUTPUT_ROOT/evidence/L2/test_variants --variant-set all --batch-size 3

# 分析
python experiments/evidence_judge/analyze_evidence.py \
    --input $OUTPUT_ROOT/evidence/L2/dev/head_probabilities.jsonl \
    --output $OUTPUT_ROOT/evidence/L2/dev/analysis.json

python experiments/evidence_judge/analyze_evidence.py \
    --input $OUTPUT_ROOT/evidence/L3/test/head_probabilities.jsonl \
    --output $OUTPUT_ROOT/evidence/L3/test/analysis.json

python experiments/evidence_judge/analyze_evidence.py \
    --input $OUTPUT_ROOT/evidence/L2/test_variants/head_probabilities.jsonl \
    --output $OUTPUT_ROOT/evidence/L2/test_variants/analysis.json
```

---

## 依赖

- Python 3.10+
- `torch`（>=2.0）
- `transformers`（带 Qwen3-VL 支持）
- `peft`, `safetensors`, `qwen-vl-utils`
- `numpy`, `scikit-learn`, `scipy`
- `Pillow`
- `Pillow` 的 `ImageEnhance`（image variant 生成）

---

## 资源需求

- **表征提取**：每个 split 约需 8–16GB VRAM（取决于 batch size 和图片分辨率），
  2 卡 H100 batch_size=3 下 train 约 15–30 分钟，dev/test 约 5–10 分钟。
  如遇显存不足，可调小 `BATCH_SIZE`（最低 1）。
- **Probe 训练**：单卡 CPU 或 GPU，5 seeds × 4 methods 约 5–15 分钟。
- **证据概率提取**：L2/L3 约 2–4GB VRAM（head-only forward），2 卡 batch_size=3 约 10–20 分钟每 split。
- **Consistency variants**：variant_set=all 时每样本最多 7 个 forward，约 3–5× 耗时。
- **磁盘**：每个 split 的 `.npz` 约 200–500MB（4 种表征 × 4096 × N）。

---

## Claim 口径

| 实验 | 可 claim | 不可 claim |
|------|----------|------------|
| Experiment A (frozen base) | 在 frozen backbone 下，visual_query_attn 与 last_token 的 probe 分类性能差异 | 模型"使用"了视觉信息；解决 hallucination；优于其他方法 |
| Experiment A (adapter) | diagnostic only：adapter 训练后特征空间变化 | 公平比较；adapter 改进了 grounding |
| Experiment B L2 | 模型对 keep/cropout 输入的概率变化方向 | 模型具有因果理解；模型正确 grounded |
| Experiment B L3 | p_answer 随可见性比例变化的单调性趋势 | 模型具有剂量感知能力；达到 human-level |
| Consistency | 模型对 query paraphrase 和 photometric 变换的敏感性 | 模型具有鲁棒性；sampling stability |

---

## 失败策略

| 失败模式 | 行为 | 恢复 |
|----------|------|------|
| 分片超时（`merge_shards` timeout） | 返回 None，不输出不完整文件 | 检查 world size 和 `.done` 文件，重跑 |
| visual token 数量不匹配 | `RuntimeError`，该样本失败 | 检查 `image_grid_thw` 和 `merge_size`，确认模型 processor 兼容 |
| Manifest 不一致 | 返回 exit code 1 | 确保三个 split 使用相同 `--base` 和 `--adapter` |
| 分片完整性错误（缺失/重复索引） | 返回 None | 检查 multiprocessing 是否正常，`--limit` 是否一致 |
| 样本处理异常 | 默认 fail-fast（`extract_representations.py` 不允许失败）；`extract_head_probabilities.py` 可通过 `--allow-errors` 继续 | 检查异常栈 |
| 缺少 L3 endpoint | 不计算 endpoint delta，不伪造 | 仅在精确 0.0 和 1.0 都存在时报告 |
| 模型加载失败 | 直接退出 | 检查 `--base`/`--adapter`/`--calibrated-head` 路径和格式 |
| Fingerprint 变化 | 自动清理旧 shard 文件 | 重跑时自动重新提取 |
