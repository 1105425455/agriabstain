# Base Native → ReGate — Native base/SFT evaluation and calibrated ReGate curve analysis

本实验比较三个方法（Base native、SFT native、ReGate with calibrated head）在三数据集
（comprehensive 2646、refcocog 362、plantwild 408）上的 selective prediction 性能。
所有 9 份 predictions 通过 `run_native_eval.sh` + `run_regate_candidates.sh` 生成，
再由 `analyze_native_regate.py` 统一评估。

---

## 目录

- [实验目的](#实验目的)
- [方法学边界](#方法学边界)
- [固定数据文件与行数](#固定数据文件与行数)
- [精确模式参数](#精确模式参数)
- [目录文件职责](#目录文件职责)
- [流程](#流程)
- [命令](#命令)
- [输出树](#输出树)
- [Claim 口径](#claim-口径)
- [失败恢复](#失败恢复)

---

## 实验目的

1. **Base native → SFT checkpoint-444 native 隔离**  
   基座模型（Qwen3-VL-8B-Instruct，无 adapter）与 SFT checkpoint-444（LoRA adapter，无 calibrated head）在不同数据集上进行纯文本生成推理，比较从 base 到 SFT 的 native decision 变化。两者不使用任何 dual-head / calibrated head。

2. **同 backbone SFT native → calibrated ReGate curve**  
   固定同一 checkpoint-444 backbone，用 calibrated head 替代文本生成。通过 `--dual-head --faithful-dual-output --dump-all-answer-candidates` 获取三分类 head 概率（p_answer/p_wo/p_nt）和最佳 answer candidate。支持 coverage 门控（threshold sweep）和三分类 argmax 两种 policy。

3. **Shared answer_candidate gate comparison**  
   ReGate 的 answer_candidate 与 SFT native 的生成文本共享相同 backbone 输出。`SFT_native_gate_shared_candidate` 方法使用 SFT native 的 answer/abstain 门控（gate），但正确性基于 ReGate 的 `answer_candidate` 字段，从而隔离生成（generation）混淆。

---

## 方法学边界（重要）

- **Base native 不含任何 adapter**，用于诊断 base backbone 的固有能力上限/下限。
- **SFT native 使用 checkpoint-444 LoRA adapter，不含 calibrated head**，纯三分类文本生成（`--system-override decision`）。
- **ReGate 使用同一 checkpoint-444 + calibrated head**，head 输出经 `score_answerable` / `score_unanswerable` 与 GT 严格比较。
- 三个方法共享**相同**的数据文件（same GT）、相同的 `resolve_inputs.py` 输入，仅模型下游不同。
- **关键隔离**：SFT native 和 ReGate 共享 backbone 权重，但 native 用生成文本 decision，ReGate 用 head 概率 + 固定 answer_candidate。`SFT_native_gate_shared_candidate` 进一步分离门控（生成）与候选正确性（head）。
- **本实验不能 claim「端到端公平比较」**，因为 ReGate 的有监督 head 数据来自 SFT 的训练集。本实验只提供在同一 backbone 上三种输出方式的对比分析。
- **所有输出写入外部目录，不污染代码目录。**

---

## 固定数据文件与行数

| 数据集 | 文件 | 预期行数 |
|--------|------|---------|
| comprehensive | `benchmark/annotations/sft/sft_test.jsonl` | 2,646 |
| refcocog | `benchmark/annotations/eval/ood_refcocog.jsonl` | 362 |
| plantwild | `benchmark/annotations/eval/plantwild_real_closed12_paired_408.jsonl` | 408 |

所有 9 份 predictions 必须与此三个文件一一对应。

---

## 精确模式参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--iou-threshold` | 0.5 | IoU 阈值（comprehensive 与 RefCOCOg 使用 bbox grounding） |
| p_answer threshold sweep | — | 从 1.0 到 0.0 每个唯一概率值为阈值，生成 ReGate curve |
| three_class_argmax | — | `p_answer >= p_wo and p_answer >= p_nt` → answer，否则 abstain |

**禁用表**（以下参数在本实验中不启用）：

| 参数/模式 | 原因 |
|-----------|------|
| `--no-plots` | 可传递，默认生成 PDF/PNG 图 |
| PlantWild box grounding | 该数据集 GT box = null，grounding_scope = `label_only` |

**统一指标定义**（见 `NativeRegateMetric`）：

| 指标 | 定义 |
|------|------|
| total | 总样本数 |
| supported | GT decision = answer 的样本数 |
| unsupported | GT decision = abstain 的样本数 |
| answered | 模型判为 answer（含正确与错误） |
| strict_3_correct | 满足 strict-3 约束（supported + 语义 label 正确 或 unsupported + correct reason）；语义 label 正确指 normalize_label 匹配，不检查 bbox/IoU |
| selective_risk_errors | answered 样本中 wrong（label / box / reason 不符合 GT） |
| grounding_correct | answered + supported 中 label 正确且 IoU >= threshold（或 label-only） |
| grounding_scope | "bbox"（comprehensive / refcocog）或 "label_only"（plantwild） |

---

## 目录文件职责

| 文件 | 职责 | 类型 |
|------|------|------|
| `__init__.py` | 包标记 | 仅标记，无运行时影响 |
| `README.md` | 本文件 | 文档 |
| `resolve_inputs.py` | 将基准数据 JSONL 中的相对图片路径解析为绝对路径，输出到 `$OUTPUT_ROOT/inputs/`；支持 fingerprint 复用 | 所有 stage 均依赖 |
| `run_native_eval.sh` | Base / SFT native 文本生成推理 launcher：调用 `eval_qwen3vl_3reason.py`，无 adapter 或仅 `--adapter`；必须先 resolve_inputs | 启动 GPU |
| `run_regate_candidates.sh` | ReGate candidate dumping launcher：调用 `eval_qwen3vl_3reason.py` 加 `--dual-head --faithful-dual-output --dump-all-answer-candidates`；必须先 resolve_inputs | 启动 GPU |
| `rescore_metrics.py` | 使用当前 fixed-GT scorer 从保存的 predictions 重建各 run 的 `metrics.json`；`--check` 检测 stale artifact | 纯 CPU |
| `analyze_native_regate.py` | 读取 9 份 predictions，计算 shared-candidate diagnostics、ReGate curve 和 publication plots；输出统一带 `native_regate_diagnostic_*` 前缀 | 纯 CPU |
| `select_dev_threshold.py` | 仅在 development 上选择达到 SFT-native retention 的最高经验阈值，冻结后应用于 test predictions；不做 test rematching 或插值 | 纯 CPU |
| `test_native_regate.py` 等 | 单元、CLI、artifact 和 paper consistency tests，纯 CPU，不 import transformers | 测试 |

**重要**：predictions 文件不存储在代码目录内。所有 predictions 写入 `$OUTPUT_ROOT`（约定 `/user/huangyongbin/newrepo/agriabstain/new_reason/outputs/base_native_regate_20260720`），通过 `run_native_eval.sh` / `run_regate_candidates.sh` 的 `OUTPUT_ROOT` 控制。

### Development-selected operating point

主结果使用 canonical development split 选择阈值。先分别生成 SFT-native development
predictions 与 ReGate development probabilities，再运行：

```bash
python -m experiments.base_native_regate.select_dev_threshold \
  --dev-native /path/to/dev/native/predictions.jsonl \
  --dev-probabilities /path/to/dev/regate/head_probabilities.jsonl \
  --test-native /path/to/test/native/predictions.jsonl \
  --test-regate /path/to/test/regate/predictions.jsonl \
  --output-dir /path/to/output
```

选择规则是在 development 上找到满足 ReGate supported retention 不低于 SFT-native
development retention 的最高经验 `p_answer` threshold。阈值随后冻结并直接应用于 test；
test 不参与选阈值、rematching 或 interpolation。GPU 推理由现有
`evaluation/eval_qwen3vl_3reason.py` 和
`experiments/evidence_judge/extract_head_probabilities.py` 完成。

checkpoint、gate 与 threshold 的选择均不使用任何 test 或 external-probe 记录。完整的
test-set sweep 单独保存，仅用作 operating-characteristic 诊断（论文中以 curve-matched
数值报告，用于刻画 routing behavior），不参与阈值选择。

匿名 release 默认提供四个 merged、path-free development/test inputs，只保留该分析实际读取的
字段，因此 compact 包即可逐样本复算 52.0%。Compact `result.json` 中的 SHA256 记录作者侧原始运行
产物 provenance；路径字段清理会改变文件级 hash，但不会改变 sample IDs、GT、probabilities、
选中阈值或报告指标。

---

## 流程

```
run_all.sh all
├── [stage=native]  run_native_eval.sh all
│   ├── resolve_inputs (idempotent, fingerprint)
│   ├── base evaluation → OUTPUT_ROOT/native/base/{comprehensive,refcocog,plantwild}/
│   └── sft evaluation → OUTPUT_ROOT/native/sft/{comprehensive,refcocog,plantwild}/
├── [stage=regate]   run_regate_candidates.sh run
│   ├── resolve_inputs (idempotent, fingerprint)
│   └── regate candidate dumping → OUTPUT_ROOT/regate/{comprehensive,refcocog,plantwild}/
└── [stage=analysis] run_analysis.sh
    ├── validate 9 predictions exist (base/sft/regate × 3 datasets)
    ├── rescore_metrics.py → 各 run 的 metrics.json
    ├── analyze_selective_grounding.py → canonical CSV/JSON
    └── analyze_native_regate.py → diagnostic CSV/JSON + plots
```

**推理资源需求**：native + regate 共需约 12 GPU·hours（4×A100-80GB），分析约 2 分钟 CPU。

---

## 命令

### 环境变量

```bash
export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
export CHECKPOINT=/path/to/sft_final/checkpoint-444
export OUTPUT_ROOT=/user/huangyongbin/newrepo/agriabstain/new_reason/outputs/base_native_regate_20260720
export DERIVED_ROOT=/path/to/derived_images   # optional
```

### 阶段式运行（推荐）

```bash
# 1. 检查参数
bash run_all.sh check

# 2. native base + sft (约 6 GPU·hours)
bash run_all.sh native

# 3. regate candidates (约 6 GPU·hours)
bash run_all.sh regate

# 4. 分析 (约 2 min CPU)
bash run_all.sh analysis

# 或完整流程
bash run_all.sh all
```

### 单步运行

```bash
# native 推理
bash run_native_eval.sh all

# regate 推理
bash run_regate_candidates.sh run

# 分析（需 OUTPUT_ROOT 环境变量）
bash run_analysis.sh
```

---

## 输出树

`$OUTPUT_ROOT`（约定 `/user/huangyongbin/newrepo/agriabstain/new_reason/outputs/base_native_regate_20260720`）：

```
{OUTPUT_ROOT}/
├── inputs/
│   ├── manifest.json              # resolve_inputs 指纹
│   ├── comprehensive_resolved.jsonl
│   ├── refcocog_resolved.jsonl
│   └── plantwild_resolved.jsonl
├── native/
│   ├── base/
│   │   ├── comprehensive/         # comprehensive predictions (JSONL)
│   │   ├── refcocog/              # refcocog predictions
│   │   └── plantwild/             # plantwild predictions
│   └── sft/
│       ├── comprehensive/         # comprehensive predictions (JSONL)
│       ├── refcocog/              # refcocog predictions
│       └── plantwild/             # plantwild predictions
├── regate/
│   ├── comprehensive/             # regate predictions with answer_candidate
│   ├── refcocog/
│   └── plantwild/
└── analysis/
    ├── summary.json               # canonical metrics 汇总
    ├── native_comparison.csv      # canonical Base/SFT/ReGate fixed points
    ├── baseline_points.csv        # canonical Base/SFT 详细指标
    ├── regate_argmax_points.csv   # canonical ReGate argmax 详细指标
    ├── matched_retention.csv      # canonical matched-retention diagnostics
    ├── {dataset}_regate_curve.csv # canonical ReGate 全曲线
    ├── native_regate_diagnostic_* # shared-candidate diagnostics
    ├── {dataset}_risk_coverage.pdf/png
    ├── {dataset}_hallucination_retention.pdf/png
    └── {dataset}_strict3_coverage.pdf/png
```

**代码目录下不存储任何 predictions**。

---

## Claim 口径

| Claim | 支持证据 |
|-------|----------|
| Base native 在 comprehensive 上的 strict-3 accuracy | `native_comparison.csv` 中 `Base_native` 行 |
| SFT checkpoint-444 提升 strict-3 accuracy 且降低 selective risk | `native_comparison.csv` 中 `Base_native` vs `SFT_native_observed` |
| ReGate matched-retention diagnostic | `matched_retention.csv` 中 empirical/interpolated 值 |
| SFT_native_gate_shared_candidate 隔离回答腿生成混淆 | 门控来自 SFT native，answer candidate 来自 ReGate；abstain reason 仍来自 SFT native |
| RefCOCOg 使用 bbox grounding，PlantWild 使用 label-only grounding | metrics 中分别为 `grounding_scope = "bbox"` 与 `"label_only"` |
| 所有 predictions 行数准确 | check 阶段强制 2646 / 362 / 408 |

**不可 claim**：

- 不能 claim「ReGate 优于 SFT native 的端到端公平比较」—— ReGate 的有监督 head 在 SFT 训练数据上训练。
- 不能 claim「base 到 SFT 的改进为架构改善」—— 两者不仅权重不同，生成方式也不同。
- 不能 claim「matched retention 的 interpolation 值为真实可达到的」—— interpolation 仅在曲线点间线性估计。
- 不横向比较 native 与 gated 方法的 `format_valid_rate`：native 衡量文本格式有效性，gated 衡量 answer candidate 有效性。

---

## 失败恢复

| 失败模式 | 诊断 | 恢复 |
|----------|------|------|
| 环境变量未设置 | check 阶段报错 | 设置对应 env 后重试 |
| 数据文件行数不对 | check 阶段报错 | 检查 benchmark 数据完整性 |
| OUTPUT_ROOT 在代码目录内 | check 阶段报错 | 改为外部目录 |
| GPU OOM | `torchrun` 内部报错 | 减少 `NPROC` 或增加 `--batch-size` |
| resolve_inputs 图片路径找不到 | `python resolve_inputs.py` 报 `FileNotFoundError` | 设置正确的 `DERIVED_ROOT` |
| analysis 阶段 predictions 缺失 | `run_analysis.sh` 报 `ERROR: Missing prediction` | 先运行缺失 stage |
| analysis 阶段 validate_all 报错 | 具体错误消息（gt mismatch / 行数不符） | 用 `test_native_regate.py` 单独验证 |
| 断点恢复 | 所有步骤幂等：native/regate 通过 `resolve_inputs` fingerprint 跳过已完成的 resolve；analysis 重复运行覆盖已存在输出 | 直接重跑失败 stage |
