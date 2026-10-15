# AgriAbstain Benchmark — AAAI 发布包

本文档逐组件说明代码与数据的职责、格式和运行方式。所有路径基准以仓库根为准，
除非特别注明。审稿人请先阅读 `REVIEWER_README.md`。

## 可复现边界

`aaai_code/` 包含 benchmark source data/annotations、构造代码、训练代码、正式评测代码、
mechanism experiments 和 CPU analysis/tests。`benchmark/data/derived/` 中的 metamorphic
图片按构造脚本重建，不作为静态副本分发。

模型运行 artifacts 按设计写入外部 `$OUTPUT_ROOT`；当前研究仓库使用 `new_reason/outputs/`。
其中 checkpoint、calibrated head、predictions、特征和分析 JSON 属于运行产物。Base
Qwen3-VL 权重与公开数据集图片由外部来源提供。匿名 supplement 包含 annotations、图片
manifests 和构造代码；完整重建需先从原始数据提供方获取 PlantDoc、COCO、RefCOCOg 和
PlantWild 图片。复核论文中的现成数值还需要随复现材料提供对应 compact 结果 artifacts，
或在取得模型权重与图片后重新运行 GPU 流程。

构造代码要求以下外部数据布局；这些大文件受体积与原数据许可约束，不进入匿名包：

- PlantDoc images 与 YOLO labels：`benchmark/data/source/plantdoc/{images,labels}/`，来源为
  PlantDoc Dataset（`https://github.com/pratikkayal/PlantDoc-Dataset`）。
- COCO 2017 train images：`benchmark/data/source/coco/images/`，来源为
  `https://cocodataset.org/#download`。
- `instances_selected.json`：从 COCO `instances_train2017.json` 按匿名包
  `benchmark/IMAGE_MANIFEST.json` 列出的 image IDs 保留对应 images、annotations 与 categories，
  写入 `benchmark/data/source/coco/annotations/instances_selected.json`。
- RefCOCOg 与 PlantWild 仅用于外部评测；所需文件名和 hash 由 `IMAGE_MANIFEST.json` 约束。

36-layer post-adaptation diagnostic 已整理到 `experiments/layer_selection/`。该目录包含
hidden-state extraction、36-layer scan、CPU tests 和运行说明；匿名 supplement 同时提供
compact scan inputs/results，不包含大体积 NPZ features。

---

## 1. 顶层设计原则

```
aaai_code/
├── benchmark/          # 只放数据（annotations + images + manifests）
├── configs/            # 构造配置（YAML）
├── evaluation/         # 正式模型推理与评测入口
├── experiments/        # 独立机制实验（表征公平比较、证据归因分析）
├── reason_abstain/     # Python package（bench 构造 + eval 评测）
├── scripts/            # 构造/审计/打包工具（可执行脚本）
├── training/           # SFT_FINAL、margin+KL 消融与本地 Qwen 训练框架
├── README.md           # 本文档
└── pyproject.toml      # ruff 配置
```

| 目录 | 职责 | 备注 |
|------|------|------|
| `benchmark/` | 数据目录，包含 images、annotations、manifests。**不包含任何代码**。 | 可通过 `scripts/package_benchmark.py` 独立分发的纯数据包。 |
| `reason_abstain/` | Python package，`bench/` 子包提供 benchmark 构造逻辑，`eval/` 子包提供评测逻辑。 | 从 `aaai_code/` 根可直接 `from reason_abstain.bench import ...`。 |
| `scripts/` | 构造、审计、打包用可执行脚本。 | 从 `aaai_code/` 根运行。 |
| `configs/` | 构造 pipeline 的 YAML 配置。 | 从 `aaai_code/` 根引用 `benchmark/data/...` 路径。 |
| `training/` | SFT_FINAL 主方法、margin+KL v1/v2 消融、共享 Qwen3-VL 训练代码。 | 不包含 checkpoint、mu/sd、z_ref、日志或预测。 |
| `experiments/` | 独立机制实验：表征公平比较、证据归因分析、Base native→ReGate 隔离比较和 36-layer diagnostic。 | 所有 artifacts（特征、probe、概率、predictions）写 `$OUTPUT_ROOT` 外部目录，不污染代码目录。 |
| `evaluation/` | SFT_FINAL 与 margin+KL 共用的正式推理评测代码。 | 正式 PlantWild 只使用 closed12 408 条。 |
| `pyproject.toml` | ruff 格式配置（line-length=100, target-version=py310）。 | 非包元数据（无 `[project]`）。 |

---

## 2. 目录树

```
aaai_code/
├── benchmark/
│   ├── MANIFEST.json                  # 数据清单（hash、行数、图片计数）
│   ├── IMAGE_MANIFEST.json            # 图片清单（路径、sha256、字节数、domain）
│   ├── PACKAGE_REPORT.json            # package_benchmark.py 运行报告
│   ├── annotations/
│   │   ├── raw/                       # 原始构造输出（每条记录一个样本）
│   │   │   ├── plantdoc_full.raw.jsonl
│   │   │   ├── plantdoc_full.raw.summary.json
│   │   │   ├── plantdoc_full.raw.config.snapshot.yaml
│   │   │   ├── coco_full.raw.jsonl
│   │   │   ├── coco_full.raw.summary.json
│   │   │   └── coco_full.raw.config.snapshot.yaml
│   │   ├── splits/                    # 切分后的 splits
│   │   │   ├── train.jsonl
│   │   │   ├── dev.jsonl
│   │   │   ├── test.jsonl
│   │   │   ├── dose_pool.jsonl
│   │   │   ├── split_manifest.json
│   │   │   └── SPLIT_METADATA_FIX_AUDIT.json
│   │   ├── sft/                       # SFT 格式数据
│   │   │   ├── sft_train.json
│   │   │   ├── sft_train_with_source.json
│   │   │   ├── sft_dev.json
│   │   │   └── sft_test.jsonl
│   │   ├── eval/                      # 评测集
│   │   │   ├── dev_L1_discrim.jsonl
│   │   │   ├── dev_L2_evidence.jsonl
│   │   │   ├── dev_L3_dose.jsonl
│   │   │   ├── test_L1_discrim.jsonl
│   │   │   ├── test_L2_evidence.jsonl
│   │   │   ├── test_L3_dose.jsonl
│   │   │   ├── ood_refcocog.jsonl
│   │   │   ├── plantwild_real_closed12_paired_408.jsonl
│   │   │   ├── REFCOCOG_IMAGE_ID_ISOLATION_AUDIT.json
│   │   │   └── BUILD_AUDIT.json
│   │   └── audits/
│   │       └── PLANTDOC_CAPABILITY_GATE_AUDIT.json # split 前 capability-controlled label-space 审计
│   └── data/
│       ├── source/
│       │   ├── plantdoc/
│       │   │   ├── images/           # 726 张图片
│       │   │   └── labels/           # 726 个 YOLO 格式 .txt 标签
│       │   └── coco/
│       │       ├── images/           # 203 张图片
│       │       └── annotations/
│       │           └── instances_selected.json  # 203 图的子集标注
│       ├── ood/
│       │   ├── refcocog/
│       │   │   ├── images/           # 181 张图片
│       │   │   └── metadata/
│       │   │       ├── val-00000-of-00003.parquet  # ~408 MB
│       │   │       └── gate_refcocog.json
│       │   └── plantwild_closed12/
│       │       └── images/           # 144 张图片，12 类子目录
│       └── derived/                  # 空目录：metamorphic 变换产物，需重建
├── configs/
│   ├── plantdoc_full.yaml
│   └── coco_full.yaml
├── reason_abstain/
│   ├── __init__.py
│   ├── bench/
│   │   ├── __init__.py
│   │   ├── boxes.py
│   │   ├── build_coco.py
│   │   ├── build_plantdoc.py
│   │   ├── export_qwen.py
│   │   ├── io.py
│   │   ├── quality.py
│   │   ├── schema.py
│   │   ├── split.py
│   │   ├── templates.py
│   │   ├── test_varied_query.py
│   │   └── transforms.py
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py
│       └── parser.py
├── training/
│   ├── deepspeed/zero2.json
│   ├── sft_final/
│   │   ├── compute_musd.py
│   │   ├── run_train.sh
│   │   └── calibrate_head.py
│   ├── margin_kl/
│   │   ├── dump_zref.py
│   │   ├── paired_dataset.py
│   │   ├── paired_sampler.py
│   │   ├── paired_trainer.py
│   │   ├── train_paired_qwen.py
│   │   ├── run_v1.sh
│   │   ├── run_v2.sh
│   │   └── eval_closed12.sh
│   └── qwenvl/                         # 本地修改版 Qwen3-VL 训练 package
│       ├── data/
│       ├── model/
│       └── train/
├── evaluation/
│   ├── eval_qwen3vl_3reason.py
│   └── run_sft_final.sh
├── experiments/
│   ├── evidence_judge/
│   │   ├── __init__.py                   # 包标记
│   │   ├── README.md                     # 本目录文档（中文）
│   │   ├── extract_representations.py    # Experiment A：四种表征提取
│   │   ├── train_probes.py               # Experiment A：probe 训练与比较
│   │   ├── extract_head_probabilities.py # Experiment B：head 概率提取
│   │   ├── analyze_evidence.py           # Experiment B：L2/L3/consistency 分析
│   │   ├── run_representation.sh         # Experiment A launcher
│   │   ├── run_evidence.sh               # Experiment B launcher
│   │   └── test_evidence_judge.py        # 纯 CPU 单元测试
│   └── base_native_regate/               # Native base/SFT → calibrated ReGate 曲线比较
│       ├── __init__.py                   # 包标记
│       ├── README.md                     # 本文档（中文+英文技术词混合）
│       ├── resolve_inputs.py             # 图片路径解析（idempotent, fingerprint）
│       ├── run_native_eval.sh            # Base/SFT 文本生成推理 launcher
│       ├── run_regate_candidates.sh      # ReGate candidate dumping launcher
│       ├── run_analysis.sh               # 分析 launcher（CPU）
│       ├── run_all.sh                    # 编排 launcher（check/native/regate/analysis/all）
│       ├── analyze_native_regate.py      # shared-candidate diagnostics + ReGate curve
│       ├── rescore_metrics.py            # predictions → current fixed-GT metrics.json
│       └── test_*.py                     # 单元、artifact 与 paper consistency tests
├── scripts/
│   ├── audit_refcocog_image_isolation.py
│   ├── build_benchmark.py
│   ├── build_eval_L1.py
│   ├── build_eval_L2.py
│   ├── build_eval_L3.py
│   ├── build_final_sft_3reason.py
│   ├── build_ood_refcocog.py
│   ├── fix_split_metadata.py
│   ├── package_benchmark.py
│   ├── split_benchmark.py
│   └── plantwild_rl/
│       └── build_closed_set_eval.py
├── README.md
└── pyproject.toml
```

---

## 3. 代码文件职责说明

### 3.1 顶层非图片文件

| 文件 | 职责 |
|------|------|
| `pyproject.toml` | ruff 配置（line-length=100, target-version=py310, ignore E501）。无 `[project]` 元数据。 |

### 3.2 `configs/` — 构造配置

| 文件 | 职责 |
|------|------|
| `plantdoc_full.yaml` | PlantDoc benchmark 构造配置。定义 26 个配置类名；calibration pool 实际有 24 个标签，4 个低于 pre-split capability threshold，另有 1 个 unverified placeholder 被配置排除，最终 benchmark 含 20 个有数据的农业标签。还定义 T1–T7、移除算子、region ladder 和同作物 T6 pairs。 |
| `coco_full.yaml` | COCO 数据集的 benchmark 构造配置。定义 annotation_path、images_dir、T1–T5/T7 变换、region ladder 档位、absent_per_positive=2、max_samples=10000。 |

### 3.3 `scripts/` — 构造/审计工具

| 脚本 | 职责 |
|------|------|
| `build_benchmark.py` | 调用 `reason_abstain.bench.build_plantdoc` / `build_coco` 执行构造。入口参数：`--dataset`（coco/plantdoc）、`--config`、`--output`。输出 raw JSONL + summary + config snapshot。 |
| `split_benchmark.py` | 将 raw JSONL 母本按策略切分为 train/dev/test/dose_pool 四个 JSONL，输出 split_manifest.json。分裂逻辑实现在脚本内，不依赖 `reason_abstain.bench.split`。 |
| `fix_split_metadata.py` | **MUTATION TOOL** — 修复 dev.jsonl/test.jsonl 中错误为 `"train"` 的 split 字段，改为 `"dev"`/`"test"`。原子写入，输出审计 JSON。默认从 `aaai_code/` 根运行，package mode 读写 `benchmark/annotations/splits/`。会改写正式数据，不要用于只读校验。只读校验请使用第 7 节命令，或 `--help` 查看参数。 |
| `build_final_sft_3reason.py` | 从 train.jsonl 构造三分类 SFT 数据（answer / wrong_object / no_target）。输出 `sft_train.json`、`sft_dev.json`、`sft_test.jsonl`。 |
| `build_eval_L1.py` | 从 test.jsonl 构造 L1 判别探针：同图既有真病（answer）又有近义病（T6, abstain）。输出 `{dev,test}_L1_discrim.jsonl`。 |
| `build_eval_L2.py` | 从 test.jsonl 构造 L2 证据探针：完整可见（keep）vs 裁到 0% 不可定位（cropout）。输出 `{dev,test}_L2_evidence.jsonl`。 |
| `build_eval_L3.py` | 从 test.jsonl + dose_pool.jsonl 构造 L3 剂量探针：T7_ladder 五档（1.0/0.75/0.5/0.25/0.0）。输出 `{dev,test}_L3_dose.jsonl`。 |
| `build_ood_refcocog.py` | 从 RefCOCOg parquet 构造通用域 OOD 评测集。answer 探针用 referring expression 作 query + GT box；absent 探针用训练见过的 COCO 类作 absent query。默认 package mode 读写 `benchmark/data/ood/refcocog/` 和 `benchmark/annotations/eval/`。输出 `ood_refcocog.jsonl`。 |
| `audit_refcocog_image_isolation.py` | **MUTATION TOOL** — 审计 ood_refcocog.jsonl 的 image isolation：回填 4 个 metadata 字段（coco_image_id、refcocog_row_index、coco_file_name、question_id）；验明 probe 的 unique COCO IDs 与训练 COCO IDs 交集为 0。默认 package mode 读写 `benchmark/data/ood/refcocog/metadata/`、`benchmark/annotations/splits/`、`benchmark/annotations/eval/`。会原子改写 ood_refcocog.jsonl，不要用于只读校验。只读校验请使用第 7 节命令或 `--help`。 |
| `package_benchmark.py` | **data-only 打包**。见第 9 节。 |
| `plantwild_rl/build_closed_set_eval.py` | 从旧 13 类 open-set（444 条）过滤为 12 类 closed-set 评测集。排除 `tomato leaf yellow virus`（SFT answer=0，12 张图 × 3 探针 = 36 条）。输出 144 张图、408 条 paired records。**需要完整研究仓库**（含旧 `new_reason/outputs/eval_probes/plantwild_real_v1/` 和 `checkpoint-444` 预测），不能仅靠 data-only benchmark 包重建。 |

### 3.4 `reason_abstain/bench/` — benchmark 构造子包

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包标记：`"""Benchmark construction utilities."""` |
| `schema.py` | 数据 schema 定义（record 字段、relation 枚举、decision 类型等）。 |
| `io.py` | 读写工具：`read_yaml`、`write_json`、`write_jsonl`、`write_yaml`。 |
| `boxes.py` | bbox 坐标转换：xywh ↔ Qwen [0,1000] 相对坐标格式；box_area 计算。不包含 IoU 或筛选逻辑。 |
| `transforms.py` | 图像变换引擎：T2 photometric、T3 geometric、T4 证据移除（gaussian_blur/mean_replace/pixelate）、T7 region ladder。 |
| `templates.py` | query 模板生成：T1 paraphrase 自动生成不同措辞的 natural language query。 |
| `quality.py` | 数据质量报告：`summarize_records` 统计各类别/relation 分布。 |
| `split.py` | 独立的 leakage-safe reusable split utility。当前正式 split 由 `scripts/split_benchmark.py` 脚本内逻辑生成。 |
| `build_plantdoc.py` | PlantDoc 数据集构造主逻辑：读取 YOLO labels、执行 metamorphic 变换、生成 record 序列。 |
| `build_coco.py` | COCO 数据集构造主逻辑：读取 COCO JSON、执行 metamorphic 变换、生成 record 序列。 |
| `export_qwen.py` | 导出 Qwen-VL-Finetune 格式工具。 |
| `test_varied_query.py` | 测试脚本：验证 metamorphic query 变体输出正确性。 |

### 3.5 `reason_abstain/eval/` — 评测子包

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包标记：`"""Evaluation utilities for structured AgriAbstain outputs."""` |
| `parser.py` | 模型输出解析：从 `<answer>...</answer>` 提取 JSON decision。 |
| `metrics.py` | 评测指标计算：IoU、confusion matrix、decision accuracy、abstention rate、MCC、metamorphic test satisfaction (MTS micro/macro, per-relation)、risk-coverage analysis、per-probe breakdown。<br>**注意**：`grounding_reason` 输出使用 `E2E_mIoU` / `E2E_mIoU_c` 作为规范名称。`A_Ground` / `A_Ground_c` 保留为向后兼容的 deprecated alias，值完全相等。新代码应优先使用 `E2E_mIoU` 系列。 |

### 3.6 `training/sft_final/` — SFT_FINAL 主方法

| 文件 | 职责 |
|------|------|
| `compute_musd.py` | 在 base Qwen3-VL 的 2,000 条训练样本上提取 `hidden_states[27]` last-token 表征，生成判别头固定标准化统计 `mu/sd`。支持 torchrun、dry-run 和确定性采样；产出的 `.pt` 不随发布包分发。 |
| `run_train.sh` | SFT_FINAL 正式训练入口。显式启用 tri-head；默认复现 checkpoint-444 对应配置：6 epochs、`head_lambda=0.4`、warmup 10、`WO_HEAD_WEIGHT=1.5`、LR `1e-4`、LoRA `16/32/0.05`、seed 42。 |
| `calibrate_head.py` | 冻结 SFT backbone，先提取固定表征，再训练三分类 calibrated head。正式 SFT_FINAL 评测使用该输出覆盖端到端 head。 |

### 3.7 `training/margin_kl/` — margin+KL v1/v2 消融

| 文件 | 职责 |
|------|------|
| `dump_zref.py` | 从 SFT_FINAL checkpoint 与 calibrated head 生成 7,072×3 的 reference logits cache。支持 torchrun、manifest 指纹和 dry-run；不分发生成的 cache。 |
| `paired_dataset.py` | 为每条训练样本附加稳定 row index，供 z_ref 查表和 paired batch 使用。 |
| `paired_sampler.py` | 按 answer / wrong_object / no_target 模式构造 batch；v2 通过 `PAIR_PATTERN` 改变采样配比。 |
| `paired_trainer.py` | 实现 snapshot-anchored margin loss、symmetric KL consistency 与 auxiliary CE，并校验 z_ref 和样本行索引对齐。 |
| `train_paired_qwen.py` | 从 SFT_FINAL adapter + calibrated head 初始化，仅解冻指定窄带 LoRA 与判别头，接入 paired dataset/sampler/trainer。 |
| `run_v1.sh` | v1 launcher：margin `0.5`、beta `1.0`、200 steps、每 50 steps 保存。论文选用 `checkpoint-50`。 |
| `run_v2.sh` | v2 launcher：margin `0.25`、beta `1.0`、pattern `0,1,0,2,0,1,0,2`、80 steps、每 20 steps 保存。论文选用 `checkpoint-60`。 |
| `eval_closed12.sh` | margin+KL checkpoint 的 PlantWild closed12 408 条 faithful eval；强制检查行数，不加载 SFT calibrated head。 |

### 3.8 共享训练与正式评测

| 文件/目录 | 职责 |
|-----------|------|
| `training/deepspeed/zero2.json` | LoRA 训练使用的 DeepSpeed ZeRO-2 配置。 |
| `training/qwenvl/data/__init__.py` | 只注册正式 `sft_train.json` 和 paired `sft_train_with_source.json`，媒体路径以 `benchmark/` 为基准。 |
| `training/qwenvl/data/data_processor.py` | Qwen 多模态样本预处理、lazy dataset 和 collator，传递 decision/reason/head label。 |
| `training/qwenvl/data/rope2d.py` | Qwen2/2.5/3-VL 多模态 RoPE position IDs。 |
| `training/qwenvl/model/qwen3vl_dual_head.py` | 二分类 dual-head 基类和判别头结构。 |
| `training/qwenvl/model/qwen3vl_dual_head_3reason.py` | SFT_FINAL 与 margin+KL 使用的三分类 tri-head 模型。 |
| `training/qwenvl/model/qwen3vl_dual_head_2reason.py` | `train_qwen.py` 保留的二分类兼容分支，不属于本文主结果。 |
| `training/qwenvl/train/argument.py` | Hugging Face model/data/training 参数定义。 |
| `training/qwenvl/train/trainer.py` | Trainer 与可选 FlashAttention monkey patch；正式 launcher 使用 SDPA 且关闭 packing。 |
| `training/qwenvl/train/aux_head.py` | `train_qwen.py` 的 auxiliary-head 兼容实现。 |
| `training/qwenvl/train/repr_head.py` | `train_qwen.py` 的 representation-head 兼容实现。 |
| `training/qwenvl/train/train_qwen.py` | 共享训练入口，包含 TriHeadTrainer、LoRA 配置、固定 mu/sd 加载和 checkpoint 保存。 |
| 各级 `__init__.py` | Python package 标记。 |
| `evaluation/eval_qwen3vl_3reason.py` | 正式三态推理与评测：HF/vLLM backend、多卡 shards、断点续跑、tri-head、calibrated head、faithful dual output 和完整指标。 |
| `evaluation/run_sft_final.sh` | 依次评测 SFT test、L1/L2/L3、RefCOCOg、PlantWild closed12；PlantWild 强制为 408 条。 |

### 3.9 `experiments/evidence_judge/` — 独立机制实验

| 文件 | 职责 |
|------|------|
| `extract_representations.py` | Experiment A：在 frozen backbone 上提取 `hidden_states[27]` 的四种表征（last_token、visual_mean、visual_max、visual_query_attn），支持 torchrun 多卡分片与 fingerprint 指纹。 |
| `train_probes.py` | Experiment A：在固定表征上训练轻量 probe MLP，独立 mu/sd、5 seeds paired bootstrap 比较四种表征的 head-level 分类性能。 |
| `extract_head_probabilities.py` | Experiment B：使用 calibrated head 提取 L2/L3 证据探针的 head-only 概率（p_answer/p_wo/p_nt），支持 query/image counterfactual variants。 |
| `analyze_evidence.py` | Experiment B：L2 keep-vs-cropout 分析、L3 dose-response 单调性分析、input equivariance consistency 分析。 |
| `run_representation.sh` | Experiment A launcher，阶段 `check\|extract\|probe\|all`，默认 `check`。使用 `AAAI_CODE` 绝对路径定位数据，`$OUTPUT_ROOT` 外部输出。 |
| `run_evidence.sh` | Experiment B launcher，阶段 `check\|l2\|l3\|consistency\|all`，默认 `check`。 |
| `test_evidence_judge.py` | 纯 CPU 单元测试（50 项），不 import transformers 模型。 |

**设计原则**：
- 所有 artifacts（特征 `.npz`、probe 权重、概率 JSONL、分析报告）写入 `$OUTPUT_ROOT` 外部目录，不污染代码目录。
- Launcher 默认阶段为 `check`，仅检查参数；必须显式指定 `all` 才启动 GPU 流程，防止误启。
- 数据路径基于脚本推导的 `AAAI_CODE` 绝对路径，不依赖调用 cwd。
- `benchmark/` 仍为纯数据目录，不包含本实验的任何代码或输出。

### 3.10 `experiments/base_native_regate/` — Native base/SFT → calibrated ReGate 曲线分析

| 文件 | 职责 |
|------|------|
| `resolve_inputs.py` | 将 benchmark JSONL 中相对图片路径解析为绝对路径，输出到 `$OUTPUT_ROOT/inputs/`；支持 fingerprint 复用，幂等。 |
| `run_native_eval.sh` | Base / SFT native 文本生成推理 launcher：调用 `eval_qwen3vl_3reason.py`，无 adapter 或仅 `--adapter`。阶段 `check\|base\|sft\|all`，默认 `check`。 |
| `run_regate_candidates.sh` | ReGate candidate dumping launcher：调用 `eval_qwen3vl_3reason.py` 加 `--dual-head --faithful-dual-output --dump-all-answer-candidates`。阶段 `check\|run`，默认 `check`。 |
| `run_analysis.sh` | 分析 launcher：验证并重算 9 份 predictions，生成 canonical CSV/JSON，再生成 diagnostic CSV/JSON/图。 |
| `run_all.sh` | 编排 launcher：阶段 `check\|native\|regate\|analysis\|all`，默认 `check`。`all` 按 native→regate→analysis 顺序运行。 |
| `rescore_metrics.py` | 从 predictions 重建当前 fixed-GT `metrics.json`，或用 `--check` 检测 stale artifact。 |
| `analyze_native_regate.py` | 计算 shared-candidate diagnostics、regate_full_curve 和 publication plots；输出使用 `native_regate_diagnostic_*` 前缀。 |
| `test_*.py` | 61 项 metric、artifact、paper consistency 与 CLI 测试；完整 `aaai_code` 当前共 194 项 CPU tests。 |

**设计原则**：
- 所有 9 份 predictions 写入 `$OUTPUT_ROOT` 外部目录（约定 `/user/.../new_reason/outputs/base_native_regate_20260720`），代码目录不存 predictions。
- 三个方法共享相同数据文件（2646 / 362 / 408 行），通过 `resolve_inputs` 统一解析图片路径。
- 默认 stage 为 `check`，仅检查参数/路径/行数，不启动 GPU。
- ReGate 使用 calibrated head 的 p_answer/p_wo/p_nt 概率实现精确门控，SFT native 和 Base native 使用纯文本生成 decision。
- RefCOCOg 和 comprehensive 使用 `bbox` grounding scope（IoU 阈值 0.5），PlantWild 因 GT box = null 使用 `label_only`。

---

## 4. benchmark 数据文件说明

### 4.1 三个 manifests

| 文件 | 用途 | 格式 |
|------|------|------|
| `MANIFEST.json` | 数据完整性清单。记录所有 annotations 文件的路径、sha256、字节数、记录数；图片计数（四组：source_plantdoc 726、source_coco 203、ood_refcocog 181、ood_plantwild_closed12 144）；path_audit 检查旧前缀（`old_prefix_occurrences_total`）和缺失引用（`missing_image_refs_total`）。 | JSON |
| `IMAGE_MANIFEST.json` | 图片级清单。每条记录含 path（相对 benchmark/）、sha256、bytes、domain（source_plantdoc / source_coco / ood_refcocog / ood_plantwild_closed12）。共 1254 条。 | JSON array |
| `PACKAGE_REPORT.json` | `package_benchmark.py` 的运行报告。记录每一步的复制结果、图片 ID 列表、总图片数（1254）、总字节数、annotation 文件数（27）、错误/警告列表。 | JSON |

### 4.2 `annotations/raw/` — 原始构造输出

| 文件 | 用途 | 格式 | 已知记录数 |
|------|------|------|-----------|
| `plantdoc_full.raw.jsonl` | PlantDoc 全量构造输出，每条一个 record。 | JSONL | 20,131 |
| `plantdoc_full.raw.summary.json` | 构造摘要：各类别/relation 的记录数统计。 | JSON dict | N/A |
| `plantdoc_full.raw.config.snapshot.yaml` | 构造时使用的 plantdoc_full.yaml 配置快照。 | YAML | — |
| `coco_full.raw.jsonl` | COCO 全量构造输出，每条一个 record。 | JSONL | 9,999 |
| `coco_full.raw.summary.json` | 构造摘要。 | JSON dict | N/A |
| `coco_full.raw.config.snapshot.yaml` | 构造时使用的 coco_full.yaml 配置快照。 | YAML | — |

### 4.3 `annotations/splits/` — 切分后数据

| 文件 | 用途 | 格式 | 已知记录数 |
|------|------|------|-----------|
| `train.jsonl` | 训练集 splits。 | JSONL | 14,701 |
| `dev.jsonl` | 验证集 splits。 | JSONL | 2,408 |
| `test.jsonl` | 测试集 splits（硬 GT）。 | JSONL | 2,646 |
| `dose_pool.jsonl` | 仅 dose 模糊档池。组成：T4 target degradation 7,674 条 + T7 dose-only 1,454 条（请求 0.25 档 1,135 条 + 请求 0.5 档但 achieved_visible_ratio < 0.5 的 319 条），共计 9,128 条。**注**：0.75 档的 T7_ladder 样本已进入 hard-label splits（train.jsonl/dev.jsonl/test.jsonl），不在 dose_pool 内。 | JSONL | 9,128 |
| `split_manifest.json` | 切分记录：每组的记录数、图片数、按 dataset/relation 的分布。 | JSON dict | N/A |
| `SPLIT_METADATA_FIX_AUDIT.json` | 修复 split 字段的审计记录。 | JSON dict | N/A |

### 4.4 `annotations/sft/` — SFT 格式数据

| 文件 | 用途 | 格式 | 已知记录数 |
|------|------|------|-----------|
| `sft_train.json` | 三分类 SFT 训练集（含 head_label 0/1/2）。每条 `image` 字段为图片路径列表。 | JSON | 7,072 |
| `sft_train_with_source.json` | 追溯增强 artifact，在 `sft_train.json` 基础上额外保留 source 字段。已发布，不由当前脚本重新生成。 | JSON | 7,072 |
| `sft_dev.json` | 三分类 SFT 验证集。 | JSON | 2,408 |
| `sft_test.jsonl` | 三分类 SFT 测试集（JSONL 格式供评估）。 | JSONL | 2,646 |

### 4.5 `annotations/eval/` — 评测集

| 文件 | 用途 | 格式 | 已知记录数 |
|------|------|------|-----------|
| `dev_L1_discrim.jsonl` | L1 判别探针（dev 集）：同图真病 vs 近义病。 | JSONL | 410 |
| `dev_L2_evidence.jsonl` | L2 证据探针（dev 集）：完整可见 vs 裁到 0%。 | JSONL | 570 |
| `dev_L3_dose.jsonl` | L3 剂量探针（dev 集）：5 档可见性阶梯。 | JSONL | 677 |
| `test_L1_discrim.jsonl` | L1 判别探针（test 集）。 | JSONL | 410 |
| `test_L2_evidence.jsonl` | L2 证据探针（test 集）。 | JSONL | 790 |
| `test_L3_dose.jsonl` | L3 剂量探针（test 集）。 | JSONL | 750 |
| `ood_refcocog.jsonl` | RefCOCOg 通用域 OOD 评测集。 | JSONL | 362 |
| `plantwild_real_closed12_paired_408.jsonl` | PlantWild 野外 closed12 评测集。 | JSONL | 408 |
| `REFCOCOG_IMAGE_ID_ISOLATION_AUDIT.json` | RefCOCOg image isolation 审计结果。 | JSON dict | N/A |
| `BUILD_AUDIT.json` | 构造审计：所有 eval 集的构造记录、hash、行数校验。 | JSON dict | N/A |

### 4.6 `annotations/audits/` — 构造前审计

| 文件 | 用途 | 格式 |
|------|------|------|
| `PLANTDOC_CAPABILITY_GATE_AUDIT.json` | PlantDoc split 前 capability-controlled label-space 审计。记录 raw Qwen3-VL-8B 在 768 条 base records 上的 24 类统计、0.40 correct-rate threshold、4 个 capability exclusions、20 个 retained labels、输入 artifact hashes，以及该筛选发生在 benchmark source-image split 之前的时序记录。 | JSON dict |

该 capability control 以整类为单位保留或排除，不做 record-level 过滤，因此是在 verified
supported-grounding label space 内定义 unsupported-query 评测。20 类 label space 在 source
image 分配到 train/dev/test 之前即已冻结，后续 split 仍按 source image 严格隔离。

---

## 5. `benchmark/data/` 子目录说明

### 5.1 `data/source/plantdoc/` — PlantDoc 源数据

- **images/**：726 张图片（train 502 / dev 110 / test 114）。
- **labels/**：726 个 YOLO 格式 `.txt` 标签文件（与图片一一对应），每行 `class_id x_center y_center width height`（归一化）。

### 5.2 `data/source/coco/` — COCO 源数据

- **images/**：203 张图片（train 142 / dev 30 / test 31）。
- **annotations/instances_selected.json**：COCO JSON 格式，仅包含 203 张选中图片的标注（1610 个 annotation，80 个 category）。

### 5.3 `data/ood/refcocog/` — RefCOCOg OOD 数据

- **images/**：181 张图片。
- **metadata/val-00000-of-00003.parquet**：RefCOCOg 标注 parquet 文件，约 408 MB。
- **metadata/gate_refcocog.json**：能力门控过滤后的 RefCOCOg 条目清单。

### 5.4 `data/ood/plantwild_closed12/` — PlantWild OOD 数据

- **images/**：144 张图片，按 12 类病害子目录存放（如 `apple scab/`、`corn rust/`、`tomato early blight/` 等）。

### 5.5 `data/derived/` — 变换产物（空）

**此目录为空**。metamorphic 变换生成的派生图片不随包发布，需通过 `scripts/build_benchmark.py`
重建（见第 8 节）。当前 raw annotations 中 unique derived image paths 共 20,815：
PlantDoc 13,666，COCO 7,149。这些是构造成功并发出 record 的路径。degenerate crop
（例如裁剪区域完全落在图片外的样本）在构造时已跳过，不进入 raw annotations，因此 raw
中不会出现对应路径。`MANIFEST.json` 中 `derived_image_refs_count: 50867`
为 annotations 中 path reference 出现次数（同一路径可能在多条记录中被引用），非图片张数。

---

## 6. 路径基准

- **annotations 内** `data/...` 路径相对 `aaai_code/benchmark`。
  - 例如 `data/source/plantdoc/images/...` 实际指向 `aaai_code/benchmark/data/source/plantdoc/images/...`。
- **configs 内** `benchmark/data/...` 路径相对 `aaai_code`。
  - 例如 `configs/plantdoc_full.yaml` 中的 `images_dir: benchmark/data/source/plantdoc/images` 实际指向 `aaai_code/benchmark/data/source/plantdoc/images`。
- **所有示例命令从 `aaai_code/` 根运行**。
- **`experiments/` 下的 launcher 脚本内部推导 `AAAI_CODE` 绝对路径**，可从任意目录运行。

---

## 7. 只读校验命令

以下命令不修改任何正式 annotations，仅用于完整性审计。

### 7.1 ruff 格式检查

```bash
ruff check . --exclude benchmark/annotations
```

`benchmark/annotations/` 下的 JSON/JSONL/YAML 非 Python，不参与 lint。

### 7.2 import 检查

```bash
python -c "from reason_abstain.bench import schema, io, boxes, transforms, templates, quality, split, build_plantdoc, build_coco; print('bench imports OK')"
python -c "from reason_abstain.eval import parser, metrics; print('eval imports OK')"
```

### 7.3 manifest 路径存在校验

```bash
python -c "
import json
with open('benchmark/MANIFEST.json') as f:
    m = json.load(f)
for a in m['annotations']:
    p = 'benchmark/' + a['path']
    assert __import__('os').path.isfile(p), f'Missing: {p}'
    print(f'  OK  {a[\"path\"]}  ({a[\"records\"]} records, {a[\"bytes\"]} bytes)')
print(f'Total annotations: {len(m[\"annotations\"])}')
"
```

### 7.4 MANIFEST hash 校验 + path_audit 断言

```bash
python -c "
import json, hashlib
with open('benchmark/MANIFEST.json') as f:
    m = json.load(f)

# path_audit 断言：旧前缀和缺失引用必须为 0
pa = m['path_audit']
assert pa['old_prefix_occurrences_total'] == 0, f'old_prefix_occurrences_total = {pa[\"old_prefix_occurrences_total\"]}'
assert pa['missing_image_refs_total'] == 0, f'missing_image_refs_total = {pa[\"missing_image_refs_total\"]}'
print(f'  OK  path_audit: old_prefix={pa[\"old_prefix_occurrences_total\"]}, missing_refs={pa[\"missing_image_refs_total\"]}')

# sha256 校验
errors = []
for a in m['annotations']:
    p = 'benchmark/' + a['path']
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    if h != a['sha256']:
        errors.append(f'HASH MISMATCH: {a[\"path\"]}')
    else:
        print(f'  OK  hash  {a[\"path\"]}')
if errors:
    for e in errors: print(e)
else:
    print('All annotation hashes match.')
"
```

### 7.5 IMAGE_MANIFEST hash/size 校验

```bash
python -c "
import json, hashlib
with open('benchmark/IMAGE_MANIFEST.json') as f:
    images = json.load(f)
print(f'IMAGE_MANIFEST entries: {len(images)}')
errors = []
for img in images:
    p = 'benchmark/' + img['path']
    if not __import__('os').path.isfile(p):
        errors.append(f'MISSING: {img[\"path\"]}')
        continue
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    if h != img['sha256']:
        errors.append(f'HASH MISMATCH: {img[\"path\"]}')
    elif __import__('os').path.getsize(p) != img['bytes']:
        errors.append(f'SIZE MISMATCH: {img[\"path\"]}')
if errors:
    for e in errors[:20]: print(e)
    print(f'... ({len(errors)} errors total)')
else:
    print('All 1254 images: hash and size match.')
"
```

### 7.6 图片计数校验

```bash
python -c "
import json
with open('benchmark/MANIFEST.json') as f:
    m = json.load(f)
ic = m['image_counts']
for k, v in ic.items():
    print(f'{k}: {v[\"total\"]} images (expected {v[\"expected\"]}, match={v[\"match\"]})')
"
```

---

## 8. Derived 重建说明

### 8.1 重要前提

**正式 annotations 已存在**。`benchmark/annotations/` 下的所有 JSONL/JSON 文件是最终版本，
无需重新生成。derived images 不在发布包中，仅用于训练时读取。

**重建 derived 前，不要训练引用 derived 的 SFT。** SFT 数据（`sft_train.json` 等）中的
`image` 字段引用了 `data/derived/...` 路径，如果 derived 目录为空，训练会失败。

### 8.2 重建命令

以下命令将 raw annotations 输出到 `/tmp` 或指定的 scratch 目录，生成 derived images
到 `benchmark/data/derived/`：

```bash
# PlantDoc（当前 raw 中 unique derived paths 13,666，需数 GB 磁盘空间）
python scripts/build_benchmark.py \
    --dataset plantdoc \
    --config configs/plantdoc_full.yaml \
    --output /tmp/plantdoc_rebuild.jsonl

# COCO（当前 raw 中 unique derived paths 7,149）
python scripts/build_benchmark.py \
    --dataset coco \
    --config configs/coco_full.yaml \
    --output /tmp/coco_rebuild.jsonl
```

重建完成后，`benchmark/data/derived/plantdoc/train/` 和 `benchmark/data/derived/coco/train/`
下将包含派生图片。此时可开始训练引用 derived 的 SFT。

**注意**：重建所得派生图片数应与 raw 中的 unique derived paths 数一致
（20,815），因为 degenerate crop 在构造时已跳过，不进入 raw，raw 中
的路径均为成功构造的镜像。

### 8.3 自定义 scratch 目录

```bash
SCRATCH=/path/to/scratch  # 替换为你的临时目录
python scripts/build_benchmark.py --dataset plantdoc --config configs/plantdoc_full.yaml --output "$SCRATCH/plantdoc_rebuild.jsonl"
```

---

## 9. `scripts/package_benchmark.py` 说明

`package_benchmark.py` 是 **data-only 打包工具**。它从完整研究仓库中复制 curated benchmark
数据到自包含的 `benchmark/` 目录，所有路径重写为相对 POSIX 路径。**不包含**
`reason_abstain/`、`configs/`、`scripts/`、`pyproject.toml` 或 README。

**重跑行为**：重跑会清理 `benchmark/` 下的受管目录（`annotations/`、`data/`、
`MANIFEST.json`、`IMAGE_MANIFEST.json`、`PACKAGE_REPORT.json`）及 legacy artifacts
（`benchmark/` 下的 `agriabstain/`、`reason_abstain/`、`scripts/`、`configs/`、
`.ruff_cache/`、`outputs/`，以及文件 `pyproject.toml`、`README.md`），然后重新复制。
**源数据不修改。**

使用方式：
```bash
# 默认（从 aaai_code/ 运行）
python scripts/package_benchmark.py

# 自定义路径
python scripts/package_benchmark.py \
    --source-root /path/to/agriabstain \
    --output-root /path/to/benchmark
```

---

## 10. 正式数据口径

| 数据集 | 图片数 | 记录数 | 说明 |
|--------|--------|--------|------|
| SFT Train | 3,252 unique image paths | **7,072** | `sft_train.json` / `sft_train_with_source.json` |
| SFT Dev | 1,345 unique image paths | **2,408** | `sft_dev.json` |
| SFT Test | 1,504 unique image paths | **2,646** | `sft_test.jsonl` |
| PlantWild closed12 | 144 | **408** | `plantwild_real_closed12_paired_408.jsonl`。12 类，每图 3 探针（answer + no_target + wrong_object）。旧 PlantWild 444 条（13 类 open-set）不在发布包内。 |
| RefCOCOg | 181 | **362** | `ood_refcocog.jsonl`。181 行/图片，每行 2 探针（answer + absent）。180 个 unique COCO IDs（RefCOCOg 中有一行对应的 COCO 图片与另一行相同，见 `REFCOCOG_IMAGE_ID_ISOLATION_AUDIT.json`）。 |

---

## 11. 依赖

- Python 3.10+
- 核心依赖（可通过 `import` 和 config 确定）：
  - `PyYAML`（config 读写）
  - `Pillow`（图像处理）
  - `pyarrow`（parquet 读写，仅 RefCOCOg 构建需要）
  - `numpy`（数值计算）
  - `tqdm`（进度显示）
  - 训练/推理：`torch`、带 Qwen3-VL 支持的 `transformers`、`peft`、`deepspeed`、`safetensors`、`qwen-vl-utils`、`scikit-learn`

基础模型权重、SFT/margin+KL checkpoints、`discrim_musd.pt`、`zref_cache.pt` 和校准头
不在发布包内。前两个 `.pt` 可由本包脚本重建。匿名 supplement 以 `--include-predictions`
模式构建，包含论文对应的 curated predictions、selected runtime logs、environment record、
Margin+KL selected metrics 和 GPT request configs；raw API responses、response IDs、
hidden features 与 checkpoint 权重仍排除。

---

## 12. 训练与评测顺序

所有命令从 `aaai_code/` 根运行。训练前必须先按第 8 节重建 `benchmark/data/derived/`。

### 12.1 SFT_FINAL

1. 生成 base-model 表征标准化统计：

```bash
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
MUSD_OUTPUT=/scratch/discrim_musd.pt \
torchrun --nproc_per_node=8 training/sft_final/compute_musd.py
```

2. 训练三分类 SFT。第 4 epoch 端点对应原实验的 `checkpoint-444`；launcher 仍完整训练 6 epochs：

```bash
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
DISCRIM_MUSD_PATH=/scratch/discrim_musd.pt \
SFT_OUTPUT_DIR=/scratch/sft_final \
bash training/sft_final/run_train.sh
```

3. 对选定 checkpoint 提取表征并训练 calibrated head：

```bash
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
CALIB_ADAPTER=/scratch/sft_final/checkpoint-444 \
CALIB_LOG_DIR=/scratch/calibration \
torchrun --nproc_per_node=4 training/sft_final/calibrate_head.py --mode extract

BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
CALIB_ADAPTER=/scratch/sft_final/checkpoint-444 \
CALIB_LOG_DIR=/scratch/calibration \
python training/sft_final/calibrate_head.py --mode train
```

4. 正式评测：

```bash
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
ADAPTER=/scratch/sft_final/checkpoint-444 \
CALIBRATED_HEAD=/scratch/sft_final/checkpoint-444/discrim_head_calibrated.pt \
OUTPUT_DIR=/scratch/eval_sft_final \
bash evaluation/run_sft_final.sh
```

正式评测沿用原实验命令的 `--system-override decision --max-pixels 262144`。MUSD 与
calibrated-head 表征提取保留原脚本的 processor 默认分辨率，以复现 checkpoint-444 的
artifact 生成口径；不要在复现实验中单独改动其中一个阶段的 image processor 参数。

### 12.2 margin+KL v1/v2

1. 先用 SFT_FINAL selected checkpoint 生成固定 z_ref cache：

```bash
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
ZREF_ADAPTER=/scratch/sft_final/checkpoint-444 \
ZREF_CACHE_PATH=/scratch/zref_cache.pt \
torchrun --nproc_per_node=8 training/margin_kl/dump_zref.py
```

2. 启动 v1 或 v2。两者都从 SFT_FINAL adapter + calibrated head 初始化：

```bash
export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
export SFT_INIT_CKPT=/scratch/sft_final/checkpoint-444
export DISCRIM_MUSD_PATH=/scratch/discrim_musd.pt
export ZREF_CACHE_PATH=/scratch/zref_cache.pt

OUTPUT_DIR=/scratch/margin_kl_v1 bash training/margin_kl/run_v1.sh
OUTPUT_DIR=/scratch/margin_kl_v2 bash training/margin_kl/run_v2.sh
```

3. 评测论文 selected checkpoints：

```bash
BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
CKPT_DIR=/scratch/margin_kl_v1/checkpoint-50 \
bash training/margin_kl/eval_closed12.sh

BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
CKPT_DIR=/scratch/margin_kl_v2/checkpoint-60 \
bash training/margin_kl/eval_closed12.sh
```
