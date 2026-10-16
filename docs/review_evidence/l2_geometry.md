# L2 cropout 负样本几何审计（离线诊断）

> 性质声明：本报告为**离线审计子集**，复用原 record 次序与派生图、标注几何计算，
> 不是正式 benchmark 修复，也不代表人工逐图目检。**“无已标注 query 残留”仅指
> 标注几何意义下 query 类别的 bbox 不在 crop 区域中，不能等同人工确认 query 完全
> absent**。存在未覆盖的标注缺失/类别映射不明时会归入“缺信息”并明确说明。

## 核心结论

- `build_eval_L2.py` 按同图 `keep_rows × crop_rows` Cartesian 配对（循环嵌套），
  生成的 395 条 `L2_cropout` 中 **247 条 query 与所裁掉目标类别不同**（交叉 label），
  148 条一致（同 instance 的“设计意图”配对）。
- 全部 395 条 crop 的裁剪区域重建（零档 `save_visibility_ladder_crop` 公式）
  与派生 crop 图实际尺寸**逐一核对一致**（geometry_verified=True）。
- 原标签下 FCA = 0.4759（模型在 395 条 abstain 目标上 answered
  188/395）。
- 几何审计下，395 条中 **136
  条无已标注 query 残留**（其中 same-label 73、
  cross-label 63）；
  104 条完整残留、
  155 条部分残留。
- 交叉 label 的 247 条 FCA=0.6559（answered 162/247），
  模型在大量 query 对象其实仍在图中（完整/部分残留）的样本上 answered，这部分“负例标签”几何上并不可靠。
- geometry-certified-no-annotated-query 子集（136 条）
  FCA=0.1471。
- 严格同 instance 去重 clean 子集 148 条
  （全部可几何核对；其中无已标注 query 残留 73 条），
  FCA=0.1757。
- matched pairs 全指标（clean 子集）：keep answer 准确率
  0.8108，crop abstain 准确率
  0.8243，双侧均对
  104/148。
- L3 endpoint：148/148
  个唯一实例在图面上与 L2 cropout 重合；同 image/query/source 目标可比的记录
  148 条，prediction 决策差异
  0。

## 计数

| 集合 | 记录数 | answered | FCA |
|---|---|---|---|
| 全部 cropout（原标签） | 395 | 188 | 0.4759 |
| same-label（query==裁掉目标） | 148 | 26 | 0.1757 |
| cross-label（query!=裁掉目标） | 247 | 162 | 0.6559 |
| geometry-certified 无 query 残留 | 136 | 20 | 0.1471 |
| clean 同 instance 去重 | 148 | 26 | 0.1757 |

## 逐条数据

- JSONL 记录：`docs/review_evidence/l2_geometry_records.jsonl`
- CSV 概览：`docs/review_evidence/l2_geometry_records.csv`
- 汇总 JSON：`docs/review_evidence/l2_geometry_summary.json`

## 输入源与哈希

| 文件 | sha256 |
|---|---|
| aaai_code/benchmark/annotations/eval/test_L2_evidence.jsonl | 10ffdeeadeccadee390b4a82b7ff9b8dd7f4da9328553e43643c823d09b7b453 |
| aaai_code/benchmark/annotations/splits/test.jsonl | 59240265e91033127ba3c493cca41af069aff3256a649c503620b3ce68f379c4 |
| new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/eval_faithful_v1/L2_evidence/predictions.jsonl | dd70de973ff37c29fdc3f3e91755c4064fa49c87c3b0be11de72b42839b32e07 |
| new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/eval_faithful_v1/L3_dose/predictions.jsonl | 5848e36b7f1f137c546926c0eb59ad0796f46586592522d040753e3d4f9f5745 |
| aaai_code/benchmark/data/source/coco/annotations/instances_selected.json | 289e363d97805425ec3d042733d8d7d4b41a6a7d9d6f69f3bcf48226cffb4d7c |
| aaai_code/benchmark/annotations/raw/plantdoc_full.raw.config.snapshot.yaml | 9e0fcda4fb43c043c2a6a6809316f7a6ad2bfc8336bb5ac0c51ca8ecb877bf42 |

- 唯一源图数：89；PlantDoc txt 标注即时读取，COCO 标注取自 instances_selected。
- 裁剪区域重建复用 `reason_abstain.bench.transforms` 的 `qwen_box_to_xyxy`/`_clip_xyxy`
  及 `save_visibility_ladder_crop` 零档坐标公式；图片仅用 PIL 读取尺寸。

## 已知限制

- “部分/完整残留”依赖 PlantDoc txt 与 COCO `instances_selected.json` 标注质量。
- 主统计对任何正交集都计残留，不将小于1%的面积默认为完全排除。
  bbox量化与像素取整可能导致小边界交集；逐条比例见JSONL，容差分析单独报告。
- 交叉 label 记录中，L3 的 prompt（crop 自身 label）与 L2 不同，prediction 仅在同 label 时可直接比较。
- 本审计不涉及任何人工目检、模型训练或推理 API。
