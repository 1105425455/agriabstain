# PlantWild Paired Benchmark (OOD_answer + OOD_absent + OOD_confusion)

## 概述

同图三路 paired benchmark，基于 PlantWild 156 张 eval 图像（PV13 13 类 × 12 张），
每张图配 **answer** + **absent** 两条探针，apple/corn/tomato 类另配 **confusion** 探针，
合计 **444 条**。

### 三类探针

| 探针 | 数量 | Query | GT | 测试目标 |
|------|------|-------|----|----------|
| OOD_answer | 156 | 图像真实 PV13 病害 | decision=answer, label=PV13, box=None | over-abstain（不应错误 abstain） |
| OOD_absent | 156 | 跨作物病害（另一作物） | abstain / no_target | no_target 正确率 |
| OOD_confusion | 132 | 同作物不同病害 | abstain / wrong_object | wrong_object 正确率 |

### 同图 Paired 设计

- 每张图恰好有 **1 条 answer + 1 条 absent**；apple/corn/tomato 图另有 **1 条 confusion**。
- potato/squash 仅 1 个病害类，无 confusion 探针。
- 排序按 base image 文件名稳定排列，每张图顺序为 answer → absent → confusion（若有）。
- 模型按 batch 处理时仅关心单条记录，顺序不影响结果。

## 文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 构造脚本 | `new_reason/scripts/build_ood_plantwild_paired.py` | 读取 262k JSONL → 新增 answer 记录 → 输出 paired JSONL |
| 输出 JSONL | `outputs/eval_probes/ood_plantwild_paired_262k.jsonl` | 444 条评估记录 |
| 统计/audit | `outputs/ood_candidates/plantwild_benchmark/paired_build_stats.json` | 构造统计 |
| 图像目录 | `outputs/ood_candidates/plantwild_benchmark/eval_images_262k/` | 156 张 262k resize 图像（不复制，引用绝对路径） |

## 构造方法

### answer 真实 label 恢复

从图像文件名解析 PlantWild 原始类名，经 `PW_TO_PV13` 映射得到项目统一 PV13 名称：

```python
# 文件名: pw_tomato_leaf_mold_4.jpg
# 解析: wild_cls = "tomato leaf mold"  →  PW_TO_PV13["tomato leaf mold"] = "tomato mold leaf"
```

断言：所有解析结果必须在 `PW_TO_PV13` 映射中，否则构造失败。

### 负样本复用

OOD_absent 和 OOD_confusion 记录直接复用 `ood_plantwild_262k.jsonl` 的 288 条记录，
仅修改 `metadata.dataset` 从 `plantwild_ood` 为 `plantwild_ood_paired`。
messages、solution、relation、reason、image 路径完全保留。

## 评估指标

评估时使用 `--system-override decision` 配置。

### 主指标

| 指标 | 说明 |
|------|------|
| **MCC** | Matthews 相关系数（二分类 answer vs abstain） |
| **A_FU** | 可回答样本的 answer 准确率（over-abstain 检测） |
| **AAC** | answer 样本的 decision + label 联合准确率 |
| **FCA** | 所有 abstain 样本的正确 abstain 率 |
| **HOR-FCA** | 高危 abstain 子集（reason=no_target）的 FCA，分母=156 |
| **U_Abs** | 非目的性的无条件 abstain 率 |

### 子集指标

| 指标 | 分母 | 说明 |
|------|------|------|
| OOD_answer strict | 156 | decision=answer + label 匹配 |
| OOD_absent strict | 156 | decision=abstain + reason=no_target |
| OOD_confusion strict | 132 | decision=abstain + reason=wrong_object |
| wrong_object conditional | 132 | confusion 子集中 reason=wrong_object |
| no_target conditional | 156 | absent 子集中 reason=no_target |

### 不报告

- **A_Ground**：无 GT box（answer 的 box=None），不评估 grounding/location。

### 评价口径

`eval/metrics.py` 对 GT answer + box=None：只要 prediction decision=answer 且 label 匹配即计 TP，不评 IoU。
因此 PlantWild 的 A_FU / AAC 可正常测量。

## 自检结果

构造脚本内置 16 项自动自检，全部通过：
1. total_records=444 ✓
2. answer=156, absent=156, confusion=132 ✓
3. unique_images=156 ✓
4. 每图恰好 answer+absent；confusion 仅出现在 apple/corn/tomato ✓
5. answer label 均为有效 PV13 名称 ✓
6. answer metadata (decision/label/box/reason) 正确 ✓
7. 负样本 decision/reason 正确 ✓
8. 13 类各 12 条 answer ✓
9. 所有图片路径指向 eval_images_262k ✓
10. HOR 分母=156（no_target 计数）✓
11. 原有负样本 messages/solution/relation/reason/image 完全保留 ✓
12. answer 不进入 HOR ✓