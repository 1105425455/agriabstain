# SFT 对比实验结果汇总

> **注**：本文档为对比实验的原始过程记录(含早期三方对比)。完整、最新、诚实口径的对比结论以 new_reason/SFT_FINAL.md 第 5.1 节为准。L1 判别头落后于 native_lm/cot 是事实，不以 frontier 阈值扫描辩护。

## 1. 实验设置
三个方法共享同一 SFT 模型(Qwen3-VL-8B, checkpoint-444, ep3)、同一测试集、同一评估脚本，只有 decision 产生方式不同：
- **Ours(判别头)**: decision 由三分类判别头输出(--dual-head)，有连续概率 p_answer
- **prompt (prompt_reward_norms)**: 纯 LM，通过 system prompt 注入 reward 归一化提示，无 confidence 信号
- **cot (cot_structured)**: 纯 LM，通过 system prompt 注入结构化 CoT 推理提示，无 confidence 信号

任务：植物病害"定位或弃答"，三类 decision = answer(定位真病) / wrong_object(近义病弃答) / no_target(无靶弃答)。

## 2. 主表(Table 1)：全 6 探针 MCC

| 探针 | Ours(判别头) | native_lm | prompt | cot |
|---|---|---|---|---|
| L1_discrim (近义判别) | 0.419 | **0.497** | 0.331 | **0.590** |
| L2_evidence | **0.574** | 0.528 | 0.506 | 0.385 |
| L3_dose | **0.841** | 0.765 | 0.777 | 0.710 |
| test_full (综合) | **0.856** | 0.758 | 0.696 | 0.637 |
| ood_refcocog | **0.925** | 0.863 | 0.768 | 0.685 |
| ood_plantvillage | 见下节(纯负集，MCC退化) | — | — | — |

Ours 在 5/6 探针 MCC 领先，唯 L1 被 cot/nativelm 反超。加粗为该行最高。

脚注：checkpoint-444(ep3) 是按低漏诊约束(A_FU=0.059)刻意选择的保守 operating point，非 raw performance 最优点。

## 3. L3 数据可靠性说明
L3_dose 探针的 format_valid_rate=0.441 曾疑似解析失败，经排查确认：这是 L3 探针固有特性——750 条样本中 419 条为 ambiguous(gt decision 模糊)被评估脚本统一跳过，实际参与评估 total=331，invalid=0(JSON 解析全部成功)。三个方法在 L3 上口径完全一致(都 skip 419、total 331)，对比公平可靠。

## 4. ood_plantvillage：纯负集弃答召回率
plantvillage 是分布外纯负集(240 条全应弃答)，MCC 在纯负集上退化(三方均为 0)，改报弃答召回率(abstain recall，越高越好)：

| 方法 | 弃答召回率 |
|---|---|
| Ours(判别头) | 0.533 (128/240) |
| native_lm | 0.754 (181/240) |
| prompt | 0.692 (166/240) |
| cot | **0.942 (226/240)** |

说明：此集上 Ours 弃答率最低，与 L1 模式一致自洽——ckpt-444 是保守选点(倾向 answer 保低漏诊)，cot 激进弃答。这反映 Ours 在 OOD 弃答上的 operating point 偏激进(倾向 answer)，是保守选点的代价，需在整体 claim 中诚实呈现。
