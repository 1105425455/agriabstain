# Grouping ablation 最终报告

## 结论

最终 grouped SFT（checkpoint-444）整体显著优于 independently sampled negatives。独立负样本仅在 RefCOCOg 上稳定获益，但在 comprehensive、PlantWild 与 L1/L2/L3 controlled diagnostics 上退化，尤其削弱了 contradiction / evidence-sensitive abstention。结果支持论文的核心主张：收益来自 same-source grouping，而不只是负样本数量、类别或状态分布。

## 实验控制与训练

- 总训练样本 7,072：`answer=2,703`、`wrong_object=1,500`、`no_target=2,869`。
- 2,703 个 positive 完全不变；4,369 个 negative 全部替换。
- 独立负样本来自 1,776 个与 benchmark/positive source 均不重叠的 source。
- 数量、category-state strata、state distribution 与 prompt target 均通过审计；所有媒体路径存在。
- 使用与最终 grouped SFT 相同的 6-epoch recipe、objective、architecture 和 hyperparameters，seed=42。
- validation-best 为 checkpoint-222（epoch 2），validation loss 0.289195；训练耗时 2,473.4 秒。
- calibrated head 在 dev 上 MCC=0.3614、accuracy=54.0%；`wrong_object` recall 仅 9.09%，表明 independent negatives 学到的 representation 难以区分 visual contradiction。

## 与最终 grouped SFT 的同口径比较

下表均为百分点；`Δ = independent - grouped`。MTS 使用与论文一致的 relation macro；MTS、MCC 越高越好，FCA 越低越好。

| Interface | Probe | MTS grouped / independent (Δ) | MCC grouped / independent (Δ) | FCA grouped / independent (Δ) |
|---|---|---:|---:|---:|
| Native | Comprehensive | 80.68 / 75.59 (-5.09) | 75.77 / 65.98 (-9.80) | 9.45 / 19.27 (+9.83) |
| Native | RefCOCOg | 82.60 / 87.57 (+4.97) | 85.77 / 92.27 (+6.50) | 4.42 / 3.87 (-0.55) |
| Native | PlantWild | 64.77 / 60.79 (-3.98) | 60.24 / 44.91 (-15.33) | 15.53 / 32.20 (+16.67) |
| ReGate | Comprehensive | 81.95 / 52.67 (-29.28) | 84.85 / 52.31 (-32.54) | 13.23 / 48.60 (+35.37) |
| ReGate | RefCOCOg | 81.22 / 86.74 (+5.52) | 92.03 / 94.04 (+2.01) | 8.29 / 5.52 (-2.76) |
| ReGate | PlantWild | 67.27 / 61.85 (-5.42) | 56.00 / 34.96 (-21.04) | 26.89 / 25.00 (-1.89) |
| ReGate | L1 discrimination | 65.12 / 50.24 (-14.88) | 39.32 / 8.54 (-30.79) | 61.46 / 76.59 (+15.12) |
| ReGate | L2 evidence | 62.15 / 41.01 (-21.14) | 56.97 / 9.07 (-47.90) | 47.59 / 87.85 (+40.25) |
| ReGate | L3 dose | 80.36 / 50.15 (-30.21) | 83.55 / 31.78 (-51.77) | 17.57 / 83.11 (+65.54) |

补充的 mechanism-sensitive 结果：

- Native Comprehensive T6：50.73% → 8.78%（-41.95 pp）。
- Native PlantWild OOD confusion：65.83% → 27.50%（-38.33 pp）。
- ReGate Comprehensive T6：57.07% → 26.34%（-30.73 pp）。
- ReGate PlantWild 虽然 FCA 略降 1.89 pp，但 A_FU 从 14.58% 升至 39.58%，MCC 同时下降 21.04 pp，不能视为整体改善。

指标口径：`U_Reason` 以全部 unsupported GT 为分母，要求模型拒答且 reason 正确；
`U_Reason_c` 才是预测拒答条件下的 reason accuracy。FCA 与 U_Abs 仅在
unsupported invalid 数为零时互补。

## 解释

Independent negatives 更容易由跨图像 appearance/domain cues 区分，因此可提升 RefCOCOg 这类 broad OOD absence probe；但它们没有提供同一视觉内容下“query 改变而 evidence 保持”的受控反事实。移除 same-source grouping 后，模型在 T6、PlantWild confusion 和 L2/L3 上出现一致的大幅下降，说明 grouping 主要改善的是 query-evidence discrimination，而非简单增加 negative exposure。

该实验目前只有一个训练 seed（42）。论文中可将其报告为严格受控的 ablation，但不应声称已估计跨 seed 方差。

## 产物

- 数据审计：`artifacts/data/audit.json`
- 最佳 checkpoint：`artifacts/checkpoints/independent_negative_seed42/checkpoint-222`
- 训练记录与曲线：`artifacts/selection/process.md`、`artifacts/selection/loss_curves.png`
- 全部评估：`artifacts/evaluation/`
- 机器可读比较：`artifacts/summary/comparison.json`
- 论文 LaTeX 草稿：`paper_grouping_ablation.tex`
