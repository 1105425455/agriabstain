# 双人标签审计操作说明

审计清单为 `label_audit_prep.csv`，三个状态各200条，共600条。仅完成抽样准备，尚无人工标注结果。

## 独立标注

两位标注者分别查看相同image-query，不查看模型预测、不互看答案。组织者应在分发视图中隐藏 `state` 与构造标签，保留样本ID、图像和query；原CSV保留这些字段用于之后比对。

每人填写：

- `answer`：query目标在图中可见且可定位。
- `wrong_object`：可识别对象与query的细粒度类别冲突；需要明确的视觉或领域依据。
- `no_target`：图中无法定位query目标。
- `uncertain`：图像/专业知识不足以作出可靠判断，不强迫三选一。

建议在标注字段中附简短理由，并另外记录多实例、共现病害、裁剪残留和疑似原始标注遗漏。disagreement和uncertain交由第三位具备领域经验的复核者处理，保留两位原始答案。

## 抽样与统计

- 测试split总体有answer1,323、wrong_object205、no_target1,118条。wrong_object抽200条；no_target中优先抽COCO负例，配置为120/200。
- 去重单位是image-query，多个样本可能来自同一source image；三个状态样本并非600张独立图像。wrong_object仅75个source image，无法同时满足200条与source唯一。
- 估计总体标签错误率时使用CSV中分层权重，分别报告各状态/来源/变换。计算Cohen's kappa与原始一致率时说明uncertain处理方式；置信区间按source image聚类bootstrap。
- 标签错误率本身不构成所有模型FCA的普适下界。需区分参考标签错误方向，并说明对遵循真实视觉状态的系统产生的测量偏差。

可在完成标注后补充双人一致率、复核后错误率和具体错误类型；当前不填任何人审数值。
