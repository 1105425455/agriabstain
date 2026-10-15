# Grouping Ablation

该实验检验 same-source counterfactual grouping 的贡献。主方法训练集中的 positive
保持不变；每条 negative 保持原 prompt、target response、queried category、三态标签及
数据域，但图像替换为独立采样的 annotated source。

## 对照定义

- 总量保持 `7,072`：answer `2,703`、wrong_object `1,500`、no_target `2,869`。
- answer 记录逐字段保持不变。
- negative 的 query、assistant target、head/reason label 保持不变。
- independent source 不得出现在 benchmark 的 train/dev/test/dose source 集合中。
- sampled image 必须不包含 queried category。
- wrong_object 还必须包含原 T6 样本对应的 visible confusable category。
- 数据构造 seed、输入/输出 SHA256、source 使用分布写入 `artifacts/data/audit.json`。

这是真正的数据消融：只 shuffle 顺序、修改 `group_id/source_image_id` 或增大 batch
不会改变 sample-wise SFT objective，因此不构成 grouping ablation。

该对照同时改变 negative image provenance 和 source diversity，因此支持 matched
construction contrast；它不将 independently sourced imagery 的所有属性进一步拆开。

## 隔离边界

所有新数据、checkpoint、日志、校准头、预测和汇总均写入：

```text
aaai_code/experiments/grouping_ablation/artifacts/
```

脚本只读主仓库的 frozen SFT 数据、原始 annotations/images、base model、mu/sd 和
evaluation probes。不会覆盖 `benchmark/`、`new_reason/outputs/` 或现有 checkpoint。

`artifacts/` 是 author-side runtime 目录，包含 checkpoint、features、predictions 和本地
路径，不进入匿名发布包。交付使用的无路径 compact 结果位于
`main/grouping_ablation_fixed_gt.json`，其中明确以 `MTS_macro` 为论文口径。

## 运行

环境：`agrivllm`；8×4090；每卡 batch size 固定为 1，gradient accumulation 为 8。

```bash
source /user/huangyongbin/miniconda3/etc/profile.d/conda.sh
conda activate agrivllm
cd /user/huangyongbin/newrepo/agriabstain/aaai_code
bash experiments/grouping_ablation/run_pipeline.sh all
```

也可单独执行 `build|train|select|calibrate|eval` 阶段。训练 recipe 与 SFT_FINAL 相同：
LoRA `r=16, alpha=32, dropout=0.05`，LR `1e-4`，6 epochs，tri-head
`lambda=0.4`、warmup 10、class weight `[1.0,1.5,1.0]`，seed 42。

checkpoint 仅按固定 100-record dev subset 的 teacher-forced LM loss 选择。选择完成后，
在 independent-negative train representations 上重新拟合 calibrated head；不会复用主方法
的 calibrated head。

## 输出

- `artifacts/data/`：对照训练集、negative assignment、审计。
- `artifacts/checkpoints/`：LoRA 和 training-stage head。
- `artifacts/selection/`：每 epoch train/dev loss、loss curve、best checkpoint。
- `artifacts/calibration/`：representation shards 和 refit 日志。
- `artifacts/evaluation/`：SFT native 与 ReGate 预测/指标。
- `artifacts/summary/comparison.json`：与 frozen grouped baseline 的逐数据集差值。
- `main/grouping_ablation_fixed_gt.json`：去除本地路径的 canonical paper artifact。

## 测试

```bash
pytest experiments/grouping_ablation -q
ruff check experiments/grouping_ablation
```
