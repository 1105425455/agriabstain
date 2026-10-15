# Grouping ablation 训练记录

- validation-best checkpoint: `/user/huangyongbin/newrepo/agriabstain/aaai_code/experiments/grouping_ablation/artifacts/checkpoints/independent_negative_seed42/checkpoint-222`
- best epoch: 2
- best validation loss: 0.289195
- final training loss: 0.351122775503823
- training time: 2473.3988 seconds
- loss curve: `/user/huangyongbin/newrepo/agriabstain/aaai_code/experiments/grouping_ablation/artifacts/selection/loss_curves.png`
- 分析：Validation loss reached its minimum before the final epoch; extending training is not supported and would risk overfitting.
- 决策：保持与主方法完全相同的六 epoch recipe，不通过增加 epoch 优化结果。
