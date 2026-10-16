# Grouping ablation 训练记录

- validation-best checkpoint: `/user/huangyongbin/newrepo/agriabstain/h27experiment/runs/grouped/seed-21/checkpoints/checkpoint-222`
- best epoch: 2
- best validation loss: 0.296342
- final training loss: 0.3922883345558121
- training time: 2286.1697 seconds
- loss curve: `/user/huangyongbin/newrepo/agriabstain/h27experiment/runs/grouped/seed-21/selection/loss_curves.png`
- 分析：Validation loss reached its minimum before the final epoch; extending training is not supported and would risk overfitting.
- 决策：保持与主方法完全相同的六 epoch recipe，不通过增加 epoch 优化结果。
