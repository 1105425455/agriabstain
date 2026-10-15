# Grouping ablation 训练记录

- validation-best checkpoint: `/user/huangyongbin/newrepo/agriabstain/new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444`
- best epoch: 4
- best validation loss: 0.278026
- final training loss: 0.3867527295698275
- training time: 2343.5684 seconds
- loss curve: `experiments/grouping_ablation/artifacts/grouped_dev_selection/selection/loss_curves.png`
- 分析：Validation loss reached its minimum before the final epoch; extending training is not supported and would risk overfitting.
- 决策：保持与主方法完全相同的六 epoch recipe，不通过增加 epoch 优化结果。
