# 本次离线审稿证据

新增CPU实验和完整完成情况见 [CPU_COMPLETION.md](CPU_COMPLETION.md)。本轮包括缓存上的小head CPU训练、L2几何审计、冻结基座对照和历史重训结果复核；未使用GPU。前次仅离线统计的范围由该完成清单更新。

入口为仓库根 `review.md`，按“review 意见：回复的内容”组织。全部结果读取现有文件，未训练、推理、访问付费API或修改原标注。

| 文件 | 用途 |
|---|---|
| `unified_metrics.csv` / `.json` | 44行统一指标；正式score函数、source路径与SHA-256 |
| `paired_selection.json` | SFT/ReGate回答集合配对与有效候选质量 |
| `threshold_provenance.json` | 原dev阈值选择记录，含2,408 dev/2,646 test规模及原来源hash |
| `data_and_gpt.json` | 训练重复/类别组成，GPT请求配置和parser invalid统计 |
| `category_composition.md` | 实际训练query标签的三态计数，含重复 |
| `gate_loss_curves.csv` | 两组日志的17个实际评估点，无插值epoch |
| `gate_protocol_provenance.json` | 两组日志样本数/类别数，以及当前代码recipe摘录与hash |
| `l2_breakdown.json` | 395个L2负例的query与实际crop目标逐条追溯，及既有配对概率分析 |

仓库根复算：

```bash
PYTHONDONTWRITEBYTECODE=1 /user/huangyongbin/miniconda3/envs/agrivllm/bin/python tools/review_metrics.py
python tools/review_data_audit.py
python tools/review_gate_logs.py
python tools/review_l2.py
```

新增脚本 Ruff E/F/I/UP 检查通过。现有 `test_native_regate.py`、`test_rescore_metrics.py`、`test_selective_grounding.py` 合计75 passed。本次没有用人工审计、训练seed重复或未运行baseline补造结论。

注意：L2构造问题和gate拟合稳定性问题只是发现并记录，原数据、原论文结果、原模型保持原样；这些问题不能被本次统计视为已经修复。
