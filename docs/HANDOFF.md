# 后续工作交接

## 最新范围修正

用户明确要求 Git 仓库不包含论文文件和交付包。`paper/`、论文配图 `figure/`、`submission_snapshot/` 已从当前 Git 版本移除，保留在本机忽略目录；对应快照验证脚本也移除。下文这些目录的路径与核验结果仅作历史记录。标准代码测试以根 README 为准，排除依赖本地论文的 12 项一致性测试。初始两个 commit 已推送，旧 Git 历史仍包含被移除文件；当前未重写历史。

## 当前结论

已建立当前 ReGate AAAI 2027 稿件对应的个人研究仓库，代码主要从原工作区原样复制，必要依赖已补齐。核心算法、论文、标注和报告数值未改动。已完成代码覆盖、内部 import、外部路径和 CPU 测试检查；未重跑训练或推理。

- 本地基线：`d181ee4`，`chore: archive paper-aligned ReGate research repository`。
- 远端：`git@github.com:1105425455/agriabstain.git`；截至本交接尚未 push。
- 复制来源清单：2,540 个文件；外部资产清单：362 个有效链接。这些是本次整理时的统计。
- 完整测试：296 passed、2 skipped、0 failed。最终日志：`docs/verification/regate-commit-tests.log`。
- 交付快照保持原样，原验证器为 17/18 通过；唯一失败是已有中文注释门禁，哈希和数值校验通过。

## 先读取哪些文件

1. 根 README：目录、主线入口及资产管理边界。
2. `docs/LOCAL_CHANGES.md`：复制后实际修改的四份脚本/测试与补齐文件。
3. `docs/VERIFICATION.md`：首节是最新结果；后续失败记录属于修复前历史，不能当成当前失败。
4. `aaai_code/README.md`：已有代码的详细运行说明。
5. 当前论文：`paper/Learning_to_Locate_or_Abstain__A_Discriminative_Head_Architecture_for_Reason_Attributed_Selective_Visual_Grounding__1_/main.tex` 和同目录 `technical_supplement.tex`。

## 论文与产物对应

以下路径均相对仓库根：

| 用途 | 路径 |
|---|---|
| 正式训练/开发/测试标注 | `aaai_code/benchmark/annotations/` |
| Stage 1 / Stage 2 | `aaai_code/training/sft_final/{run_train.sh,calibrate_head.py}` |
| 正式 evaluator / metrics | `aaai_code/evaluation/eval_qwen3vl_3reason.py`、`aaai_code/reason_abstain/eval/metrics.py` |
| 最终模型 | `new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/` |
| Base/SFT/ReGate 主表与曲线 | `new_reason/outputs/base_native_regate_20260720/` |
| dev 选阈值 | `new_reason/outputs/matched_retention_dev_20260725/` |
| grouping 消融 | `aaai_code/experiments/grouping_ablation/{main,artifacts}/` |
| 表征与证据分析 | `new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/` |
| 层选择 | `aaai_code/experiments/layer_selection/`、`new_reason/outputs/head_boundary_refit/` |
| 层选择 samples/split | `new_reason/outputs/rl_grpo_trihead/rl_data/`；历史目录名，当前诊断需要 |
| prompt-interface 对照 | `new_reason/baselines/prompt_cot/` |
| GPT 对照 | `aaai_code/experiments/gpt56_sol_full/outputs/` |
| margin+KL | `aaai_code/training/margin_kl/`、`new_reason/outputs/sft_runs/paired_retrain/` |
| 交付快照 | `submission_snapshot/`，表图映射见其中 `REVIEWER_README.md` §8 |

## 安全的日常检查

从仓库根确认 Git 状态，随后进入 `aaai_code/` 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /user/huangyongbin/miniconda3/envs/agrivllm/bin/python \
  -m pytest -q -p no:cacheprovider \
  reason_abstain experiments scripts/test_build_anonymous_release.py
```

仅核查论文表格/正文及发布布局可将最后一行换成：

```text
experiments/base_native_regate/test_paper_metric_consistency.py scripts/test_build_anonymous_release.py
```

该子集 74 项通过。基线测试没有执行模型推理；不要把 shell 中的 `run_*.sh` 当成普通检查命令直接启动。

## 仍需了解的边界

- 本机链接依赖来源工作区；迁移时按 EXTERNAL_ASSETS 恢复文件。模型和图片不在 Git 中。
- 部分作者侧 JSON 和历史入口保留原绝对路径，完整性通过不等同于零配置迁移。
- Fig1/Fig2 成品和素材在仓库中，可编辑设计源文件尚未确认；不阻断代码接手。
- 当前没有待修复的 CPU 测试失败；下一步应按用户的新任务继续，勿自动重复整理或重跑 GPU 实验。
