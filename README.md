# AgriAbstain / ReGate 论文研究仓库

对应当前论文 **ReGate: Mitigating Hallucinations in Visual Grounding via Metamorphic Counterfactual Supervision**（AAAI 2027 submission）。本仓库从作者现有工作区复制整理，保留开发代码、论文、原始结果与提交快照之间的对应关系。

## 内容与入口

| 路径 | 内容 |
|---|---|
| `aaai_code/` | 当前论文的 benchmark 构造、两阶段训练、推理、评测、机制实验及现有测试 |
| `paper/` | 当前正文、technical supplement、checklist、参考文献与最终图表 |
| `figure/` | 任务示意图素材及源图片清单 |
| `new_reason/outputs/` | 与论文相关的作者侧配置、日志、逐样本预测与分析结果；大型权重及特征为本地链接 |
| `new_reason/baselines/prompt_cot/` | 当前 prompt-interface 对照的现有代码与结果 |
| `submission_snapshot/` | 已交付的匿名代码数据包快照，保留原 RELEASE_MANIFEST；与作者开发版分开保存 |
| `docs/COPY_MANIFEST.json` | 每份复制文件的原路径、字节数和 SHA-256 |
| `docs/EXTERNAL_ASSETS.json` | 数据、模型、特征等外部资产的位置及本地链接 |
| `docs/history/` | 原有研究记录，选点和指标以当前论文与正式结果为准 |

训练和评测从 `aaai_code/` 执行，具体参数沿用 [开发版说明](aaai_code/README.md)：

- Stage 1：`training/sft_final/run_train.sh`。
- Stage 2：`training/sft_final/calibrate_head.py`。
- 推理与评测：`evaluation/eval_qwen3vl_3reason.py`、`evaluation/run_sft_final.sh`。
- 主表与曲线：`experiments/base_native_regate/`。
- 分组消融：`experiments/grouping_ablation/`。
- 表征与证据分析：`experiments/evidence_judge/`、`experiments/layer_selection/`。
- margin+KL：`training/margin_kl/`。

## 数据与模型

当前服务器通过 Git 忽略的本地符号链接使用原研究工作区的数据、派生图片、checkpoint 和特征；Git clone 不会下载这些资产。完整位置见 `docs/EXTERNAL_ASSETS.json`，图片清单见 `aaai_code/benchmark/IMAGE_MANIFEST.json`。基座为 `Qwen3-VL-8B-Instruct`，正式 endpoint 为 `fusion_3reason_v3_hl04/checkpoint-444` 与其中的 `discrim_head_calibrated.pt`。

迁移机器时需要按资产清单恢复大文件，并通过现有脚本的环境变量指定模型和产物目录。原始研究代码与 JSON 中保留了作者真实路径，以便追溯；部分历史入口仍依赖原工作区位置。匿名快照的可移植复现说明见 [REVIEWER_README](submission_snapshot/REVIEWER_README.md)。

## 版本边界

本次整理直接复制现有实现，不重新实现算法、不改变论文数值、不混入早期 SFT/RL 方案或后续多 seed、linear-gate 实验。`asrabstain/` 属于独立项目。

当前论文图文件齐全；Fig1/Fig2 的可编辑设计源文件尚未在论文图目录中发现。匿名包对分组消融及部分表征实验仅保存 compact 结果；作者侧相应材料在本仓库的结果目录和外部资产清单中另行追溯。

## 检查

提交前完整 CPU 测试在记录的 `agrivllm` Python 3.11 环境下为 **296 passed、2 skipped，零失败**。已适配当前论文格式及现有发布布局，保留产物数值断言；详情及原始日志见 [核验记录](docs/VERIFICATION.md)。

```bash
cd aaai_code
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /user/huangyongbin/miniconda3/envs/agrivllm/bin/python \
  -m pytest -q -p no:cacheprovider reason_abstain experiments scripts/test_build_anonymous_release.py
```

复制文件的校验结果、测试结果及已知限制见 `docs/VERIFICATION.md`。提交快照的 SHA-256 清单仅适用于 `submission_snapshot/`，不用于覆盖或解释作者代码。
