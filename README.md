# AgriAbstain / ReGate 研究仓库

对应论文 **ReGate: Mitigating Hallucinations in Visual Grounding via Metamorphic Counterfactual Supervision**。仓库包含正式代码、配置、标注、预测、审稿分析，以及论文使用的最终模型和不可公开恢复的数据。论文TeX/PDF、配图及交付包不在当前版本中。

**新agent先读 [AGENTS.md](AGENTS.md)、[交接说明](docs/HANDOFF.md) 和 [资产恢复说明](docs/ASSET_RECOVERY.md)。**

## 下载与恢复

需要Git LFS。最终资产共2,777个文件、约4.51 GB；当前清单检查识别1,537个LFS路径（路径数与去重后的上传对象数不同）。

```bash
git lfs install
git clone git@github.com:1105425455/agriabstain.git
cd agriabstain
git lfs pull
python tools/verify_remote_assets.py --hash
python tools/restore_assets.py --execute
```

Qwen3-VL-8B-Instruct基座和COCO原图单独下载，入口、路径及校验方式见资产恢复说明。PlantDoc/PlantWild精确图片、加工标签与派生图已包含。恢复脚本会解压派生图并创建兼容目录关联。

## 保存范围

| 内容 | 位置 |
|---|---|
| 正式构造、训练、校准、推理、评测代码 | `aaai_code/` |
| Benchmark标注及图片校验清单 | `aaai_code/benchmark/annotations/`、`IMAGE_MANIFEST.json` |
| 主方法最终LoRA与calibrated head | `new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/` |
| Independent最终checkpoint | `aaai_code/experiments/grouping_ablation/artifacts/checkpoints/independent_negative_seed42/checkpoint-222/` |
| Margin+KL选定checkpoint | `new_reason/outputs/sft_runs/paired_retrain/{v1/checkpoint-50,v2/checkpoint-60}/` |
| 三次重训选定checkpoint与已有结果 | `research_assets/h27experiment/` |
| 精确图片、加工标注、派生图归档 | `research_assets/` |
| 审稿回应及可复算证据 | `review.md`、`docs/review_evidence/`、`tools/review_*.py` |
| 最终远端资产白名单 | `docs/FINAL_ASSETS.json` |

中间epoch权重、optimizer/RNG状态和表征缓存不随Git提供。可从最终adapter开展新的训练或评测；CPU head重新拟合需要先重建缓存。既有预测与最终head可以直接读取。历史清单 `COPY_MANIFEST.json`、`EXTERNAL_ASSETS.json` 描述来源工作区，不能当成当前上传清单。

## 代码入口

从 `aaai_code/` 执行，参数详见 [开发说明](aaai_code/README.md)：

- Stage 1：`training/sft_final/run_train.sh`。
- Stage 2：`training/sft_final/calibrate_head.py`。
- 评测：`evaluation/eval_qwen3vl_3reason.py`、`evaluation/run_sft_final.sh`。
- 主表与阈值：`experiments/base_native_regate/`。
- 构造消融：`experiments/grouping_ablation/`。
- 表征/证据分析：`experiments/{evidence_judge,layer_selection}/`。

通过`BASE_MODEL`、`ADAPTER`、`OUTPUT_ROOT`等参数指定模型和输出目录。旧日志和部分历史入口保留原绝对路径，运行前检查；不要默认另一台机器有原作者目录。

## 验证

作者环境为Python 3.11，依赖见`aaai_code/requirements.txt`及`aaai_code/environment/`。本机可使用`/user/huangyongbin/miniconda3/envs/agrivllm/bin/python`；其他机器使用自己配置的环境。

```bash
cd aaai_code
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider \
  reason_abstain experiments scripts/test_build_anonymous_release.py \
  --ignore=experiments/base_native_regate/test_paper_metric_consistency.py
```

已有完整测试基线为296 passed、2 skipped，包含12项需要本地论文的测试。上面的代码仓库命令排除这12项。CPU审稿实验与结果核验见 [CPU完成清单](docs/review_evidence/CPU_COMPLETION.md)；资产上传及取回核验见 [传输记录](docs/ASSET_TRANSFER_STATUS.md)。

首次接手请以`git status`、`git log`和`git remote -v`确认实际版本。论文仍在作者本地/Overleaf管理，不要重新加入Git。
