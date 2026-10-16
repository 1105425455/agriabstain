# AgriAbstain / ReGate 研究仓库

对应论文 **ReGate: Mitigating Hallucinations in Visual Grounding via Metamorphic Counterfactual Supervision**。仓库包含正式代码、配置、标注、预测、审稿分析，以及论文使用的最终模型和不可公开恢复的数据。论文TeX/PDF、配图及交付包不在当前版本中。

**接手先读本README；agent另读 [AGENTS.md](AGENTS.md)。** 具体实验操作查 [代码说明](aaai_code/README.md)，审稿结论查 [review.md](review.md)。

## 下载与恢复

权重和大型数据存放在仓库的 **Git LFS** 中，没有单独创建GitHub Release。最终资产共2,777个文件、约4.51 GB，清单见 [FINAL_ASSETS.json](docs/FINAL_ASSETS.json)。

```bash
git lfs install
git clone git@github.com:1105425455/agriabstain.git
cd agriabstain
git lfs pull
python tools/verify_remote_assets.py --hash
python tools/restore_assets.py --execute
```

恢复脚本解压派生图并创建兼容目录关联。仅下载主方法权重时，可先使用 `GIT_LFS_SKIP_SMUDGE=1 git clone ...`，再运行：

```bash
git lfs pull --include='new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/**,research_assets/discrim_musd.pt'
```

部分下载适合只取模型；完整资产校验和数据恢复仍需先拉取全部LFS文件。

### 另外下载什么

| 资源 | 获取与放置 |
|---|---|
| Qwen3-VL-8B-Instruct基座 | [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)，下载后设置`BASE_MODEL` |
| COCO原图 | [官方下载](https://cocodataset.org/#download)，按`aaai_code/benchmark/IMAGE_MANIFEST.json`中的文件名与SHA-256恢复源COCO和RefCOCOg图片 |
| PlantDoc/PlantWild精确副本 | 已提供，分别为`research_assets/exact_images/plantdoc/`的726张和`plantwild_closed12/`的144张 |
| 加工标签、RefCOCOg元数据和派生图 | 已提供，位于`research_assets/`；派生图归档为`datasets/derived_images.tar.gz` |

COCO图片放入`aaai_code/benchmark/data/source/coco/images/`和`aaai_code/benchmark/data/ood/refcocog/images/`。已有公共图片目录可用`python tools/restore_assets.py --execute --public-data-root /path/to/public-data`连接；该目录应包含上述`source/coco/images`和`ood/refcocog/images`相对布局。

## 保存范围

| 内容 | 位置 |
|---|---|
| 正式构造、训练、校准、推理、评测代码 | `aaai_code/` |
| Benchmark标注及图片校验清单 | `aaai_code/benchmark/annotations/`、`aaai_code/benchmark/IMAGE_MANIFEST.json` |
| 主方法最终LoRA与calibrated head | `new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/` |
| Independent最终checkpoint | `aaai_code/experiments/grouping_ablation/artifacts/checkpoints/independent_negative_seed42/checkpoint-222/` |
| Margin+KL选定checkpoint | `new_reason/outputs/sft_runs/paired_retrain/{v1/checkpoint-50,v2/checkpoint-60}/` |
| 三次重训选定checkpoint与已有结果 | `research_assets/h27experiment/` |
| 精确图片、加工标注、派生图归档 | `research_assets/` |
| 审稿回应及可复算证据 | `review.md`、`docs/review_evidence/`、`tools/review_*.py` |
| 最终远端资产白名单 | `docs/FINAL_ASSETS.json` |

中间epoch权重、optimizer/RNG状态和表征缓存不随Git提供。可从最终adapter开展新的训练或评测；CPU head重新拟合需要先重建缓存。既有预测与最终head可以直接读取。历史清单 `COPY_MANIFEST.json`、`EXTERNAL_ASSETS.json` 描述来源工作区，不能当成当前上传清单。

主方法Stage-1 head包含在adapter safetensors中，Stage-2使用`discrim_head_calibrated.pt`；margin+KL使用adapter内更新的head。选定checkpoint保留配置、tokenizer及`trainer_state.json`用于追溯。固定训练归一化文件为`research_assets/discrim_musd.pt`。

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

已有完整测试基线为296 passed、2 skipped，包含12项需要本地论文的测试。上面的代码仓库命令排除这12项。CPU审稿实验与结果核验见 [CPU完成清单](docs/review_evidence/CPU_COMPLETION.md)。

已核对10个选定checkpoint的必需文件；28份最终权重共约1.24 GB的SHA-256与提交的LFS指针全部一致。远端上传已完成，并实际取回一个head验证哈希。尚未在脱离旧工作区的隔离clone中重新下载整套资产并完整执行恢复，不能将当前核验等同于全新机器端到端测试。

首次接手请以`git status`、`git log`和`git remote -v`确认实际版本。论文仍在作者本地/Overleaf管理，不要重新加入Git。
