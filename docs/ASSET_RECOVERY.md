# 最终模型与数据恢复

## 保存范围

按用户最终要求，Git LFS只保存**最终模型及论文使用的选定权重**，不保存中间epoch、优化器、RNG状态或可重建的特征缓存。论文、配图及交付包也不上传。

正式清单为 `docs/FINAL_ASSETS.json`。`EXTERNAL_ASSETS.json`、COPY_MANIFEST以及曾生成的REMOTE_ASSETS属于来源/工作区历史清单，不表示每个历史文件都在Git当前版本内。

选定checkpoint保留`trainer_state.json`、adapter配置和tokenizer信息，用于选点与运行配置追溯；不包含optimizer/RNG或中间权重。

保留的权重包括：

- Grouped主方法 checkpoint-444：LoRA adapter和Stage-2 calibrated head；Stage-1 head与归一化buffer包含于adapter safetensors。
- Independent构造对照 checkpoint-222。
- Margin+KL v1 checkpoint-50、v2 checkpoint-60，直接使用adapter中的更新head。
- 三次完整重训每组每seed的dev-selected checkpoint，共6个。位置位于`research_assets/h27experiment/runs/`。
- 审稿CPU实验的最终选中MLP/linear heads，位于`docs/review_evidence/`；训练过程checkpoint不上传。
- 固定训练归一化`research_assets/discrim_musd.pt`。

## 下载

```bash
git lfs install
git clone git@github.com:1105425455/agriabstain.git
cd agriabstain
git lfs pull
python tools/verify_remote_assets.py
python tools/restore_assets.py --execute
```

先下载主方法时可限定：

```bash
git lfs pull --include='new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/**,research_assets/discrim_musd.pt'
```

检查实际权重文件大小；约百字节的LFS pointer需要先`git lfs pull`。LFS下载依赖GitHub账户访问权限和服务端配额。

## 公开资源与项目数据

| 资源 | 获取方式 |
|---|---|
| Qwen3-VL-8B-Instruct完整基座 | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct 或 https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct；下载后设置`BASE_MODEL` |
| COCO train2017原图 | https://cocodataset.org/#download；源COCO203张及RefCOCOg181张按`aaai_code/benchmark/IMAGE_MANIFEST.json`定位并核对SHA-256 |
| PlantDoc项目精确副本 | LFS：`research_assets/exact_images/plantdoc/`，726张；上游公开入口 https://github.com/pratikkayal/PlantDoc-Dataset，但加工版本可能不同 |
| PlantWild closed12项目精确副本 | LFS：`research_assets/exact_images/plantwild_closed12/`，144张、12类 |
| 项目加工YOLO标签与COCO子集标注 | LFS：`research_assets/plantdoc_yolo_labels/`、`research_assets/coco_annotations/` |
| RefCOCOg元数据、parquet、gate记录 | LFS：`research_assets/refcocog_metadata/` |
| 本项目生成的变换图片 | LFS：`research_assets/datasets/derived_images.tar.gz`，恢复脚本解压至`aaai_code/benchmark/data/derived/` |

恢复脚本创建兼容目录关联，保留已存在实体文件。公共COCO图片下载后按manifest放入`aaai_code/benchmark/data/source/coco/images/`及`aaai_code/benchmark/data/ood/refcocog/images/`。可使用`--public-data-root`连接已整理为source/ood布局的数据根目录。

## Agent继续工作的边界

完整训练/评测代码、配置、最终权重、标注、已有预测和审稿统计均保留。原始日志中的绝对路径用于provenance；正式入口优先使用`aaai_code`并通过`BASE_MODEL`、`ADAPTER`、`OUTPUT_ROOT`指定运行位置。

**原优化器/RNG和中间epoch已按要求排除，不能声称可逐step精确恢复旧训练。** 可以从所提供最终adapter开始新的训练或评测。特征缓存不上传；CPU-only head重新拟合需要先由对应checkpoint重建表征。已完成CPU实验的最终head、逐样本概率与统计仍可读取。

`research_assets/papermodel_lineargate/`、`h27experiment/`、`devtau_multiseed/`保留必要代码和小型结果，读取缺少的缓存时应明确提示重新提取，不能自动访问旧机器。论文材料继续只在作者本地/Overleaf管理。
