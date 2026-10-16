# 后续工作交接

## 当前状态

正式代码、审稿回复、CPU证据和论文选定的最终权重已推送。资产提交为`981882a`；1,532个LFS对象约4.5 GB上传成功，最终白名单包含2,777个文件。具体最新提交以`git log`为准。

原稿与修订稿只在本地/Overleaf管理，不在当前Git版本中。中间epoch、optimizer/RNG和表征缓存按用户要求排除；从最终adapter继续工作需要公开基座和对应数据。

## 接手顺序

1. 查看根README及`ASSET_RECOVERY.md`，执行LFS拉取、资产核验、目录恢复。
2. 读取`FINAL_ASSETS.json`确认实际上传内容；下载公开Qwen基座和COCO原图。
3. 读取根`review.md`了解逐条审稿回应；`review_evidence/CPU_COMPLETION.md`列出已完成实验与剩余条件。
4. `VERIFICATION.md`记录原代码整理和CPU测试历史，旧失败记录已在后续修复，不代表当前状态。
5. `aaai_code/README.md`提供详细代码入口；旧作者路径需按当前clone位置配置。

## 主要入口与结果

| 用途 | 仓库相对路径 |
|---|---|
| 训练/校准 | `aaai_code/training/sft_final/` |
| 正式推理/指标 | `aaai_code/evaluation/eval_qwen3vl_3reason.py`、`aaai_code/reason_abstain/eval/metrics.py` |
| 原论文主方法 | `new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/` |
| 主表与曲线 | `new_reason/outputs/base_native_regate_20260720/` |
| dev阈值 | `new_reason/outputs/matched_retention_dev_20260725/` |
| 构造消融 | `aaai_code/experiments/grouping_ablation/` |
| 已有完整重训 | `research_assets/h27experiment/` |
| CPU新head结果 | `docs/review_evidence/cpu_gate/`、`frozen_and_prior/` |
| L2几何审计 | `docs/review_evidence/l2_geometry_summary.json`及逐条记录 |
| 数据恢复 | `tools/restore_assets.py` |
| 清单与LFS实体核验 | `tools/verify_remote_assets.py` |

## 审稿分析的重要事实

- Grouped在多次完整重训及固定backbone head拟合中保持选择指标/MTS优势。
- L2原构造按同图keep×crop组合，存在query与裁剪目标错配；原标注未覆盖修改，几何诊断子集结果单列。
- 发布Grouped gate的归一化buffer与checkpoint-444重提取统计不匹配，来源尚未确定。
- Frozen Base＋MLP已完成CPU对照，论文dev100可通过original_index与SFT记录唯一匹配。
- 新增retention与成功定位须分开解释，具体配对结果在review.md。
- 人工双人审计目前只有600条准备清单，无人工答案；aux-loss=0、指定近邻模型同协议比较等仍缺新增模型运行。

## 继续工作

优先复用保存的最终权重、预测和已验证脚本，不重复训练以重建已有结果。重新拟合head需要先提取未随Git提供的特征缓存。论文文件有单独的原稿/修订正文/修订附录入口，作者本地的`REVIEW_INTEGRATION_NOTES.md`记录整合过程；代码repo不需要论文即可进行常规开发。
