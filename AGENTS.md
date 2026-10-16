# Agent接手约定

先读根README、`docs/HANDOFF.md`、`docs/ASSET_RECOVERY.md`。当前资产范围以`docs/FINAL_ASSETS.json`为准。

## 当前边界

- Git包含正式代码、标注、已有预测、审稿分析、最终checkpoint/head和不可下载的精确数据，模型与大型数据使用Git LFS。
- 中间epoch权重、optimizer/RNG、特征缓存不上传。不得把旧`EXTERNAL_ASSETS.json`中每条路径都当作可下载资产。
- `paper/`、`figure/`、`submission_snapshot/`当前均排除。旧commit曾包含这些材料；未经授权不要重写历史或force-push。
- 原研究工作区`/user/huangyongbin/newrepo/agriabstain`保持原样；`asrabstain/`是独立项目，跳过。
- 优先复用现有实现，不顺带重写算法或重构。论文数值、标注和指标修改需有明确任务依据。

## 环境与恢复

- 新clone先`git lfs pull`，再`python tools/verify_remote_assets.py --hash`、`python tools/restore_assets.py --execute`。公开Qwen基座/COCO按恢复说明下载。
- 本机使用`/user/huangyongbin/miniconda3/envs/agrivllm/bin/python`；默认系统Python有NumPy/PyArrow兼容问题。其他机器按环境记录安装依赖。
- 正式命令从`aaai_code/`执行，模型及输出使用环境变量或CLI参数。不要自动依赖旧机器路径。
- 日常验证使用根README的CPU测试命令，排除需要本地论文的`test_paper_metric_consistency.py`。
- 无用户明确任务，不启动GPU训练、推理或付费API。既有CPU脚本也可能拟合head，执行前读说明。
- Python修改使用ruff；文档中文、技术术语保留英文。保持改动最小，不全量格式化历史代码。

## Git

- 接手先运行`git status --short --branch`、`git log -3 --oneline`、`git remote -v`。
- 仅在用户要求时commit/push；不要把commit授权等同force-push授权。
- 新最终模型须加入FINAL_ASSETS并用LFS，不能误提交中间状态、密钥或本机外部symlink。
- `COPY_MANIFEST.json`是复制来源基线，已记录的改动见`LOCAL_CHANGES.md`；不能以原哈希要求所有当前代码不变。
