# Agent 接手约定

最新用户要求：Git LFS仅保存论文使用的最终checkpoint/head和不可下载数据；中间epoch、optimizer/RNG与特征缓存排除。先读 `docs/ASSET_RECOVERY.md` 与 `docs/FINAL_ASSETS.json`。公开基座/COCO原图按说明下载。不要沿用下文旧“模型只在本机”或“全部缓存都已上传”的历史说明。论文及交付包仍排除。

最新范围：按用户要求，`paper/`、`figure/`、`submission_snapshot/` 已从 Git 当前版本移除，仅本机保留且忽略；不要重新提交这些文件。日常 CPU 测试按根 README 排除需要论文 TeX 的 `test_paper_metric_consistency.py`。早期核验/交接文档中的论文及快照路径属于本地或历史资料。远端已推送过初始版本，旧 commit 仍含这些资料；未获授权不要 force-push。

先读根 `README.md`，再读 `docs/HANDOFF.md`。验证记录以 `docs/VERIFICATION.md` 的“提交前完整性收尾”节为当前状态，后文保存历史过程。

## 工作范围

- 当前工作区：`/user/huangyongbin/newrepo/agriabstain-paper-repo`。正式代码在 `aaai_code/`，常规命令从该目录执行。
- 来源工作区 `/user/huangyongbin/newrepo/agriabstain` 保留原样；不要清理、迁移或覆盖其中内容。`asrabstain/` 是独立项目，跳过。
- 用户要求优先复制、复用现有实现；仅做任务需要的最小修改，避免重写算法和顺带重构。
- 论文、标注、训练目标、评测指标的变更须有明确任务依据。`submission_snapshot/` 是交付快照，保持原样，勿用于覆盖开发版。
- 当前任务验证范围是代码完整性与 CPU 测试；不要自行启动训练、模型推理或付费 API 请求。

## 环境与文件

- 使用 `/user/huangyongbin/miniconda3/envs/agrivllm/bin/python`。默认系统 Python 存在 NumPy/PyArrow 兼容问题。
- 数据、模型和大型特征通过 Git 忽略的 symlink 接入；完整位置见 `docs/EXTERNAL_ASSETS.json`。禁止递归复制、提交这些链接目标，禁止删除原始资产。
- `docs/COPY_MANIFEST.json` 是源文件基线，存在明确记录的修改，不能要求所有当前文件仍匹配原哈希；例外见 `docs/LOCAL_CHANGES.md`。
- 保留原始结果中的绝对路径用于 provenance；跨机器运行前需恢复资产、配置入口，不能宣称 fresh clone 可直接运行 GPU 流程。
- Python 修改使用 ruff；文档中文，技术术语保留英文。不要为消除历史 lint 全量格式化原始代码。

## 验证与 Git

- 完整 CPU 命令、主线入口和结果路径见 `docs/HANDOFF.md`；基线为 296 passed、2 skipped。
- 不删除数值断言、不调整论文数值来迎合测试。测试定位需读取当前未注释正文。
- 每次接手先运行 `git status --short --branch`、`git log -3 --oneline` 和 `git remote -v` 确认现场。
- 已有本地论文基线 commit `d181ee4`；截至本次交接未 push。后续状态以 Git 实际查询为准。
- 只在用户明确要求时 commit/push；不要把先前的 commit 授权推断为 push 授权。
