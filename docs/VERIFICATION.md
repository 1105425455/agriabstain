# 论文版本整理核验

本次只读核查原工作区并复制至独立 Git 工作区。原始项目和 `asrabstain/` 未改动。远端 `git@github.com:1105425455/agriabstain.git` 已通过 SSH 连接，检查时为空仓库。用户已授权本地 commit；本次不 push。

## 提交前完整性收尾（以此节为当前状态）

- 最终完整 CPU 测试 **296 passed、2 skipped、0 failed**，139.59 秒；命令包含 `reason_abstain experiments scripts/test_build_anonymous_release.py`，日志 `verification/regate-commit-tests.log`。18 条 warning 来自退化样例的 bootstrap CI。未重跑训练或推理。
- 提交前独立 Reviewer 审查未发现阻断风险，确认数值断言没有被放宽、三个训练数据代码文件逐字节一致。
- COPY_MANIFEST 现含 2,540 个源文件；新增部分均与原文件逐字节一致。明确修改的四个脚本/测试见 LOCAL_CHANGES，论文、模型实现和指标公式未改。
- 外部资产共 362 个本地链接；正式代码 `.py/.sh` 相对源 aaai_code 无遗漏，一阶 `qwenvl/reason_abstain/experiments` import 静态解析无缺失。
- 补齐被目录过滤误排除的 `qwenvl/data/` 三份代码；补齐层选择输入、GPT L3 分析和 epoch1 recovery 选点模型元数据/链接。56 个发布构建器声明来源全部存在。
- 两个 prompt baseline 脚本改用仓库内正式 evaluator/metrics，`--help` 和离线 import 验证通过；未启动训练或推理。
- 旧稿字符串测试已适配当前未注释正文，12 项论文测试通过；发布布局测试 62 项通过。原始历史失败日志保留在 verification 中供追溯。
- 全部归档 Python/Shell 文件通过静态语法检查。大型 tensor 仍仅核验文件存在性与链接，不加载模型。

## 复制与实际资产

- 复制 2,497 个已有文件，复制完成后逐文件 SHA-256 校验全部一致。
- 后续仅修改一个现有测试的论文目录路径，见 `LOCAL_CHANGES.md`；原始哈希继续保留。
- 342 个本地大型资产链接全部可达，未纳入 Git 候选文件。
- `aaai_code/benchmark/annotations/sft/sft_train.json` 的 7,072 条记录对应图片全部存在。
- 九份 dev/test/external probe 文件及综合测试的图片引用检查无缺失：dev L1/L2/L3、test L1/L2/L3、RefCOCOg、PlantWild closed12、sft_test。
- 正式 checkpoint-444 的 `adapter_model.safetensors`（109,572,070 bytes）、`adapter_config.json`、`discrim_head_calibrated.pt`（8,955,629 bytes）和 `trainer_state.json` 存在。大型 tensor 只检查存在性和大小，未加载推理。
- 当前正文与补充引用的六张图均存在。Fig1 素材清单的 17 个源图路径均存在。Fig1/Fig2 的可编辑设计源文件尚未确认。

## 测试环境及结果

系统 Python 的 NumPy/PyArrow 二进制兼容问题阻断测试收集，改用现有 `agrivllm` Python 3.11 环境；该环境补装了 pytest 9.1.1、pluggy 1.6.0、iniconfig 2.3.1。未修改模型训练依赖。

```bash
cd aaai_code
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /user/huangyongbin/miniconda3/envs/agrivllm/bin/python \
  -m pytest -q -p no:cacheprovider reason_abstain experiments
```

最终完整测试结果为 **224 passed、2 skipped、10 failed**（165.16 秒）。首轮结果为 222 passed、2 skipped、12 failed；12 项均位于 `test_paper_metric_consistency.py`，首次因旧论文目录后缀失败。仅修正路径后，该文件重跑为 2 passed、10 failed，随后完整套件再次确认上述最终结果。失败涉及旧稿原句、表格标签和公式字符串匹配；现有测试尚未适配当前论文的排版和措辞。保留失败，不修改论文、算法或指标来迎合测试。

独立核对当前正文主表九组 Base/SFT/ReGate 数值，以及分组消融三个主要数据集的 MTS，均与保存的正式结果一致。论文测试中已执行的主表/GPT 数值断言与训练集数量断言通过，随后失败于旧稿字符串匹配。其他未执行到的断言不视为已通过。

## 提交快照

在 `submission_snapshot/` 运行复制的原有 `../tools/verify_supplement.py`：18 项中 17 项通过，1 项失败。失败项为“无中文残留”，25 个原文件仍含中文；为保持交付版本原样，不做改写。

通过项包括：292 文件 RELEASE_MANIFEST 哈希和大小、反向清单无多余或缺失、27 份标注哈希、66,036 条图片引用清单覆盖、Python 编译、运行环境记录，以及 Table 3、margin+KL、L2/L3 和 dev-selected threshold 数值校验。

## 已知限制

- 本机可通过外部资产链接使用现有数据和权重；全新 clone 需恢复 `EXTERNAL_ASSETS.json` 中的资产。部分作者代码及原始 JSON 保留原绝对路径。
- 未重跑 GPU 训练或推理，未承诺跨机器无配置运行。
- `test_paper_metric_consistency.py` 在原配置下有既存 Ruff I001/RUF100 问题；仅修路径，未顺带格式化。AST 语法检查通过；LSP 因 basedpyright 未安装不可用。
- 凭据格式扫描仅命中 `test_build_anonymous_release.py` 的测试样例；不输出匹配值。扫描不等同于全面安全证明。
- 原始检查日志保存在 `verification/`。

## 独立审查

Reviewer 确认复制清单仅有已声明的一处路径差异、342 个链接与清单匹配且均被 Git 忽略、提交快照 292 个文件校验通过；未发现额外算法改写或可确认的硬编码凭据。审查提出的验证文档缺失问题在审查进行期间已通过新增本文件解决；根 README 已补充最终失败数及实际 Python 环境入口。旧稿文本测试与快照中文门禁的既有失败继续明确保留，后续修订需要与版本快照分开处理。
