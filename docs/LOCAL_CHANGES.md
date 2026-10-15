# 复制后的最小修改

所有复制文件的原始版本及 SHA-256 记录于 `COPY_MANIFEST.json`。该清单描述复制时的来源基线，以下改动为明确记录的例外。

| 文件 | 改动 | 原因 |
|---|---|---|
| `aaai_code/experiments/base_native_regate/test_paper_metric_consistency.py` | 修正论文目录，适配当前表标签、缩写、合并表和正文数值；排除注释旧稿 | 保留产物数值、数据计数和实现断言，12 项通过；当前稿已删除的描述明确按 artifact-only 追溯 |
| `aaai_code/scripts/test_build_anonymous_release.py` | fixture 和布局预期改为现有 REVIEWER_README/manifest；release notes 从返回报告读取 | 与现有发布器行为一致，预测数量、匿名化及清单检查保留，62 项通过 |
| `new_reason/baselines/prompt_cot/run_baselines.py` | 从脚本位置定位仓库，调用已复制的正式 evaluator/训练包；基座支持 `BASE_MODEL`；添加现有 closed12 标注入口 | 消除未复制的旧 evaluator 和 `newstory` 依赖，保留推理参数与 prompt |
| `new_reason/baselines/prompt_cot/recompute_plantwild_metrics.py` | 指标导入改为仓库内 `aaai_code/reason_abstain` | 消除未复制旧 Python 包的依赖，使用正式论文指标 |

新增根 README、`.gitignore` 和 `docs/` 中的整理记录用于导航与资产管理。模型、训练目标、评测指标、数据标注和论文数值保持原样。

完整性复核额外原样补齐 `training/qwenvl/data/` 三份训练数据代码、层选择 samples/split assignment、GPT L3 分析目录和 epoch1 recovery 元数据；对应来源已加入 COPY_MANIFEST，大型模型链接加入 EXTERNAL_ASSETS。正式 aaai_code 范围的 `.py/.sh` 相对源工作区无遗漏（不含未进入论文的 groundingme/openref 实验）。
