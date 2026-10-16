# 审稿补充整合稿说明

## 最新Overleaf布局（以此节为准）

同一项目中保留原稿及修订稿两个版本，分别选择Main document：

- 原正文：`main.tex`；原附录：`technical_supplement.tex`。
- 修订正文：`main_review_integrated.tex`；修订附录：`technical_supplement_review.tex`。
- 原bib：`aaai2027.bib`；修订正文使用`aaai2027_review.bib`。

上传到原Overleaf项目的文件共3个：修订正文、修订附录、修订bib。图片和模板继续共享。44行统一指标表已内嵌到修订附录，没有额外input文件。

修订附录以原技术补充为基础，保留原构造细节、图、margin+KL结果、prompt和协议，更新L2配对问题、Stage-2归一化/seed范围及重复记录说明，并加入新增CPU实验、完整重训统计和统一指标表。修订正文已移除末尾appendix；两个入口分别通过无缺失ref、无重复label、环境/括号配对检查。原main.tex、technical_supplement.tex和aaai2027.bib哈希未变。下文为拆分前的过程记录。

## 文件

- `main_review_integrated.tex`：独立整合稿；以英文论文正文方式组织新增分析。
- `review_common_metrics.tex`：由已核验44行指标JSON生成的统一附表，由新稿自动input。
- `aaai2027_review.bib`：原bib的独立副本，增加NOPE、HallusionBench、HieA2G三条已核实引用。
- 原`main.tex`、`aaai2027.bib`、technical supplement和图片均未改动。

原main.tex SHA-256：`058066acdab72a1895e3c07f0e614597e4d9558fa670b18d9c046a424616c76c`。

## 主要整合

1. 摘要、引言与结论突出MCS，明确FCA/retention收益和E2E代价，去除所有probe普遍强的泛化表述。
2. 数据章节增加resampling具体步骤、唯一记录/类别来源、T7几何公式、类别筛选偏差、label语义范围。
3. 主表加入Comprehensive dev-selected点，正文解释新增retention对应的无效候选计数。
4. 同源构造部分同时解释native与gate增益，并加入历史三次完整重训、固定backbone三次head拟合、Stage-1 head对照和Frozen Base对照。
5. L2保留原指标用于追溯，但明确标记原标签受构造错误影响，另报告严格零交集子集及容差敏感性。未把子集结果替换成新的完整benchmark。
6. 增加reason accuracy分解、GPT配置/invalid率、L3二元baseline和明确的局限性章节。
7. 新稿末尾附录包含共同指标及运行重复统计，正文只保留核心结果。

## 验证

- 静态LaTeX检查：引用全部解析、无重复label/bib key、环境成对、花括号平衡、所有引用图片存在。
- 44行统一附表直接由`docs/review_evidence/unified_metrics.json`生成。
- 原main.tex哈希保持一致；原bib保持一致。
- 当前环境没有pdflatex/latexmk/xelatex/lualatex/tectonic，未生成新PDF、未核验页数和overfull box。此稿是完整内容整合版，尚未按会议主文页数上限压缩。
- 独立review检查事实、实验范围和论文叙述一致性；具体勘误在本说明末尾记录。

## 使用

在该目录以 `main_review_integrated.tex` 为主文件编译，例如在有TeX环境时运行：

```bash
latexmk -pdf -interaction=nonstopmode main_review_integrated.tex
```

原technical_supplement仍是旧稿，可能与新正文的L2审计、归一化追溯和新增实验说明不一致。提交前需要基于新稿统一补充材料，不能直接将旧supplement视为已同步版本。

所有论文材料仍位于Git忽略目录，本次未commit/push，也未将论文重新纳入代码仓库。

## 独立review后的修订

Reviewer核对了实验身份、主要数值、L2原标签与诊断子集、未完成人审及引用完整性。按审查意见，已将Frozen对照的因果表述收窄为特征可分性与路由行为分析；明确adapted argmax具有更高MCC/retention/E2E及更高FCA；将148个同instance配对与136条严格零交集子集区分；收窄PlantWild transfer措辞并修正routing表题。原稿未变。

再次核验：统一附表44行、308个数值单元全部与`unified_metrics.json`一致；重复实验附表6行、36组mean/std全部与`prior_raw_verification.json`及`cpu_gate/scored_heads.json`的四位小数结果一致。正文先陈述Grouped在native、gate及重复实验中的实测优势，再说明泛化与归一化观察；没有更改数值或隐藏E2E代价及L2构造问题。

第二轮独立review未发现主要数值、舍入或sample-SD错误。已澄清L3违例方向，将T4/crop-out边际通过率与配对联合成功率区分，删除未经消融支持的representation-level机制归因。Comprehensive GPT配对bootstrap区间已从`aaai_code/experiments/gpt56_sol_full/outputs/decision_prompt_v1/test_full/analysis/comparison.json`的`paired_gpt_minus_regate`核实并写入正文，保留其record-level不确定性口径。
