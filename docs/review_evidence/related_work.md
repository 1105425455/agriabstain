# 相关工作核验

本节依据论文官方摘要或数据库摘要，不包含新baseline实验。不同数据集原文中的分数不直接与ReGate数值比较。

| 工作 | 已核实任务与联系 | 来源 |
|---|---|---|
| Lovenia等，Negative Object Presence Evaluation (NOPE) to Measure Object Hallucination in Vision-Language Models，ALVR 2024 | VQA不存在对象评测；LLM生成29.5k negative-pronoun样本，考察模型识别对象不存在的能力。没有grounding box接口。 | [ACL官方摘要](https://aclanthology.org/2024.alvr-1.4/)，DOI 10.18653/v1/2024.alvr-1.4 |
| Guan等，HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models，CVPR 2024 | 图像语境QA诊断，覆盖language hallucination与visual illusion。与视觉证据/语言先验冲突相关，但QA问题对指标与grounding FCA/retention定义不同。 | [DOI](https://doi.org/10.1109/CVPR52733.2024.01363) |
| Wang等，Hierarchical Alignment-enhanced Adaptive Grounding Network for Generalized Referring Expression Comprehension，AAAI 2025 | HieA2G处理generalized REC及目标数量，通过adaptive grounding counter处理零/多目标。可在统一数据与empty-set映射下比较二元选择和定位指标。 | [DOI](https://doi.org/10.1609/aaai.v39i8.32867) |
| Hemanthage等，RECANTFormer: Referring Expression Comprehension with Varying Numbers of Targets，EMNLP 2024 | 处理不同数量目标，覆盖zero/one/multiple referents；与本文unsupported场景有任务交集。 | [ACL](https://aclanthology.org/2024.emnlp-main.1214/)，DOI 10.18653/v1/2024.emnlp-main.1214 |
| Yu与Li，Revisiting Counterfactual Problems in Referring Expression Comprehension，CVPR 2024 | 针对query目标无法在图中找到的counterfactual REC，联合定位与counterfactual判定，属于应比较的近邻工作。 | [DOI](https://doi.org/10.1109/CVPR52733.2024.01276) |
| Li等，AgroVG: A Large-Scale Multi-Source Benchmark for Agricultural Visual Grounding，2026 | 官方摘要将农业grounding建模为单/多/无目标的set prediction，10,071个annotation-grounded image-query pairs、10个来源、6类任务，含box/mask和target-absent设置。与本文农业及absent-target设置相关。 | [arXiv官方摘要](https://arxiv.org/abs/2605.22034)，2026-05-21 |
| 审稿所列Zhang等，Exposing and evaluating hallucinations for GUI grounding，2026 | 本次OpenAlex/Crossref/Semantic Scholar检索未核实该确切题名及条目，保留待核对。未找到条目不证明论文不存在，也不使用其他GUI论文替代。 | 审稿原文所列引用待核实 |

MCS是Metamorphic Counterfactual Supervision，利用七类确定性关系构造annotation-defined证据状态和同源对照。其模型无关性限于固定label space后的构造算子及reference定义；农业类别筛选使用了Base模型能力。T6 contradiction与T5 unavailable是本文操作性划分，不能直接与其他基准的全部负例语义视为完全等价。

对于支持empty-set的grounding方法，可在相同记录上约定empty-set→abstain，再计算FCA、retention和MCC；多目标输出还需明确box集合到本文定位指标的匹配协议。原因输出缺失影响Strict-3/FullStrict中的reason正确性，仍可比较相应binary版本。NOPE与HallusionBench主要提供任务和问题定位，不自动成为可直接运行的grounding模型baseline。

现有文件没有记录最初未采用AgroVG的决策原因，因此不推测其历史可用性或标注质量。
