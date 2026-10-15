# AAAI 论文故事线 —— AgriAbstain: 非对称风险约束下的选择性 Grounding

---

## 1. 一句话卖点 / TL;DR

**非对称风险约束下的选择性 grounding——绘制细粒度选择性 grounding 在何处可迁移、何处不可，并给出诚实机理诊断。不宣称"取胜"跨域 wrong_object/no_target，而是刻画边界：域内安全（A_FU 0.026）且通用 grounding 不损（RefCOCOg 0.925）；absence 检测跨域鲁棒（96.8%）；但跨数据集时 answer/abstain 工作点整体偏向弃答，真实病 over-abstain 30.1%，且已测试的 levers 无法外推该 frontier。**

**三个贡献一句话概括（诚实版）：**

1. **细粒度、带 reason 归因的选择性 grounding 框架**——三分类 head（answer/wrong_object/no_target），域内选择性安全强（A_FU 0.026）且不损通用 grounding（RefCOCOg 0.925）。
2. **严格构造的 in-the-wild paired 同图三路 OOD benchmark（PlantWild，156 图 × {answer,no_target,wrong_object}，对 PlantDoc 严格去重，官方 test split）**，把细粒度 near-miss 弃答从采集域 confound 中隔离出来。（benchmark 贡献）
3. **对"选择性 grounding 里什么能迁移"的实证刻画 + 机理诊断**：no_target/absence 检测跨域鲁棒（96.8%）；去掉野外→实验室 confound 后 wrong_object 迁移显著改善（5.6%→59.8%）；但 answer/abstain 工作点在跨数据集时整体偏向弃答，真实病 over-abstain 30.1%——这是 matched-risk frontier 在跨数据集下的表现，且已测试的 decision-form / cost-weighting levers 无法外推。

**（可选 4）matched-risk frontier 作为非对称风险选择性 grounding 的评估视角**，非方法贡献，**明确不宣称理论 Pareto 下界**。

**主结果三个关键结论：** A_FU 0.026 | RefCOCOg 0.925 | PlantWild no_target 弃答 96.8% | 但真实病 over-abstain 30.1%、frontier 未被已测 levers 外推。

---

## 2. 任务定位

### 2.1 问题定义

本工作针对农业病害视觉 grounding 中的选择性弃答（selective abstention）问题。核心场景：给定一幅农作物病害图像，模型需输出病害区域的 bounding box 与病害类别标签，同时有权选择"弃答"（不输出预测），以避免在不确定时给出错误诊断，造成农户经济损失。

### 2.2 模型与决策头

基座模型：**Qwen3-VL-8B**（多模态大语言模型），经 LoRA 微调。决策层采用**三分类判别 head**，输出三种决策：

| 决策类别 | 语义 | 行为 |
|---------|------|------|
| **answer** | 模型确信可正确识别并定位 | 输出 bounding box + 病害标签 |
| **wrong_object** | 检测到目标病害，但属于近义病害（同作物不同病），应弃答 | 输出"无法确认" |
| **no_target** | 目标病害不在图像中，应弃答 | 输出"图中无此病害" |

### 2.3 非对称风险红线

农业病害诊断场景中存在**非对称风险（asymmetric risk）**：真病误弃会直接漏掉真实病害，因此必须作为独立红线约束，而不能用更高的近义弃答率抵消。我们采用以下高风险指标：

- **A_FU**：按本项目既定评测口径，衡量本应 `answer` 的真病样本被误判为 `wrong_object` 或 `no_target` 的比例，即真病误弃/漏诊风险。
- **红线目标：A_FU 必须保持在项目预设安全红线内。** 主模型达到 0.026，已满足该红线；不以显著抬高 A_FU 为代价追求更漂亮的 OOD 近义弃答数字。

---

## 3. 核心贡献（3 条）

### C1：细粒度带 Reason 归因的选择性 Grounding 框架

现有选择性分类/grounding 工作通常做**二分类**弃答（answer vs. abstain），不区分弃答原因。我们提出**三分类 reason 归因**，将弃答拆解为：

- **wrong_object（近义病害混淆）**：模型识别到病害，但难以区分同作物的不同病害（如番茄晚疫病 vs. 番茄早疫病），此时弃答并提供"可能是近义病害"的 reason。
- **no_target（目标缺席）**：目标病害不在图中（如图中为健康叶片或无关场景），弃答并输出"图中无目标"的 reason。

这一拆分不仅增强了可解释性，也为下游（如农技站人工复核）提供 actionable 信息。**域内安全性强**：在主模型 operating point 下，真病漏诊 A_FU 0.026，且不损通用 grounding 能力（RefCOCOg 0.925）。

**主模型（3reason ckpt-444，主线模型）性能：**

| 指标 | 数值 |
|------|------|
| test_full MCC | 0.856 |
| test_full A_FU（真病漏诊） | **0.026** ✅ 红线达标 |
| L1 近义判别 MCC | 0.419 |
| L1 近义判别 A_FU | 0.059 |
| L1 近义弃答率（同分布） | 0.395 |
| L2 grounding MCC | 0.574 |
| L3 grounding MCC | 0.841 |
| RefCOCOg（通用 grounding） | **0.925** |

> 注：上述数字中，test_full A_FU 0.026 在文件 `frontier_key_points.md` 中记录为 ~0.024-0.026（阈值依赖），此处以决策者口径为准。L1 A_FU 0.059 在文件中记录为 0.0585，以决策者口径为准。

### C2：Matched-Risk Frontier 作为评估视角/经验诊断（非方法贡献）

传统 OOD 检测评估用单一阈值下的一个点（如 PV 近义弃答 = X%）评价模型好坏，忽略弃答与漏诊之间的风险权衡。我们提出**matched-risk frontier**作为**评估视角**，**明确不宣称理论 Pareto 下界**。

**方法**：对样本级 `P(answer)` 做阈值扫描，在不同 A_FU 容忍度下测量 PlantVillage 近义弃答率（以及 L1_mismatch 弃答率），将结果绘制为 matched-risk frontier。

**三个关键发现：**

1. **稳定权衡存在**——将 PV 近义弃答从基线 0.056 提升到 0.35 需要 A_FU 从 0.026 升至 0.08（真病漏诊翻 3 倍）；提升到 0.50 需要 A_FU 升到 0.12（翻近 5 倍）。在项目预设 A_FU 红线下，不能用这种代价追求更高的 PV 近义弃答。

2. **改变已测试的决策形式仍未外推 frontier**——2reason 完整重训模型（二分类 decision + LM reason，严格单变量调整）的 ckpt-222/ckpt-333 工作点（A_FU ~0.114[^1]，PV近义弃答 ~0.50）落在 3reason 同一条 matched-risk frontier 上，没有把 frontier 向外推出。这支持该权衡并非 3reason 单一设计造成；结论范围限于当前 Qwen3-VL-8B backbone 与已测试方案。

   [^1]: 文件 `frontier_key_points.md` 中 ckpt-222 A_FU=0.1081、ckpt-333 A_FU=0.1104，以决策者口径"~0.114"为准。

3. **旧 SFT 高弃答的真相**——曾误以为旧二分类 SFT 模型 PV 近义弃答 0.90 是优秀表现。实测 ckpt-86 的"高弃答"建立在对 L1 真病误弃达 63%（A_FU 0.632）的代价上。其应弃 0.879 vs 应答误弃 0.632，gap 仅 0.25，几乎"什么都拒"。旧 SFT 的 reason 仅为统一 `target_absent`，无 wrong_object/no_target 细分，严格指标高是口径假象——换细粒度口径归零。旧 SFT 与 3reason 落在同一条 Pareto frontier 上，只是极端偏拒端（不可用）vs. 低漏诊端。

**关键工作点小表（阈值 → A_FU vs 弃答率）：**

| 模型/目标工作点 | test_full A_FU | PV 近义弃答 | 含义 |
|----------------|---------------|-------------|------|
| 3reason ckpt-444 当前点 | **0.026** | 0.056 | 主线：满足真病误弃红线 |
| 3reason 阈值扫描目标点 | 0.08 | 0.35 | 近义弃答提升需真病漏诊约翻 3 倍 |
| 3reason 阈值扫描目标点 | 0.12 | 0.50 | 近义弃答提升需真病漏诊约翻 5 倍 |
| 2reason ckpt-222/333 | ~0.114[^2] | ~0.50 | 同一 frontier 上更激进的 operating point |
| 旧 SFT ckpt-86 | 0.632 | ~0.90 | 极端偏拒端（不可用） |

[^2]: 文件中两个 2reason checkpoint 的逐点数值略有差异；此处采用决策者给出的概括工作点 `(A_FU 0.114, PV近义 0.50)`。

### C3：诚实的跨采集域边界刻画 + 机理归因（含 PlantWild 证据）

**核心叙事定位**：本文对"选择性 grounding 里什么能迁移"给出实证刻画 + 机理诊断。不宣称"取胜"跨域 wrong_object/no_target，而是诚实绘制边界。

#### C3a：PlantVillage 原始发现（采集域 confound 的对照单点）

PlantVillage OOD 测试集拆解为两个子场景：

| 场景 | 样本数 | GT 类型 | 决策弃答率 | 含义 |
|------|--------|---------|-----------|------|
| OOD_absent（无目标） | 132 条 | no_target | **0.924** | 缺席检测跨域鲁棒 ✅ 正面 |
| OOD_confusion（近义混淆） | 108 条 | wrong_object | **0.056**[^8] | 细粒度跨域脆弱 ⚠️ 边界 |

[^8]: 文件 `frontier_key_points.md`/`REPORT.md` 中 tau=0.5 时记录为 0.0370、tau=0.510 时 0.0463。此处 0.056 以决策者口径为准。

**根因归因（已实证，非猜测）：**

1. ✅ **不是没教过**：训练数据的近义病害对 100% 覆盖了 PlantVillage 测试中出现的全部 20 个病害对（mismatch_pairs 配置 + 1500 样本验证）。
2. ✅ **不是作物 OOD**：PV 近义样本全部属于 seen crop（tomato / corn / apple），是同作物不同病，非作物级别新颖度问题。
3. ✅ **是"野外→实验室"域迁移 + 细粒度病害判别本质难**：模型在野外数据（PlantDoc 采集）上训练，在实验室数据（PlantVillage 采集）上测试。细粒度病害判别依赖于病害的病理特征（病斑纹理、形状、颜色），而非粗粒度的背景/光照/采集条件。模型学到的是域特定捷径（background / illumination shortcut），而非病害病理本质，跨采集域后失效。这本质是 **shortcut learning** 问题。
4. ✅ **OOD gate（feature 距离/作物新颖度）不适用**：作物本身没有变化（都是 seen crop），病害之间的细粒度特征在 feature space 中高度重叠，gate 无法在 decision 层面分开。

#### C3b：PlantWild 验证——"采集域 confound"假说被证实

**已被 PlantWild paired 结果纠正**：PlantVillage 的 wrong_object 弃答 0.056 不能泛化为"细粒度近义判别跨域不可做"。去掉野外→实验室采集域 confound 后，wrong_object 弃答大幅改善至 59.8%，证伪了"纯细粒度"不可迁移的猜测，揭示主因是域 shortcut。

详见下方新增章节「PlantWild paired benchmark 与结果」。

---

### C3c / 新增：PlantWild paired benchmark 与结果（权威）

#### Benchmark 构造（严格，已 review 无阻断）

PlantWild 与 PlantDoc 均为 web-scraped / in-the-wild，去掉了 PlantVillage 的"野外→实验室"采集域差异。

- 官方 test split only；PV13 的 13 类 100% 覆盖；每类 12 图，共 **156 unique images**。
- 与 PlantDoc 全量 2567 图做 dHash≤8 交叉去重，排除 27 张；内部去重；人工排除 2 张异常图。
- 图片统一 262,144 pixels（解决 10MP OOM，与 Qwen 评估声明的 max_pixels 一致）。
- 同一张图三问：真实病害 answer(156)、跨作物 no_target(156)、同作物不同病 wrong_object(132；potato/squash 无 siblings)，合计 **444 条**。
- **A_Ground 不适用于 PlantWild（无 GT box），不可报**；grounding 能力仅由 RefCOCOg 0.925 + 域内 L2/L3 佐证。
- **HOR-FCA 分母 = 156（全部 no_target 条目）**，定义为 no_target 弃答样本中出现高遮挡/幻觉错误 box 的比例。

#### 权威结果（job553011）

metrics 路径：`new_reason/outputs/ood_plantwild_eval/sftfinal_ckpt444_paired_262k/metrics.json`；逐样本 head 概率 `predictions.jsonl` 含 p_answer/p_wo/p_nt。

| 维度 | 指标 | 数值 | 备注 |
|------|------|------|------|
| **Overall** | format valid | 1.0 | 全部输出合规 |
| | MCC | **0.490** | 综合判别 |
| | FCA | 0.201 | |
| | HOR-FCA | **0.032 (5/156)** | 弃答时幻觉 box 极少 ✅ |
| | U_Abs | 0.799 | |
| | A_FU | **0.301** | 见下方诚实定性 |
| | AAC | 0.635 | |
| **OOD_answer（真实病 156）** | decision answer | 109/156 = **69.9%** | 正确判定为 answer |
| | strict answer+label | 99/156 = **63.5%** | 含 label 正确 |
| | 回答后 label 正确 | 99/109 = 90.8% | 回答时 label 高可靠 |
| | over-abstain A_FU | **47/156 = 30.1%** | 真实病被误弃 |
| **OOD_absent（跨作物 156）** | decision abstain | 151/156 = **96.8%** | 缺席检测跨域鲁棒 ✅ |
| | strict no_target | 117/156 = **75.0%** | 含 reason 正确 |
| | 弃答后 reason 正确 | 77.5% | |
| | HOR-FCA | 3.2% (5/156) | 弃答时幻觉 box 极少 ✅ |
| **OOD_confusion（同作物近义 132）** | decision abstain | 79/132 = **59.8%** | 近义跨域弃答大幅改善 |
| | strict wrong_object | 66/132 = **50.0%** | 含 reason 正确 |
| | 弃答后 reason 正确 | 66/79 = **83.5%** | 弃答时 reason 质量高 |
| **同图全部正确** | all decision 正确 | 54/156 = 34.6% | 同图三问全部 decision 正确 |
| | all strict 正确 | 39/156 = 25.0% | 离线独立计算（within_image 字段恒 null，注明 provenance） |

#### 与 PlantVillage 对比（关键机理证据）

| 场景 | PlantVillage（野外→实验室） | PlantWild（同野外域） | 变化 |
|------|---------------------------|---------------------|------|
| wrong_object decision | 5.6% | **59.8%** | **大幅改善** ✅ |
| no_target decision | 92.4% | **96.8%** | 稳定鲁棒 ✅ |

**结论**：去掉野外→实验室采集域后 wrong_object 大幅改善（5.6%→59.8%），佐证 PlantVillage 的 wrong_object 崩塌主因是**采集域 confound**，而非细粒度任务本身不可做。no_target 检测在跨域条件下保持稳定（92.4%→96.8%）。

#### 诚实定性（不得隐藏 A_FU 30.1%）

**不能宣称"跨数据集同时低漏诊 + 高弃答"**。PlantWild 真实病 over-abstain 30.1% 是同一 operating point 的代价面，必须与 59.8%/96.8% 并列呈现。跨数据集时 answer/abstain 工作点整体偏向弃答——这是真实 frontier 代价，非单一 scalar 可修复。

---

## 4. 主结果表

**3reason ckpt-444（主线模型）全指标汇总：**

| 评测维度 | 指标 | 数值 | 备注 |
|---------|------|------|------|
| **test_full** | MCC | 0.856 | 综合判别 |
| | A_FU | **0.026** | 🔴 项目预设红线达标 |
| **L1 近义判别** | MCC | 0.419 | 近义病害区分 |
| | A_FU | 0.059 | |
| | 弃答率 | **0.395** | 同分布近义弃答 |
| **L2 grounding** | MCC | 0.574 | |
| **L3 grounding** | MCC | 0.841 | |
| **RefCOCOg** | Accuracy | **0.925** | 通用 grounding 不掉点 |
| **OOD PlantWild（主 OOD benchmark）** | | | **同野外域，主表** |
| — OOD_answer（真实病 156） | decision answer | **69.9%** | |
| | over-abstain A_FU | **30.1%** | 诚实并列，不可隐藏 |
| — OOD_absent（跨作物 156） | decision abstain | **96.8%** | 缺席跨域鲁棒 ✅ |
| | HOR-FCA | **3.2% (5/156)** | 弃答时幻觉 box 极少 ✅ |
| — OOD_confusion（近义 132） | decision abstain | **59.8%** | 近义跨域改善（5.6%→59.8%） |
| | strict wrong_object | **50.0%** | |
| **Overall** | MCC | 0.490 | |
| | FCA | 0.201 | |
| **OOD PlantVillage** | | | **采集域 confound 对照**（完整表见附录） |
| — OOD_absent | 弃答率 | **0.924** | 无目标跨域鲁棒 ✅ |
| — OOD_confusion | 弃答率 | **0.056** | 近义跨域脆弱 ⚠️；被 PlantWild 纠正为主因是采集域 confound |

### 4.1 指标呈现规范

- **摘要**：域内 A_FU 0.026、RefCOCOg 0.925、PlantWild no_target 弃答 96.8%；诚实写明 wrong_object 5.6%→59.8%（去 confound）但真实病 over-abstain 30.1%、frontier 未被已测 levers 外推。
- **HOR-FCA 3.2%（5/156，分母 156）** 作为"弃答质量"正面指标：no_target 弃答时极少幻觉错误 box。
- **A_FU 30.1%** 必须与正面数并列、不得隐藏。
- **同图 strict 25.0%（39/156）** 放附录或次要行，注明离线计算 provenance（within_image=null）。
- **A_Ground 不适用 PlantWild**（无 GT box），不可报。

---

## 5. Frontier 分析（论文核心图叙事）

### 5.1 曲线画法

横轴：**test_full A_FU**（真病漏诊率）
纵轴：**PlantVillage 近义弃答率**（或 L1_mismatch 弃答率）
方法：对 3reason ckpt-444 的 `P(answer)` 做逐样本候选阈值扫描。每个阈值对应一个 operating point，分别计算 test_full 的 A_FU 与 OOD_confusion 的 PV 近义弃答率，连线得到 matched-risk frontier。

### 5.2 曲线上的三类点

论文核心图包含三个类别的 operating point：

| 点类别 | 含义 | 位置 |
|--------|------|------|
| ⚪ **3reason 多阈值点（连续曲线）** | 同一模型扫全部 tau，给出整条 frontier 曲线 | 沿曲线从左下（低漏诊低弃答）到右上（高漏诊高弃答） |
| 🔵 **2reason ckpt-222/333（tau=0.5）** | 完整重训的二分类架构，不参与扫阈值 | 落在同一条曲线上（不外推） |
| 🔴 **旧 SFT ckpt-86（tau=0.5）** | 旧二分类 SFT，曾被误认为"高弃答好模型" | 曲线右上极端端（A_FU 0.632） |

### 5.3 曲线告诉我们的三件事

1. **权衡清晰**：从 A_FU 0.026→0.08 才能将 PV 近义弃答推至 0.35；要达到 0.50 需 A_FU 0.12，即真病漏诊约翻 5 倍——红线下不可接受。
2. **已测试方案未能外推**：2reason 两个 checkpoint 落在 3reason 阈值扫描形成的同一 frontier 附近。说明这不是简单更换 decision/reason 形式即可解决的问题；不把该结论外推到未测试 backbone。
3. **旧 SFT 不是"奇迹"**：ckpt-86 在曲线最右上端——高弃答以极高的漏诊为代价，在非对称风险约束下不可用。

### 5.4 风险-覆盖语言挂靠

本曲线借鉴 selective prediction 的 **risk-coverage** 分析思想，但两个轴按本任务的非对称风险重新定义：横轴是真病误弃 A_FU，纵轴是近义样本正确弃答率。它用于展示两个安全目标之间的经验 Pareto frontier，不将其等同于标准 selective classification 中的 error-vs-coverage 定义，也不宣称理论下界。

### 5.5 PlantWild 单数据集 Operating-Point Frontier（新诊断，已离线算出）

**与旧"域内 A_FU ↔ PlantVillage matched-risk frontier"区分**：本节是 PlantWild 自身三路样本的 P(answer) 阈值扫描，无 PlantVillage 参与。

**数据来源**：对 `predictions.jsonl` 的 p_answer 做阈值扫描（规则：p_answer≥τ 判 answer，否则按 p_wo/p_nt 弃答），无新增 GPU。

#### OOD_answer 的 p_answer 分布

- median 0.992，q25 0.233，q10 0.037，min 0
- 呈**双峰**——多数真实病高置信，约 25–30% 落入低置信尾部。

#### 阈值扫描结果（τ：A_FU / wrong_object 弃答 / no_target 弃答）

| τ | A_FU（真实病 over-abstain） | wrong_object 弃答 | no_target 弃答 |
|---|---------------------------|-------------------|---------------|
| 0.50（当前 argmax 工作点） | 0.301 | 0.591 | 0.962 |
| 0.05 | 0.115 | 0.265 | 0.904 |
| 0.02 | 0.071 | 0.152 | 0.878 |
| 0.01 | 0.045 | 0.098 | 0.833 |
| A_FU 地板 | ≈0.045（约 7/156 真实病 p_answer≈0，模型彻底判为 absent 的硬失败） | — | — |

#### 结论

- 要把 A_FU 从 0.30 降到 ~0.11，必须把 wrong_object 弃答从 0.59 拉到 0.27；降到 ~0.045 时 wrong_object 崩到 ~0.10。
- over-abstained 真实病与正确弃答的 wrong_object 在 P(answer) 空间**重叠**——这是**真实 frontier，不是可用单一 scalar 修复的全局 miscalibration**。
- no_target 全程 0.81–0.96 可分、鲁棒。

#### 直接含义（写入 rebuttal/limitation）

单纯 test-time threshold/temperature 只能沿此 frontier 移动，无法把真实病 over-abstain 与 wrong_object 弃答解耦；因此"重新校准阈值"或"paired margin 外推"对**跨数据集**无望。用 PlantWild 官方 val 的 label 选阈值属 domain adaptation（会改定位），只能作为 oracle 上界诊断、不能当部署方法。

**建议后续生成一张 PlantWild frontier 图**（answer-recall 或 A_FU 轴 vs wrong_object/no_target 弃答），属绘图，不需重训。

---

## 6. 消融

### 6.1 架构消融：2reason vs. 3reason

| 比较维度 | 3reason ckpt-444 | 2reason ckpt-222 | 2reason ckpt-333 |
|---------|-----------------|-----------------|-----------------|
| Decision head | 三分类：answer/wrong_object/no_target | 二分类：answer/abstain | 二分类：answer/abstain |
| Reason 输出 | 由 head 决定 reason 类别 | 由 LM 生成统一 reason | 由 LM 生成统一 reason |
| test_full A_FU | 0.026 | 0.108 | 0.110 |
| PV 近义弃答 | 0.056 | 0.472 | 0.583 |
| **Frontier 位置** | 低漏诊端 | 同一条曲线上 | 同一条曲线上 |

**结论**：2reason 通过更高的 A_FU 容忍（~0.11）换来更高的近义弃答（~0.47-0.58），但工作点落在 3reason 的同一条 matched-risk frontier 上，未外推 frontier。这说明当前权衡并非简单由 3reason 的 decision/reason 形式造成；结论不外推到未测试 backbone 与数据域。

### 6.2 阈值扫描消融

通过逐样本概率扫阈值生成整条 frontier，而非仅报告默认阈值的一个点。关键结果是：PV 近义弃答达到 0.35/0.50 时，test_full A_FU 分别升至 0.08/0.12。该消融给出完整的风险收益图景，避免把不同漏诊代价下的单点数字直接比较。

### 6.3 旧 SFT 反例消融

旧二分类 SFT 模型 ckpt-86 的关键解剖：

| 指标 | ckpt-86 |
|------|---------|
| L1 应答误弃（A_FU） | 0.632 |
| L1 应弃正确弃答 | 0.879 |
| 应弃与误弃 gap | 0.247（非常窄） |
| Reason 粒度 | 统一 `target_absent`，无 wrong_object/no_target 拆分 |
| 细粒度口径下 reason 正确率 | 归零（口径假象） |

**结论**：旧 SFT 的"高弃答(0.90)"是"什么都拒"策略的副产物，与 3reason 在同一 Pareto frontier 的极端端，非样本外新 frontier。其 reason 粒度不够，严格指标虚高是口径 artifact。

### 6.4 Reason 粒度消融

| Reason 方案 | 可解释性 | 是否支持细粒度归因 | L1 近义丢弃率 |
|------------|---------|------------------|-------------|
| 统一 `target_absent` | 低 | 否 | N/A（无法区分） |
| wrong_object / no_target 拆分 | 高 ✅ | 是 ✅ | 0.395（有意义的近义拒绝率） |

Reason 粒度拆分让我们能够独立分析"近义混淆"和"目标缺席"两种失败模式，在 OOD 场景中分别得到 0.056（细粒度跨域脆弱）和 0.924（缺席跨域鲁棒）——如果没有拆分，只会看到一个混合的"弃答率 X%"。

---

## 7. Limitation（诚实，不藏短板）

### 7.1 跨域细粒度近义弃答受限——PlantWild 与 PlantVillage 并存解读

**PlantWild = 主 OOD benchmark，正文主表。PlantVillage 保留为采集域 confound 的对照单点。**

- **PlantWild（同野外域，主 benchmark）**：wrong_object decision 弃答 **59.8%**（vs. PlantVillage 5.6%），证明去掉野外→实验室 confound 后近义判别显著改善；但真实病 over-abstain A_FU 30.1%，是跨数据集 operating point 的诚实代价。
- **PlantVillage（野外→实验室，对照）**：wrong_object 弃答仅 5.6%，根因为采集域 confound（shortcut learning），完整表格见附录。两者都不删——只留 PlantWild 会丢失机理对照；只留 PlantVillage 会丢失更好 benchmark 且像藏好数字。

**诚实定性**：0.056 不是"调参失败的糟糕数字"，而是**被刻画的边界**——我们不仅发现了这个局限，还通过 matched-risk frontier 证明了它是当前已测试方案下的经验权衡而非欠训练/欠数据，通过归因实验锁定了 root cause（采集域 confound），明确了 method scope。PlantWild 59.8% 进一步佐证该边界的主要成因是域 shortcut 而非细粒度本身不可做。

- **已排除的替代解释**：
  - ❌ 不是没教过：训练集 100% 覆盖了 PV 测试的 20 个病害对。
  - ❌ 不是作物 OOD：所有 PV 近义样本均为 seen crop（tomato/corn/apple）。
  - ❌ 不是表征不可分：train/test 内部探针 AUC 高，表征内部可区分，分布漂移是症结。
- **实际含义**：在跨采集域的细粒度病害判别场景下，当前模型无法可靠弃答。这不否认在"同域"（same domain L1_mismatch，弃答 0.395）或在"典型 OOD 检测"（缺席检测 0.924-0.968）场景下的表现。

### 7.2 数据物种与病害覆盖有限

- 训练数据仅覆盖 **3 种作物**（tomato、corn、apple）及其病害，母本数据 973 条已耗尽。
- 更多作物的近义病害对缺乏标注，无法通过加入更多 wrong_object 数据缓解。
- 这限制了结论对更广泛作物/病害空间的泛化性声明。

### 7.3 单 Backbone

所有实验基于 **Qwen3-VL-8B** 单一 backbone。当前"未外推"结论仅验证了同一 family 的架构变化（2reason vs. 3reason）以及 cost-CE lever，尚未在完全不同的 backbone（如 LLaVA、InternVL）上验证。

### 7.4 OOD Gate 不适用

我们测试了基于 feature 距离和作物新颖度的 OOD gate 方案，发现在 seen crop 框架下，gate 无法在 decision 层区分细粒度病害，因为病害特征在 feature space 中高度重叠。这一路径已正式排除。

---

## 8. 已排除的路径（附录性质）

| 已排除路径 | 核心理由 | 一句话结论 |
|-----------|---------|-----------|
| **SupCon 提升表征 AUC** | 动机被削弱：train/test 内部探针 AUC 本身已高，问题不是表征不可分，而是跨域分布漂移 | 杠杆不对 |
| **Extraction gap / 重训标准头** | 公平对比后：部署校准头已接近重训头性能；之前 gap 是 dump 未挂 calibrated head 的错基线 | Gap 不存在 |
| **OOD gate（feature 距离/作物新颖度）** | 作物新颖度假设不成立——PV 近义全部是 seen crop；病害特征高度重叠，gate 分不开 | 假设不成立 |
| **加 wrong_object 数据** | 母本 973 条已耗尽，仅 3 作物，无更多近义标注可用 | 数据不可得 |
| **RL（含 box RPO）** | decision 由 head 控制，policy gradient 优化 LM token 是错杠杆；frontier 证明只能沿线移动无法外推；有破坏 A_FU 0.026 的风险；box 已 0.925 不值得引入复杂度 | 错杠杆全砍 |
| **用 DPO 修 decision** | DPO 优化的是 LM token 概率，不可靠地改变 head 行为 | 错杠杆 |

---

## 9. 审稿人质疑应对（Rebuttal 预案）

### (a) "PlantWild 真实病 A_FU 30.1%，方法跨数据集不可用"

> **我们回应**：我们承认，在当前已测试方案下，不能宣称方法跨数据集可用。
>
> 1. **PlantWild single-dataset threshold frontier 诊断**：PlantWild 单数据集内阈值扫描（tau 0.5→0.05）给出 operating point 沿 frontier 移动：A_FU 从 0.301 降至 0.115，同时 wrong_object 弃答从 0.591 降至 0.265，no_target 弃答从 0.962 降至 0.904。这证明单一 scalar（threshold/temperature）只能沿此 frontier 移动，无法将真实病 over-abstain 与 wrong_object 弃答解耦；no_target 检测全程保持 0.90+ 稳健。
> 2. **PlantVillage 5.6% vs PlantWild 59.8% 是采集域 confound 证据，非细粒度不可做**：去掉野外→实验室 confound 后 wrong_object 弃答从 5.6% 大幅改善至 59.8%，佐证 PlantVillage 的崩塌主因是域 shortcut，而非细粒度判别本身不可做。但 PlantWild 真实病 over-abstain 30.1% 是跨数据集 operating point 的诚实代价——该 frontier 在当前已测试 lever 下未被外推。
> 3. **缺席检测是亮点**：OOD_absent 弃答率 0.924（PlantVillage）/ 0.968（PlantWild），说明"目标不存在"检测在跨域场景下是鲁棒的。

### (b) "新意薄，只是微调了现成 VLM"

> **我们回应**：本工作的新意体现在三个层面：
>
> 1. **评估视角/经验诊断**：matched-risk frontier 作为评估视角，将逐样本风险扫描用于选择性 grounding，揭示真病误弃↔近义弃答在已测试方案下的经验权衡。该 frontier 是实证诊断结果，**明确不宣称理论 Pareto 下界**。
> 2. **归因框架**：wrong_object/no_target 的三分类 reason 拆分，明确区分两种弃答模式，并独立评估其跨域迁移性。
> 3. **非对称风险约束**：以 A_FU 为红线，不用显著增加真病误弃来换取单一 OOD 指标；核心定位是优先控制真实病害漏诊风险。
>
> 需诚实说明：benchmark 构造（PlantWild 同图三路 paired）、三分类 reason 归因框架、以及跨域迁移的实证诊断，才是本文的主要贡献主体；matched-risk frontier 本身是评估视角，非理论方法创新。

### (c) "农业 niche，泛化性不够"

> **我们回应**：我们使用 **selective prediction / risk-coverage** 的通用语言来框定工作，不限于农业。
>
> - Risk-coverage curve 是 selective classification 领域的核心工具（Geifman & El-Yaniv 2017, 2019），本工作将其拓展到 grounding + reason 场景。
> - Shortcut learning 与 domain shift 是 ML 通用问题，农业场景是具体 case study。
> - RefCOCOg 0.925 表明主模型仍具备较强的通用 grounding 能力；是否相对无选择性基线退化，应以最终公平对照表为准。

### (d) "为何不用 RL / DPO / policy gradient 优化弃答"

> **我们回应**：经多轮实验论证，RL 是**错杠杆**：
>
> 1. 弃答决策由三分类 head 的 logits 控制，policy gradient 直接优化的是 LM token 的生成概率，不是 head 的 decision boundary——是不对齐的优化目标。
> 2. Frontier 实证：2reason 与 3reason 落在同一条 frontier 上，说明已测试的决策形式主要改变 operating point，未外推 frontier。
> 3. 风险：A_FU 0.026 的红线极容易因 RL 波动而破坏，且 grounding 已达 0.925（RefCOCOg），无改进空间不值得引复杂度。
> 4. 已测试 cost-sensitive CE（λ=3/5/8）未外推 frontier，结论为 null；当前 frontier 在所有已测试 lever 下稳健。

### (e) "数据/作物太少，结论无法泛化"

> **我们回应**：我们诚实承认数据限制（3 作物、母本 973 耗尽）。但：
>
> 1. 本文的定位不是"通用农业病害 grounding 系统"，而是"在有限资源下，刻画非对称风险约束下的选择性 grounding 边界"。
> 2. 在当前 3 作物范围内，多个阈值工作点与 2reason 重训结果呈现一致权衡，足以支持本文的域内结论；跨作物、跨 backbone 泛化仍需后续验证。
> 3. RefCOCOg 0.925 表明 grounding 能力在通用域不掉点，说明问题不在于"VLM 能力差"而在"domain-specific 细粒度判别难"。

---

## 10. 下一步 / 待办

### 10.1 Cost-sensitive CE 实验（已完成·null）——纠正旧"建议补做"

**旧版 §10.1 将 cost-CE 列为"建议优先补做"，该实验已做完，结论为 null，不再列为待办。**

**设置**：仅提高 wrong_object 类 CE 权重 λ=3/5/8，其余完全相同；各用校准头同口径评估。

| λ | test A_FU | MCC | PV 近义弃答 |
|---|-----------|-----|------------|
| λ=3 | 0.101 | 0.834 | 0.352 |
| λ=5 | 0.110 | 0.837 | 0.139（不单调） |
| λ=8 | 0.116 | 0.829 | 0.500 |

**结论**：无 A_FU≤0.035 的点且不单调；基础成功与强成功判据均失败；仍沿原 frontier，仅可作消融支持"简单 class weighting 推不外"。RL/DPO/SupCon/OOD gate 保持已排除，不再推荐。

### 10.2 论文写作挂靠文献方向

| 方向 | 关键引用 | 本工作连接点 |
|------|---------|------------|
| Selective prediction / risk-coverage | Geifman & El-Yaniv (2017, 2019), MSP | Matched-risk frontier 直接继承其曲线方法论 |
| Selective classification with abstention | Cortes et al. (2016), Wiener & El-Yaniv (2015) | 非对称风险约束（asymmetric cost） |
| Shortcut learning / domain generalization | Geirhos et al. (2020), Arjovsky et al. (2019) | 跨域细粒度判别失败归因 |
| OOD detection / open-set recognition | Hendrycks & Gimpel (2017), Liang et al. (2018) | 对照范式——不追单一 OOD 数字，而刻画经验权衡 |
| Visual grounding / referring expression | Mao et al. (2016), Kazemzadeh et al. (2014) | RefCOCOg 作为 grounding 通用基准 |

### 10.3 下一步具体行动（按优先级）

1. **（最高优先）落地已算出的 PlantWild 单数据集 operating-point frontier**（复用 predictions.jsonl，无需重训）：出图 + 写进正文，定性 30.1% 缺口的性质（真实 frontier、非简单 miscalibration 可解耦）。
2. **冻结故事线**：定位为"域内安全选择性 grounding + no_target 跨域鲁棒 + 跨数据集 frontier 的诚实刻画（PlantWild/PlantVillage confound 对照）"；摘要脱离"取胜 wrong_object/no_target"措辞；进入写作。
3. **不做**：不为主会跑 IRM/GroupDRO/style-aug/paired-margin 重训（动机弱、多变量、耗卡），除非有新证据显示 frontier 有可利用 slack；不做跨数据集 capability gate（属筛类刷分，不诚实）；不再复跑已证伪的 cost-CE/RL/SupCon/OOD gate。把 style-aug 仅作"future work、动机部分成立"列出。

---

## 11. 投稿评估（诚实定级）

### 11.1 强项

| 维度 | 详情 |
|------|------|
| **红线达标** | A_FU 0.026 达到项目预设真病误弃红线，符合核心风险约束 |
| **Grounding 能力强** | RefCOCOg 0.925，表明主模型保留较强的通用 grounding 能力 |
| **Benchmark 严格** | PlantWild 同图三路 paired 构造（156 图 ×3，去重、统一像素、人工审查），将细粒度近义弃答从采集域 confound 中隔离 |
| **诚实归因** | 不藏短板，multi-dataset 并存：PlantWild 主 + PlantVillage 对照，诚实呈现 30.1% over-abstain 与 frontier 诊断 |
| **通用框架语言** | 用 risk-coverage / selective prediction 等通用术语框定，不被农业 niche 限制 |

### 11.2 弱项

| 维度 | 详情 |
|------|------|
| **OOD 真实病 over-abstain 30.1%** | 跨数据集时 answer 工作点整体偏向弃答，绝对值可能被审稿人质疑"不可用" |
| **正向方法成果集中在域内** | OOD 部分本质是 negative/characterization 结果（wrong_object strict 50%、over-abstain 30.1%），缺干净的方法 win |
| **数据规模小** | 仅 3 作物、973 条母本，可能被认为量不够 |
| **架构局限** | 单 backbone（Qwen3-VL-8B），2reason/3reason 在同一 family 内比较 |
| **无部署端实验** | 未展示实际场景（如手机端实时推理）的端到端效果 |

### 11.3 被拒风险点

| 风险等级 | 风险点 | 缓解策略 |
|---------|--------|---------|
| 🔴 高 | 被判为"执行良好的负结果/分析，缺方法贡献，农业 niche，OOD 数字不足以宣称跨域可用" | ✅ 用 frontier 诊断（over-abstain 与 wrong_object 弃答在 P(answer) 空间重叠、单 scalar 不可解耦）+ shortcut/域动机预先回应 |
| 🟡 中 | "为何不试 IRM/style-aug/DG" | ✅ PlantWild 属同野外域，style-aug 动机偏弱，不作为本次投稿实验；future work 列出 |
| 🟡 中 | "新意不足"——被归类为"又一个微调 VLM 做农业" | ✅ 诚实定位为 benchmark/empirical-study 取向，非方法取胜 |
| 🟢 低 | "数据太少"——generalization claim 被质疑 | ✅ 诚实定位为边界刻画而非通用系统 |

### 11.4 最大化接收策略

1. **摘要前置诚实定位**：第一句就写"绘制选择性 grounding 迁移边界 + 机理诊断"，而非"取胜跨域 OOD"。
2. **前置安全结果**：A_FU 0.026 + RefCOCOg 0.925 + no_target 96.8% 在第一段就给。
3. **通用语言框定**：用 risk-coverage / selective prediction 开篇，而非只按农业应用展开。让审稿人在方法论层面先理解贡献，再展示农业 domain case study。
4. **Frontier 诊断入正文**：PlantWild 单数据集 threshold scan 已离线算出，无需重训，直接出图纳入正文支撑"单一 threshold/temperature 不可解耦"论点。

### 11.5 诚实把握定级（AAAI 主会）

| 状态 | 把握 |
|------|------|
| **现状（含本次 offline frontier 诊断）** | **Borderline / 偏弱 borderline**。强项：benchmark 严格、诊断诚实、域内安全强、no_target 迁移稳。弱项：正向 method 成果集中在域内；OOD 部分本质是 negative/characterization 结果（over-abstain 30.1%、wrong_object strict 50%）；novelty 偏实证而非方法。 |
| **更适合定位** | 更适合 benchmark/empirical-study 取向的主会投稿，或需一个真正 method win 才有把握。不要过度包装。 |

---

## 附录：证据链/来源

以下为合作者可追溯的仓库内文件路径：

| 内容 | 文件路径 |
|------|---------|
| Matched-risk frontier 关键数字表 | `outputs/matched_risk/frontier_key_points.md` |
| Matched-risk 离线阈值分析完整报告 | `outputs/matched_risk/REPORT.md` |
| 3reason ckpt-444 模型卡片 | `outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/README.md` |
| 3reason 全 tau 扫描结果 CSV | `outputs/matched_risk/frontier_3reason.csv` |
| 2reason ckpt-222/333 工作点 CSV | `outputs/matched_risk/operating_points_2reason.csv` |
| Frontier 曲线图 | `outputs/matched_risk/frontier_plot.png` |
| A_FU vs. 整体指标图 | `outputs/matched_risk/frontier_overall.png` |
| Tau 扫描多指标曲线图 | `outputs/matched_risk/tau_sweep.png` |
| 旧 SFT 分析（反例） | 参见 `outputs/sft_runs/fusion/fusion_hl05/README.md` 及 `fusion_hl05/checkpoint-92/README.md` |
| **PlantWild paired metrics（新增）** | `new_reason/outputs/ood_plantwild_eval/sftfinal_ckpt444_paired_262k/metrics.json` |
| **PlantWild 逐样本 head 概率** | 同目录 `predictions.jsonl` |
| **PlantWild 构造脚本（含 12 项自检）** | `new_reason/scripts/build_ood_plantwild_paired.py` |
| **PlantWild 辅助构造脚本** | `build_ood_plantwild.py`、`prepare_ood_plantwild_eval_images.py` |
| **PlantWild benchmark 数据** | `new_reason/outputs/eval_probes/ood_plantwild_paired_262k.jsonl` |
| **PlantWild 构造统计** | `new_reason/outputs/ood_candidates/plantwild_benchmark/paired_build_stats.json` |

---

> **文档版本**：v2
> **编写日期**：2026-07-11
> **版本说明（v2 主要变更）**：
> 1. 新增 PlantWild paired 同图三路 OOD benchmark 结果 + 采集域 confound 对照分析
> 2. 新增 PlantWild 单数据集 within-dataset operating-point frontier 诊断
> 3. cost-sensitive CE 实验从"建议补做"改为"已完成·null"
> 4. 贡献重写（C1–C3, §1 TL/DR）：从"取胜"叙事改为"刻画边界 + 机理诊断"
> 5. 投稿把握重新定级为 Borderline / 偏弱 borderline，下一步行动明确为 3 项
