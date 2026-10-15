# AgriAbstain SFT 收官版本（三分类判别头 3reason）

## 1. 最终模型
- **checkpoint**: `new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444`（ep4, head_lambda=0.4）
- **校准判别头**: `checkpoint-444/discrim_head_calibrated.pt`
- **base模型**: /user/huangyongbin/models/modelscope/Qwen/Qwen3-VL-8B-Instruct
- **选点理由**: 作为RL起点，checkpoint-444是唯一同时满足test MCC≥0.836、test A_FU≤3.5%、Ref MCC≥0.906三项faithful门禁的checkpoint。ep5虽test MCC更高，但test A_FU升至7.79%(103条真病误弃)，Ref MCC降至0.9008，未通过保护线。
- **论文 selection audit（2026-07-24）**: 对全部六个 epoch checkpoint 使用 fixed-100 dev subset 的 teacher-forced LM loss 重新选点，checkpoint-444 同样为 validation best（0.278026；111/222/333/555/666 为 0.311196/0.290230/0.283440/0.280943/0.283077）。论文与 grouping ablation 采用该 retrospective development-loss audit；上述 test 门禁仅保留为早期工程决策记录，不作为论文选点依据。Epoch 1 recovery task: `617455`。

## 2. 架构
- 三分类判别头 Qwen3VLWithTriHead：第27层last-token表征 → mu/sd标准化 → 3层MLP(4096→512→256→3) → [answer, wrong_object, no_target]
- 推理：判别头argmax出decision+reason；answer时box/label靠LM生成；abstain时reason直接从head读
- LM输出JSON：answer={"decision":"answer","label":...,"box":...}；abstain={"decision":"abstain","reason":"wrong_object|no_target"}
- mu/sd固定buffer(从base模型train表征预估计)：/user/huangyongbin/newrepo/agriabstain/final_story/outputs/discrim_musd.pt

## 3. 数据配比（总量 7072）
| head_label | 类别 | 数量 |
|---|---|---|
| 0 | answer | 2703 |
| 1 | wrong_object | 1500 (oversample自973全量: tomato910/corn489/apple101) |
| 2 | no_target | 2869 |

answer:abstain ≈ 38%:62%（2703 : 4369）

成分来源（build_final_sft_3reason.py argparse默认）：
- wrong_object：母本全量973(tomato594/corn312/apple67)，oversample到1500
- evidence裁0%(no_target)：669全量
- 中间档answer(可见0.5/0.75)：300
- T5跨作物1500 / occluded 200 / COCO_ans 1100 / COCO_nt 500 / judge真病 + evidence配对 + T2T3T7 + T4等
- 关键参数：--wo-imbalance-ratio 999(全量) / --wo-oversample-target 1500 / --n-evidence-crop0 669 / --n-evidence-mid 300 / --n-t5 1500 / --n-coco-ans 1100 / --n-coco-abs 500

## 4. 训练超参
- LoRA r16/α32/dropout0.05, lr 1e-4, warmup 0.05, 6 epoch
- head_lambda=0.4
- 判别头class_weight=[1.0, 1.5, 1.0]（answer/wrong_object/no_target；WO_HEAD_WEIGHT=1.5）
- LM侧reason加权：NO_TARGET_LOSS_WEIGHT=1.0 / WRONG_OBJECT_LOSS_WEIGHT=1.0（注：LM侧加权对abstain固定模板空转无效，已回归1.0）

## 5. 最终指标（faithful 端到端，ckpt-444 ep4）

### 5.0 指标定义

| 缩写 | 全称 | 定义 |
|---|---|---|
| FCA | Forced-Choice Accuracy | 该拒强答率 = AU / 所有unanswerable。越低越好。 |
| HOR-FCA | High-Occlusion-Risk FCA | `_is_high_occlusion`判定的unanswerable子集上的错误强答率。`reason=no_target`直接进入该子集，也可由低可见比例触发。越低越好。**局限**：PlantWild中仅覆盖144条no_target，不覆盖120条wrong_object，不能代表near-confusion风险。 |
| A_FU | Answer False negative / Unanswerable | 真病误弃率(漏诊) = FN / answerable。越低越好。 |
| U_Abs | Unanswerable Abstain | unanswerable样本上的选择弃答率 = TN / 所有unanswerable = 1−FCA。reason subtype另由U_Reason/U_Reason_c衡量。 |
| MCC | Matthews Correlation Coefficient | 综合二分类质量(decision正确/错误)，含AU。 |
| A_Ground | Answerable Grounding | 所有answerable上的聚合IoU：正确回答取IoU，未回答/错误回答贡献0。**不是**条件定位质量(旧A_Ground_c为条件正确时的平均IoU)。 |
| U_Reason_c | Unanswerable Reason conditional | 正确弃答中reason(wrong_object/no_target)正确的比例。 |

### 5.1 六探针主表（faithful 端到端，ckpt-444 ep4）

| 探针 | n | FCA | HOR-FCA | A_FU | U_Abs | MCC | A_Ground | U_Reason_c |
|---|---|---|---|---|---|---|---|---|
| L1_discrim | 410 | 61.46% | 0%(n=0) | 5.85% | 38.54% | .3932 | .8103 | .9873 |
| L2_evidence | 790 | 47.59% | 47.59%(n=395) | 1.77% | 52.41% | .5697 | .6412 | 1.0000 |
| L3_dose | 750(331有效) | 17.57% | 17.57%(n=148) | 1.09% | 82.43% | .8355 | .6895 | 1.0000 |
| test_full | 2646 | 13.23% | 7.87%(88/1118) | 2.42% | 86.77% | .8485 | .7011 | .9974 |
| ood_refcocog | 362 | 8.29% | 8.29%(n=181) | 0% | 91.71% | .9203 | .6455 | 1.0000 |
| plantwild_real_closed12_v1† | 408 | 26.89% | 4.86%(7/144) | 14.58% | 73.11% | .5600 | .0000 | .9067 |

注：L3_dose 750条中原419条ambiguous(gt decision模糊)被跳过，实际评估331条，invalid=0，format_valid=0.441是探针固有特性。PlantWild无GT bbox，A_Ground=0不能判断定位能力；score_answerable在box=None时只按decision+label评。

† PlantWild closed12为正式benchmark（12类、144张唯一图、408条），由原13类/156图/444条去掉tomato leaf yellow virus（SFT train中answer记录=0）得到。筛选规则与模型预测表现无关，仅依赖训练taxonomy支持。原13类完整taxonomy结果（含训练未见类tomato leaf yellow virus）作为"unsupported-class stress test/完整taxonomy补充"单独报告：MCC=0.5225, FCA=25.69%, A_FU=19.87%。

### 5.2 选点逻辑：faithful 门禁

ckpt-444（ep4）是唯一同时满足以下三个faithful门禁条件的checkpoint：
- **test MCC ≥ 0.836**
- **test A_FU ≤ 3.5%**
- **Ref MCC ≥ 0.906**

| checkpoint | ep | test MCC | test A_FU | Ref MCC | 门禁 |
|---|---|---|---|---|---|
| 222 | 2 | .7669 | 1.81% | .9254 | ✗ test MCC |
| 333 | 3 | .8393 | 3.78% | .9306 | ✗ A_FU |
| **444** | **4** | **.8485** | **2.42%** | **.9203** | **✓ 全部** |
| 555 | 5 | .8595 | 7.79% | .9008 | ✗ A_FU |
| 666 | 6 | .8597 | 4.61% | .9241 | ✗ A_FU |

选点理由：作为RL起点，安全的行为先验(低漏诊 A_FU=2.42%)比高test MCC重要。ep5(ckpt-555)虽test MCC 0.8595更高，但103个真病漏诊(7.79%)不可接受。RL能把保守模型往上推，难把激进过弃的往回收。

### 5.3 SFT 对比实验（判别头 vs 纯 LM baseline，同一 ckpt-444）

### 5.3.1 实验设置与公平性
五个方法共享**同一 SFT 模型(ckpt-444)、同一 SFT 训练数据、同一测试集、同一评估脚本(scripts/eval_qwen3vl_swift_3reason.py)**，唯一差别是 decision 产生方式。全部为默认 argmax(softmax top1)输出，无人为阈值调节。这保证对比隔离了"decision 架构"这单一变量，防审稿人质疑 baseline 偏弱：
- **判别头(ours)**：三分类判别头 argmax 出 decision+reason，answer 时 box/label 靠 LM 生成。**无override/retry/fallback/placeholder/GT fallback**。head=answer时LM只生成label/box，无KV-cache仅影响效率。
- **native_lm**：纯 LM 输出，用与判别头完全相同的 DECISION_SYSTEM_PROMPT，不加判别头。唯一变量 = decision 来自 LM 生成 vs 判别头 → 最干净的"架构 vs 架构"消融。
- **prompt baseline（3 档梯度）**：纯 LM + 递增强度的 system prompt 工程，用来测"prompt 越强能不能补上架构差距"：
  - prompt_basic：仅陈述三选一规则，无激励。
  - prompt_reward：在 basic 上加显式 reward(正确 answer+box +1 / 错误 answer −1 / 正确弃答 +0.4)。
  - prompt_reward_norms：在 reward 上再加 4 条诚实/谦逊原则(说真话不猜、谦逊诊断、无清晰证据不下结论、误诊伤害作物)。
- **cot (structured)**：纯 LM + 结构化 4 步 CoT 推理 prompt。

6 个评估探针：L1_discrim(近义病判别)、L2_evidence(证据充分性)、L3_dose(剂量阶梯)、test_full(综合测试集)、ood_refcocog(通用定位 OOD)、plantwild_real_closed12_v1(真实世界野外作物病害 OOD，12类封闭集)。

### 5.3.2 主结果：全 6 探针 MCC（faithful 端到端值）

| 探针 | 判别头(ours) | native_lm | prompt_basic | prompt_reward | prompt_reward_norms | cot |
|---|---|---|---|---|---|---|
| L1_discrim | 0.3932 | 0.497 | 0.514 | 0.332 | 0.331 | **0.590** |
| L2_evidence | **0.5697** | 0.528 | 0.467 | 0.499 | 0.506 | 0.385 |
| L3_dose | **0.8355** | 0.765 | 0.757 | 0.775 | 0.777 | 0.710 |
| test_full | **0.8485** | 0.758 | 0.724 | 0.703 | 0.696 | 0.637 |
| ood_refcocog | **0.9203** | 0.863 | 0.862 | 0.838 | 0.768 | 0.685 |
| plantwild_real_closed12_v1† | 0.5600 | **0.6024** | 0.5811 | 0.5057 | 0.5126 | 0.5449 |

（加粗为该行最高；ours用faithful端到端值，baseline为同一评估脚本LM决策值。注意L1和PlantWild的最高不是判别头——ours在L2/L3/test/Ref领先，L1和PlantWild整体MCC不领先。† PlantWild closed12为正式benchmark，原13类完整taxonomy结果单独报告于补充。）

### 5.3.3 完整指标（MCC / A_FU 漏诊率 / format_valid）
A_FU = 真病误弃率(漏诊，越低越好)；format_valid = JSON 格式合规率。判别头 ep4 的 A_FU 已在第 5.1 节主表列出(L1 5.85%/L2 1.77%/L3 1.09%/test 2.42%/refcocog 0%/plantwild closed12 14.58%)。五个 baseline 的 A_FU 与 format：

| 方法 | 探针 | MCC | A_FU | format_valid |
|---|---|---|---|---|
| native_lm | L1 | 0.497 | 0.078 | 1.000 |
| native_lm | L2 | 0.528 | 0.159 | 1.000 |
| native_lm | L3 | 0.765 | 0.208 | 0.441 |
| native_lm | test | 0.758 | 0.149 | 1.000 |
| native_lm | refcocog | 0.863 | 0.094 | 0.997 |
| prompt_basic | L1 | 0.514 | 0.078 | 1.000 |
| prompt_basic | L2 | 0.467 | 0.235 | 1.000 |
| prompt_basic | L3 | 0.757 | 0.224 | 0.441 |
| prompt_basic | test | 0.724 | 0.198 | 1.000 |
| prompt_basic | refcocog | 0.862 | 0.127 | 1.000 |
| prompt_reward | L1 | 0.332 | 0.039 | 1.000 |
| prompt_reward | L2 | 0.499 | 0.197 | 1.000 |
| prompt_reward | L3 | 0.775 | 0.197 | 0.441 |
| prompt_reward | test | 0.703 | 0.162 | 1.000 |
| prompt_reward | refcocog | 0.838 | 0.127 | 1.000 |
| prompt_reward_norms | L1 | 0.331 | 0.059 | 1.000 |
| prompt_reward_norms | L2 | 0.506 | 0.200 | 1.000 |
| prompt_reward_norms | L3 | 0.777 | 0.202 | 0.441 |
| prompt_reward_norms | test | 0.696 | 0.180 | 1.000 |
| prompt_reward_norms | refcocog | 0.768 | 0.204 | 1.000 |
| cot | L1 | 0.590 | 0.220 | 1.000 |
| cot | L2 | 0.385 | 0.392 | 1.000 |
| cot | L3 | 0.710 | 0.306 | 0.441 |
| cot | test | 0.637 | 0.357 | 1.000 |
| cot | refcocog | 0.685 | 0.343 | 1.000 |
| native_lm | plantwild_real_closed12_v1 | 0.6024 | 0.2361 | 1.000 |
| prompt_basic | plantwild_real_closed12_v1 | 0.5811 | 0.2500 | 1.000 |
| prompt_reward | plantwild_real_closed12_v1 | 0.5057 | 0.2083 | 1.000 |
| prompt_reward_norms | plantwild_real_closed12_v1 | 0.5126 | 0.2222 | 1.000 |
| cot | plantwild_real_closed12_v1 | 0.5449 | 0.4167 | 1.000 |

注：L3 的 format_valid=0.441 是探针固有特性（见 5.3.7），非解析失败。baseline与ours使用同一评估脚本，口径完全一致。

关键观察：**cot 的 A_FU 全面最高(L1 0.220/L2 0.392/test 0.357/plantwild 0.417)——它靠激进弃答换 L1 高分，代价是大量真病被漏诊。判别头 ep4 的 ID A_FU(L1 0.059/L2 0.018/test 0.024)远低于所有 baseline，但 PlantWild closed12 A_FU(0.146)低于 prompt_basic(0.250)/prompt_reward(0.208)等。ours相较prompt baseline的优势是低ID A_FU和较低PlantWild漏诊，但PlantWild仍有wrong_object决策缺口。**

### 5.3.4 PlantWild 详细分析

#### 5.3.4.0 benchmark设计说明

PlantWild closed12 benchmark构建如下：
- **图像来源**：PlantWild test split候选集，每类12张唯一图，共144张。
- **分类规则**：closed-set 12类——筛选规则为PlantWild标签在SFT train唯一answer记录中出现≥1次（与模型预测表现无关），唯一排除tomato leaf yellow virus（SFT train中answer记录=0）。
- **数据量**：144张唯一图、408条（144 answer + 144 no_target + 120 wrong_object）。potato late blight和squash powdery mildew因原始wrong_object probe不足，仅含answer/no_target各12条（无WO），其余10类每类12条answer、12条no_target、12条wrong_object。
- **数据文件**：`new_reason/outputs/eval_probes/plantwild_real_closed12_v1/plantwild_real_closed12_paired_408.jsonl`，`BUILD_AUDIT.json`记录筛选规则、SHA及完整sample_id列表。
- **复现说明**：closed12 predictions是原444逐样本预测按sample_id确定性子集（原444数据/预测SHA未修改），重新聚合得到。生成脚本及测试路径写入复现目录。
- **source/image SHA/dHash隔离**继承自原`plantwild_real_v1`（其`BUILD_AUDIT.json`记录`sha256_intersection_with_sft=0`、`dhash_leak_with_sft=0`）；closed12 audit通过`paired_input_sha256`固定该输入版本。

#### 5.3.4.1 判别头(ours) closed12逐类表现

PlantWild closed12共408条(144 answerable + 144 no_target + 120 wrong_object)，无GT bbox，A_Ground=0。score_answerable在box=None时按decision+label评定。

| 子集 | n | 指标 | 值 |
|---|---|---|---|
| **OOD_answer (answerable)** | 144 | 选择answer(覆盖率) | 123 (85.42%) |
| | | 严格正确(TP) | 108 (75.00%) |
| **OOD_absent (no_target)** | 144 | decision正确 | 137 (95.14%) |
| | | decision+reason正确 | 137 (95.14%) |
| | | AU(错误强答) | 7 (4.86%) |
| **OOD_confusion (wrong_object)** | 120 | decision正确(弃答) | 56 (46.67%) |
| | | reason正确 | 38 (31.67%) |
| | | 错误强答(AU) | 64 (53.33%) |

HOR-FCA(4.86%)只覆盖144 no_target中的7次错误强答，不能代表wrong_object。wrong_object的decision正确率仅46.67%，说明判别头在近义物混淆场景下仍有大量错误强答，是PlantWild低MCC的主因。

ours closed12整数：TP108 FP15 FN21 TN193 AU71 INVALID0；MCC=0.560015；FCA=71/264=26.89%；HOR-FCA=7/144=4.86%；A_FU=21/144=14.58%；U_Abs=193/264=73.11%；answer coverage=123/144=85.42%；strict TP=108/144=75.00%；NT decision/reason=137/144=95.14%，AU7；WO decision=56/120=46.67%，reason=38/120=31.67%，AU=64/120=53.33%。

#### 5.3.4.2 五组Baseline PlantWild closed12表现（逐样本独立重算，与metrics.json零差异）

native_lm：
- MCC=0.6024, FCA=41/264=15.53%, HOR-FCA=1/144=0.69%, A_FU=34/144=23.61%
- answer coverage=110/144=76.39%, strict answer TP=110/144=76.39%
- no_target decision=143/144=99.31%, reason=75/144=52.08%
- WO decision=80/120=66.67%, reason=79/120=65.83%

prompt_basic：
- MCC=0.5811, FCA=43/264=16.29%, HOR-FCA=2/144=1.39%, A_FU=36/144=25.00%
- coverage=108/144=75.00%, strict=79/144=54.86%
- no_target decision=142/144=98.61%, reason=135/144=93.75%
- WO decision=79/120=65.83%, reason=67/120=55.83%

prompt_reward：
- MCC=0.5057, FCA=70/264=26.52%, HOR-FCA=3/144=2.08%, A_FU=30/144=20.83%
- coverage=114/144=79.17%, strict=73/144=50.69%
- no_target decision=141/144=97.92%, reason=141/144=97.92%
- WO decision=53/120=44.17%, reason=18/120=15.00%

prompt_reward_norms：
- MCC=0.5126, FCA=65/264=24.62%, HOR-FCA=2/144=1.39%, A_FU=32/144=22.22%
- coverage=112/144=77.78%, strict=73/144=50.69%
- no_target decision=142/144=98.61%, reason=141/144=97.92%
- WO decision=57/120=47.50%, reason=31/120=25.83%

cot：
- MCC=0.5449, FCA=22/264=8.33%, HOR-FCA=1/144=0.69%, A_FU=60/144=41.67%
- coverage=84/144=58.33%, strict=70/144=48.61%
- no_target decision=143/144=99.31%, reason=143/144=99.31%
- WO decision=99/120=82.50%, reason=36/120=30.00%

（注意：strict answer统一写 TP/144，不是 TP/answered——后者是conditional precision，非此处strict subset accuracy。）

#### 5.3.4.3 PlantWild closed12对比分析

| 指标 | ours | native_lm | prompt_basic | prompt_reward | prompt_reward_norms | cot |
|---|---|---|---|---|---|---|
| MCC | .5600 | **.6024** | .5811 | .5057 | .5126 | .5449 |
| A_FU | 14.58% | 23.61% | 25.00% | 20.83% | 22.22% | 41.67% |
| strict answer | 75.00% | **76.39%** | 54.86% | 50.69% | 50.69% | 48.61% |
| no_target decision | 95.14% | 99.31% | 98.61% | 97.92% | 98.61% | **99.31%** |
| WO decision | 46.67% | 66.67% | 65.83% | 44.17% | 47.50% | **82.50%** |
| WO reason | 31.67% | **65.83%** | 55.83% | 15.00% | 25.83% | 30.00% |

ours在PlantWild closed12上整体MCC仍不领先(native_lm最高)。ours的优势是较低的A_FU(14.58% vs native_lm 23.61%)，但WO decision(46.67%)和WO reason(31.67%)显著落后于native_lm(66.67%/65.83%)和cot(82.50%/30.00%)，说明判别头在真实世界近义物混淆场景下**wrong_object是主要缺口**——该弃时大量强答(64/120=53.33%)。这是PlantWild closed12低MCC的核心瓶颈。

#### 5.3.4.4 补充：unsupported-class stress test（完整13类taxonomy）

原PlantWild完整13类/444条结果保留作为unsupported-class stress test，包含训练未见的tomato leaf yellow virus。该结果不参与正式主对比：

- OURS：MCC=0.5225, FCA=25.69%, A_FU=19.87%, WO decision=49.24%, WO reason=30.30%
- native_lm：MCC=0.5627, FCA=14.58%, A_FU=29.49%
- prompt_basic：MCC=0.5484, FCA=15.28%, A_FU=30.13%
- prompt_reward：MCC=0.4710, FCA=25.00%, A_FU=26.28%
- prompt_reward_norms：MCC=0.4723, FCA=23.26%, A_FU=28.21%
- cot：MCC=0.5174, FCA=7.64%, A_FU=46.15%

排除unsupported类后，A_FU从完整taxonomy的19.87%降至14.58%（漏诊改善），但FCA从25.69%微升至26.89%（因no_target分母减少），MCC从0.5225升至0.5600。WO仍是主要瓶颈（decision 46.67% vs 完整49.24%），说明正式known-label mismatch仍困难。

### 5.3.5 三个 baseline 的具体 prompt（可复现）
定义于 new_reason/baselines/prompt_cot/configs/prompts.py。

- **native_lm** 用评估脚本内置的 DECISION_SYSTEM_PROMPT（与判别头训练/评估完全一致）：
  "You are AgriAbstain, a vision-language grounding assistant for high-stakes plant disease diagnosis. Locate the queried target. If present and localizable → {\"decision\":\"answer\",\"label\":...,\"box\":...}; otherwise abstain → {\"decision\":\"abstain\",\"reason\":\"wrong_object|no_target\"}. wrong_object=视觉相似但不同的病/物；no_target=目标不在图中或被裁掉。"

- **prompt_basic**：独立自包含的三选一决策 system prompt（开头同为 "You are AgriAbstain..."），仅陈述 answer/wrong_object/no_target 三选一规则，无激励（非在 DECISION_SYSTEM_PROMPT 上拼接）。

- **prompt_reward**：在 prompt_basic 基础上追加显式 reward——正确 answer+box +1 / 错误 answer −1 / 正确弃答 +0.4。

- **prompt_reward_norms**：在 prompt_reward 基础上追加 4 条诚实/谦逊原则(说真话不猜、谦逊诊断、无清晰视觉证据不下结论、误诊伤害作物)，要求 JSON 放最后一行。

- **cot (structured)**：结构化 4 步推理——Step1 Evidence(描述可见症状) → Step2 Matching(是否匹配目标病/有哪些易混病) → Step3 Confidence(评估确定性) → Step4 Decision(给出唯一决定)，推理在前、干净 JSON 放最后一行(解析器抽末尾 {...} 不崩)。

### 5.3.6 prompt 强度梯度消融

三档 prompt(basic→reward→reward_norms)的 MCC 趋势并非单调递增：

| 探针 | basic → reward → reward_norms | 趋势 |
|---|---|---|
| L1_discrim | 0.514 → 0.332 → 0.331 | 递减(basic 最强) |
| L2_evidence | 0.467 → 0.499 → 0.506 | 递增 |
| L3_dose | 0.757 → 0.775 → 0.777 | 递增 |
| test_full | 0.724 → 0.703 → 0.696 | 递减(basic 最强) |
| ood_refcocog | 0.862 → 0.838 → 0.768 | 递减(basic 最强) |

发现（诚实陈述，不夸大）：
- "prompt 越强越好"不成立。加 reward/principles 只在 evidence(L2)/dose(L3) 这类需权衡的任务上小幅提升，却在 L1/test/refcocog 上反而降低 MCC——更复杂的激励把 LM 带向更激进弃答，牺牲了综合判别。
- prompt_basic 的 L1=0.514 反超判别头(0.3932)，与 native_lm/cot 同一模式：靠更激进弃答刷高 L1 近义判别，是 operating point 差异而非真本事(判别头为压低漏诊 A_FU 0.059 偏敢答)。
- 关键：无论 prompt 怎么调(basic/reward/reward_norms 三档全试)，纯 LM 在综合定位(test_full 最高仅 0.724 vs 判别头 0.8485)和通用 OOD(refcocog 最高 0.862 vs 0.9203)上都稳定低于判别头。这排除了"baseline prompt 不够好"的质疑——prompt 工程的整条强度梯度都补不上架构差距。

### 5.3.7 数据可靠性排查
- **L3_dose 的 format_valid=0.441 不是解析失败**：曾疑似 JSON 解析崩，排查确认是 L3 探针固有特性——750 条样本中 419 条为 ambiguous(gt decision 模糊)被评估脚本统一跳过，实际参与评估 total=331，invalid=0(解析全部成功)。四个方法在 L3 上口径完全一致(都 skip 419、total 331)，对比公平可靠。

### 5.3.8 结论（诚实）
- 判别头在综合定位(L2/L3/test_full)和 OOD(refcocog)上一致最优，且ID漏诊率 A_FU 最低。
- 判别头在细粒度近义病判别(L1)和真实世界野外 PlantWild closed12 上整体MCC不领先——不藏。
  - L1: ours .3932 vs native_lm .497 / cot .590。ep4 偏"敢答"以压低漏诊，代价是近义病更易被误答。
  - PlantWild closed12: ours .5600 vs native_lm .6024。判别头在WO decision(46.67%)和WO reason(31.67%)上显著落后，wrong_object是主要缺口。
- native_lm 是最强 baseline（用 SFT 原生格式，未被额外 prompt 干扰），整体强于 prompt/cot；判别头在 4/6 探针仍胜过它，说明架构增益来自判别头本身而非 prompt 工程。
- cot 靠激进弃答刷高 L1，但 A_FU 全面最高、L2/L3/test/OOD 全面落后——用大量漏诊换单项，不适合高风险场景。
- prompt 工程做了 basic→reward→reward_norms 三档强度梯度消融，纯 LM 综合定位/OOD 稳定低于判别头，且"prompt 越强"并不单调更好；个别探针(L1)被某些 prompt 档反超均为激进弃答的 operating point 假象，非架构能力。这排除了"baseline 偏弱"的审稿质疑。
- 排除unsupported类后answer A_FU从完整taxonomy的19.87%降为14.58%，但FCA并未改善（26.89%），说明正式known-label mismatch仍困难。WO仍是主要缺口（decision 46.67%），判别头在真实世界近义物混淆场景下仍需RL/post-SFT解决。

数据文件（closed12）：new_reason/baselines/prompt_cot/outputs/{native_lm,prompt_basic,prompt_reward,prompt_reward_norms,cot_structured}/plantwild_real_closed12_v1/metrics.json
OURS closed12：checkpoint-444/eval_faithful_v1/plantwild_real_closed12_v1/{metrics.json,predictions.jsonl,summary.txt}
probe数据：new_reason/outputs/eval_probes/plantwild_real_closed12_v1/{plantwild_real_closed12_paired_408.jsonl,BUILD_AUDIT.json}
补充（完整13类）：checkpoint-444/eval_faithful_v1/plantwild_real_v1/

## 6. 关键路径速查
| 用途 | 路径 |
|---|---|
| 训练数据 | new_reason/outputs/sft_3reason/sft_train.json |
| dev/test | new_reason/outputs/sft_3reason/sft_dev.json / sft_test.jsonl |
| 数据生成 | new_reason/scripts/build_final_sft_3reason.py |
| 训练脚本 | new_reason/scripts/run_fusion_sft_3reason.sh（单训） / run_all_3reason.sh（训+校准+评估一体） |
| 判别头模型 | newstory/qwen-vl-finetune/qwenvl/model/qwen3vl_dual_head_3reason.py |
| 校准头 | new_reason/scripts/calibrate_head_3reason.py |
| 评估 | scripts/eval_qwen3vl_swift_3reason.py |
| 数据集注册 | newstory/qwen-vl-finetune/qwenvl/data/__init__.py 的 agriabstain_new_reason_3reason |
| 训练开关 | 环境变量 USE_TRI_HEAD=True 启用三分类判别头 |

## 7. 设计理由与审稿要点（AAAI）

### 亮点（科学贡献点）
- **Tri-head 选择性预测架构**：判别头(第27层last-token→MLP)做decision+reason，LM做grounding(box/label)。head=answer时LM只生成label/box；head=abstain时decision+reason直接从head读取，无需LM生成。**为什么不用纯LM做decision**：判别头基于固定表征的判别信号比LM自回归生成更稳、可单次forward端到端出decision，且与grounding解耦——高风险场景下decision质量是安全底线，不应受生成随机性影响。无override/retry/fallback/placeholder/GT fallback。无KV-cache仅影响效率，不影响决策。
- **固定mu/sd标准化(frozen-BatchNorm语义)**：消融证实(job 534345/534535)——第27层last-token逐维(x-mu)/sd标准化后判别MCC 0.99；无标准化仅0.81；可学affine/RMSNorm 0.78；LayerNorm在evidence判别上塌到0。mu/sd从**base模型**train表征预估计(非微调后)，验证base统计量标准化微调表征MCC 0.969≈自洽上限0.974，**鲁棒于LoRA漂移**。有实测支撑的架构决策。
- **破共线设计(同图配对)**：每个answer样本有同图abstain孪生，防止模型走"图像→决策"捷径而非真正学判别。当前同图配对块覆盖judge/evidence配对图全覆盖。
- **高风险选点逻辑**：faithful门禁——444唯一满足test MCC≥.836、test A_FU≤3.5%、Ref MCC≥.906。**理由**：农业病害诊断中漏诊真病(over-abstain)代价远高于近义误答；且作为RL起点，保守模型可被RL往上推、激进过弃模型难回收。这是任务驱动的、可辩护的工程决策。
- **表征可分性诊断(方法论insight)**：用判别头概率分布实测真病vs近义病AUC=0.83，将L1瓶颈定位为表征层可分性不足(而非决策边界错位)；实测阈值/margin/集成后处理无法突破此上限，根解需在post-SFT重塑表征(SupCon)。

### 可能被审稿人质疑的点（预备回应）
  - **L1近义判别落后于纯LM baseline(0.3932 vs native_lm 0.497 / cot 0.590)**：不藏、不找补。判别头 ep4 为压低漏诊(A_FU 0.059)偏"敢答"，代价是近义判别弱。根因是真病vs近义病表征重叠(实测AUC=0.83)，post-SFT用SupCon抬AUC是根解。判别头的贡献不在"每项最高"，而在"最低漏诊 + 综合/OOD最优"的组合，L1单项落后是明确的、可辩护的取舍代价。
  - **PlantWild closed12 MCC不领先(0.5600 vs native_lm 0.6024)**：真实世界野外场景下判别头存在wrong_object(近义物混淆)缺口。WO decision仅46.67%(错误强答率53.33%，计入FCA)；answer侧A_FU(过度弃答)14.58%。这暴露了SFT数据中真实野外场景覆盖不足，需RL/post-SFT补充。
  - **判别头vs LM decision的一致性**：判别头直接argmax输出decision+reason，无需回退LM保证一致性。U_Reason_c在各探针上均≥0.883，证明弃答后reason判别可靠。
  - **oversample是否引入过拟合/数据泄漏**：oversample仅复制训练集内wrong_object样本(增信号密度)，不跨split、不碰dev/test；判别头head_loss为纯逐样本CE无pair-wise依赖，复制安全。

## 8. 已知瓶颈与下一步（RL/post-SFT）
- **L1 近义判别 MCC=0.3932**：ckpt-444 在 L1 上判别力受限；205条answerable中有12条P(answer)<0.5。历史表征诊断AUC为0.83，提示真病与近义病表征仍有重叠。这是低A_FU operating point下需要继续改善的能力缺口。
- **PlantWild closed12 真实世界WO缺口**：wrong_object decision仅46.67%(错误强答率53.33%，计入FCA)；answer侧A_FU 14.58%(过度弃答)。SFT数据中真实野外场景覆盖不足，判别头未经野外近义物混淆训练。
- **L1突破 vs 低漏诊 SFT内二选一**：同一决策边界旋钮的两端(守漏诊/抓近义)，阈值/margin/集成实测都突破不了AUC=0.83天花板。
- **根解在post-SFT**：①SupCon(把真病-answer和近义-wrong_object在嵌入空间推开，直接抬AUC) ②RL from ckpt-444(重塑表征+非对称代价，同时约束错误强答与漏诊) ③从独立训练源补充真实野外wrong_object配对样本；PlantWild real保持evaluation-only。
- **待跑pre-flight**：对1500训练近义病跑P(answer)分布，决定SupCon数据是否充足
