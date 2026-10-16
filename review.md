# ReGate 审稿意见回复

以下按审稿问题逐项回应，合并两位人工审稿人与 AI review 的重叠意见。补充结果包括已有预测的离线复算、固定 backbone 的 CPU head 实验，以及已完成的历史重训结果复核。

**结果来源约定：**“论文结果”指原运行；“CPU replay”指在保存表征和候选上重新评分；“head 三次拟合”固定 backbone，仅改变 head 初始化；“三次完整重训”指历史上已完成的三组 Stage-1/Stage-2 运行。均值后的 ± 为样本标准差。

## 1. Stage 2 的作用与两种 operating point

**review 意见：** SFT native 在 Table 3 的多项指标优于 ReGate。请解释第二阶段的作用，提供隔离它的消融，并将 dev-selected 点列入表格。（Pty5；mDZK W5、W6）

**回复的内容：** Stage 2 的主要作用是改变选择决策及风险—覆盖率工作点。Table 3 保持 adapted backbone 和 grounding decoder 固定，比较 native LM decision 与 Stage-2 gate routing。我们补充了同一 backbone 下 Stage-1 辅助头与 Stage-2 发布头的直接对照，以及 argmax 和 dev-selected 两个工作点的统一结果。

### 1.1 同一 backbone 下的 head 对照

以下均在 Comprehensive 的同一份保存表征和 grounding 候选上评分。Stage-1 三分类 head 及其归一化参数从对应 checkpoint 的 adapter 权重读取。

| Backbone | Head | FCA↓ | Retention↑ | MCC↑ | Strict-3↑ | E2E-mIoU↑ |
|---|---|---:|---:|---:|---:|---:|
| Grouped | Stage-1 auxiliary | 0.2933 | 0.9267 | 0.6493 | 0.7766 | 0.7039 |
| Grouped | Stage-2 published，CPU replay | 0.1323 | 0.9758 | 0.8485 | 0.8560 | 0.7011 |
| Independent | Stage-1 auxiliary | 0.2834 | 0.9123 | 0.6413 | 0.7755 | 0.6958 |
| Independent | Stage-2 published，CPU replay | 0.4845 | 0.9554 | 0.5244 | 0.6640 | 0.6780 |

Grouped 的 Stage 2 明显改善 FCA、MCC 和 Strict-3，定位指标基本持平。Independent 的 Stage 2 提高了 retention，但 FCA、MCC 和定位指标变差。这表明 refit 的效果取决于训练条件；Independent 的 head 泛化问题需要结合第3节的拟合实验分析。

CPU replay 与原运行相比，Grouped 的选择决策完全一致，Independent 有2/2,646条决策不同。上表保留 CPU replay 实测值；论文原 Independent FCA 为0.4860。

### 1.2 统一报告 argmax 与 dev-selected 点

阈值 τ=0.9888232350349426 由2,408条 dev 数据选定，随后固定用于 Comprehensive test。

| 方法 / 工作点 | FCA↓ | Retention↑ | MCC↑ | Strict-3↑ | E2E-mIoU↑ | MTS↑ | FullStrict↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base native | 0.2109 | 0.8330 | 0.6227 | 0.6602 | 0.6557 | 0.6638 | 0.6119 |
| SFT native | 0.0945 | 0.8511 | 0.7577 | 0.8193 | 0.7046 | 0.8068 | 0.7929 |
| ReGate argmax | 0.1323 | 0.9758 | 0.8485 | 0.8560 | 0.7011 | 0.8195 | 0.8299 |
| ReGate dev-selected τ | 0.0454 | 0.8896 | 0.8461 | 0.8745 | 0.6657 | 0.8395 | 0.8503 |
| ReGate τ=0.5，固定阈值诊断 | 0.1315 | 0.9758 | 0.8492 | 0.8560 | 0.7011 | 0.8195 | 0.8299 |

dev-selected 点将 unsupported answer 从125/1,323降至60/1,323，FCA相对下降52%；supported answer route 从1,126/1,323增至1,177/1,323。同时，E2E-mIoU从0.7046降到0.6657。因此，“52% reduction”明确对应 Comprehensive 上相对 SFT native 的 FCA 改善，定位质量存在代价。

阈值点的 E2E、MTS 和 FullStrict 由保存的 answer candidate 重新路由计算。三分类 argmax 与 τ=0.5 的二元阈值规则分别报告。FullStrict 定义见第6节。

**证据：** [head对照](docs/review_evidence/cpu_gate/scored_heads.json)、[统一指标表](docs/review_evidence/unified_metrics.csv)、[dev阈值记录](docs/review_evidence/threshold_provenance.json)。

## 2. 额外 retention 是否带来成功定位

**review 意见：** 请确认 E2E/retention 的条件质量变化，并解释 argmax 增加的回答是否带来端到端收益。（mDZK 第6点）

**回复的内容：** 确认。E2E-mIoU除以supported retention，得到被选择回答的supported集合上的平均label-gated IoU；错误标签或无效候选贡献零。Comprehensive的SFT/ReGate分别为0.8279/0.7185，RefCOCOg为0.7168/0.6455。

按样本配对 SFT native 与 argmax ReGate：

| 数据集 | 两者均回答 | 仅SFT回答 | 仅ReGate回答 | 两者均不回答 | 仅ReGate中的supported | 其中成功答案 |
|---|---:|---:|---:|---:|---:|---:|
| Comprehensive | 1,210 | 41 | 256 | 1,139 | 171 | 0/171 |
| RefCOCOg | 171 | 0 | 25 | 166 | 18 | 0/18 |
| PlantWild | 146 | 5 | 48 | 209 | 15 | 0/15 |

三份主表预测中，“仅ReGate回答”的保存候选均为invalid。按正文§3.3的定义，它们计入gate的answer-selection，正确性计零。因此，这部分额外retention没有转化为成功答案。Comprehensive的E2E净变化为−0.00349，RefCOCOg为0。PlantWild没有bbox，成功答案仅按label判断。

**证据：** [回答集合配对](docs/review_evidence/paired_selection.json)。

## 3. Independent gate 的拟合与稳定性

**review 意见：** Grouped的MTS增益可能被Independent gate的退化放大。请报告两组拟合设置、dev loss曲线，并同时报告native对比。（mDZK W1）

**回复的内容：** 论文原运行中，Grouped/Independent的gate MTS为0.819504/0.526741，差值0.292764；native MTS为0.806774/0.755871，差值0.050903。gate接口下的差异同时涉及表征、head拟合和决策边界。

### 3.1 拟合协议与原始曲线

当前实现使用4096→512→256→3 MLP、GELU、dropout 0.1；各组从adapted training features计算mean/std。优化器为full-batch Adam，lr=1e-3、weight decay=1e-3；Stage 2使用无类别权重的cross entropy，最多1,000 epoch，每50 epoch评估，以dev loss选点，patience为5次评估。

两组日志均记录7,072条训练表征、100条dev表征，dev三类样本数为34/33/33。提取代码以`random.Random(42)`分层抽样；历史日志未保存全部运行参数及逐条dev ID。Stage-2 MLP初始化没有显式固定torch seed。

| 原始gate拟合记录 | Grouped | Independent |
|---|---:|---:|
| 第一次记录epoch / dev loss | 50 / 0.5087 | 50 / 3.0538 |
| 最优epoch / dev loss | 300 / 0.4644 | 50 / 3.0538 |
| 停止epoch / 最后dev loss | 550 / 0.6532 | 300 / 5.2438 |
| 恢复best后train多类MCC | 0.9811 | 0.9813 |
| 恢复best后dev多类MCC | 0.8354 | 0.3614 |
| dev wrong_object accuracy | 90.91% | 9.09% |

Independent从第一次记录开始即有明显train–dev差距。全部17个已记录评估点见 [loss曲线数据](docs/review_evidence/gate_loss_curves.csv)。

### 3.2 固定 backbone 的三次 CPU head 拟合

固定论文的Grouped checkpoint-444及Independent checkpoint-222，使用各自缓存的7,072条train和100条dev表征，以相同recipe、初始化seed7/21/42重新拟合。所有运行仅按dev loss选epoch，三次全部纳入统计。

| Comprehensive，mean ± sd | Grouped | Independent |
|---|---:|---:|
| FCA↓ | 0.0484 ± 0.0027 | 0.4654 ± 0.0332 |
| Retention↑ | 0.8987 ± 0.0055 | 0.9574 ± 0.0091 |
| Binary MCC↑ | 0.8515 ± 0.0060 | 0.5432 ± 0.0152 |
| AUROC↑ | 0.9828 ± 0.0029 | 0.8983 ± 0.0047 |
| Strict-3↑ | 0.8731 ± 0.0037 | 0.6759 ± 0.0156 |
| E2E-mIoU↑ | 0.6623 ± 0.0043 | 0.6801 ± 0.0024 |
| MTS↑ | 0.8374 ± 0.0028 | 0.5417 ± 0.0172 |

Grouped三次best dev loss为0.532467、0.544444、0.438411；Independent为3.266253、3.284905、3.023980。Independent的高FCA在三个初始化下持续存在，差距并非只出现在原来的一次head初始化中；其head泛化问题仍然存在。

归一化核验发现：Grouped发布head的mu/sd与checkpoint-444重提取训练表征统计不匹配，最大绝对差为4.060974/1.455295，RMS差为0.513682/0.189540；Independent完全匹配。该差异的历史来源尚未确定。Grouped发布head与新refit的比较因此同时包含归一化统计变化。

**证据：** [CPU拟合及评分](docs/review_evidence/cpu_gate/scored_heads.json)、[日志与代码配置摘录](docs/review_evidence/gate_protocol_provenance.json)。

## 4. Reason accuracy 的提升来自哪里

**review 意见：** 0.504→0.865的提升是否主要由拒答率解释，而非reason辨别能力改善？（mDZK W1）

**回复的内容：** 主要增益来自拒答覆盖。Comprehensive有1,323个unsupported样本：

| 指标 | Grouped | Independent |
|---|---:|---:|
| 拒答率 | 1,148/1,323 = 0.867725 | 680/1,323 = 0.513983 |
| 固定unsupported分母的U_Reason | 1,145/1,323 = 0.865457 | 667/1,323 = 0.504157 |
| 已拒答条件下reason正确率 | 1,145/1,148 = 0.997387 | 667/680 = 0.980882 |

两组拒答覆盖相差35.37个百分点，条件reason正确率相差1.65个百分点。U_Reason增益主要体现更多unsupported记录被正确拒答。两组选择的拒答子集不同，对固定难例集合的reason辨别能力仍需单独比较。

**证据：** 两组`regate/comprehensive/metrics.json`及`reason_abstain/eval/metrics.py`。

## 5. 完整训练的重复实验与方差

**review 意见：** 主表缺少完整训练多seed结果；probe seeds与记录bootstrap无法说明训练方差。（mDZK W2；AI review）

**回复的内容：** 原论文主表及构造对照各对应一次Stage-1训练，Stage-1 seed为42；Table 3、5、6共享同一主模型。五seed probe衡量固定backbone上的probe拟合变化，bootstrap衡量记录采样变化。

另有已完成的历史完整重训`h27experiment/runs/{grouped,independent}/seed-{7,21,42}`。我们从12份Comprehensive逐样本预测重新计算，结果与原汇总一致。每次运行使用自身dev-selected checkpoint，以下与论文单次结果分开报告。

| 三次完整重训，mean ± sd | FCA↓ | Retention↑ | MCC↑ | Strict-3↑ | E2E-mIoU↑ | MTS↑ |
|---|---:|---:|---:|---:|---:|---:|
| Grouped native | 0.0887 ± 0.0190 | 0.8065 ± 0.0162 | 0.7219 ± 0.0269 | 0.7919 ± 0.0374 | 0.6621 ± 0.0163 | 0.7808 ± 0.0229 |
| Grouped gate | 0.0600 ± 0.0083 | 0.9022 ± 0.0049 | 0.8429 ± 0.0117 | 0.8482 ± 0.0154 | 0.6251 ± 0.0244 | 0.8062 ± 0.0170 |
| Independent native | 0.1940 ± 0.0093 | 0.8511 ± 0.0249 | 0.6580 ± 0.0168 | 0.7968 ± 0.0127 | 0.6993 ± 0.0196 | 0.7457 ± 0.0110 |
| Independent gate | 0.4870 ± 0.0084 | 0.9562 ± 0.0039 | 0.5234 ± 0.0084 | 0.6659 ± 0.0140 | 0.6828 ± 0.0188 | 0.5280 ± 0.0123 |

Grouped在Comprehensive的gate选择指标和MTS上持续优于Independent；Independent native的定位指标仍有竞争力。Grouped argmax retention的重训均值为0.9022，低于论文单次的0.9758，表明原单次operating point存在运行间变化。

三次重训的dev-selected结果中，ReGate test FCA为0.0393±0.0099、retention为0.8677±0.0142；native分别为0.0887±0.0190、0.8065±0.0162。平均相对FCA下降55.89%±1.56个百分点。各次阈值分别在该次运行的dev数据上选择。

Table 5全部prompt接口及Table 6 GPT比较尚无同等完整的三次重训结果。

**证据：** [逐seed原预测复算](docs/review_evidence/frozen_and_prior/prior_raw_verification.json)、[dev-selected汇总](docs/review_evidence/frozen_and_prior/dev_selected_multiseed.json)。

## 6. 统一指标与完整任务严格指标

**review 意见：** 各表指标不一致；希望统一报告，并增加同时要求定位和正确拒答的指标。（Pty5；mDZK；AI review）

**回复的内容：** 已补充 [44行统一指标表](docs/review_evidence/unified_metrics.csv)，覆盖原主表、两个阈值点、GPT六探针、prompt interfaces、构造消融及Grouped L1/L2/L3。统一列出FCA、retention、MCC、Strict-3、E2E-mIoU、MTS和FullStrict，保留样本数与预测来源。

FullStrict按所有具有明确GT状态的记录计算：supported须answer、label正确且IoU≥0.5；unsupported须abstain并reason正确。Comprehensive的SFT native、ReGate argmax、ReGate dev-selected分别为0.7929、0.8299、0.8503。PlantWild没有bbox，使用label-only正确性，E2E记N/A。L3固定指标使用331个端点，419个partial records用于graded分析。

A_FU为supported-query false rejection rate，等于1−retention；E2E-mIoU以全部supported记录为分母。MTS在各probe内按relation宏平均，方法比较在同一probe内进行。给定类别比例、FCA与retention后，MCC可由二元混淆矩阵确定，附表保留它用于跨方法查阅。

## 7. L2构成、裁剪几何与L3端点

**review 意见：** L2 FCA=0.476与其他crop-out通过率及L3 endpoint结果冲突，395条unsupported中只有148对进入配对分析。（mDZK W4）

**回复的内容：** L2构造存在query与裁剪目标错配。构造器按source image聚合后，对全部base keep记录与zero-crop记录做笛卡尔积，query取keep的label，缺少目标一致性约束。

| L2负例分层 | n | 判为answer | FCA |
|---|---:|---:|---:|
| 全部 | 395 | 188 | 0.475949 |
| query与crop目标类别一致 | 148 | 26 | 0.175676 |
| query与crop目标类别不同 | 247 | 162 | 0.655870 |
| PlantDoc来源 | 58 | 3 | 0.051724 |
| COCO来源 | 337 | 185 | 0.548961 |

### 7.1 逐条几何审计

对395条负例恢复零档裁剪区域，与实际派生图片尺寸逐条核对，全部一致；随后检查query类别的所有已标注bbox与裁剪区域的交集。COCO子集的1,610条标注与原COCO文件中对应203张图的全部标注一致。

| 几何状态 | 全部 | 类别一致 | 跨类别 |
|---|---:|---:|---:|
| 严格零交集 | 136 | 73 | 63 |
| 正交集，且无query bbox保留≥99.9% | 155 | 75 | 80 |
| 至少一个query bbox保留≥99.9% | 104 | 0 | 104 |

104条跨类别负例几乎完整保留了已标注query目标bbox，与no_target参考标签冲突。严格零交集子集有136条，其中20条被回答，FCA=0.147059；其中类别一致的73条有12条被回答，FCA=0.164384。

bbox量化与像素取整会产生边界残片。以最大保留比例≤1%作为单独的容差诊断时，子集为210条、answered为34条，FCA=0.161905。主统计采用严格零交集。几何结果依赖原标注，完整人工标签修复尚未完成。

### 7.2 与原有crop-out及L3的关系

Comprehensive的T2_occluded、T3_occluded、base_occluded分别为5/80、7/80、24/170条被回答，对应FCA 0.0625、0.0875、0.141176。它们与L2扩展后的395条具有不同构成。

L2严格同instance配对并去重后为148对：keep侧label+IoU正确120/148，crop侧正确拒答122/148，双侧均正确104/148。其crop目标ID与L3零档端点全部重合，保存的二元decision无差异，均回答26/148条。

原概率敏感性分析按`source_instance + target_label`分组，记录397组、去重245条，得到148个完整组和249个不完整组。在完整组上，平均Δp=0.785811，方向正确率96.6216%。这解释了概率配对分析与完整L2分类评估的分母差异。

**证据：** [几何汇总](docs/review_evidence/l2_geometry_summary.json)、[逐条记录](docs/review_evidence/l2_geometry_records.jsonl)、[容差敏感性](docs/review_evidence/l2_geometry_tolerance.json)。

## 8. Resampling、重复比例与类别组成

**review 意见：** 请补充训练resampling和类别组成，说明973条contradiction扩展至1,500条的重复情况。（Pty5；mDZK W3）

**回复的内容：** SFT共7,072个实例；按完整训练输入和目标去重后为6,138条。

| 状态 | 训练实例 | 唯一记录 |
|---|---:|---:|
| answer | 2,703 | 2,422 |
| wrong_object | 1,500 | 973 |
| no_target | 2,869 | 2,743 |

wrong_object额外复制527条，占最终该状态的35.13%。构造程序以`seed+1000`建立独立随机数生成器，打乱原记录pool，取527条追加；最终527条各出现两次、446条各出现一次。resampling在split分配之后仅作用于训练数据，dev/test保持原样。

wrong_object按crop为apple101、corn489、tomato910。十个query类别依次为：apple rust/scab 51/50；corn gray leaf spot/leaf blight/rust 197/130/162；tomato early blight/bacterial spot/late blight/mold/septoria 168/165/178/204/195。

按图片来源，COCO为answer1,100、no_target888；PlantDoc为answer1,603、wrong_object1,500、no_target1,981。Table 1的500条COCO unavailable对应专门抽样分支，另388条来自crop0补充分支。全部99个实际query标签的三态计数见 [类别组成表](docs/review_evidence/category_composition.md)。

**证据：** [数据统计](docs/review_evidence/data_and_gpt.json)、`aaai_code/scripts/build_final_sft_3reason.py`。

## 9. T7可见度定义与hard/graded边界

**review 意见：** retained visibility如何计算？哪些记录进入hard supervision？（Pty5）

**回复的内容：** retained visibility是标注目标bbox的保留面积比例。原像素bbox为B、裁后区域为C时，r=area(B∩C)/area(B)。裁剪沿一个轴移除目标bbox的`1−target_ratio`，像素取整后计算achieved ratio，保留四位小数；bbox重新映射到裁后图像的[0,1000]坐标。

请求档位为1、0.75、0.5、0.25、0。achieved≥0.5且存在有效新bbox时使用supported hard label；请求0且成功排除量化目标bbox时使用no_target；其余中间档标记dose_only。名义0.5可能因像素取整落在阈值两侧，因此使用实际achieved值。尺寸或长宽比不满足实现约束的crop会跳过。

该比例描述bbox几何保留程度；疾病症状可识别度、其他同类实例和原标注遗漏需要另外评估。

**证据：** `reason_abstain/bench/transforms.py::save_visibility_ladder_crop`、`build_plantdoc.py`的T7构造分支。

## 10. L3单调性的baseline

**review 意见：** 21.4%的可见度违例率缺少baseline。（mDZK）

**回复的内容：** 在相同183组、567个相邻档位上，补充GPT与ReGate的二元answer决策单调性：可见度提高而answer由1降到0计违例，ties不计。

| 指标 | GPT | ReGate argmax |
|---|---:|---:|
| 二元决策相邻下降数 / 邻接数 | 23/567 | 5/567 |
| 二元决策相邻下降比例 | 4.0564% | 0.8818% |

连续ReGate p_answer的下降比例为122/567=21.5168%；按group等权平均为约21.45%，对应论文约21.4%。连续概率与二元决策分别报告。

| 可见度档位 | GPT answer率 | ReGate answer率 | ReGate mean p_answer |
|---|---:|---:|---:|
| 0 | 0.3378 | 0.1757 | 0.1872 |
| 0.25 | 0.7363 | 0.5000 | 0.5044 |
| 0.5 | 0.8444 | 0.9037 | 0.8993 |
| 0.75 | 0.8824 | 0.9706 | 0.9671 |
| 1 | 0.9235 | 0.9891 | 0.9828 |

**证据：** [L3对照](docs/review_evidence/additional_cpu/l3_gpt_vs_regate.json)。

## 11. Frozen backbone、辅助损失与confidence曲线

**review 意见：** 请补充Frozen backbone＋gate、Stage 1辅助loss=0，以及SFT native自身confidence的FCA–retention曲线。（mDZK W5、第7点、第9点）

**回复的内容：** 已补充Frozen Base＋MLP的正式Comprehensive对照。使用冻结Base的layer-27缓存，train7,072条、test2,646条；通过缓存original_index对应SFT记录，再用image/query/state唯一匹配，精确恢复论文100条dev（34/33/33）。使用Stage-2式MLP recipe、seed42，best dev loss为0.4784（epoch50），epoch300停止。测试候选来自Base native，按sample ID对齐。

| Comprehensive | FCA↓ | Retention↑ | MCC↑ | Strict-3↑ | E2E-mIoU↑ | MTS↑ | AUROC↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base native | 0.2109 | 0.8330 | 0.6227 | 0.6602 | 0.6557 | 0.6638 | — |
| Frozen Base＋MLP，dev100 | 0.0952 | 0.8639 | 0.7693 | 0.8122 | 0.5895 | 0.7690 | 0.9672 |
| Frozen Base＋linear，full-dev选C | 0.0401 | 0.8020 | 0.7716 | 0.8220 | 0.5650 | 0.7820 | 0.9685 |
| Adapted＋发布MLP，论文结果 | 0.1323 | 0.9758 | 0.8485 | 0.8560 | 0.7011 | 0.8195 | 0.9834 |

Frozen Base已具有较强选择能力。适配系统的MCC和E2E更高，同时decoder与工作点发生变化。linear按完整2,408条dev选得C=0.01，单独标明其选择规模；表中gate结果均为对应模型保存候选上的离线路由评分。

Stage 1辅助loss=0尚需完整训练。当前SFT native的dev/test预测只有结构化输出与raw文本，未保存logprob、entropy或其他自身confidence，因此现有记录不足以建立连续confidence曲线。

**证据：** [Frozen MLP结果](docs/review_evidence/frozen_and_prior/runs/mlp_dev100/metrics.json)、[dev100映射](docs/review_evidence/frozen_and_prior/exact_dev100.json)、[confidence字段检查](docs/review_evidence/additional_cpu/sft_native_confidence_and_baselines.json)。

## 12. 标签可靠性与双人人工审计

**review 意见：** annotation absence未必代表image absence；需要每状态约200条、两位标注者独立审计，并验证reason语义。（mDZK W3；AI review）

**回复的内容：** 原始标注不完整可能导致负例标签错误。T6 contradiction采用“已标注confusable类存在、query类未标注”的操作定义；T5跨crop query及目标排除定义为unavailable。多实例、共现症状和未标注目标是审计重点。

已准备三态各200条、共600条的审计清单，保留分层权重、来源ID和两位标注者的空白记录。wrong_object覆盖205条总体中的200条；no_target包含120条COCO与80条PlantDoc。抽样单位为image-query，source image可重复，后续区间按source聚类计算。

人工标注尚未完成，当前没有人工错误率或标注一致率。清单和操作说明见 [审计CSV](docs/review_evidence/additional_cpu/label_audit_prep.csv)、[标注指南](docs/review_evidence/additional_cpu/ANNOTATOR_GUIDE.md)。

## 13. 类别筛选对Base的偏差

**review 意见：** Base能力筛选保留20/24类可能影响结果，请报告完整24类表现。（mDZK第11点；AI review）

**回复的内容：** 筛选在source-image split之前进行，使用768条PlantDoc base records、24个实际类别，以Base的正确label及IoU≥0.5正确率达到0.40为保留条件。原校准池统计如下：

| 集合 | 正确 / 样本数 | Micro正确率 | Macro类别正确率 |
|---|---:|---:|---:|
| 全24类 | 542/768 | 0.7057 | 0.6729 |
| 保留20类 | 531/726 | 0.7314 | 0.7621 |
| 排除4类 | 11/42 | 0.2619 | 0.2271 |

保留类别后的Base micro提高2.57个百分点、macro提高8.92个百分点，说明筛选改变了类别难度分布。排除类为bell pepper leaf spot、blueberry leaf、tomato leaf、tomato leaf mosaic virus。以上是原筛选池的描述统计；三系统完整24类独立评测尚未补充。

**证据：** [24/20类统计](docs/review_evidence/additional_cpu/capability_24_vs_20.json)。

## 14. GPT版本、温度与解析失败率

**review 意见：** 请报告GPT的具体版本、temperature和parse-failure rate，并与FCA分开。（mDZK第10点）

**回复的内容：** 六个run均记录`model_requested=gpt-5.6-sol`，响应`api_model`相同；temperature=0、image_detail=original、stream=false。服务记录未提供独立的不可变snapshot ID。全部请求记录完成，无缺失，使用相同task schema和逐样本query，无JSON repair call。

| 探针 | 响应数 | Parser invalid | 比例 |
|---|---:|---:|---:|
| Comprehensive | 2,646 | 20 | 0.7559% |
| L1 | 410 | 3 | 0.7317% |
| L2 | 790 | 1 | 0.1266% |
| L3全部档位 | 750 | 2 | 0.2667% |
| RefCOCOg | 362 | 1 | 0.2762% |
| PlantWild closed12 | 408 | 9 | 2.2059% |

invalid涵盖JSON、schema和box有效性，在选择指标中计non-answer，正确性计零。L3两条invalid中，一条为supported endpoint、一条为partial；331条固定指标记录中有1条invalid。

**证据：** [请求配置与解析统计](docs/review_evidence/data_and_gpt.json)。

## 15. 最近方法、贡献范围与运行效率

**review 意见：** 三态输出差异不能排除二元指标比较；相关工作定位及contradiction能力需要说明，generation-skipping收益也缺少测量。（mDZK W5；AI review）

**回复的内容：** Counterfactual REC、RECANTFormer和HieA2G覆盖counterfactual或zero-target grounding。在相同记录上约定empty-set→abstain后，可比较FCA、retention、MCC及supported定位；reason输出差异影响三态正确性指标。这些指定方法的同协议实验尚未补充。现有GroundingME/OpenRef结果是本模型在其他数据上的评测。

NOPE研究VQA中的不存在对象，HallusionBench诊断语言幻觉与视觉错觉。MCS的具体贡献是从既有定位标注构造七类确定性关系、保留同源对照并提供构造定义的reason。AgroVG的官方摘要覆盖农业单/多/无目标set prediction，共10,071个image-query pair、10个来源、6类任务，是相关农业资源；现有记录未说明最初未采用它的原因。审稿所列GUI hallucination论文的确切条目尚待核实。

细粒度contradiction仍是主要短板：L1 FCA=0.6146、T6 strict pass约0.571、PlantWild contradiction accuracy约0.317。当前较强证据集中在Comprehensive选择指标、target-unavailability处理和受控证据响应。

现有155份日志中117份包含进度或耗时文本，但卡数、分片和输出方式不同，缺少同硬件、同batch的受控decode-skip测量，目前没有可报告的加速比。代码已上传研究仓库，数据公开范围与许可声明尚待整理。

**证据：** [相关工作核验](docs/review_evidence/related_work.md)、[已有baseline协议](docs/review_evidence/additional_cpu/sft_native_confidence_and_baselines.md)、[耗时记录检查](docs/review_evidence/additional_cpu/decode_skip_timing.json)。

---

完整实验与复算入口：[CPU完成清单](docs/review_evidence/CPU_COMPLETION.md)。论文版本及源文件哈希：[版本记录](docs/review_evidence/paper_version.json)。
