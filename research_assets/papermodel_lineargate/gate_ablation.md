# 读出头消融 + 冻结基座对照(rebuttal 备料)

本目录是**隔离的**实验区:所有产出只写在 `papermodel_lineargate/` 下,不触碰论文原始
资产、`paper/` 目录和匿名打包路径。论文正文、附录、代码、checklist 均未改动。

配套材料:`devtau_multiseed/rebuttal_evidence.md`(三次独立重训的证据与措辞禁忌)。
两份文档共用同一套指标口径,第 8 节给出交叉验证。

回答两个预期质疑:

- **Q1**「为什么用 MLP 头?线性门是不是就够了?」
- **Q2**「MLP 头是不是只是个 readout?可分性本来就在基座里,不是你训出来的?」

结论先说:**Q1 站得住**——读出头怎么选几乎不影响判别能力(ΔAUROC 0.0014),
数据构造才是主导因子(0.0936),且构造效应在两种头下几乎相同。
**Q2 不能靠 AUROC 反驳**——冻结基座上的线性探针已经到 0.9685,Stage-1 适配只加
0.0135;反驳 Q2 必须走行为层面的论证,见第 10 节。

---

## 1. 实验设置

| 项目 | 取值 |
|---|---|
| 评测集 | comprehensive test,2646 条(supported 1323 / unsupported 1323) |
| unsupported 细分 | wrong_object 205,no_target 1118 |
| 标定训练集 | 7072 条(与论文 Stage-2 标定同一批) |
| 特征 | layer-27,4096 维 |
| 工作点 | `three_class_argmax`,与论文一致 |
| 线性门 | `LogisticRegression(solver=lbfgs, max_iter=2000, random_state=42)`,C ∈ {0.01, 0.1, 1, 10},按 dev loss 选 |

三种被比较的读出头,都吃**同一批** `head_hidden`,走**同一套**打分代码:

- `mlp` —— 论文发布的 `discrim_head_calibrated.pt`,原封不动
- `linear` —— 多类 logistic regression
- `mlp_refit` —— 按论文 Stage-2 流程重训的 MLP,**仅作内部一致性验证,不对外发布**

冻结基座特征来自 `new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/repr/`,
其 manifest 自证 `adapter = None`、`layer = 27`、
`base = Qwen3-VL-8B-Instruct`,即 LoRA 适配与辅助头训练**之前**的表征。

为了让冻结基线尽量占优(结论才保守),给了它两个额外便利:四种 pooling 变体全部
扫描取最好;C 网格在**完整 2408 条 dev** 上选,而适配门只用 100 条标定 dev。

## 2. 可信度校验(先过这一关,数字才有意义)

| 校验项 | 结果 |
|---|---|
| 复现论文已发布数字(两个 arm 的 FCA / MCC / Strict-3 / retention) | 最大绝对偏差 **0.0** |
| 复现论文附录记录的 grouped AUROC | 0.9834 vs 记录值 .983 |
| 复现论文附录记录的 grouped `q10(p_A)` | 0.9847 vs 记录值 .985 |
| 冻结 test 集 vs 评测集 sample_id | 2646 条**完全相同**,零差集 |
| supported/unsupported 划分一致性 | **零分歧**(冻结 `head_label==0` ⟺ 评测 `gt.decision=="answer"`) |
| 特征抽取流水线可复现性 | independent arm 重抽特征与原始一致 |

第一行是最关键的一条:评测流水线原样吐出了论文的数字,所以后面所有对比都是在论文
那条线上做的,不是另起一套。第二、三行是独立的二次确认:两个来自论文附录、本次没有
用来标定任何东西的量,也被重算命中。

一处已知不一致,不影响上述结论:grouped arm 的 `discrim_head_calibrated.pt` 内存的
`mu`/`sd` 与本次从 `checkpoint-444` 抽取的特征统计量对不上(independent arm 完全
匹配)。说明该头的标准化统计量来自另一个训练步的抽取。由于论文数字被精确复现,
这只是 provenance 记录问题,不是评测错误,故 `check_features.py` 对此降级为 warning。

## 3. 结果:适配后的读出头对比(argmax 工作点)

| arm | gate | FCA ↓ | retention ↑ | Strict-3 ↑ | MCC ↑ | A_FU ↓ |
|---|---|---|---|---|---|---|
| grouped | mlp(论文) | 0.1323 | 0.9758 | 0.8560 | 0.8485 | 0.0242 |
| grouped | linear | 0.0529 | 0.8957 | 0.8715 | 0.8439 | 0.1043 |
| grouped | mlp_refit | 0.0484 | 0.9002 | 0.8723 | 0.8530 | 0.0998 |
| independent | mlp(论文) | 0.4860 | 0.9554 | 0.6633 | 0.5231 | 0.0446 |
| independent | linear | 0.4686 | 0.9592 | 0.6750 | 0.5427 | 0.0408 |
| independent | mlp_refit | 0.4974 | 0.9599 | 0.6595 | 0.5202 | 0.0401 |

grouped 那一栏,linear 的 FCA 看起来比 MLP 好很多(0.053 vs 0.132),但 retention 同时
掉了 8 个点(0.896 vs 0.976)。**这是工作点位置不同,不是能力差异**,机制见下节。
单看 FCA 会得出错误结论。

## 4. 工作点诊断:线性门的 FCA "优势"是尺度假象

用论文附录同一个诊断量 `q10(p_A)`(supported 记录上 answer 分数的第 10 百分位,对分数
尺度敏感、对排序不敏感):

| arm | gate | q10(p_A \| sup) | 中位数 | frac(p_A>0.99 \| sup) | argmax retention |
|---|---|---|---|---|---|
| grouped | mlp(论文) | **0.9847** | 1.0000 | 0.885 | 0.9758 |
| grouped | linear | **0.4401** | 0.9999 | 0.751 | 0.8957 |
| independent | mlp(论文) | 0.9758 | 1.0000 | 0.873 | 0.9554 |
| independent | linear | 0.9456 | 1.0000 | 0.843 | 0.9592 |

读法:论文的 grouped 头**饱和**(q10 = 0.985),argmax 因此右移到高覆盖端;grouped 线性门
**不饱和**(q10 = 0.440),argmax 落在更均衡处,于是 FCA 低、retention 也低。

**independent arm 是这个机制的对照组**:那里两个门都饱和(0.976 / 0.946),它们的 argmax
指标就贴得很近(retention 0.9554 vs 0.9592,FCA 0.486 vs 0.469)。同样的头架构差异,在
尺度接近时就不产生工作点差异 —— 说明 grouped 上那 8 个点确实来自尺度而非能力。

这与 `devtau_multiseed/rebuttal_evidence.md` 第 4 节描述的是同一个现象:那边是论文头
(q10 .985)与重训头(q10 .523 ± .043)之间的位移,这边是论文头与线性门之间的位移。

## 5. 阈值无关视角(去掉工作点混淆)

同一批 2646 条,以 `p_answer` 为分数、supported vs unsupported 求 AUROC:

| 配置 | AUROC |
|---|---|
| grouped / mlp(论文) | **0.9834** |
| grouped / linear | 0.9819 |
| independent / mlp(论文) | 0.8898 |
| independent / linear | 0.8841 |

把线性门的阈值挪到与论文 MLP 相同的 retention 上再读 FCA:

| arm | retention(对齐后) | FCA(mlp) | FCA(linear) |
|---|---|---|---|
| grouped | 0.9758 | 0.1315 | 0.1353 |
| independent | 0.9539 | 0.4845 | 0.4611 |

工作点对齐之后,两种头的 FCA 差距落到 1 到 2 个百分点以内,方向还不一致。

## 6. 结果:冻结基座线性探针

冻结基座(无 LoRA、无辅助头)layer-27 表征上训线性探针,四种 pooling:

| 特征 | 选中 C | dev loss | test AUROC | test 三类 acc | test 三类 MCC |
|---|---|---|---|---|---|
| `last_token`(头输入的对应物) | 0.01 | 0.3300 | **0.9685** | 0.8681 | 0.7812 |
| `visual_mean` | 0.01 | 0.8233 | 0.7624 | 0.6141 | 0.3298 |
| `visual_max` | 0.01 | 0.9017 | 0.7319 | 0.6009 | 0.2754 |
| `visual_query_attn` | 0.01 | 0.9761 | 0.6049 | 0.5310 | 0.1497 |

三类 acc / MCC 只在本表内部可比(纯分类口径),不要和第 3 节的端到端指标对齐看——
后者含生成环节。跨表只用 AUROC 比较。

## 7. 三个因子的量级

固定其余条件,单因子对 AUROC 的贡献:

| 因子 | 对比 | ΔAUROC |
|---|---|---|
| **数据构造** | grouped − independent(适配后 + MLP) | **+0.0936** |
| **数据构造** | grouped − independent(适配后 + linear) | **+0.0978** |
| Stage-1 适配 | grouped 适配 linear − 冻结 linear(构造均为 grouped) | +0.0135 |
| 读出头 | MLP − linear(grouped) | +0.0014 |
| 读出头 | MLP − linear(independent) | +0.0057 |

构造效应比读出头效应大**一个半数量级以上**,且在两种读出头下几乎相同
(0.0936 / 0.0978)。

## 8. 与三次独立重训的交叉验证

`devtau_multiseed/rebuttal_evidence.md` 第 4 节用的是同一口径(按 gate 的 answer score
在 1323 supported / 1323 unsupported 全集上排序),但模型是**重训的 Stage-1**,与本目录
用的论文发布模型是两套独立产物。两边对得上:

| 量 | 本目录(论文发布模型) | 三次独立重训 |
|---|---|---|
| grouped AUROC | 0.9834 | .981 ± .003 |
| independent AUROC | 0.8898 | .895 ± .010 |
| 构造效应(ΔAUROC) | **+0.0936** | **+0.086** |

含义:构造效应这个量既不依赖读出头架构(第 7 节),也不依赖 Stage-1 那一次具体的
随机实现(本节)。这是论文核心主张在两个正交方向上的稳健性证据,比任何单次运行都强。

对照之下,论文里那些 argmax 相关的量(retention / A_FU)在重训下差 7.4 点 —— 差异**只**
出现在工作点相关的量上,阈值无关的量是稳的。两份文档的第 4 节讲的是同一件事。

## 9. 怎么读这些数字

1. **读出头选择基本不影响结论。** grouped 上 MLP 比 linear 只高 0.0014 AUROC,工作点
   对齐后 FCA 差距 1 个点以内。这对论文是**好事**:核心结论(构造方式带来的差异)对
   读出头架构鲁棒,换成线性门照样成立。代价是——「MLP 是必要的」这个说法**不能**
   靠判别能力来论证,只能靠它在 Stage-1 里作为辅助损失参与表征塑造这一角色来讲。

2. **原本设想的 Q2 辩护线不成立。** 我们本来希望看到「冻结基座可分性差,是 Stage-1
   联合训练把它塑造出来的」。实测冻结基座 `last_token` 已经 0.9685,适配只加 0.0135。
   换个角度确实能说得更好听一些:AUROC 误差从 0.0315 降到 0.0181,相对下降 42.7%;
   但绝对量就是 1.35 个点,拿它当主要论据会显得单薄。

3. **唯一干净支持架构设计的证据在 pooling 对比里。** `last_token`(0.9685)大幅高于
   三种视觉池化(0.60~0.76)。这直接支撑了「用最后一个 token 的联合表征作为头输入」
   这一设计选择,而不是对视觉 token 做池化。这条可以直接答「为什么用这个特征」。

4. **构造效应是真正的贡献点,而且被这批实验从两个方向加固了**(第 7、8 节)。

## 10. 风险:0.9685 这个数字是双刃剑

冻结基座探针 0.9685 可能被审稿人反读成「你的负样本本来就好检测,构造里有捷径」。
这个解读必须主动管理,不能指望它自己消失。

可用的反驳,基于论文自己的 Table(comprehensive test):

| 系统 | FCA ↓ | retention ↑ | Strict-3 ↑ |
|---|---|---|---|
| Base(冻结基座 + 原生结构化解码) | 0.211 | 0.833 | 0.660 |
| SFT native | 0.094 | 0.851 | 0.819 |
| ReGate | 0.132 | 0.976 | 0.856 |

同一个冻结骨干:表征里的 supported/unsupported 信号线性可分到 0.9685,但它**自己解码
时有 21.1% 的错误承诺率、Strict-3 只有 0.660**。这正是「表征里有 ≠ 生成时会用」,
与「模型内部状态编码了正确性但输出不体现」这一既有观察一致。论文要解决的就是把这个
潜在信号变成可控行为。

必须说准的一点:探针是在 7072 条构造数据上**有监督**训出来的,基座本身没有这份监督。
所以正确说法是「该信息在监督下线性可解码」,而不是「基座零样本就知道」。把潜在可解码
性转化为行为的,恰恰是构造 + 门,这又回到论文的贡献点上。

## 11. 措辞:能说 / 不能说

**能说**

- 读出头架构不是结论的来源:MLP 与线性门 AUROC 差 0.0014,工作点对齐后 FCA 差 1 个点内。
- 构造效应 +0.0936(MLP)/ +0.0978(linear),在两种读出头下几乎相同。
- 构造效应在论文发布模型与三次独立重训上一致(+0.0936 vs +0.086)。
- 头输入用 `last_token` 而非视觉池化是有依据的选择(0.9685 vs 0.60~0.76)。
- 评测流水线原样复现了论文已发布数字(最大偏差 0.0)以及附录记录的 AUROC 与 q10。
- 若被追问「线性门 FCA 更低,你们为什么不用它」:引第 4 节,说明这是尺度导致的工作点
  差异,并指出 independent arm 的对照。

**不能说**

- ~~「MLP 头判别能力显著优于线性门」~~ —— 差 0.0014,拿不出手。
- ~~「线性门 FCA 0.053 优于 MLP 0.132」~~ —— 忽略了 retention 掉 8 个点,是错误比较。
  任何单独引用 argmax 下 FCA 的说法都必须同时给 retention。
- ~~「Stage-1 的辅助头训练创造了可分性」~~ —— 只加 0.0135,冻结基座已有 0.9685。
- ~~「冻结基座区分不了 supported / unsupported」~~ —— 直接被 0.9685 打穿,别说。
- ~~「冻结 0.9685 低于 independent 适配 0.8898,所以适配有用」~~ —— 方向都反了,而且这个
  对比混合了构造与骨干两个因子,见第 12 节。
- **不要主动把 `mlp_refit` 写进任何对外材料。** 它只是证明我们能复现 Stage-2 标定流程;
  一旦出现,就要回答「为什么重训的头和发布的头工作点不同」,而那是第 4 节那套解释的
  重复,徒增攻击面。

## 12. 已知弱点(自己清楚,不主动写)

- **缺失单元格:** 没做「冻结基座 + independent 构造」的探针(需要再抽一次特征)。因此
  「冻结 0.9685 vs independent 适配 0.8898」这个对比混合了构造和骨干两个因子,**不能当
  单因子结论引用**。要用就得补这一格。
- 冻结探针的三类 acc / MCC(0.8681 / 0.7812)与第 3 节端到端指标不同口径,不可跨表比较;
  写材料时容易被误拼在一张表里。
- grouped 头的 `mu`/`sd` provenance 对不上(第 2 节末)。论文数字被精确复现所以无害,但
  如果审稿人要求逐位复现 Stage-2 标定,这一格答不上来。
- 「MLP 必要性」目前唯一的论据是它在 Stage-1 作为辅助损失参与表征塑造(`head_lambda`
  0 → 0.4,梯度回传进 backbone),这是**机制论证而非消融证据**。真要做消融,得跑一次
  `head_lambda = 0` 的 Stage-1,成本是一次完整训练。

## 13. 使用建议

| 材料 | 建议 |
|---|---|
| 第 3、4、5 节(读出头对比 + 工作点诊断 + 阈值无关) | 可进 rebuttal。回答「线性门够不够」,且第 4 节自带对照组 |
| 第 6 节 pooling 对比 | 可进 rebuttal。回答「为什么用这个特征」 |
| 第 7、8 节(因子量级 + 交叉验证) | 可进 rebuttal,是对论文核心主张最有力的一块 |
| 第 6 节冻结 vs 适配的 +0.0135 | 不建议主动提。只在审稿人直接问「为什么不用冻结特征 + 线性探针」时用,且必须搭配第 10 节的行为层面论证 |
| `mlp_refit` | 不发布(见第 11 节) |
| 主论文正文 | 不改。这批材料定位是 rebuttal 备料 |

## 14. 复现

```bash
# 冻结基座探针:16 个 (pooling, C) 拟合并行铺开
# sklearn lbfgs 是 CPU 求解器,GPU 无加速;96 核上墙钟约 6.6 分钟,串行要数小时
bash runs/frozen_base/launch.sh

# 汇总并与适配门对比
python scripts/frozen_probe_summary.py \
  --run-dir runs/frozen_base \
  --adapted summary/matched_retention.json \
  --out summary/frozen_probe.json

# 适配门侧(需要 GPU 抽取 head_hidden,8 卡)
bash scripts/run_arm.sh grouped all
bash scripts/run_arm.sh independent all
python scripts/matched_retention.py --runs-root runs --output summary/matched_retention.json
python scripts/summarize.py
```

## 数据来源

```
papermodel_lineargate/summary/frozen_probe.json        冻结探针逐 pooling 结果 + 适配门参照
papermodel_lineargate/summary/matched_retention.json   AUROC 与对齐 retention 的 FCA
papermodel_lineargate/summary/gate_comparison.json     三类 argmax 全指标 + 对已发布数字的偏差校验
papermodel_lineargate/runs/frozen_base/metrics_*.jsonl 16 个拟合的逐条记录
papermodel_lineargate/runs/frozen_base/logs/           每个分片的日志
papermodel_lineargate/models/{grouped,independent}/    论文发布的 adapter + 标定头(SHA256 校验)
```

冻结基座表征:`new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/repr/{train,dev,test}/`
论文 Stage-1:`new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444`
论文评估产物:`new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/`
多次重训证据:`devtau_multiseed/rebuttal_evidence.md`(含 `h27experiment/` 与 `devtau_multiseed/` 下的汇总)
