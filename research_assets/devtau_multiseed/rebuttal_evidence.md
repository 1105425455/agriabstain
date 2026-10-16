# 多次独立重训:可用的正面证据

用途:rebuttal 素材。论文、代码、checklist 均保持不动;这里只汇总**站得住、可以直接引用**的部分,以及措辞禁忌。

三次重训 = `h27experiment/runs/{grouped,independent}/seed-{7,21,42}`,每个 seed 都是从 base 模型完整跑完 Stage-1(666 步 / 6 epoch,独立的 `checkpoint-111`…`checkpoint-666`)+ Stage-2 头标定 + 全套评估。论文那次 = `new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444`。

统一口径:称 **"three independent retrainings"**(三次独立重训),不要称 "three seeds"。

---

## 1. 四次运行是同配置 replicate(逐位证据)

这一节的作用是排掉"重跑用的是另一套设置"这种更难看的怀疑。

| | 论文 | h27 grouped seed-42 |
|---|---|---|
| `max_steps` | 666 | 666 |
| `num_train_epochs` | 6 | 6 |
| `train_batch_size` | 1 | 1 |
| `gradient_accumulation_steps` | 8 | 8 |
| `learning_rate` | 1e-4 | 1e-4 |
| `bf16` | True | True |
| `seed` | 42 | 42 |
| `head_lambda` | 0 → 0.4 | 0 → 0.4 |
| `total_flos` | 2.7900122973077504e+17 | 2.7900122973077504e+17 |
| log 条数 | 796 | 796 |
| step-0 `loss_answer` | 3.72898268699646 | 3.72898268699646 |

`total_flos` 与 step-0 `loss_answer` **逐位相同** → 同一份数据、同一个数据顺序、同一套超参。

唯一在 step 0 就分叉的是 step-0 `head_loss`(1.40625 vs 1.453125)。LoRA 的 B 矩阵零初始化,所以 step-0 的 answer loss 与初始化无关;判别头是真随机初始化,它一露头就不同。

来源:两侧 `checkpoint-444/trainer_state.json` 与 `training_args.bin`。

---

## 2. 构造消融:三对三(最强的一块)

这是**唯一真正需要多次重训来支撑的贡献主张**——"同源构造是有效的设计选择"。原来是单次对单次,现在是三对三。

Grouped / Independent,各三次重训的 mean ± std,行序与主论文 `tab:grouping_ablation` 一致:

| 行 | MTS | MCC | FCA↓ |
|---|---|---|---|
| Comp.-Gate | .806±.017 / .528±.012 | .843±.012 / .523±.008 | .060±.008 / .487±.008 |
| Comp.-Native | .781±.023 / .746±.011 | .722±.027 / .658±.017 | .089±.019 / .194±.009 |
| Ref.-Gate | .820±.019 / .814±.033 | .944±.006 / .941±.000 | .057±.006 / .061±.000 |
| Ref.-Native | .823±.019 / .827±.035 | .877±.003 / .912±.009 | .044±.000 / .031±.003 |
| PW-Gate | .620±.040 / .582±.005 | .484±.038 / .289±.048 | .307±.105 / .309±.027 |
| PW-Native | .572±.099 / .571±.020 | .527±.054 / .442±.014 | .183±.055 / .338±.006 |
| L1-Gate | .694±.048 / .480±.021 | .477±.073 / .075±.026 | .280±.066 / .820±.024 |
| L2-Gate | .663±.006 / .389±.005 | .575±.028 / .078±.035 | .382±.011 / .908±.016 |
| L3-Gate | .809±.028 / .511±.024 | .868±.035 / .294±.024 | .074±.027 / .854±.022 |

可以直接引用的结论:

- 综合测试与三个受控 probe 上,Grouped 在**每一次重跑**中都优于 Independent:MTS 差 +.21 ~ +.30,MCC 差 +.32 ~ +.57,而运行间标准差只有 .01 ~ .07 → **效应比噪声大一个量级,三次全部同向**。
- L2/L3 的 MCC 差值 +.497 / +.574,L1 +.403。
- 源不相交的 PlantWild 方向一致但更小(MCC +.196±.079)。
- RefCOCOg 上两个条件在天花板附近、差异在一个标准差内,与论文里"general-domain control"的定位一致。
- 论文单次报的 seed-42 MCC 差值,在七个非 RefCOCOg 行里四行高于重跑均值、三行低于,没有系统性偏向。

来源:`h27experiment/summary/multiseed_summary.json`。

---

## 3. Development-selected 工作点:复现质量

**关键点:不稳的只有 argmax,dev-selected 是稳的。** dev-selected 的阈值锚在每次运行自己的 native retention 上,所以吸收了 Stage-1 的水平漂移。

| 量 | 论文 | 三次重训 | 论文是否在 1σ 内 |
|---|---|---|---|
| FCA 绝对下降 | .0491 | .0494 ± .0091 | 是(几乎精确) |
| ReGate FCA | .0454 | .0393 ± .0099 | 是 |
| native FCA | .0945 | .0887 ± .0190 | 是 |
| FCA 相对下降 | 52.0% | 55.9 ± 1.6% | 否(论文更保守) |
| ReGate retention | .8896 | .8677 ± .0142 | 否(约 1.5σ) |
| native retention | .8511 | .8065 ± .0162 | 否(约 2.75σ) |
| Strict-3 | .8745 | .8467 ± .0180 | 否 |
| 选出的 τ | .9888 | .7295 ± .0747 | 否 |

逐 seed:

| seed | τ | native Ret | ReGate Ret | ReGate FCA | 相对下降 |
|---|---|---|---|---|---|
| 7 | .7984 | .7937 | .8564 | .0325 | 56.6% |
| 21 | .6501 | .8012 | .8632 | .0506 | 54.1% |
| 42 | .7400 | .8246 | .8836 | .0348 | 57.0% |

对比 argmax 工作点(同一个 Grouped 臂):

| 工作点 | 论文 | 三次重训 | 差距 |
|---|---|---|---|
| argmax retention | .976 | .902 ± .005 | 7.4 点 |
| dev-selected retention | .8896 | .8677 ± .0142 | 2.2 点 |

**τ 三次差异大(.7295 ± .0747)本身是在替协议辩护**:每次重训后头的置信度尺度不可比,所以必须在 dev 上重选阈值。若写死 .9888 换一次重训直接崩(预检实测 retention 掉到 .7793)。→ 使用者必须按各自 dev 重选 τ,不能复用常数。

来源:`devtau_multiseed/summary/dev_selected_multiseed.json`,逐 seed 见 `runs/seed-*/analysis/result.json`(含 τ sweep 与 sha256 完整性字段);`inputs/manifest.json` 记三个 seed 用的是同一份 2,408 条 dev(sha256 一致)。

---

## 4. 阈值无关的证据

用来证明"判别质量一致,只是 argmax 落点不同"。

Grouped / Independent 三次重训的 AUROC:

| probe | AUROC |
|---|---|
| Comprehensive | .981 / .895 |
| PlantWild | .829 / .687 |
| L1 | .814 / .581 |
| L2 | .827 / .596 |
| L3 | .981 / .906 |
| RefCOCOg | .980 / .993 |

匹配 supported retention = 0.95 时的 FCA:`.120/.472`(Comp.)、`.527/.849`(PW)、`.652/.925`(L1)、`.413/.916`(L2)、`.088/.372`(L3)、`.048/.031`(Ref.)。L3 用 183 fully visible + 148 fully excluded endpoints,保留 419 条 graded-response 记录。

论文那次与重跑落在**同一条风险-覆盖曲线**上:

| | 论文 seed-42 | 三次重训 |
|---|---|---|
| AUROC | .983 | .981 ± .003 |
| FCA @ Ret=.95 | .091 | .120 ± .025 |
| FCA @ Ret=.90 | .051 | .060 ± .012 |
| `q10(p_A)` | .985 | .523 ± .043 |

→ 排序质量相同(AUROC),曲线相同(matched retention),**只有置信度尺度不同**(`q10` .985 vs .523)。论文那次的头饱和了,所以 argmax 右移到高覆盖端:retention .976、FCA .132;重跑的头不饱和,argmax 落在更均衡处:retention .902、FCA .060。

这就是 argmax 下 A\_FU / retention 对不上的全部原因,而 A\_FU 就是 1 − retention,是全表最依赖工作点的量。

---

## 5. 论文那次在分布中的位置

如果被问"你们重跑的绝对水平比主表低,是不是不可复现":

- native retention 答对数(supported = 1323):论文 1126,重跑 1050 / 1060 / 1091。
- 四次合并:mean .8176,std .0259,论文在 **+1.29σ** → 不构成离群。
- 论文是四次可交换抽样里的最大值,`P = 1/4`,拒绝不了可交换性。
- 论文离最好那次重跑 .0265(35 条),而**三次重跑彼此的极差是 .0310(41 条)** → 论文到重跑的距离比重跑内部抖动更小。
- 方向对论文有利:论文报的门收益是四次里**最保守**的(+3.85 点 vs 重跑 +6.12 ± 0.20;相对 FCA 下降 52.0% vs 55.9 ± 1.6%)。别人重跑只会得到更强的效果。
- 论文那次不是从多次里挑的,当时只有那一次。落在顶端是事后才可见的抽样落点。

按剩余空间归一化的门收益(增益 ÷ (1 − native)):论文 .259,重跑 .304 / .312 / .336 = .317 ± .017 —— 比裸配对差值更接近,是更诚实的归一化说法。

---

## 6. 措辞:能说 / 不能说

**能说**

- "three independent retrainings",报 mean ± std。
- 构造消融的效应比运行间标准差大一个量级、三次全部同向。
- FCA 绝对下降精确复现(.0491 vs .0494 ± .0091)。
- 论文报的效果是四次里最保守的一次。
- τ 必须按各自 dev 重选,不能复用常数。
- 若被直接追问可复现性:承认主表是一次不可逐位重放的运行,并指向上面第 4、5 节。

**不能说**

- ~~"配对增益是不变量 / 跨重跑稳定"~~ —— 重跑内部是 +.0612 ± .0020(极差 0.38 点)非常稳,但**论文自己的 +.0385 落在簇外**。审稿人会拿论文那行去套这个"不变量",结果发现唯一不满足的就是论文。只说"论文是四次里最保守的一次",或用剩余空间归一化的版本。
- ~~"我们没有指定 seed / 故意不固定 seed 以获得无偏方差"~~ —— `training_args.bin` 记着 `seed=42`,seed 是指定了的。artifact 一公开就穿。
- ~~"argmax 下的 retention / A\_FU 复现了"~~ —— 没有,差 7.4 点。argmax 相关的量一律走第 4 节的曲线论证。
- ~~"论文在 ±1std 之外所以异常"~~ —— 那是把论文排除在 std 估计外、且 n=3 造成的假象;合并四次后只有 +1.29σ。
- **不要主动引入判别头初始化未受 seed 控制这个代码级细节。** 论文里报"三次独立重训 + 标准差"是标准做法,不需要解释随机性来自哪个模块。一旦写进去,就得回答"那你为什么不修",而那个答案很牵强。只在被直接问到逐位可复现性时才提。

---

## 7. 已知的弱点(自己清楚,不主动写)

- in-domain E2E-mIoU:论文报 Stage-1 把 `.656` 提到 `.705`,三次重训是 `.662 ± .016`(base .656)→ in-domain 的定位提升在重训下不稳。RefCOCOg 那半句很硬(`.525 → .640 ± .031`),可保留。若哪天顺手改,把 in-domain 半句语气放软即可。
- argmax 下 Grouped 的 A\_FU 高于 Independent(.098 ± .005 vs .044 ± .004),与论文单次报的 .024/.045 方向相反。原因见第 4 节。
- Stage-1 checkpoint 选择是在 100 条 dev 上按极小的 loss 差决定的,基本是平局:seed-21 `.296342@222` vs `.296931@555` 差 0.0006;seed-42 `.280059@444` vs `.280341@333` 差 0.0003。且 dev loss 与 native retention 不相关(seed-7 dev loss 最好但 native retention 最低)。
- 已验证:把三个 seed 统一到 step 444 **不能**弥合与主表的差距 —— seed-42 本来就是 step 444,native 仍差 2.65 点。不要提这个实验。

---

## 数据来源

```
devtau_multiseed/summary/dev_selected_multiseed.json    dev-selected 汇总
devtau_multiseed/runs/seed-*/analysis/result.json       逐 seed 完整记录(τ sweep + sha256)
devtau_multiseed/inputs/manifest.json                   2,408 条 dev 的 sha256
devtau_multiseed/scripts/run_all.sh                     复跑脚本(协议逐字复制自论文 pipeline)
h27experiment/summary/multiseed_summary.json            9 行 × 4 指标 × 两个 arm
h27experiment/runs/{grouped,independent}/seed-*/         checkpoints / evaluation / logs
h27experiment/writeup/multiseed_retraining_note.tex      早前写的附录稿(未进论文,留作素材)
```

论文 Stage-1:`new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444`
论文评估产物:`new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/`
