# 冻结基座门控（frozen-base gate）与历史多 seed 核验（review evidence）

本目录是 CPU-only 复核的证据区。不训练 backbone、不推理 API、不修改源数据；所有拟合都在
`agrisam3` conda 环境内以 `CUDA_VISIBLE_DEVICES=''` + 4 个 BLAS 线程完成。源数据与源脚本所在仓库
`/user/huangyongbin/newrepo/agriabstain` 保持原样，本目录只收结果、provenance 副本与复算脚本。

主要结论（先给数字，后给口径）：

| 方法（comprehensive，3-class argmax，base-native candidate） | FCA↓ | supported retention↑ | binary MCC↑ | Strict-3↑ | E2E-mIoU↑ | MTS macro↑ | state acc | 3-class MCC | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base native（已发布/复算一致） | 0.2109 | 0.8330 | 0.6227 | 0.6602 | 0.6557 | 0.6638 | — | — | — |
| **Frozen-base linear gate**（C=0.01，full-dev 选） | **0.0401** | **0.8020** | **0.7716** | **0.8220** | **0.5650** | **0.7820** | **0.8689** | **0.7823** | **0.9685** |
| **Frozen-base Stage2 MLP**（seed42，full-dev 选，best ep50） | **0.0983** | **0.8647** | **0.7670** | **0.8122** | **0.5930** | **0.7654** | **0.8685** | **0.7691** | **0.9671** |
| Paper ReGate argmax（adapted 单次 seed42，仅上下文参考） | 0.1323 | 0.9758 | 0.8485 | 0.8560 | 0.7011 | 0.8195 | — | — | — |

注意：**MCC_binary 与 MCC_3class 是两个不同的量**。MCC_binary 是「supported vs answered」的二元
Matthews 相关系数（FCA/A-FU 对应的选择轴）；MCC_3class 是三类状态 assignment 的多分类 MCC
（flattened contingency）。本表把两者分开列；不要把 3-class MCC（frozen linear 0.7823）当作二元
MCC（0.7716）。上表以 `MCC_binary` 为原论文 Table 的 MCC 对应量。

---

## 1. Frozen-base linear gate

### 1.1 拟合与 C 选择（CPU，`tools/review_frozen/fit_frozen_linear.py`）

- 特征缓存：`<source>/new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/repr/{train,dev,test}`
  - manifest 校验：adapter=None、layer=27、n=(7072, 2408, 2646)，`head_label` 分布
    train [2703,1500,2869] / dev [1204,205,999] / test [1323,205,1118]，与
    `source/papermodel_lineargate/summary/frozen_probe.json` 完全一致。
- 特征 = last-token pooling（frozen_probe 的 comparable_pool / best_pool）。
- 拟合：multinomial `LogisticRegression(C, lbfgs, max_iter=2000, random_state=42)`，train 集标准化
  （mu/sd 来自 7072 条 frozen train）。
- **C 选择：min dev-loss 于完整的 2408 条 frozen dev。** （与 adapted arm 的 100 条标定 dev 不同，
  见 §3。）
- 复算输出（`runs/linear/metrics.jsonl`）与原始 `frozen_probe.json` 对照：

| C | dev_loss (复算/原) | test AUROC (复算/原) | test 3-class MCC (复算/原) |
|---|---:|---:|---:|
| 0.01 | 0.3300 / 0.3300 | 0.9685 / 0.96848 | 0.7823 / 0.78119 |
| 0.1 | 0.3961 / 0.3971 | 0.9646 / 0.96433 | 0.7558 / 0.75428 |
| 1.0 | 0.6427 / 0.6410 | 0.9541 / 0.95445 | 0.6985 / 0.70076 |
| 10.0 | 1.2414 / 1.2875 | 0.9523 / 0.95350 | 0.6722 / 0.66849 |

差异 ≤ 0.002（lbfgs/BLAS 线程数造成的数值噪声），CPU 复现成立。选定 C=0.01。

### 1.2 Comprehensive 正式评估（`tools/review_frozen/reroute_frozen_base.py`）

- 与 base-native 预测按 **sample_id 严格字符串对齐**（2646/2646，无缺失/多余）。
- 候选来源：**base-native 解码本身**（`base_native_regate_20260720/native/base/comprehensive/`），
  不用任何 adapted candidate。
  - base-native 回答 → 共享 candidate 有效（label+box）；
  - base-native 弃权 → candidate 无效（valid=False），被门选答时正确性计零（与论文口径一致）。
- 指标函数直接 import 论文模块
  `source/aaai_code/experiments/base_native_regate/analyze_native_regate.py`（`regate_curve_point`、
  `native_point`）与 `reason_abstain/eval/metrics.py`（`metamorphic_test_satisfaction`）。
- Base-native 参考点同时给出「存储的已发布 metrics.json」与「从原 predictions.jsonl 复算」两列，
  二者及 review.md Table 中 Base native 行完全一致（FCA 0.2109 / retention 0.8330 / MCC 0.6227 /
  Strict-3 0.6602 / E2E 0.6557 / MTS 0.6638），用于校验管线本身没漂移。

结果（frozen linear，3-class argmax，base-native candidate）：

```
FCA                     0.04006
supported_retention     0.80197      A_FU 0.19803
Strict-3                0.82199
E2E_mIoU (论文口径)      0.56501      (E2E conditional binary 0.7964)
MTS_macro               0.78198
MCC_binary              0.77159
state_accuracy           0.86886
MCC_3class              0.78230      ← 不是二元 MCC
AUROC                   0.96848
selective_risk          0.24147
n_supported             1323         n_unsupported 1323
```

解读（口径内）：冻结基座 last-token 表征线性可分性很高（AUROC 0.9685），门把 FCA 从 base-native
0.2109 压到 0.0401；代价是 retention 从 0.8330 降到 0.8020（argmax 落点偏保守），Strict-3 升到
0.8220。E2E-mIoU 0.5650 低于 base native 0.6557（门选答的 supported 子集内定位正确率不如全量
native 解码，保持门控的常见 trade-off）。与 adapted ReGate（paper 单次）不对等比较要谨慎：
门的数据来源（frozen vs Stage-1 adapted）、C/dev 选择口径均不同。

## 2. Frozen-base Stage2 MLP gate（同架构复刻，CPU）

`tools/review_frozen/fit_frozen_mlp.py`，与被复刻的 `fit_gates.py`/`calibrate_head_3reason.py` 完全
一致的结构与超参：

- 架构 4096-512-256-3（GELU + dropout 0.1），全批量 Adam `lr=1e-3, wd=1e-3`，
  每 50 epoch 评一次 dev-loss，patience 5，seed 42，max 1000 epoch。
- 训练/校验集：frozen train 7072 / **frozen dev 2408（full-dev 选择，见 §3 的差别声明）**。
- 结果：best_epoch=50，best_dev_loss=0.3649；test AUROC 0.9671、3-class MCC 0.7691、
  argmax 准确率 0.8685。

Reroute（base-native candidate，同 §1.2 管线）：

```
FCA                     0.09826
supported_retention     0.86470      A_FU 0.13530
Strict-3                0.81217
E2E_mIoU (论文口径)      0.59295
MTS_macro               0.76542
MCC_binary              0.76697
state_accuracy           0.86848
MCC_3class              0.76913
AUROC                   0.96713
```

注意：这里的 MLP 是「未经 Stage-1 适配、直接在冻结基座表征上训的头」，训练/选择盘子与
adapted arm 的 100 条标定 dev 不同，不能把两者 dev 口径说成一致。它的作用是对照——
即使换 MLP、有同样的判别能力（AUROC 0.967 vs 0.969、MCC 0.767 vs 0.772），冻结表征上的门
仍比 adapted ReGate（paper 单次）在 retention/Strict-3/E2E 上有差距，说明 Stage-1 joint
训练塑造的超平面/表征确实带来了不同行为，而不是"基座自带可分性即可等价替换"。

## 3. Paper 100 条 stratified dev：重建与匹配性（重要差别声明）

论文 Stage-2 的 head 选择用 dev.jsonl 上按三类分层、`random.Random(42)` 抽出的 **100 条 dev**
（`<source>/new_reason/scripts/calibrate_head_3reason.py` 口径：
每类 33 条 + 1 条补齐 + 末次 shuffle）。

`tools/review_frozen/rebuild_paper_dev.py` 逐字复刻该抽样后：

- 得到 100 个 sample_id（answer=34 / wrong_object=33 / no_target=33），存
  `paper_100_dev_subset.tsv`；
- **该 100 条在 frozen dev 缓存里无法按 sample_id 匹配**：
  - frozen dev npz 的 `sample_id` 只是 sft-json 行号（'0'…'2407'，无语义 id）；
  - frozen dev 的 derived-image 体系与 dev.jsonl 不是同一 build（如
    `region-ladder-100` vs `ladder-100`），精确匹配数仅 729/2408，region-strip 后 971/2408，
    其余 1437 条无法对应；
  - 详见 `paper_100_dev_matchability.json`。

因此本复核不能把「100 条标定 dev 选择」搬到 frozen cache 上，一律以 **full 2408 冻结 dev 选择**
作为 frozen baseline（拥有比 adapted 100-条口径更多的选择信号），并在每一处与 adapted 的 dev
口径对照中明确标注这一差别。凡声称 dev 口径一致处需另建 identity 映射，本证据区不作出该声明。

## 4. 历史 multi-seed 核验（只读）

只读对照（不改源文件）：

- `source/h27experiment/summary/multiseed_summary.json`
- `source/devtau_multiseed/summary/dev_selected_multiseed.json`
- `source/devtau_multiseed/rebuttal_evidence.md`

核验产出 `multiseed_verification.json`（由 `tools/review_frozen/verify_multiseed.py` 生成）。

核验结论：

1. **完整性**：`frozen_seeds=[7,21,42]`；`missing=[]`；`all_runs_reported=True`；rows 覆盖
   3×3×N，seed 计数齐全。
2. **aggregate 自洽**：160/160 条 aggregate（mean/min/max/std，std 为 ddof=1 样本标准差）
   与逐 seed rows 重算完全一致（≤1e-6）。
3. **rebuttal §2 表格**：Comp./Ref./PW/L1/L2/L3 各 `mlp_regate`、`native` 的 MTS/MCC/FCA
   「mean±std」（grouped / independent）逐项与 `rebuttal_evidence.md` §2 九行表一致
   （例如 Comp.-Gate MTS .806±.017/.528±.012、MCC .843±.012/.523±.008、FCA .060±.008/.487±.008
   完全吻合）。
4. **方向性**：3 seeds 在 comprehensive 上 grouped−independent 的 FCA<0、MTS/MCC>0 三次全部
   同向（例：MTS_macro +0.273/+0.256/+0.306；MCC +0.328/+0.297/+0.334；FCA −0.435/−0.411/−0.435），
   支撑“构造消融效应比运行间噪声大、三次同向”。
5. **devtau per-seed**：seed 7/21/42 的 `selected_threshold / dev_target_retention / dev_selected_
   retention / test_native_retention / test_native_fca / test_regate_retention / test_regate_fca /
   test_regate_strict3 / relative_fca_reduction` 字段齐全且与 `rebuttal_evidence.md` §3 逐 seed
   表格一致（threshold .7984/.6501/.7400；相对 FCA 下降 56.6%/54.1%/57.0%）。aggregate 的
   mean±std 与其 per-seed 一致。
6. **论文单次 vs 三次重训**：multiseed_summary 的 rows 与 dev_selected_multiseed.json 均把论文
   那次（`new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444`）
   与 h27 重训分开记录；本核验只引用 rows 里真实存在的 seed 结果，**不把 h27evaluations/experiment
   或不同 arm 混在一起**。

逐 seed comprehensive 明细（copy：`multiseed_comprehensive_rows.csv`）：

| seed | arm | interface | FCA | MCC | Strict-3 | MTS_macro | E2E_mIoU | state_acc |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 7 | grouped | linear_gate | 0.0582 | 0.8464 | 0.8488 | — | — | 0.9214 |
| 7 | grouped | mlp_regate | 0.0605 | 0.8470 | 0.8503 | 0.8106 | 0.6318 | — |
| 7 | grouped | native | 0.0748 | 0.7251 | 0.8084 | 0.7896 | 0.6553 | — |
| 7 | independent | linear_gate | 0.4558 | 0.5481 | 0.6886 | — | — | 0.7479 |
| 7 | independent | mlp_regate | 0.4958 | 0.5193 | 0.6686 | 0.5380 | 0.6978 | — |
| 7 | independent | native | 0.1965 | 0.6686 | 0.8080 | 0.7566 | 0.7148 | — |
| 21 | grouped | linear_gate | 0.0529 | 0.8324 | 0.8382 | — | — | 0.9131 |
| 21 | grouped | mlp_regate | 0.0680 | 0.8297 | 0.8318 | 0.7874 | 0.5981 | — |
| 21 | grouped | native | 0.1104 | 0.6936 | 0.7491 | 0.7547 | 0.6503 | — |
| 21 | independent | linear_gate | 0.4649 | 0.5396 | 0.6852 | — | — | 0.7430 |
| 21 | independent | mlp_regate | 0.4792 | 0.5330 | 0.6784 | 0.5316 | 0.6889 | — |
| 21 | independent | native | 0.2018 | 0.6667 | 0.7993 | 0.7459 | 0.7059 | — |
| 42 | grouped | linear_gate | 0.0476 | 0.8574 | 0.8651 | — | — | 0.9252 |
| 42 | grouped | mlp_regate | 0.0514 | 0.8520 | 0.8624 | 0.8206 | 0.6455 | — |
| 42 | grouped | native | 0.0809 | 0.7471 | 0.8182 | 0.7979 | 0.6806 | — |
| 42 | independent | linear_gate | 0.5042 | 0.5189 | 0.6481 | — | — | 0.7271 |
| 42 | independent | mlp_regate | 0.4860 | 0.5178 | 0.6508 | 0.5143 | 0.6617 | — |
| 42 | independent | native | 0.1837 | 0.6387 | 0.7831 | 0.7346 | 0.6773 | — |

> `linear_gate` 无 MTS/E2E 列（multiseed_summary 里该行的 `E2E_mIoU`/`MTS_macro` 为 null），
> 引用时不要给这些格填值。

**措辞边界（沿用 rebuttal_evidence.md 的禁忌，仅供引用时自查）**

- “三次独立重训 / three independent retrainings”可引用；逐 seed 财务报表见上表与
  `multiseed_verification.json`。
- **不声称模型逐位相同**：rebuttal §1 只证明 `total_flos`、step-0 `loss_answer` 逐位一致
  （同数据、同顺序、同超参）；step-0 `head_loss` 在初始化处即分叉，因此是「同配置 replicate」，
  不是「bitwise 同一模型」。引用时用配置层面的措辞。
- 不把 `h27experiment`（三次独立重训）与 `devtau_multiseed`（dev-selected 协议）的结果混为同一实验；
  也不把 grouped 与 independent 两个臂混着引用同一行做配对差。
- argmax 下的 retention/A_FU 未复现（差约 7.4 点）→ 凡原表带这些量时要用 §4 的曲线/阈值无关论证，
  不要写“复现”。

## 5. 文件与路径清单

本目录下：

```
RESULTS.md                              ← 本文件
runs/linear/                            frozen linear：fit.log, metrics.jsonl, linear_gate_best.npz,
                                            metrics.json, frozen_gate_probabilities.jsonl
runs/mlp/                               frozen MLP：fit.log, mlp_ours.pt, mlp_ours_manifest.json,
                                            test_proba.npy, metrics.json,
                                            frozen_gate_probabilities.jsonl
frozen_probe.json                       ← source/papermodel_lineargate/summary/frozen_probe.json 副本
frozen_base_metrics.jsonl / frozen_base_launch.sh   ← frozen_probe 原始 runs/frozen_base 证据副本
matched_retention.json                  ← 阈值无关对照（AUROC + matched-retention FCA，原始副本）
gate_comparison.json                    ← adapted arms 门控对比（原始副本，单次 seed42）
multiseed_summary.json                  ← h27 三次重训汇总（原始副本）
dev_selected_multiseed.json             ← dev-selected per-seed（原始副本）
rebuttal_evidence.md                    ← 措辞/引用边界（原始副本）
multiseed_verification.json             ← 本复核只读核验报告（生成）
multiseed_comprehensive_rows.csv        ← 逐 seed comprehensive 明细（生成）
paper_100_dev_subset.tsv                ← 论文 100 条 stratified dev 重建（生成）
paper_100_dev_matchability.json         ← 100-dev 在 frozen cache 不可匹配的证明（生成）
```

`tools/review_frozen/`：

```
fit_frozen_linear.py     复算 frozen LR gate（§1.1）
reroute_frozen_base.py   base-native candidate 离线 rerouting（§1.2/§2）
fit_frozen_mlp.py        frozen Stage2 MLP（§2）
rebuild_paper_dev.py     100-dev 重建 + 匹配性（§3）
verify_multiseed.py      h27/devtau 只读核验（§4）
frozen_common.py         shared helpers（对齐/softmax/指标包装）
original_scripts/        原脚本字节副本 + README.md（路径适配说明）
```

环境与约束：`/user/huangyongbin/miniconda3/envs/agrisam3/bin/python`（与原始 frozen_probe 同环境）；
每次运行 `export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`。
未使用 `run_all.sh` / `run_calibrate` 的 GPU extraction 路径；全部为显式 CPU fit。
`review.md` 未修改。