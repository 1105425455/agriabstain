# SFT对比实验: 判别头 vs Prompt vs CoT

目标: 证明三分类判别头架构打败纯prompt/CoT baseline (AAAI对照)

## 隔离原则
- 所有baseline产出放 baselines/prompt_cot/outputs/ 下, 不碰主训练目录
- 同一SFT模型 ckpt-444, 同一测试探针, 只差decision方式

## 三个对比对象
1. **Ours (dual-head)**: 判别头出decision. 基线指标见 SFT_FINAL.md
   - L1 MCC 0.419 / L2 0.574 / L3 0.841 / test 0.856 / refco 0.925
2. **Prompt-baseline**: 同模型,纯LM+prompt生成decision (不用判别头)
   - variants: basic / reward-scheme / reward+norms / two-stage / few-shot, tune后报best
3. **CoT-baseline**: 同模型,纯LM+CoT先推理再决策
   - structured 4-step: evidence→matching→confidence→decision

## 公平性 (防审稿人质疑)
- 同base模型 Qwen3-VL-8B ckpt-444
- baseline充分tune, 报best variant
- 完整指标: C@risk / AURC / ECE + MCC/AAC

## 测试探针 (同 ours)
L1_discrim(410) / L2_evidence(790) / L3_dose(750) / test_full(2646) / ood_refcocog(362) / ood_plantvillage(240)
