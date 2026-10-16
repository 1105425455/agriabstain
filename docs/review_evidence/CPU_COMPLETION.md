# 无需GPU补充完成清单

## 已完成

| 审稿要求 | 完成内容 | 输出 |
|---|---|---|
| Stage 2独立作用 | 从同checkpoint adapter读取Stage-1三态head与mu/sd，与发布Stage-2同hidden/candidate CPU评分 | `cpu_gate/scored_heads.json` |
| gate稳定性 | 固定两组论文backbone，各seed7/21/42重新拟合小MLP，共6次CPU训练，dev-only选点 | `cpu_gate/*_seed*/mlp_ours_manifest.json` |
| gate归一化追溯 | 两组cache train统计与发布head mu/sd逐维比较 | `cpu_gate/scored_heads.json` |
| 多次完整训练结果 | 复用历史h27三次完整重训，12份Comprehensive原预测重新计算并核对原汇总 | `frozen_and_prior/prior_raw_verification.json` |
| Frozen+gate | frozen linear及MLP CPU拟合，正式Comprehensive、Base native候选；MLP补充精确论文dev100 | `frozen_and_prior/runs/mlp_dev100/metrics.json` |
| dev100身份 | original_index→SFT记录→image/query/state唯一对齐100/100 | `frozen_and_prior/exact_dev100.json` |
| L2构造与几何 | 395条负例逐条重建crop、核对尺寸、检查所有query标注交集，严格零交集与容差分开 | `l2_geometry_summary.json` / `l2_geometry_tolerance.json` |
| L2/L3联系 | 148个目标实例ID重合、已有decision差异为0；原标签仍保留 | `l2_geometry_summary.json` |
| L3 baseline | 相同183组567邻接，GPT与ReGate二元单调性、连续p_answer另外统计 | `additional_cpu/l3_gpt_vs_regate.json` |
| Base筛类偏差 | 原校准池24/20/4类micro/macro正确率 | `additional_cpu/capability_24_vs_20.json` |
| 人工审计准备 | 三态各200条分层清单、空标注栏、权重、标注操作说明 | `additional_cpu/label_audit_prep.csv` / `ANNOTATOR_GUIDE.md` |
| confidence可用性 | 检查native dev/test原始字段，无自身logprob/entropy | `additional_cpu/sft_native_confidence_and_baselines.json` |
| 最近方法现有结果 | 确认groundingme/openref是本模型外部数据评测，不是指定近邻方法结果 | 同上 |
| decode-skip耗时 | 审计155份日志，无同硬件batch受控延迟对比 | `additional_cpu/decode_skip_timing.json` |
| 相关工作 | 官方/数据库摘要核验，保留无法核实GUI条目 | `related_work.md` |

## 实际仍缺的条件

- 两名人类的独立标注及复核尚未执行；计算不能替代人工答案。
- Stage-1 auxiliary loss=0完整训练、指定近邻模型同协议预测、三系统全24类独立测试、SFT自身token confidence需要不存在的模型运行产物。
- 受控GPU latency/显存比较需要新的硬件运行。
- Frozen缓存提取分辨率与adapted calibration的处理路径可能不同；dev100样本身份已匹配，仍不将两个完整系统解释为只改一个因素的纯因果比较。

## 核验与勘误

- Stage-1真实三分类head来自adapter safetensors；同目录`discrim_head_2class.pt`输出为2维，未用于评分。
- 新head reason使用自身p_wo/p_nt；未沿用旧head概率。
- 论文发布head的Independent CPU replay与原运行有2/2646条decision差异，原论文与replay表分列。
- `paper_100_dev_matchability.json`早期直接sample-ID匹配结论已被纠正：通过original_index及SFT导出记录可以100/100唯一匹配。最终使用`exact_dev100.json`。
- L2主结果采用严格零交集；初版1%容差统计已改为单独敏感性，不混入“完全排除”定义。
- Frozen同dev100结果用正式aggregate_metrics和MTS独立重算，Strict3/E2E/MTS全部一致。
- 原工作区、论文、正式benchmark、发布head和GPU训练配置均未改动；新增模型文件仍被Git忽略。
