# Selective Grounding

本目录评估 ReGate 在不同 `p_supported` 阈值下的 grounding 风险、coverage 和 unsupported hallucination。正式 ReGate 使用 SFT_FINAL `checkpoint-444` 与 calibrated tri-head；PlantWild 仅使用 closed12 408。

## 生成 ReGate candidates

输出必须放在外部 runtime 目录，不能写入 `benchmark/`：

```bash
export OUTPUT_ROOT=/path/to/runtime/selective_grounding
bash experiments/selective_grounding/run_regate_candidates.sh
```

脚本先将 annotation 中的 image path 解析到 `${OUTPUT_ROOT}/inputs/`，不会修改或写入 `benchmark/`。默认从只读的 `new_reason/outputs/benchmarks/transformed_images` 解析未打包的 derived images；可用 `DERIVED_IMAGE_ROOT` 覆盖。推理固定使用 `--system-override decision`，确保 RefCOCOg 自带的 CoT prompt 不改变判别头输入，并复现正式 faithful evaluation。可在末尾追加 evaluator 参数，例如 `--limit 1`。每条 prediction 必须包含 `p_answer`、`p_wo`、`p_nt` 和 `answer_candidate`。

## 离线分析

从 `aaai_code` 目录执行：

```bash
python -m experiments.selective_grounding.analyze_selective_grounding \
  --regate comprehensive=/path/to/comprehensive/predictions.jsonl \
  --regate refcocog=/path/to/refcocog/predictions.jsonl \
  --regate plantwild=/path/to/plantwild_closed12/predictions.jsonl \
  --baseline native_lm:comprehensive=/path/to/native_lm/test_full/predictions.jsonl \
  --output-dir /path/to/runtime/selective_grounding_analysis
```

`--baseline METHOD:DATASET=PATH` 可重复传入。分析器要求 baseline 与 ReGate 的 `sample_id` 集合及 GT 完全一致，不匹配时立即报错。

输出包括每个数据集的完整 unique-threshold curve、baseline points、ReGate argmax point、empirical/interpolated matched-retention 表，以及 Risk-Coverage 和 Hallucination-Retention 图。`selective_risk` 将 unsupported answer、格式错误 candidate、错误 label 和 IoU 未达标均计为错误；GT box 为 null 时按 label correctness。

## Prompt-interface fixed-GT artifact

`main/prompt_interface_metrics_fixed_gt.json` 是论文 prompt-interface appendix 表的
canonical compact artifact。它使用当前 `reason_abstain.eval.metrics.aggregate_metrics`
从逐样本 predictions 重算，保存完整 fixed-GT metrics、样本数和每个 predictions 文件的
SHA256。重建命令：

```bash
python -m experiments.selective_grounding.rescore_prompt_interfaces \
  --prompt-root /path/to/runtime/prompt_cot/outputs \
  --regate-root /path/to/runtime/base_native_regate/regate \
  --output experiments/selective_grounding/main/prompt_interface_metrics_fixed_gt.json
```
