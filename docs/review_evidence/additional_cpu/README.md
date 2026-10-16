# additional_cpu：第二轮离线审稿证据

全部结果仅读取现有文件，无模型推理、无训练、无付费 API、无 GPU。

| 文件 | 用途 |
|---|---|
| `l3_gpt_vs_regate.json/.md` | L3 GPT 二元决策 vs ReGate head 概率的相邻可见度 baseline（183 组/567 相邻对） |
| `capability_24_vs_20.json/.md` | Base 校准池全 24 类 per-label n/correct 重算，retained20/excluded4 micro·macro |
| `label_audit_prep.csv/.json/.md` | 双人标签审计抽样准备（每状态 200，T6/COCO 负例优先，分层+权重+source 去重） |
| `sft_native_confidence_and_baselines.json/.md` | SFT native 无自身 confidence 的证据；groundingme/openref 非 CounterfactualREC/GREC/RECANTFormer |
| `decode_skip_timing.json/.md` | 现有日志不能量化 decode-skip；仅报告缺失 |
| `sources_hashes.json` | 上述所有输入文件的 SHA-256 |

复算：

```bash
PYTHONDONTWRITEBYTECODE=1 /user/huangyongbin/miniconda3/envs/agrivllm/bin/python tools/review_additional_cpu.py
```
