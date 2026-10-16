## Capability audit — 全 24 类 vs retained 20 的 Base 校准池重算

按 Base 校准池（PlantDoc base records 768 条）与 frozen Qwen3-VL 原始预测，用整理的 `parsed.decision==answer` 且归一化 label 匹配且 IoU≥0.5 逐条重算，与已发布 audit 的 per-label n/correct_rate 完全一致（差异 0）。这是 Base 池复算，不是独立 test，也不是三系统全 24 类的正式结果；保留具体计数。

| 集合 | n | correct | micro correct_rate | macro（per-label 均值） |
|---|---:|---:|---:|---:|
| 全部 24 类（observed） | 768 | 542 | 0.7057 | 0.6729 |
| retained 20 | 726 | 531 | 0.7314 | 0.7621 |
| excluded 4 | 42 | 11 | 0.2619 | 0.2271 |

excluded 4 明细与全 24 类 per-label 计数：

| label | n | correct | correct_rate |
|---|---:|---:|---:|
| apple_leaf | 38 | 35 | 0.9211 |
| apple_rust_leaf | 42 | 33 | 0.7857 |
| apple_scab_leaf | 43 | 40 | 0.9302 |
| bell_pepper_leaf | 10 | 5 | 0.5000 |
| bell_pepper_leaf_spot (excluded4) | 17 | 6 | 0.3529 |
| blueberry_leaf (excluded4) | 9 | 3 | 0.3333 |
| cherry_leaf | 4 | 4 | 1.0000 |
| corn_gray_leaf_spot | 52 | 31 | 0.5962 |
| corn_leaf_blight | 97 | 66 | 0.6804 |
| corn_rust_leaf | 75 | 65 | 0.8667 |
| grape_leaf | 18 | 18 | 1.0000 |
| peach_leaf | 9 | 8 | 0.8889 |
| potato_leaf_late_blight | 37 | 18 | 0.4865 |
| raspberry_leaf | 12 | 9 | 0.7500 |
| soybean_leaf | 8 | 6 | 0.7500 |
| squash_powdery_mildew_leaf | 64 | 53 | 0.8281 |
| strawberry_leaf | 4 | 4 | 1.0000 |
| tomato_early_blight_leaf | 42 | 31 | 0.7381 |
| tomato_leaf (excluded4) | 9 | 2 | 0.2222 |
| tomato_leaf_bacterial_spot | 45 | 28 | 0.6222 |
| tomato_leaf_late_blight | 53 | 23 | 0.4340 |
| tomato_leaf_mosaic_virus (excluded4) | 7 | 0 | 0.0000 |
| tomato_mold_leaf | 34 | 21 | 0.6176 |
| tomato_septoria_leaf_spot | 39 | 33 | 0.8462 |

证据：`aaai_code/benchmark/annotations/audits/PLANTDOC_CAPABILITY_GATE_AUDIT.json`、重算结果 `docs/review_evidence/additional_cpu/capability_24_vs_20.json`（含 `checked_against_published_audit`）。
