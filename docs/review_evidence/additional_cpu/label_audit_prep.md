## 双人标签审计 — 抽样准备文件

**状态**：'sampling_prep_only'，未收集任何人工标注结果；`annotator_1`/`annotator_2` 均为空，待两名标注者填写。

**总体**：从 raw test split（`splits/test.jsonl`，共 2646 条）抽样，不用超采样 train（`sft_train.json`）以防重复。每状态 200 条，`seed=42`，按 (dataset, transform) 分层；每状态内对 (image_path, query) 去重，确保不复制图片、不伪造标签。

| state | population | selected | 优先级 |
|---|---:|---:|---|
| answer | 1323 | 200 | proportional |
| wrong_object | 205 | 200 | whole state = T6 |
| no_target | 1118 | 200 | COCO negatives 60% |

选中的行数为 600（应等于 600），每行含 `sample_id`、`image_path`、`query`、`state`、`transform`、population/权重分层与两个空 `annotator` 字段。`sample_weight` = 层 population / 层 selected（survey design weight）。

文件：`docs/review_evidence/additional_cpu/label_audit_prep.csv`（可直接评审）与 `.json`（含计划、逐行权重与分层）。
