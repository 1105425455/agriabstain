# Margin+KL selected operating points

Frozen evaluation artifacts for the two post-SFT margin+KL continuations reported in the
technical supplement. They are the original run outputs, with absolute filesystem paths
replaced by `<REPO_ROOT>` / `<MODEL_ROOT>` placeholders.

## Layout

| Path | Content |
|------|---------|
| `v1_checkpoint-50/trainer_state.json` | Optimiser/loss trace of the Version 1 continuation (margin 0.5, 200 steps, evaluated at checkpoint 50) |
| `v1_checkpoint-50/plantwild_real_closed12_v1/metrics.json` | Fixed-GT metrics on the 12-class PlantWild probe (408 records) |
| `v1_checkpoint-50/plantwild_real_closed12_v1/predictions.jsonl` | Per-record parsed predictions |
| `v2_checkpoint-60/...` | Same three files for Version 2 (margin 0.25, answer-emphasised state pattern, 80 steps, evaluated at checkpoint 60) |

## Reported values

The supplement's margin+KL table reads the following fields:

| Column | JSON path |
|--------|-----------|
| FCA | `metrics.headline.FCA` |
| A_FU | `metrics.safeguard.A_FU` |
| MCC | `metrics.safeguard.MCC` |
| Strict-3 | `mts.MTS_micro` |
| Supported | `mts.per_relation.OOD_answer` |
| Contradiction | `mts.per_relation.OOD_confusion` |
| Unavailable | `mts.per_relation.OOD_absent` |

Resulting values, rounded to three decimals:

| Method | FCA | A_FU | MCC | Strict-3 | Supported | Contradiction | Unavailable |
|--------|-----|------|-----|----------|-----------|---------------|-------------|
| Margin+KL v1 | 0.155 | 0.257 | 0.584 | 0.740 | 0.708 | 0.500 | 0.972 |
| Margin+KL v2 | 0.174 | 0.222 | 0.592 | 0.733 | 0.722 | 0.458 | 0.972 |

The two comparison rows in the same table (SFT native and ReGate argmax) come from the
Base/SFT/ReGate evaluation, not from this directory.

## Reproduction

Training and evaluation entry points are `training/margin_kl/run_v1.sh`,
`training/margin_kl/run_v2.sh`, and `training/margin_kl/eval_closed12.sh`. Both
continuations initialise from the Stage-1 adapter and its calibrated head; see
`REVIEWER_README.md` §6 for the required environment variables.
