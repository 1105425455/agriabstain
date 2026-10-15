# Canonical metrics

The canonical tables in this directory are recomputed by
`experiments.selective_grounding.analyze_selective_grounding` from the saved
`predictions.jsonl` files.

| File | Contents |
|------|----------|
| `native_comparison.csv` | Unified fixed-GT metrics for Base, SFT native and ReGate argmax |
| `baseline_points.csv` | Detailed metrics for Base and SFT native |
| `regate_argmax_points.csv` | Detailed metrics for ReGate argmax |
| `matched_retention.csv` | Empirical test-curve matched-retention diagnostics |
| `summary.json` | Machine-readable summary from the same run as the CSVs |
| `{comprehensive,plantwild,refcocog}_regate_curve.csv` | Full FCA–retention operating curves |
| `*.pdf` / `*.png` | Figures drawn from those curves |

The three FCA–retention figures of the technical supplement are
`comprehensive_hallucination_retention_paper`, `plantwild_hallucination_retention` and
`refcocog_hallucination_retention`. The `_paper` variant zooms into the matched-retention
operating region and carries an inset with the complete sweep; plain
`comprehensive_hallucination_retention` is that complete sweep on its own. The risk–coverage
and Strict-3–coverage figures are additional diagnostics that the paper does not use.

All rates use the full ground-truth universe. An invalid supported output counts as a false
rejection and contributes 0 to Strict-3 and E2E-mIoU. The E2E-mIoU denominator is every
ground-truth supported record; `E2E_mIoU_c` is the conditional variant.

Running `experiments.base_native_regate.rescore_metrics` on any
`repro_artifacts/base_native_regate/predictions/**/predictions.jsonl` regenerates that run's
`metrics.json`. Outputs of the separate `analyze_native_regate` analyser carry the
`native_regate_diagnostic_*` prefix and do not overwrite the canonical files above.
