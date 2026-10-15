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

All rates use the full ground-truth universe. An invalid supported output counts as a false
rejection and contributes 0 to Strict-3 and E2E-mIoU. The E2E-mIoU denominator is every
ground-truth supported record; `E2E_mIoU_c` is the conditional variant.

The `metrics.json` in each run directory is rebuilt from that run's predictions by
`experiments.base_native_regate.rescore_metrics`. Outputs of the separate
`analyze_native_regate` analyser use the `native_regate_diagnostic_*` prefix so they cannot
overwrite the canonical schema above.
