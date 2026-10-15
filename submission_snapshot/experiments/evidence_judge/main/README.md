# Evidence-sensitivity analyses (L2 / L3, test split)

Frozen outputs of the gate-only evidence-sensitivity analyses reported in the technical
supplement. They are the original analysis outputs, with absolute filesystem paths replaced
by `<REPO_ROOT>` / `<MODEL_ROOT>` placeholders.

Both analyses score the probabilities of the final refitted gate directly.

## Layout

| Path | Content |
|------|---------|
| `L2_test/analysis.json` | Paired target-removal statistics on the balanced L2 fixed-point probe |
| `L2_test/manifest.json` | Input records, checkpoint, and calibrated-head provenance for L2 |
| `L3_test/analysis.json` | Per-group visibility/`p_A` correlations plus per-group detail for L3 |
| `L3_test/manifest.json` | Input records, checkpoint, and calibrated-head provenance for L3 |

The per-record `head_probabilities.jsonl` inputs of these analyses are shipped under
`repro_artifacts/evidence_judge/L2/` and `repro_artifacts/evidence_judge/L3/`.

## Reported values

L2 (`L2_test/analysis.json`, key `L2`):

| Quantity | JSON field | Value |
|----------|-----------|-------|
| Mean drop in `p_A` after target removal | `delta_p_answer_mean` | 0.786 |
| 95% CI (BCa) | `bootstrap_95ci` | [0.720, 0.840] |
| Share of pairs with the expected direction | `correct_direction_ratio` | 96.6% |
| Complete pairs | `n_pairs` | 148 |
| Groups excluded as incomplete after balancing | `incomplete_pairs` | 249 |

L3 (`L3_test/analysis.json`, key `L3`):

| Quantity | JSON field | Value |
|----------|-----------|-------|
| Test groups | `n_groups` | 183 |
| Mean Spearman rho | `mean_spearman_rho` | 0.750 (CI [0.694, 0.794]) |
| Mean Kendall tau | `mean_kendall_tau` | 0.682 (CI [0.625, 0.729]) |
| Adjacent-level violation rate | `mean_adjacent_violation_rate` | 21.4% (CI [18.3%, 24.9%]) |

All intervals are BCa bootstrap intervals over source groups with `bootstrap_seed = 42`.
The L3 analysis uses every available visibility level within a group and does not impute
missing levels.

## Reproduction

`experiments/evidence_judge/extract_head_probabilities.py` dumps the gate probabilities and
`experiments/evidence_judge/analyze_evidence.py` produces these JSON files;
`experiments/evidence_judge/run_evidence.sh` runs both.
