# Grouping Ablation: Canonical Results

This directory contains the compact, path-free artifact used by the paper. The
author-side runtime artifacts, checkpoints, features, logs, and predictions stay
under `../artifacts/` and are excluded from the anonymous release.

`grouping_ablation_fixed_gt.json` records:

- data-control audits and source hashes;
- the full-retrain and calibrated-head protocol;
- validation losses for all six epoch checkpoints in both conditions under the same
  fixed-100 selection rule, including the grouped epoch-1 recovery provenance;
- Native and ReGate metrics for all reported probes;
- both `MTS_macro` and `MTS_micro`, with `MTS_macro` designated as the paper metric;
- fixed-GT invalid-output and reason-attribution policies.

Rebuild it from the author-side artifacts with:

```bash
python -m experiments.grouping_ablation.build_release_artifact \
  --artifact-root experiments/grouping_ablation/artifacts \
  --output experiments/grouping_ablation/main/grouping_ablation_fixed_gt.json
```
