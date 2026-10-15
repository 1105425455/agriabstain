# Grouping ablation: canonical results

This directory contains the compact, path-free artifact the paper reads. The checkpoints,
extracted features and raw predictions of the two ablation runs are not distributed. Run
logs for the independent-negative condition are under `repro_artifacts/logs/grouping/`; the
grouped condition is the main SFT run, whose log is `repro_artifacts/logs/sft_final/train.log`.

`grouping_ablation_fixed_gt.json` records:

- data-control audits;
- the full-retrain and calibrated-head protocol;
- validation losses for all six epoch checkpoints in both conditions under the same
  fixed-100 selection rule, including the grouped epoch-1 recovery provenance;
- Native and ReGate metrics for all reported probes;
- both `MTS_macro` and `MTS_micro`, with `MTS_macro` designated as the paper metric;
- fixed-GT invalid-output and reason-attribution policies.

Rebuild it from those run outputs with:

```bash
python -m experiments.grouping_ablation.build_release_artifact \
  --artifact-root /path/to/grouping_ablation/artifacts \
  --output experiments/grouping_ablation/main/grouping_ablation_fixed_gt.json
```
