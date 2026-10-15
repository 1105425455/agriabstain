# Layer selection

## Pre-adaptation candidate comparison

`main/pre_adaptation_candidate_selection.json` is the compact, path-free artifact for the
coarse depth comparison run on the frozen base model over API indices 9, 18, and 27, before
joint adaptation. It uses 1,736 training-side records split source-group-disjoint into 1,599
probe-fitting and 137 probe-development records, and selects the candidate maximizing minimum
per-state development accuracy. It is a different study from the frozen-base *readout*
comparison of the technical supplement, which contrasts four pooling positions on the
benchmark splits and is backed by `experiments/evidence_judge/probe_results.json`.

Rebuild it from the extraction manifest and development scan with:

```bash
python -m experiments.layer_selection.build_candidate_artifact \
  --manifest /path/to/frozen_base/manifest_extract_alllayers.json \
  --dev-scan /path/to/frozen_base/layer_scan/dev_scan.json \
  --min-score-curve /path/to/frozen_base/layer_scan/min_score_curve.json \
  --output experiments/layer_selection/main/pre_adaptation_candidate_selection.json
```

The `.npz` feature archives and the raw scan outputs of this comparison are not distributed.

## Post-adaptation 36-layer diagnostic

Utilities for the depth diagnostic of the technical supplement. The pipeline has two stages:

1. **`extract_all_layers.py`** — Extract generation-prompt last-token hidden
   states from all 36 transformer layers (`hidden_states[1:]`, excluding
   embedding) for the training and holdout splits. Output is a pair of NPZ
   files (`hidden_train.npz`, `hidden_holdout.npz`) plus a manifest JSON.

2. **`scan_layers.py`** — Load the NPZ files, perform a seed-42 group-disjoint
   internal train/dev split (by `source_image_id`), then for each layer fit a
   `LogisticRegression(C∈{0.01,0.1,1.0})` with per-layer normalisation. The
   score is `min(answer_accuracy, wrong_object_accuracy, no_target_accuracy)`.
   Outputs `dev_scan.json`, `best_layer.json`, and `min_score_curve.json`.

## Usage

```bash
# Stage 1: extract hidden states
python -m experiments.layer_selection.extract_all_layers \
    --base /path/to/base_model \
    --adapter /path/to/checkpoint-444 \
    --sft-train /path/to/sft_train.json \
    --selection-records /path/to/selection_records.jsonl \
    --split-assignment /path/to/split_assignment.json \
    --output-dir /path/to/output

# Stage 2: scan layers
python -m experiments.layer_selection.scan_layers \
    --train-npz /path/to/output/hidden_train.npz \
    --holdout-npz /path/to/output/hidden_holdout.npz \
    --out-dir /path/to/output/scan
```

## Mapping note

- NPZ array index `L` corresponds to `hidden_states[L+1]` = transformer layer
  `L+1`.
- The gate reads transformer layer 27 = NPZ index 26.
