# Post-adaptation depth diagnostic (36-layer scan)

Frozen outputs of the post-adaptation layer scan reported in the technical supplement.
Files are byte-identical copies of the original scan outputs; only absolute filesystem
paths were replaced with `<REPO_ROOT>` / `<MODEL_ROOT>` placeholders. No probe was
refitted to produce this directory.

This scan is run **after** joint adaptation, with the adapted backbone and grounding
generator frozen. It is distinct from the pre-adaptation coarse probe recorded in
`../pre_adaptation_candidate_selection.json`, which compares only API indices 9, 18, and 27
on the frozen base model.

## Layout

| Path | Content |
|------|---------|
| `dev_scan.json` | All 36 depths: per-state accuracies, confusion matrices, and the worst-state score |
| `best_layer.json` | Selected depth and its full scan entry |
| `min_score_curve.json` | Worst-state score as a function of depth |
| `manifest_extract_alllayers.json` | Hidden-state extraction provenance (records, checkpoint, split) |

## Index convention

Each entry carries three indices. `transformer_layer` is the API index used throughout the
paper: API index `L` is `hidden_states[L]`, so API index 27 is the output of transformer
block 26 under zero-based block numbering. `npz_index` is the position inside the extracted
feature archive and equals `transformer_layer - 1`.

When cross-checking the supplement's depth table, match on `transformer_layer`, not on
`layer` or `npz_index`.

## Reported values

Selection criterion is the worst per-state validation accuracy (`min_score`). Probes are
logistic classifiers with `C` in {0.01, 0.1, 1.0}, fitted on 1,599 group-disjoint records
and evaluated on 137 probe-development records.

| API index | `min_score` |
|-----------|-------------|
| 1 | 0.6383 |
| 14 | 0.7846 |
| 18 | 0.9231 |
| 22 | 0.9362 |
| 26 | 0.9362 |
| **27** | **0.9574** |
| 28 | 0.9362 |
| 36 | 0.9149 |

API index 27 is the global maximum across all 36 depths, which independently confirms the
readout position fixed before joint adaptation.

## Reproduction

`experiments/layer_selection/extract_all_layers.py` dumps per-layer hidden states and
`experiments/layer_selection/scan_layers.py` fits the probes; see
`experiments/layer_selection/README.md`. The large `.npz` feature archives are not
distributed.
