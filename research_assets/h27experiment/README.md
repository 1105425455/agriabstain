# H27 multi-seed experiments

This directory contains isolated follow-up experiments for AgriAbstain. The
released `aaai_code/` tree is a read-only dependency and is not modified.

## Frozen protocol

- Conditions: `grouped` and `independent`.
- Full-pipeline seeds: `7`, `21`, and `42`.
- The grouped and independent annotation files are fixed across seeds.
- Runnable annotation copies retain the original absolute media paths. This is
  required because the anonymous package intentionally omits
  `benchmark/data/derived/`; non-image training fields are identical to the
  anonymous annotations.
- The independent source assignment is the released seed-42 assignment; it is
  **not** regenerated for different training seeds.
- Stage-1 LoRA/training-head initialization and Stage-2 MLP initialization use
  the run seed.
- The class-stratified Stage-2 development sample is fixed with sampling seed
  `42`, so development composition does not vary across runs.
- Each condition uses the same base model, fixed base-representation
  normalization, six-epoch recipe, checkpoint candidates, and fixed first-100
  SFT development records.
- Checkpoint selection minimizes teacher-forced LM loss on that fixed
  development subset. Test outputs never select a checkpoint, gate epoch, or
  LogisticRegression regularization strength.
- Every scheduled run is reported; no run is selected using test performance.

This protocol was frozen after the original test set had already been analyzed.
The follow-up therefore estimates run-level robustness under a fixed
retrospective protocol rather than providing a new untouched-test confirmation.

## Linear-gate control

For every adapted run, a multinomial LogisticRegression control is fitted to the
same extracted H27 training representations used by the nonlinear Stage-2 MLP.
The regularization grid is `C in {0.01, 0.1, 1.0, 10.0}`. Selection uses minimum
three-class cross-entropy on the same fixed 100-record development subset, with
smaller `C` breaking exact ties.

The evaluation pass stores one shared grounding candidate and H27 representation
per record. MLP and linear routing are then compared offline with the same
candidate, labels, parser, and denominators; only the routing probabilities
change.

## Layout

```text
h27experiment/
  code/                 local wrappers and analysis
  data/                 frozen annotation copies and manifests
  runs/<condition>/seed-<seed>/
  summary/
```

Run `python code/prepare_data.py` before launching. A full run is:

```bash
bash code/run_condition.sh grouped 7 all
bash code/run_condition.sh independent 7 all
```

The script also accepts the stages `train`, `select`, `calibrate`, `linear`,
`eval`, and `reroute`.
