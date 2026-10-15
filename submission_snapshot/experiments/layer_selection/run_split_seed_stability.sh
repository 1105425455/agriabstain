#!/usr/bin/env bash
set -eo pipefail

: "${TRAIN_NPZ:?Set TRAIN_NPZ to the all-layer training feature archive}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to an external runtime output directory}"

mkdir -p "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m experiments.layer_selection.scan_layer_stability \
  --train-npz "${TRAIN_NPZ}" \
  --output "${OUTPUT_DIR}/layer_split_seed_stability_7_21_42_87_123.json" \
  --seeds "7,21,42,87,123" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
