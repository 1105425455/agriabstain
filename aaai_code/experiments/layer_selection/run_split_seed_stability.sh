#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="/user/huangyongbin/newrepo/agriabstain"
OUTPUT_DIR="${REPO_ROOT}/new_reason/outputs/head_boundary_refit/layer_scan/split_seed_stability_v1"
TRAIN_NPZ="${REPO_ROOT}/new_reason/outputs/head_boundary_refit/features_alllayers/hidden_train.npz"

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
