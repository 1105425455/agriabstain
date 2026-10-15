#!/usr/bin/env bash
# run_representation.sh - Experiment A launcher: representation extraction and probe training
#
# Usage:
#   # By default only the arguments are validated; no GPU work is started.
#   export BASE=/path/to/Qwen3-VL-8B-Instruct
#   export OUTPUT_ROOT=/scratch/evidence_judge
#   bash run_representation.sh          # validate only
#
#   # Full run (extract + probe); DERIVED_ROOT is optional
#   export BASE=/path/to/Qwen3-VL-8B-Instruct
#   export OUTPUT_ROOT=/scratch/evidence_judge
#   export DERIVED_ROOT=/path/to/derived_images  # defaults to benchmark/data/derived
#   bash run_representation.sh all
#
#   # extraction only
#   bash run_representation.sh extract
#
#   # probe training only, which requires extraction to have finished
#   bash run_representation.sh probe
#
# Stage: check|extract|probe|all. The default is check, so no GPU work starts by accident.

set -euo pipefail

# --- Paths, derived from this script rather than from the caller's cwd ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# experiments/evidence_judge/ -> aaai_code/
AAAI_CODE="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK_DIR="$AAAI_CODE/benchmark"
SFT_DIR="$BENCHMARK_DIR/annotations/sft"

STAGE="${1:-check}"

case "$STAGE" in
  check|extract|probe|all) ;;
  *)
    echo "Usage: $0 {check|extract|probe|all}"
    echo "  check    - validate the environment and arguments only, starting nothing (default)"
    echo "  extract  - extract representations for the three splits only"
    echo "  probe    - train the probes only, which requires extraction to have finished"
    echo "  all      - full pipeline, extract + probe"
    exit 1
    ;;
esac

# --- Required environment variables ---
REQUIRED_VARS=(
  "BASE:path to the base model (Qwen3-VL-8B-Instruct)"
  "OUTPUT_ROOT:output root, which must be outside this package"
)

MISSING=0
for var_spec in "${REQUIRED_VARS[@]}"; do
  var_name="${var_spec%%:*}"
  var_desc="${var_spec#*:}"
  if [ -z "${!var_name:-}" ]; then
    echo "ERROR: missing environment variable \$${var_name} - ${var_desc}"
    MISSING=1
  fi
done

if [ "$MISSING" -eq 1 ]; then
  echo ""
  echo "Set the required environment variables and retry, for example:"
  echo "  export BASE=/path/to/Qwen3-VL-8B-Instruct"
  echo "  export OUTPUT_ROOT=/scratch/evidence_judge"
  exit 1
fi

# Optional environment variables
NPROC="${NPROC:-4}"
BATCH_SIZE="${BATCH_SIZE:-3}"

# BATCH_SIZE must be a positive integer
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BATCH_SIZE must be a positive integer, got: '$BATCH_SIZE'"
  exit 1
fi

# DERIVED_ROOT defaults to the derived directory inside this package
if [ -z "${DERIVED_ROOT:-}" ]; then
  DERIVED_ROOT="$AAAI_CODE/benchmark/data/derived"
fi

# ADAPTER is held in a bash array so that paths with spaces survive; no string splitting
ADAPTER_ARGS=()
if [ -n "${ADAPTER:-}" ]; then
  ADAPTER_ARGS=(--adapter "$ADAPTER")
fi

echo "=== run_representation.sh ==="
echo "  STAGE:        $STAGE"
echo "  AAAI_CODE:    $AAAI_CODE"
echo "  BASE:         $BASE"
echo "  DERIVED_ROOT: $DERIVED_ROOT"
echo "  OUTPUT_ROOT:  $OUTPUT_ROOT"
echo "  NPROC:        $NPROC"
echo "  BATCH_SIZE:   $BATCH_SIZE"
if [ "${#ADAPTER_ARGS[@]}" -gt 0 ]; then
  echo "  ADAPTER:      ${ADAPTER_ARGS[1]} (diagnostic mode)"
else
  echo "  ADAPTER:      (unset - frozen base model)"
fi
echo ""

# --- Check that the paths exist ---
ERRORS=0

# Base model
if [ ! -d "$BASE" ] && [ ! -f "$BASE" ]; then
  echo "ERROR: BASE does not exist: $BASE"
  ERRORS=1
fi

# Adapter, when one is given
if [ "${#ADAPTER_ARGS[@]}" -gt 0 ]; then
  if [ ! -d "${ADAPTER_ARGS[1]}" ]; then
    echo "ERROR: ADAPTER does not exist: ${ADAPTER_ARGS[1]}"
    ERRORS=1
  fi
fi

# OUTPUT_ROOT must not sit inside this package
OUTPUT_ROOT_REAL="$(cd "$OUTPUT_ROOT" 2>/dev/null && pwd || echo "$OUTPUT_ROOT")"
AAAI_CODE_REAL="$(cd "$AAAI_CODE" && pwd)"
if [[ "$OUTPUT_ROOT_REAL" == "$AAAI_CODE_REAL"/* ]] || [ "$OUTPUT_ROOT_REAL" = "$AAAI_CODE_REAL" ]; then
  echo "ERROR: OUTPUT_ROOT ($OUTPUT_ROOT) is inside this package ($AAAI_CODE_REAL)"
  echo "  point OUTPUT_ROOT at an external directory, for example /scratch/..., to keep artifacts out of the package"
  ERRORS=1
fi

# Data files
for f in "sft_train.json" "sft_dev.json" "sft_test.jsonl"; do
  if [ ! -f "$SFT_DIR/$f" ]; then
    echo "ERROR: data file not found: $SFT_DIR/$f"
    ERRORS=1
  fi
done

# Derived-image directory
if [ ! -d "$DERIVED_ROOT" ]; then
  echo "WARNING: DERIVED_ROOT does not exist: $DERIVED_ROOT"
  echo "  references to derived images will fail."
  echo "  rebuild them as described in REVIEWER_README.md, section 5,"
  echo "  or point DERIVED_ROOT at a directory where they have been rebuilt."
fi

if [ "$ERRORS" -ne 0 ]; then
  echo ""
  echo "Fix the errors above and retry"
  exit 1
fi

# In check mode, validate and exit
if [ "$STAGE" = "check" ]; then
  echo "Arguments validated. To run, pass a stage: all / extract / probe"
  exit 0
fi

# --- Output paths ---
REPR_DIR="$OUTPUT_ROOT/repr"

# --- 1. Extract representations ---
if [ "$STAGE" = "extract" ] || [ "$STAGE" = "all" ]; then
  echo ">>> stage 1/2: extracting representations for the three splits"

  for split_name in train dev test; do
    case "$split_name" in
      train) data_file="$SFT_DIR/sft_train.json" ;;
      dev)   data_file="$SFT_DIR/sft_dev.json" ;;
      test)  data_file="$SFT_DIR/sft_test.jsonl" ;;
    esac

    output_dir="$REPR_DIR/$split_name"
    echo "  [extract] split=$split_name -> $output_dir"

    torchrun --nproc_per_node="$NPROC" "$SCRIPT_DIR/extract_representations.py" \
      --base "$BASE" \
      "${ADAPTER_ARGS[@]}" \
      --derived-root "$DERIVED_ROOT" \
      --data "$data_file" \
      --output-dir "$output_dir" \
      --split-name "$split_name" \
      --batch-size "$BATCH_SIZE"

    echo "  [extract] split=$split_name done"
  done

  echo ">>> extraction complete"
fi

# --- 2. Train the probes ---
if [ "$STAGE" = "probe" ] || [ "$STAGE" = "all" ]; then
  echo ">>> stage 2/2: training the probes and comparing the four representations"

  python "$SCRIPT_DIR/train_probes.py" \
    --train "$REPR_DIR/train/train.npz" \
    --dev "$REPR_DIR/dev/dev.npz" \
    --test "$REPR_DIR/test/test.npz" \
    --output "$REPR_DIR/probe_results.json"

  echo ">>> probe training complete"
  echo "    results: $REPR_DIR/probe_results.json"
fi

echo "=== run_representation.sh complete ==="