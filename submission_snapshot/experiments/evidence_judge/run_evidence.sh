#!/usr/bin/env bash
# run_evidence.sh - Experiment B launcher: head-probability extraction and analysis
#
# Usage:
#   # By default only the arguments are validated; no GPU work is started.
#   export BASE=/path/to/Qwen3-VL-8B-Instruct
#   export ADAPTER=/path/to/checkpoint-444
#   export CALIBRATED_HEAD=/path/to/checkpoint-444/discrim_head_calibrated.pt
#   export OUTPUT_ROOT=/scratch/evidence_judge
#   bash run_evidence.sh             # validate only
#
#   # Full run (l2 + l3 + consistency + analysis); DERIVED_ROOT is optional
#   export BASE=/path/to/Qwen3-VL-8B-Instruct
#   export ADAPTER=/path/to/checkpoint-444
#   export CALIBRATED_HEAD=/path/to/checkpoint-444/discrim_head_calibrated.pt
#   export OUTPUT_ROOT=/scratch/evidence_judge
#   export DERIVED_ROOT=/path/to/derived_images  # defaults to benchmark/data/derived
#   bash run_evidence.sh all
#
#   # L2 only
#   bash run_evidence.sh l2
#
# Stage: check|l2|l3|consistency|all. The default is check, so no GPU work starts by accident.

set -euo pipefail

# --- Paths, derived from this script rather than from the caller's cwd ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# experiments/evidence_judge/ -> aaai_code/
AAAI_CODE="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCHMARK_DIR="$AAAI_CODE/benchmark"
EVAL_DIR="$BENCHMARK_DIR/annotations/eval"

STAGE="${1:-check}"

case "$STAGE" in
  check|l2|l3|consistency|all) ;;
  *)
    echo "Usage: $0 {check|l2|l3|consistency|all}"
    echo "  check        - validate the environment and arguments only (default)"
    echo "  l2           - L2 evidence probe (keep versus cropout)"
    echo "  l3           - L3 dose probe (dose-response)"
    echo "  consistency  — Input equivariance consistency (variant_set=all)"
    echo "  all          - full pipeline (l2 + l3 + consistency + analysis)"
    exit 1
    ;;
esac

# --- Required environment variables ---
REQUIRED_VARS=(
  "BASE:path to the base model (Qwen3-VL-8B-Instruct)"
  "ADAPTER:path to the LoRA adapter (checkpoint-444)"
  "CALIBRATED_HEAD:path to the calibrated head .pt"
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
  echo "  export ADAPTER=/path/to/checkpoint-444"
  echo "  export CALIBRATED_HEAD=/path/to/checkpoint-444/discrim_head_calibrated.pt"
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

# DERIVED_ROOT
if [ -z "${DERIVED_ROOT:-}" ]; then
  DERIVED_ROOT="$AAAI_CODE/benchmark/data/derived"
fi

echo "=== run_evidence.sh ==="
echo "  STAGE:           $STAGE"
echo "  AAAI_CODE:       $AAAI_CODE"
echo "  BASE:            $BASE"
echo "  ADAPTER:         $ADAPTER"
echo "  CALIBRATED_HEAD: $CALIBRATED_HEAD"
echo "  DERIVED_ROOT:    $DERIVED_ROOT"
echo "  OUTPUT_ROOT:     $OUTPUT_ROOT"
echo "  NPROC:           $NPROC"
echo "  BATCH_SIZE:      $BATCH_SIZE"
echo ""

# --- Check that the paths exist ---
ERRORS=0

# Base model
if [ ! -d "$BASE" ] && [ ! -f "$BASE" ]; then
  echo "ERROR: BASE does not exist: $BASE"
  ERRORS=1
fi

# Adapter
if [ ! -d "$ADAPTER" ]; then
  echo "ERROR: ADAPTER does not exist: $ADAPTER"
  ERRORS=1
fi

# Calibrated head
if [ ! -f "$CALIBRATED_HEAD" ]; then
  echo "ERROR: CALIBRATED_HEAD does not exist: $CALIBRATED_HEAD"
  ERRORS=1
fi

# OUTPUT_ROOT must not sit inside this package
OUTPUT_ROOT_REAL="$(cd "$OUTPUT_ROOT" 2>/dev/null && pwd || echo "$OUTPUT_ROOT")"
AAAI_CODE_REAL="$(cd "$AAAI_CODE" && pwd)"
if [[ "$OUTPUT_ROOT_REAL" == "$AAAI_CODE_REAL"/* ]] || [ "$OUTPUT_ROOT_REAL" = "$AAAI_CODE_REAL" ]; then
  echo "ERROR: OUTPUT_ROOT ($OUTPUT_ROOT) is inside this package ($AAAI_CODE_REAL)"
  echo "  point OUTPUT_ROOT at an external directory, for example /scratch/..., to keep artifacts out of the package"
  ERRORS=1
fi

# Evaluation data files
for f in "dev_L2_evidence.jsonl" "test_L2_evidence.jsonl" "dev_L3_dose.jsonl" "test_L3_dose.jsonl"; do
  if [ ! -f "$EVAL_DIR/$f" ]; then
    echo "ERROR: data file not found: $EVAL_DIR/$f"
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
  echo "Arguments validated. To run, pass a stage: all / l2 / l3 / consistency"
  exit 0
fi

# --- Output paths ---
EVIDENCE_ROOT="$OUTPUT_ROOT/evidence"

# --- Helpers ---
run_extract() {
  local split="$1"    # dev|test
  local level="$2"    # L2|L3
  local suffix="$3"   # _variants, or empty
  local variant_set="$4"  # none|all

  local data_file="${EVAL_DIR}/${split}_${level}_evidence.jsonl"
  if [ "$level" = "L3" ]; then
    data_file="${EVAL_DIR}/${split}_${level}_dose.jsonl"
  fi

  local out_dir="${EVIDENCE_ROOT}/${level}/${split}${suffix}"

  echo "  [extract] ${split} ${level}${suffix} (variant_set=${variant_set})"
  torchrun --nproc_per_node="$NPROC" "$SCRIPT_DIR/extract_head_probabilities.py" \
    --base "$BASE" \
    --adapter "$ADAPTER" \
    --calibrated-head "$CALIBRATED_HEAD" \
    --derived-root "$DERIVED_ROOT" \
    --data "$data_file" \
    --output "$out_dir" \
    --variant-set "$variant_set" \
    --batch-size "$BATCH_SIZE"
  echo "  [extract] done -> $out_dir"
}

run_analysis() {
  local level="$1"    # L2|L3
  local split="$2"    # dev|test
  local suffix="$3"   # optional suffix

  local in_dir="${EVIDENCE_ROOT}/${level}/${split}${suffix}"
  local input_file="${in_dir}/head_probabilities.jsonl"
  local output_file="${in_dir}/analysis.json"

  echo "  [analysis] ${level} ${split}${suffix}"
  python "$SCRIPT_DIR/analyze_evidence.py" \
    --input "$input_file" \
    --output "$output_file"
  echo "  [analysis] done -> $output_file"
}

# ── L2 ───────────────────────────────────────────────────────────────────────
run_l2() {
  echo ">>> L2 evidence probe (keep versus cropout)"

  # extract dev and test with variant_set=none
  run_extract "dev" "L2" "" "none"
  run_extract "test" "L2" "" "none"

  # analysis
  run_analysis "L2" "dev" ""
  run_analysis "L2" "test" ""
}

# ── L3 ───────────────────────────────────────────────────────────────────────
run_l3() {
  echo ">>> L3 dose probe (dose-response)"

  run_extract "dev" "L3" "" "none"
  run_extract "test" "L3" "" "none"

  run_analysis "L3" "dev" ""
  run_analysis "L3" "test" ""
}

# ── Consistency ──────────────────────────────────────────────────────────────
run_consistency() {
  echo ">>> Input Equivariance Consistency"

  # L2 + L3 with variant_set=all
  run_extract "dev" "L2" "_variants" "all"
  run_extract "test" "L2" "_variants" "all"
  run_extract "dev" "L3" "_variants" "all"
  run_extract "test" "L3" "_variants" "all"

  # analysis
  run_analysis "L2" "dev" "_variants"
  run_analysis "L2" "test" "_variants"
  run_analysis "L3" "dev" "_variants"
  run_analysis "L3" "test" "_variants"
}

# --- Dispatch ---
case "$STAGE" in
  l2)
    run_l2
    ;;
  l3)
    run_l3
    ;;
  consistency)
    run_consistency
    ;;
  all)
    run_l2
    run_l3
    run_consistency
    ;;
esac

echo "=== run_evidence.sh complete ==="