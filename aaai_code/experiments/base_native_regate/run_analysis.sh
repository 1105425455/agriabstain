#!/usr/bin/env bash
# run_analysis.sh — Validate 9 predictions and run analyze_native_regate.py.
#
# Requires:
#   OUTPUT_ROOT   — External output root (e.g. /scratch/...)
#
# Checks that 9 prediction JSONL files exist:
#   native/base/{comprehensive,refcocog,plantwild}/
#   native/sft/{comprehensive,refcocog,plantwild}/
#   regate/{comprehensive,refcocog,plantwild}/
#
# Then calls analyze_native_regate.py with --iou-threshold 0.5.
# Output goes to ${OUTPUT_ROOT}/analysis/.
set -euo pipefail

# ── Locate package root (aaai_code/) and cd there for python -m imports ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAAI_CODE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$AAAI_CODE"

# ── Environment ─────────────────────────────────────────────────────────────
: "${OUTPUT_ROOT:?Must set OUTPUT_ROOT (external output directory)}"
OUTPUT_REAL="$(cd "$OUTPUT_ROOT" 2>/dev/null && pwd || echo "$OUTPUT_ROOT")"

DATASETS=(comprehensive refcocog plantwild)
METHODS=(base sft regate)

echo "=== run_analysis.sh ==="
echo "  AAAI_CODE:  $AAAI_CODE"
echo "  OUTPUT_ROOT: $OUTPUT_REAL"
echo ""

# ── Validate 9 predictions ──────────────────────────────────────────────────
ERRORS=0
declare -a BASE_ARGS SFT_ARGS REGATE_ARGS BASELINE_ARGS RUN_DIRS

for method in "${METHODS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    if [ "$method" = "regate" ]; then
      pred_dir="${OUTPUT_REAL}/regate/${ds}"
    else
      pred_dir="${OUTPUT_REAL}/native/${method}/${ds}"
    fi

    shopt -s nullglob
    jsonl_files=("$pred_dir"/*.jsonl)
    shopt -u nullglob

    if [ ${#jsonl_files[@]} -eq 0 ]; then
      echo "ERROR: Missing prediction for ${method}/${ds}" >&2
      echo "       (no *.jsonl found in ${pred_dir})" >&2
      ERRORS=1
      continue
    fi
    if [ ${#jsonl_files[@]} -ne 1 ]; then
      echo "ERROR: Expected exactly one prediction JSONL for ${method}/${ds}, found ${#jsonl_files[@]}" >&2
      ERRORS=1
      continue
    fi

    case "$method" in
      base)
        BASE_ARGS+=(--base "$ds=${jsonl_files[0]}")
        BASELINE_ARGS+=(--baseline "Base_native:$ds=${jsonl_files[0]}")
        ;;
      sft)
        SFT_ARGS+=(--sft "$ds=${jsonl_files[0]}")
        BASELINE_ARGS+=(--baseline "SFT_native_observed:$ds=${jsonl_files[0]}")
        ;;
      regate)  REGATE_ARGS+=(--regate "$ds=${jsonl_files[0]}") ;;
    esac
    RUN_DIRS+=("$pred_dir")
  done
done

if [ "$ERRORS" -ne 0 ]; then
  echo ""
  echo "Expected 9 prediction outputs at:" >&2
  echo "  ${OUTPUT_REAL}/native/base/{comprehensive,refcocog,plantwild}/*.jsonl" >&2
  echo "  ${OUTPUT_REAL}/native/sft/{comprehensive,refcocog,plantwild}/*.jsonl" >&2
  echo "  ${OUTPUT_REAL}/regate/{comprehensive,refcocog,plantwild}/*.jsonl" >&2
  exit 1
fi

echo "All 9 predictions found."

# ── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_REAL}/analysis"
mkdir -p "$OUTPUT_DIR"
echo "  Analysis output: $OUTPUT_DIR"
echo ""

# ── Re-score saved predictions with the current fixed-GT implementation ─────
echo "=== Re-scoring run metrics from predictions ==="
python -m experiments.base_native_regate.rescore_metrics "${RUN_DIRS[@]}"

# ── Canonical paper metrics ─────────────────────────────────────────────────
echo "=== Running canonical selective-grounding analysis ==="
python -m experiments.selective_grounding.analyze_selective_grounding \
  "${REGATE_ARGS[@]}" \
  "${BASELINE_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --iou-threshold 0.5 \
  --no-plots

# ── Shared-candidate diagnostics and publication plots ──────────────────────
echo "=== Running native/ReGate diagnostics ==="
python -m experiments.base_native_regate.analyze_native_regate \
  "${BASE_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${REGATE_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --iou-threshold 0.5

echo ""
echo "=== run_analysis.sh complete ==="
echo "  Canonical summary: $OUTPUT_DIR/summary.json"
echo "  Canonical CSV:     $OUTPUT_DIR/native_comparison.csv"
echo "                     $OUTPUT_DIR/baseline_points.csv"
echo "                     $OUTPUT_DIR/regate_argmax_points.csv"
echo "                     $OUTPUT_DIR/matched_retention.csv"
echo "  Diagnostics:       $OUTPUT_DIR/native_regate_diagnostic_*"
echo "  Plots:    $OUTPUT_DIR/*.pdf / *.png"
