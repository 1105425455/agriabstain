#!/usr/bin/env bash
# run_all.sh — Orchestrated launcher for base_native_regate experiment.
#
# Stages:
#   check    — Verify environment variables and call launcher checks (default)
#   native   — run_native_eval.sh all (base + SFT inference, GPU)
#   regate   — run_regate_candidates.sh run (ReGate candidate dumping, GPU)
#   analysis — run_analysis.sh (validate 9 predictions + CPU metrics)
#   all      — native → regate → analysis (sequentially)
#
# Usage:
#   export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
#   export CHECKPOINT=/path/to/sft_final/checkpoint-444
#   export OUTPUT_ROOT=/scratch/base_native_regate
#   bash run_all.sh [check|native|regate|analysis|all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_SCRIPT="$SCRIPT_DIR/run_native_eval.sh"
REGATE_SCRIPT="$SCRIPT_DIR/run_regate_candidates.sh"
ANALYSIS_SCRIPT="$SCRIPT_DIR/run_analysis.sh"
ANALYZE_PY="$SCRIPT_DIR/analyze_native_regate.py"

STAGE="${1:-check}"

case "$STAGE" in
  check|native|regate|analysis|all) ;;
  *)
    echo "Usage: $0 {check|native|regate|analysis|all}"
    echo "  check    — Check environment variables and launcher setups (default)"
    echo "  native   — Run base + SFT native inference (GPU required)"
    echo "  regate   — Run ReGate candidate dumping (GPU required)"
    echo "  analysis — Run analysis on existing predictions (CPU)"
    echo "  all      — native → regate → analysis (sequentially)"
    exit 1
    ;;
esac

# ── Required environment variables for all stages ───────────────────────────
: "${BASE_MODEL:?Must set BASE_MODEL (path to Qwen3-VL-8B-Instruct)}"
: "${OUTPUT_ROOT:?Must set OUTPUT_ROOT to an external runtime output directory}"

# ── Stage: check ────────────────────────────────────────────────────────────
do_check() {
  echo "=== run_all.sh check ==="
  echo "  BASE_MODEL:   $BASE_MODEL"
  echo "  CHECKPOINT:   ${CHECKPOINT:-<not set>}"
  echo "  OUTPUT_ROOT:  $OUTPUT_ROOT"
  echo ""

  # Call individual launcher checks
  bash "$SCRIPT_DIR/run_native_eval.sh" check
  bash "$SCRIPT_DIR/run_regate_candidates.sh" check

  # Verify analysis script is importable
  if python -c "import sys; sys.path.insert(0, '$(dirname "$SCRIPT_DIR")/..'); from experiments.base_native_regate.analyze_native_regate import main; print('analyze_native_regate: import OK')" 2>&1; then
    echo "  analyze_native_regate: import OK"
  else
    echo "ERROR: analyze_native_regate.py import failed" >&2
    return 1
  fi

  echo "=== run_all.sh check passed ==="
}

# ── Stage: native ───────────────────────────────────────────────────────────
do_native() {
  echo "=== run_all.sh native ==="
  bash "$SCRIPT_DIR/run_native_eval.sh" all
  echo "=== run_all.sh native complete ==="
}

# ── Stage: regate ───────────────────────────────────────────────────────────
do_regate() {
  echo "=== run_all.sh regate ==="
  bash "$SCRIPT_DIR/run_regate_candidates.sh" run
  echo "=== run_all.sh regate complete ==="
}

# ── Stage: analysis ─────────────────────────────────────────────────────────
do_analysis() {
  echo "=== run_all.sh analysis ==="
  bash "$ANALYSIS_SCRIPT"
  echo "=== run_all.sh analysis complete ==="
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Execute ─────────────────────────────────────────────────────────────────
case "$STAGE" in
  check)
    do_check
    ;;
  native)
    do_native
    ;;
  regate)
    do_regate
    ;;
  analysis)
    do_analysis
    ;;
  all)
    do_native
    do_regate
    do_analysis
    ;;
esac
