#!/usr/bin/env bash
# run_native_eval.sh — Native base / SFT evaluation on the three fixed datasets.
#
# Launches the formal eval_qwen3vl_3reason.py script with **no adapter** (base)
# or with **only --adapter CHECKPOINT** (SFT).  No dual-head / calibrated /
# faithful flags are passed.
#
# Usage
# -----
#   # check only (default)
#   export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
#   export CHECKPOINT=/path/to/checkpoint-444
#   export OUTPUT_ROOT=/scratch/native_eval
#   bash run_native_eval.sh
#
#   # run base
#   bash run_native_eval.sh base
#
#   # run SFT
#   bash run_native_eval.sh sft
#
#   # run both
#   bash run_native_eval.sh all
#
# Environment variables
# ---------------------
#   BASE_MODEL      Path to the base Qwen3-VL model (required for inference).
#   CHECKPOINT      Path to the SFT checkpoint (required for SFT stage).
#   OUTPUT_ROOT     External output root (must NOT be inside aaai_code).
#   DERIVED_ROOT    Root for derived/transformed images (optional, default auto).
#   NPROC           Number of torchrun processes (default: 4).
#
set -euo pipefail

# ── Path derivation ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# experiments/base_native_regate/ -> experiments/ -> aaai_code/
AAAI_CODE="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="$AAAI_CODE/evaluation/eval_qwen3vl_3reason.py"
BENCHMARK_DIR="$AAAI_CODE/benchmark"
RESOLVE_SCRIPT="$SCRIPT_DIR/resolve_inputs.py"

# ── Stage ──────────────────────────────────────────────────────────────────────
STAGE="${1:-check}"
case "$STAGE" in
  check|base|sft|all) ;;
  *)
    echo "Usage: $0 {check|base|sft|all}" >&2
    echo "  check  - validate the environment and arguments only (default)" >&2
    echo "  base   - run the base model, with no adapter" >&2
    echo "  sft    - run the adapted model" >&2
    echo "  all    — base + sft" >&2
    exit 1
    ;;
esac

# ── Environment variables ─────────────────────────────────────────────────────
: "${BASE_MODEL:?Must set BASE_MODEL (path to Qwen3-VL-8B-Instruct)}"
: "${OUTPUT_ROOT:?Must set OUTPUT_ROOT to an external runtime output directory}"
CHECKPOINT="${CHECKPOINT:-}"
DERIVED_ROOT="${DERIVED_ROOT:-}"
NPROC="${NPROC:-4}"

# ── Path validation (check stage) ─────────────────────────────────────────────
ERRORS=0

# BASE_MODEL
if [ ! -d "$BASE_MODEL" ] && [ ! -f "$BASE_MODEL" ]; then
  echo "ERROR: BASE_MODEL path does not exist: $BASE_MODEL" >&2
  ERRORS=1
fi

# CHECKPOINT (only required for sft/all)
if [ "$STAGE" = "sft" ] || [ "$STAGE" = "all" ]; then
  if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT must be set for stage '$STAGE'" >&2
    ERRORS=1
  elif [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT path does not exist: $CHECKPOINT" >&2
    ERRORS=1
  fi
fi

# EVAL_SCRIPT
if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "ERROR: Eval script not found: $EVAL_SCRIPT" >&2
  ERRORS=1
fi

# OUTPUT_ROOT must be outside aaai_code
OUTPUT_REAL="$(cd "$OUTPUT_ROOT" 2>/dev/null && pwd || echo "$OUTPUT_ROOT")"
AAAI_REAL="$(cd "$AAAI_CODE" && pwd)"
if [[ "$OUTPUT_REAL" == "$AAAI_REAL"/* ]] || [ "$OUTPUT_REAL" = "$AAAI_REAL" ]; then
  echo "ERROR: OUTPUT_ROOT ($OUTPUT_ROOT) is inside aaai_code ($AAAI_REAL)" >&2
  ERRORS=1
fi

# Verify benchmark data files exist
for f in \
  "$BENCHMARK_DIR/annotations/sft/sft_test.jsonl" \
  "$BENCHMARK_DIR/annotations/eval/ood_refcocog.jsonl" \
  "$BENCHMARK_DIR/annotations/eval/plantwild_real_closed12_paired_408.jsonl"
do
  if [ ! -f "$f" ]; then
    echo "ERROR: Data file not found: $f" >&2
    ERRORS=1
  fi
done

# Hard checks: line counts
N_SFT=$(wc -l < "$BENCHMARK_DIR/annotations/sft/sft_test.jsonl")
if [ "$N_SFT" -ne 2646 ]; then
  echo "ERROR: sft_test.jsonl: expected 2646 lines, got $N_SFT" >&2
  ERRORS=1
fi
N_REF=$(wc -l < "$BENCHMARK_DIR/annotations/eval/ood_refcocog.jsonl")
if [ "$N_REF" -ne 362 ]; then
  echo "ERROR: ood_refcocog.jsonl: expected 362 lines, got $N_REF" >&2
  ERRORS=1
fi
N_PW=$(wc -l < "$BENCHMARK_DIR/annotations/eval/plantwild_real_closed12_paired_408.jsonl")
if [ "$N_PW" -ne 408 ]; then
  echo "ERROR: plantwild file: expected 408 lines, got $N_PW" >&2
  ERRORS=1
fi

if [ "$ERRORS" -ne 0 ]; then
  echo "" >&2
  echo "Please fix errors above and retry." >&2
  exit 1
fi

echo "=== run_native_eval.sh ==="
echo "  STAGE:         $STAGE"
echo "  AAAI_CODE:     $AAAI_CODE"
echo "  BASE_MODEL:    $BASE_MODEL"
echo "  CHECKPOINT:    ${CHECKPOINT:-(not set, not required for this stage)}"
echo "  OUTPUT_ROOT:   $OUTPUT_ROOT"
echo "  DERIVED_ROOT:  ${DERIVED_ROOT:-auto}"
echo "  NPROC:         $NPROC"
echo ""

# check mode exits here
if [ "$STAGE" = "check" ]; then
  echo "Parameter check passed. Run with: base / sft / all"
  exit 0
fi

# ── Resolve inputs (idempotent via fingerprint) ──────────────────────────────
RESOLVE_ARGS=(
  --benchmark-root "$BENCHMARK_DIR"
  --output-root "$OUTPUT_ROOT"
)
if [ -n "$DERIVED_ROOT" ]; then
  RESOLVE_ARGS+=(--derived-root "$DERIVED_ROOT")
fi

python "$RESOLVE_SCRIPT" "${RESOLVE_ARGS[@]}"

# Derived paths for resolved inputs
COMPREHENSIVE_DATA="$OUTPUT_ROOT/inputs/comprehensive_resolved.jsonl"
REFCOCOG_DATA="$OUTPUT_ROOT/inputs/refcocog_resolved.jsonl"
PLANTWILD_DATA="$OUTPUT_ROOT/inputs/plantwild_resolved.jsonl"

COMMON_ARGS=(
  --backend hf
  --system-override decision
  --max-new-tokens 384
  --max-pixels 262144
)

# ── Helper: run one stage ────────────────────────────────────────────────────
run_stage() {
  local stage_name="$1"       # "base" or "sft"
  local adapter_arg="$2"      # empty string for base, "--adapter PATH" for sft

  local data_csv="${COMPREHENSIVE_DATA},${REFCOCOG_DATA},${PLANTWILD_DATA}"
  local out_csv="${OUTPUT_ROOT}/native/${stage_name}/comprehensive,${OUTPUT_ROOT}/native/${stage_name}/refcocog,${OUTPUT_ROOT}/native/${stage_name}/plantwild"
  local tag_csv="comprehensive,refcocog,plantwild"

  echo ""
  echo "=== Running ${stage_name} evaluation ==="

  # shellcheck disable=SC2086
  torchrun --nproc_per_node="$NPROC" --master_port="$((20000 + RANDOM % 10000))" \
    "$EVAL_SCRIPT" \
    --base "$BASE_MODEL" \
    $adapter_arg \
    --data "$data_csv" \
    --out "$out_csv" \
    --tag "$tag_csv" \
    "${COMMON_ARGS[@]}"

  echo "=== ${stage_name} evaluation done ==="
}

# ── Execute ──────────────────────────────────────────────────────────────────
if [ "$STAGE" = "base" ] || [ "$STAGE" = "all" ]; then
  # BASE: no adapter
  run_stage "base" ""
fi

if [ "$STAGE" = "sft" ] || [ "$STAGE" = "all" ]; then
  # SFT: only --adapter CHECKPOINT, no head flags
  run_stage "sft" "--adapter ${CHECKPOINT}"
fi

echo "=== run_native_eval.sh complete ==="
