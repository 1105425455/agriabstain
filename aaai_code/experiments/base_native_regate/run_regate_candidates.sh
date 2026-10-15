#!/usr/bin/env bash
# run_regate_candidates.sh — Regate answer-candidate dumping with dual-head.
#
# Launches the formal eval_qwen3vl_3reason.py with:
#   --adapter CHECKPOINT
#   --calibrated-head CHECKPOINT/discrim_head_calibrated.pt
#   --dual-head --faithful-dual-output --dump-all-answer-candidates
#   --system-override decision --max-new-tokens 384 --max-pixels 262144
#
# No GT fallback or non-faithful branch is used.
#
# Usage
# -----
#   # check only (default)
#   export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
#   export CHECKPOINT=/path/to/checkpoint-444
#   export OUTPUT_ROOT=/scratch/regate
#   bash run_regate_candidates.sh
#
#   # run
#   bash run_regate_candidates.sh run
#
# Environment variables
# ---------------------
#   BASE_MODEL      Path to the base Qwen3-VL model (required).
#   CHECKPOINT      Path to the SFT checkpoint with discrim_head_calibrated.pt (required).
#   OUTPUT_ROOT     External output root (must NOT be inside aaai_code).
#   DERIVED_ROOT    Root for derived/transformed images (optional, default auto).
#   NPROC           Number of torchrun processes (default: 4).
#
set -euo pipefail

# ── Path derivation ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAAI_CODE="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="$AAAI_CODE/evaluation/eval_qwen3vl_3reason.py"
BENCHMARK_DIR="$AAAI_CODE/benchmark"
RESOLVE_SCRIPT="$SCRIPT_DIR/resolve_inputs.py"

# ── Stage ──────────────────────────────────────────────────────────────────────
STAGE="${1:-check}"
case "$STAGE" in
  check|run) ;;
  *)
    echo "Usage: $0 {check|run}" >&2
    echo "  check  - validate the environment and arguments only (default)" >&2
    echo "  run    - dump the gate candidates" >&2
    exit 1
    ;;
esac

# ── Environment variables ─────────────────────────────────────────────────────
: "${BASE_MODEL:?Must set BASE_MODEL (path to Qwen3-VL-8B-Instruct)}"
: "${CHECKPOINT:?Must set CHECKPOINT (path to checkpoint-444)}"
: "${OUTPUT_ROOT:?Must set OUTPUT_ROOT to an external runtime output directory}"
DERIVED_ROOT="${DERIVED_ROOT:-}"
NPROC="${NPROC:-4}"

CALIBRATED_HEAD="${CHECKPOINT}/discrim_head_calibrated.pt"

# ── Path validation ───────────────────────────────────────────────────────────
ERRORS=0

if [ ! -d "$BASE_MODEL" ] && [ ! -f "$BASE_MODEL" ]; then
  echo "ERROR: BASE_MODEL path does not exist: $BASE_MODEL" >&2
  ERRORS=1
fi

if [ ! -d "$CHECKPOINT" ]; then
  echo "ERROR: CHECKPOINT path does not exist: $CHECKPOINT" >&2
  ERRORS=1
fi

if [ ! -f "$CALIBRATED_HEAD" ]; then
  echo "ERROR: Calibrated head not found: $CALIBRATED_HEAD" >&2
  ERRORS=1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "ERROR: Eval script not found: $EVAL_SCRIPT" >&2
  ERRORS=1
fi

# OUTPUT_ROOT outside aaai_code
OUTPUT_REAL="$(cd "$OUTPUT_ROOT" 2>/dev/null && pwd || echo "$OUTPUT_ROOT")"
AAAI_REAL="$(cd "$AAAI_CODE" && pwd)"
if [[ "$OUTPUT_REAL" == "$AAAI_REAL"/* ]] || [ "$OUTPUT_REAL" = "$AAAI_REAL" ]; then
  echo "ERROR: OUTPUT_ROOT ($OUTPUT_ROOT) is inside aaai_code ($AAAI_REAL)" >&2
  ERRORS=1
fi

# Hard checks: line counts
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

N_SFT=$(wc -l < "$BENCHMARK_DIR/annotations/sft/sft_test.jsonl")
if [ "$N_SFT" -ne 2646 ]; then echo "ERROR: sft_test.jsonl: expected 2646 lines, got $N_SFT" >&2; ERRORS=1; fi
N_REF=$(wc -l < "$BENCHMARK_DIR/annotations/eval/ood_refcocog.jsonl")
if [ "$N_REF" -ne 362 ]; then echo "ERROR: ood_refcocog.jsonl: expected 362 lines, got $N_REF" >&2; ERRORS=1; fi
N_PW=$(wc -l < "$BENCHMARK_DIR/annotations/eval/plantwild_real_closed12_paired_408.jsonl")
if [ "$N_PW" -ne 408 ]; then echo "ERROR: plantwild file: expected 408 lines, got $N_PW" >&2; ERRORS=1; fi

if [ "$ERRORS" -ne 0 ]; then
  echo "" >&2
  echo "Please fix errors above and retry." >&2
  exit 1
fi

echo "=== run_regate_candidates.sh ==="
echo "  AAAI_CODE:       $AAAI_CODE"
echo "  BASE_MODEL:      $BASE_MODEL"
echo "  CHECKPOINT:      $CHECKPOINT"
echo "  CALIBRATED_HEAD: $CALIBRATED_HEAD"
echo "  OUTPUT_ROOT:     $OUTPUT_ROOT"
echo "  DERIVED_ROOT:    ${DERIVED_ROOT:-auto}"
echo "  NPROC:           $NPROC"
echo ""

if [ "$STAGE" = "check" ]; then
  echo "Parameter check passed. Run with: run"
  exit 0
fi

# ── Resolve inputs ────────────────────────────────────────────────────────────
RESOLVE_ARGS=(
  --benchmark-root "$BENCHMARK_DIR"
  --output-root "$OUTPUT_ROOT"
)
if [ -n "$DERIVED_ROOT" ]; then
  RESOLVE_ARGS+=(--derived-root "$DERIVED_ROOT")
fi

python "$RESOLVE_SCRIPT" "${RESOLVE_ARGS[@]}"

COMPREHENSIVE_DATA="$OUTPUT_ROOT/inputs/comprehensive_resolved.jsonl"
REFCOCOG_DATA="$OUTPUT_ROOT/inputs/refcocog_resolved.jsonl"
PLANTWILD_DATA="$OUTPUT_ROOT/inputs/plantwild_resolved.jsonl"

DATA_CSV="${COMPREHENSIVE_DATA},${REFCOCOG_DATA},${PLANTWILD_DATA}"
OUT_CSV="${OUTPUT_ROOT}/regate/comprehensive,${OUTPUT_ROOT}/regate/refcocog,${OUTPUT_ROOT}/regate/plantwild"
TAG_CSV="comprehensive,refcocog,plantwild"

echo ""
echo "=== Running regate candidate dumping ==="

torchrun --nproc_per_node="$NPROC" --master_port="$((20000 + RANDOM % 10000))" \
  "$EVAL_SCRIPT" \
  --base "$BASE_MODEL" \
  --backend hf \
  --adapter "$CHECKPOINT" \
  --calibrated-head "$CALIBRATED_HEAD" \
  --dual-head \
  --faithful-dual-output \
  --dump-all-answer-candidates \
  --system-override decision \
  --max-new-tokens 384 \
  --max-pixels 262144 \
  --data "$DATA_CSV" \
  --out "$OUT_CSV" \
  --tag "$TAG_CSV"

echo "=== run_regate_candidates.sh complete ==="
