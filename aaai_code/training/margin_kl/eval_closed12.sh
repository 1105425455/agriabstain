#!/usr/bin/env bash
# Direct faithful evaluation of a margin_kl checkpoint on the authoritative
# PlantWild closed12 split (408 samples). No 444-sample post-hoc aggregation.
# Uses aaai_code/evaluation/eval_qwen3vl_3reason.py (not calibrated-head override).
#
# Usage:
#   export CKPT_DIR=/path/to/checkpoint-50    # the margin_kl checkpoint to evaluate
#   export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
#   bash training/margin_kl/eval_closed12.sh
set -eo pipefail

# ==============================================================================
# Derive the package root from the location of this script
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAAI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ==============================================================================
# Required arguments
# ==============================================================================
: "${CKPT_DIR:?set CKPT_DIR to the margin_kl checkpoint to evaluate}"
: "${BASE_MODEL:?set BASE_MODEL to the Qwen3-VL-8B-Instruct directory}"

# ==============================================================================
# Paths
# ==============================================================================
EVAL_SCRIPT="$AAAI_ROOT/evaluation/eval_qwen3vl_3reason.py"
# The closed-set evaluation data, produced by build_closed_set_eval.py
DATA="$AAAI_ROOT/benchmark/annotations/eval/plantwild_real_closed12_paired_408.jsonl"

# ==============================================================================
# Validate the inputs
# ==============================================================================
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[FATAL] checkpoint not found: $CKPT_DIR" >&2
    exit 1
fi
if [[ ! -f "$CKPT_DIR/adapter_model.safetensors" ]]; then
    echo "[FATAL] adapter_model.safetensors not found: $CKPT_DIR" >&2
    exit 1
fi
if [[ ! -f "$DATA" ]]; then
    echo "[FATAL] closed12 data not found: $DATA" >&2
    exit 1
fi

# The probe must hold exactly 408 records
N_SAMPLES=$(wc -l < "$DATA")
if [[ "$N_SAMPLES" -ne 408 ]]; then
    echo "[FATAL] expected 408 closed12 samples, got $N_SAMPLES: $DATA" >&2
    exit 1
fi

RUN_NAME=$(basename "$(dirname "$CKPT_DIR")")
STEP=$(basename "$CKPT_DIR")
OUT_DIR="$CKPT_DIR/eval_paired/plantwild_real_closed12"
LOG_DIR="$AAAI_ROOT/training/margin_kl/logs"
LOG_FILE="$LOG_DIR/eval_closed12_${RUN_NAME}_${STEP}.log"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$AAAI_ROOT"

# ==============================================================================
# Environment
# ==============================================================================
export PYTHONPATH="$AAAI_ROOT/training:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "[$(date '+%Y-%m-%d %H:%M:%S')] margin_kl closed12 faithful eval"
echo "  checkpoint: $CKPT_DIR"
echo "  base model: $BASE_MODEL"
echo "  data:       $DATA ($N_SAMPLES samples)"
echo "  output:     $OUT_DIR"
echo "  mode:       adapter tri-head, no calibrated-head override"

# ==============================================================================
# Run the evaluation on the 408-record closed set, without a calibrated head
# ==============================================================================
torchrun --nproc_per_node=4 "$EVAL_SCRIPT" \
    --data "$DATA" \
    --base "$BASE_MODEL" \
    --adapter "$CKPT_DIR" \
    --out "$OUT_DIR" \
    --tag plantwild_real_closed12 \
    --backend hf \
    --dual-head \
    --faithful-dual-output \
    --system-override decision \
    --max-new-tokens 384 \
    --max-pixels 262144 \
    2>&1 | tee "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] completed: $OUT_DIR"
