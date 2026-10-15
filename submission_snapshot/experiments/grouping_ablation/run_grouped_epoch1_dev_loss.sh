#!/bin/bash
# Evaluate the recovered grouped epoch-1 checkpoint on the same fixed 100-record
# dev subset. This is a retrospective audit and performs no training.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAAI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AAAI_ROOT/.." && pwd)"

: "${BASE_MODEL:?Set BASE_MODEL to a local Qwen3-VL-8B-Instruct directory}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04_epoch1_recovery/checkpoint-111}"
DEV_SFT="${DEV_SFT:-$REPO_ROOT/new_reason/outputs/sft_3reason/sft_dev.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/artifacts/grouped_dev_selection}"
LOG_PATH="$OUTPUT_ROOT/logs/dev_loss_checkpoint-111-recovery.log"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$AAAI_ROOT/training:$AAAI_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$(dirname "$LOG_PATH")"

python "$SCRIPT_DIR/eval_dev_loss.py" \
    --base "$BASE_MODEL" \
    --adapter "$CHECKPOINT" \
    --data "$DEV_SFT" \
    --limit 100 \
    --max-pixels 262144 2>&1 | tee "$LOG_PATH"
