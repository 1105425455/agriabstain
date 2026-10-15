#!/bin/bash
# Select the grouped SFT checkpoint with the same 100-record dev-loss rule used
# for the independent-negative condition. This script performs no training.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAAI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AAAI_ROOT/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-/user/huangyongbin/models/modelscope/Qwen/Qwen3-VL-8B-Instruct}"
GROUPED_ROOT="${GROUPED_ROOT:-$REPO_ROOT/new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04}"
EPOCH1_CHECKPOINT="${EPOCH1_CHECKPOINT:-$REPO_ROOT/new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04_epoch1_recovery/checkpoint-111}"
EPOCH1_RECOVERY_TASK_ID="${EPOCH1_RECOVERY_TASK_ID:-}"
DEV_SFT="${DEV_SFT:-$REPO_ROOT/new_reason/outputs/sft_3reason/sft_dev.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/artifacts/grouped_dev_selection}"
LOG_DIR="$OUTPUT_ROOT/logs"
SELECTION_DIR="$OUTPUT_ROOT/selection"
CHECKPOINT_STEPS=(222 333 444 555 666)

export PYTHONUNBUFFERED=1
export PYTHONPATH="$AAAI_ROOT/training:$AAAI_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$LOG_DIR" "$SELECTION_DIR"

if [ ! -f "$DEV_SFT" ]; then
    echo "[grouped-dev-selection][FATAL] missing dev data: $DEV_SFT"
    exit 1
fi

BASE_MODEL="$BASE_MODEL" \
CHECKPOINT="$EPOCH1_CHECKPOINT" \
DEV_SFT="$DEV_SFT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
    bash "$SCRIPT_DIR/run_grouped_epoch1_dev_loss.sh"

pids=()
for index in "${!CHECKPOINT_STEPS[@]}"; do
    step="${CHECKPOINT_STEPS[$index]}"
    checkpoint="$GROUPED_ROOT/checkpoint-$step"
    log_path="$LOG_DIR/dev_loss_checkpoint-$step.log"
    if [ ! -f "$checkpoint/adapter_model.safetensors" ]; then
        echo "[grouped-dev-selection][FATAL] missing checkpoint: $checkpoint"
        exit 1
    fi
    (
        CUDA_VISIBLE_DEVICES="$index" python \
            "$SCRIPT_DIR/eval_dev_loss.py" \
            --base "$BASE_MODEL" \
            --adapter "$checkpoint" \
            --data "$DEV_SFT" \
            --limit 100 \
            --max-pixels 262144 >"$log_path" 2>&1
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
    echo "[grouped-dev-selection][FATAL] at least one dev-loss worker failed"
    exit 1
fi

TASK_ID_ARGS=()
if [ -n "$EPOCH1_RECOVERY_TASK_ID" ]; then
    TASK_ID_ARGS=(--extra-checkpoint-task-id "111=$EPOCH1_RECOVERY_TASK_ID")
fi

python "$SCRIPT_DIR/select_checkpoint.py" \
    --checkpoint-root "$GROUPED_ROOT" \
    --dev-log-dir "$LOG_DIR" \
    --output-dir "$SELECTION_DIR" \
    --extra-checkpoint "$EPOCH1_CHECKPOINT" \
    --extra-dev-log "111=$LOG_DIR/dev_loss_checkpoint-111-recovery.log" \
    "${TASK_ID_ARGS[@]}" \
    | tee "$LOG_DIR/select_checkpoint.log"
