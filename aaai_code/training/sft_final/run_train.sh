#!/bin/bash
# Stage-1 training with the three-state discriminative head; checkpoint-444 is the reported endpoint.
# Usage:
#   BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct DISCRIM_MUSD_PATH=/path/to/discrim_musd.pt \
#     bash training/sft_final/run_train.sh [output_dir]
# Or:
#   bash training/sft_final/run_train.sh /path/to/Qwen3-VL-8B-Instruct /path/to/output
#
# 8 GPUs under torchrun with DeepSpeed ZeRO-2.
# The training data is selected through the dataset registry, via the DATASET variable.
set -eo pipefail

# ==============================================================================
# Derive the package root from the location of this script
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAAI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ==============================================================================
# Base model: BASE_MODEL if set, otherwise the first positional argument
# ==============================================================================
if [ -n "${BASE_MODEL:-}" ]; then
    MODEL="$BASE_MODEL"
elif [ $# -ge 1 ]; then
    MODEL="$1"
else
    echo "[FATAL] set BASE_MODEL, or pass the base model path as the first argument"
    exit 1
fi

# ==============================================================================
# Output directory: SFT_OUTPUT_DIR, the second positional argument, or a default
# ==============================================================================
if [ -n "${SFT_OUTPUT_DIR:-}" ]; then
    OUT="$SFT_OUTPUT_DIR"
elif [ $# -ge 2 ]; then
    OUT="$2"
else
    OUT="$AAAI_ROOT/training/sft_final/outputs/sft_final_3reason"
fi

# ==============================================================================
# DISCRIM_MUSD_PATH is required
# ==============================================================================
if [ -z "${DISCRIM_MUSD_PATH:-}" ]; then
    echo "[FATAL] DISCRIM_MUSD_PATH is required and must point at discrim_musd.pt"
    exit 1
fi

# ==============================================================================
# Environment
# ==============================================================================
cd "$AAAI_ROOT"
export PYTHONPATH="$AAAI_ROOT/training:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export TOKENIZERS_PARALLELISM=false

# Select the three-state head. This variable is what distinguishes it from the two-way head.
export USE_TRI_HEAD=true

# Reported hyperparameters: 6 epochs, head_lambda 0.4, head warmup 10, WO_HEAD_WEIGHT 1.5, lr 1e-4, seed 42
HEAD_LAMBDA=${HEAD_LAMBDA:-0.4}
HEAD_WARMUP=${HEAD_WARMUP:-10}
WO_HEAD_WEIGHT=${WO_HEAD_WEIGHT:-1.5}
NO_TARGET_LOSS_WEIGHT=${NO_TARGET_LOSS_WEIGHT:-1.0}
WRONG_OBJECT_LOSS_WEIGHT=${WRONG_OBJECT_LOSS_WEIGHT:-1.0}
DATASET=${DATASET:-agriabstain_new_reason_3reason}
NPROC=${NPROC:-8}
LR=${LR:-1e-4}
SEED=${SEED:-42}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
RUN_NAME=${RUN_NAME:-sft_final_3reason}
DS_CFG="$AAAI_ROOT/training/deepspeed/zero2.json"

# Echo the resolved arguments
echo "[run_train] AAAI_ROOT=$AAAI_ROOT"
echo "[run_train] MODEL=$MODEL"
echo "[run_train] OUT=$OUT"
echo "[run_train] DATASET=$DATASET"
echo "[run_train] NPROC=$NPROC"
echo "[run_train] HEAD_LAMBDA=$HEAD_LAMBDA"
echo "[run_train] HEAD_WARMUP=$HEAD_WARMUP"
echo "[run_train] WO_HEAD_WEIGHT=$WO_HEAD_WEIGHT"
echo "[run_train] DISCRIM_MUSD_PATH=$DISCRIM_MUSD_PATH"
echo "[run_train] LR=$LR"
echo "[run_train] SEED=$SEED"
echo "[run_train] DS_CFG=$DS_CFG"

# Verify the DeepSpeed config
if [ ! -f "$DS_CFG" ]; then
    echo "[FATAL] DeepSpeed config not found: $DS_CFG"
    exit 1
fi

mkdir -p "$OUT"

# ==============================================================================
# Launch training
# ==============================================================================
torchrun --nproc_per_node="$NPROC" --master_port=$(shuf -i 20001-29999 -n 1) \
    qwenvl/train/train_qwen.py \
    --deepspeed "$DS_CFG" \
    --model_name_or_path "$MODEL" \
    --dataset_use "$DATASET" \
    --data_flatten False \
    --data_packing False \
    --lora_enable True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm False \
    --bf16 True \
    --output_dir "$OUT" \
    --num_train_epochs 6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_pixels 262144 --min_pixels 784 \
    --eval_strategy no \
    --save_strategy epoch --save_total_limit 5 \
    --learning_rate "$LR" --weight_decay "$WEIGHT_DECAY" --warmup_ratio 0.05 \
    --max_grad_norm 1 --lr_scheduler_type cosine \
    --logging_steps 5 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --run_name "$RUN_NAME" \
    --report_to none \
    --seed "$SEED" \
    --use_dual_head True \
    --head_lambda "$HEAD_LAMBDA" \
    --head_warmup_steps "$HEAD_WARMUP"

echo "[run_train] training complete -> $OUT"
