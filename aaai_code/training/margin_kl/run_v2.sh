#!/bin/bash
# Margin-and-anchor retraining, variant 2: smaller margin, revised mixture and fewer steps, aimed at A-FU
#
# Usage:
#   export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
#   export SFT_INIT_CKPT=/path/to/checkpoint-444
#   export DISCRIM_MUSD_PATH=/path/to/discrim_musd.pt
#   export ZREF_CACHE_PATH=/path/to/zref_cache.pt
#   export OUTPUT_DIR=/path/to/output_v2
#   bash training/margin_kl/run_v2.sh
#
# Reported checkpoint: checkpoint-60, the best-balanced point within the 80 steps
# The launcher does not stop at that checkpoint; it writes one every save_steps=20.
set -eo pipefail

# ==============================================================================
# Derive the package root from the location of this script
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# margin_kl/../.. = training/.. = aaai_code/
AAAI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ==============================================================================
# Required arguments
# ==============================================================================
: "${BASE_MODEL:?set BASE_MODEL to the Qwen3-VL-8B-Instruct directory}"
: "${SFT_INIT_CKPT:?set SFT_INIT_CKPT to the checkpoint-444 directory}"
: "${DISCRIM_MUSD_PATH:?set DISCRIM_MUSD_PATH to discrim_musd.pt}"
: "${ZREF_CACHE_PATH:?set ZREF_CACHE_PATH to zref_cache.pt}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the output directory}"

# ==============================================================================
# Environment
# ==============================================================================
cd "$AAAI_ROOT"
export PYTHONPATH="$AAAI_ROOT/training:$SCRIPT_DIR:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export TOKENIZERS_PARALLELISM=false

# --- Variables specific to variant 2 ---
export USE_TRI_HEAD=true
export DISCRIM_MUSD_PATH="$DISCRIM_MUSD_PATH"
export ZREF_CACHE_PATH="$ZREF_CACHE_PATH"
export SFT_INIT_CKPT="$SFT_INIT_CKPT"
export PAIR_MARGIN=0.25
export PAIR_BETA=1.0
export PAIR_PATTERN="0,1,0,2,0,1,0,2"

NPROC=${NPROC:-8}
OUT="$OUTPUT_DIR"

echo "[margin_kl v2] AAAI_ROOT=$AAAI_ROOT"
echo "[margin_kl v2] BASE_MODEL=$BASE_MODEL"
echo "[margin_kl v2] SFT_INIT_CKPT=$SFT_INIT_CKPT"
echo "[margin_kl v2] DISCRIM_MUSD_PATH=$DISCRIM_MUSD_PATH"
echo "[margin_kl v2] ZREF_CACHE_PATH=$ZREF_CACHE_PATH"
echo "[margin_kl v2] OUTPUT_DIR=$OUT"
echo "[margin_kl v2] NPROC=$NPROC"
echo "[margin_kl v2] PAIR_MARGIN=0.25 PAIR_BETA=1.0 PATTERN=0,1,0,2,0,1,0,2 max_steps=80 save_steps=20"

# DeepSpeed ZeRO-2, which is sufficient for LoRA
DS_CFG="$AAAI_ROOT/training/deepspeed/zero2.json"
if [ ! -f "$DS_CFG" ]; then
    echo "[FATAL] DeepSpeed config not found: $DS_CFG"
    exit 1
fi

mkdir -p "$OUT"

torchrun --nproc_per_node="$NPROC" --master_port=$(shuf -i 20001-29999 -n 1) \
  "$SCRIPT_DIR/train_paired_qwen.py" \
  --deepspeed "$DS_CFG" \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_use agriabstain_new_reason_3reason_paired \
  --data_flatten False \
  --data_packing False \
  --lora_enable True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm False \
  --bf16 True \
  --output_dir "$OUT" \
  --max_steps 80 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_pixels 262144 --min_pixels 784 \
  --eval_strategy no \
  --save_steps 20 --save_total_limit 5 \
  --learning_rate 1e-5 --weight_decay 0 --warmup_ratio 0.05 --lr_scheduler_type cosine \
  --head_lambda 0.4 --head_warmup_steps 0 \
  --max_grad_norm 1 \
  --logging_steps 5 \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --run_name margin_kl_v2 \
  --report_to none \
  --use_dual_head True

echo "[margin_kl v2] complete -> $OUT"