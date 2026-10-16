#!/bin/bash
# Isolated full-pipeline run for one condition and one frozen seed.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
AAAI_ROOT="$REPO_ROOT/aaai_code"
DATA_ROOT="$EXPERIMENT_ROOT/data"

CONDITION="${1:-}"
RUN_SEED="${2:-}"
STAGE="${3:-all}"
if [[ "$CONDITION" != "grouped" && "$CONDITION" != "independent" ]]; then
    echo "Usage: bash code/run_condition.sh {grouped|independent} {7|21|42} {train|select|calibrate|linear|eval|reroute|all}"
    exit 2
fi
if [[ "$RUN_SEED" != "7" && "$RUN_SEED" != "21" && "$RUN_SEED" != "42" ]]; then
    echo "[h27][FATAL] run seed must be one of 7, 21, 42"
    exit 2
fi

RUN_ROOT="$EXPERIMENT_ROOT/runs/$CONDITION/seed-$RUN_SEED"
TRAIN_DIR="$RUN_ROOT/checkpoints"
LOG_DIR="$RUN_ROOT/logs"
SELECTION_DIR="$RUN_ROOT/selection"
CALIB_DIR="$RUN_ROOT/calibration"
LINEAR_DIR="$RUN_ROOT/linear_gate"
EVAL_DIR="$RUN_ROOT/evaluation"
LINEAR_EVAL_DIR="$RUN_ROOT/linear_evaluation"

BASE_MODEL="${BASE_MODEL:-/user/huangyongbin/models/modelscope/Qwen/Qwen3-VL-8B-Instruct}"
DISCRIM_MUSD_PATH="${DISCRIM_MUSD_PATH:-$DATA_ROOT/discrim_musd.pt}"
NPROC="${NPROC:-8}"
DEV_SFT="$DATA_ROOT/sft_dev.json"
DEV_RAW="$DATA_ROOT/dev.jsonl"
if [[ "$CONDITION" == "grouped" ]]; then
    TRAIN_DATA="$DATA_ROOT/sft_train_grouped.json"
    MEDIA_ROOT="/"
else
    TRAIN_DATA="$DATA_ROOT/sft_train_independent.json"
    MEDIA_ROOT="/"
fi

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$RUN_SEED"
export PYTHONPATH="$AAAI_ROOT/training:$AAAI_ROOT:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export USE_TRI_HEAD=true
export WO_HEAD_WEIGHT=1.5
export NO_TARGET_LOSS_WEIGHT=1.0
export WRONG_OBJECT_LOSS_WEIGHT=1.0
export DISCRIM_MUSD_PATH
export H27_TRAIN_DATA="$TRAIN_DATA"
export H27_MEDIA_ROOT="$MEDIA_ROOT"

mkdir -p "$TRAIN_DIR" "$LOG_DIR" "$SELECTION_DIR" "$CALIB_DIR" \
    "$LINEAR_DIR" "$EVAL_DIR" "$LINEAR_EVAL_DIR"

require_inputs() {
    local inputs=(
        "$BASE_MODEL"
        "$DISCRIM_MUSD_PATH"
        "$TRAIN_DATA"
        "$DEV_SFT"
        "$DEV_RAW"
        "$DATA_ROOT/comprehensive.jsonl"
        "$DATA_ROOT/L1_discrim.jsonl"
        "$DATA_ROOT/L2_evidence.jsonl"
        "$DATA_ROOT/L3_dose.jsonl"
        "$DATA_ROOT/refcocog.jsonl"
        "$DATA_ROOT/plantwild.jsonl"
    )
    for path in "${inputs[@]}"; do
        if [[ ! -e "$path" ]]; then
            echo "[h27][FATAL] missing input: $path"
            exit 1
        fi
    done
}

record_environment() {
    {
        date -Iseconds
        echo "condition=$CONDITION"
        echo "run_seed=$RUN_SEED"
        echo "dev_subset_sampling_seed=42"
        echo "train_data=$TRAIN_DATA"
        echo "train_data_sha256=$(sha256sum "$TRAIN_DATA" | awk '{print $1}')"
        echo "base_model=$BASE_MODEL"
        echo "discrim_musd=$DISCRIM_MUSD_PATH"
        echo "nproc=$NPROC"
        python - <<'PY'
import platform
import sklearn
import torch
import transformers

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"cuda={torch.version.cuda}")
PY
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader
    } | tee "$LOG_DIR/environment.log"

    python - <<'PY'
import subprocess

rows = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
for row in rows.splitlines():
    index, total, used = [int(value.strip()) for value in row.split(",")]
    ratio = used / total
    print(f"[gpu-start] gpu={index} memory={used}/{total} MiB ({ratio:.1%})", flush=True)
    if ratio > 0.90:
        print(f"[gpu-start][WARN] gpu={index} memory usage exceeds 90%", flush=True)
PY
}

train_model() {
    torchrun --nproc_per_node="$NPROC" --master_port=29631 \
        "$SCRIPT_DIR/train_qwen.py" \
        --deepspeed "$AAAI_ROOT/training/deepspeed/zero2.json" \
        --model_name_or_path "$BASE_MODEL" \
        --dataset_use h27_multiseed \
        --data_flatten False \
        --data_packing False \
        --lora_enable True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm False \
        --bf16 True \
        --output_dir "$TRAIN_DIR" \
        --num_train_epochs 6 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --max_pixels 262144 --min_pixels 784 \
        --eval_strategy no \
        --save_strategy epoch --save_total_limit 6 \
        --learning_rate 1e-4 --weight_decay 0 --warmup_ratio 0.05 \
        --max_grad_norm 1 --lr_scheduler_type cosine \
        --logging_steps 5 \
        --model_max_length 2048 \
        --gradient_checkpointing False \
        --dataloader_num_workers 4 \
        --run_name "h27_${CONDITION}_seed${RUN_SEED}" \
        --report_to none \
        --seed "$RUN_SEED" \
        --use_dual_head True \
        --head_lambda 0.4 \
        --head_warmup_steps 10 2>&1 | tee "$LOG_DIR/train.log"
}

select_checkpoint() {
    local checkpoints=("$TRAIN_DIR"/checkpoint-*)
    if [[ ! -d "${checkpoints[0]}" ]]; then
        echo "[h27][FATAL] no checkpoints under $TRAIN_DIR"
        exit 1
    fi
    local pids=()
    local gpu=0
    for checkpoint in "${checkpoints[@]}"; do
        local step="${checkpoint##*-}"
        local log_path="$LOG_DIR/dev_loss_checkpoint-$step.log"
        (
            CUDA_VISIBLE_DEVICES="$gpu" python \
                "$AAAI_ROOT/experiments/grouping_ablation/eval_dev_loss.py" \
                --base "$BASE_MODEL" \
                --adapter "$checkpoint" \
                --data "$DEV_SFT" \
                --limit 100 \
                --max-pixels 262144 >"$log_path" 2>&1
        ) &
        pids+=("$!")
        gpu=$((gpu + 1))
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "[h27][FATAL] at least one development-loss worker failed"
        exit 1
    fi
    python "$AAAI_ROOT/experiments/grouping_ablation/select_checkpoint.py" \
        --checkpoint-root "$TRAIN_DIR" \
        --dev-log-dir "$LOG_DIR" \
        --output-dir "$SELECTION_DIR" 2>&1 | tee "$LOG_DIR/select_checkpoint.log"
}

best_checkpoint() {
    python - "$SELECTION_DIR/checkpoint_selection.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["best_checkpoint"])
PY
}

calibrate_head() {
    local checkpoint
    checkpoint="$(best_checkpoint)"
    torchrun --nproc_per_node="$NPROC" --master_port=29632 \
        "$SCRIPT_DIR/run_calibrate.py" \
        --run-seed "$RUN_SEED" \
        --mode extract \
        --base "$BASE_MODEL" \
        --adapter "$checkpoint" \
        --sft-train "$TRAIN_DATA" \
        --dev "$DEV_RAW" \
        --dev-max 100 \
        --log-dir "$CALIB_DIR" 2>&1 | tee "$LOG_DIR/calibrate_extract.log"

    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/run_calibrate.py" \
        --run-seed "$RUN_SEED" \
        --mode train \
        --base "$BASE_MODEL" \
        --adapter "$checkpoint" \
        --sft-train "$TRAIN_DATA" \
        --dev "$DEV_RAW" \
        --dev-max 100 \
        --log-dir "$CALIB_DIR" 2>&1 | tee "$LOG_DIR/calibrate_train.log"

    if [[ ! -f "$checkpoint/discrim_head_calibrated.pt" ]]; then
        echo "[h27][FATAL] calibrated MLP head was not written"
        exit 1
    fi
}

train_linear() {
    python "$SCRIPT_DIR/train_linear_gate.py" \
        --calibration-dir "$CALIB_DIR" \
        --output-dir "$LINEAR_DIR" \
        --run-seed "$RUN_SEED" 2>&1 | tee "$LOG_DIR/train_linear_gate.log"
}

evaluate_models() {
    local checkpoint
    checkpoint="$(best_checkpoint)"
    local calibrated_head="$checkpoint/discrim_head_calibrated.pt"
    local evaluator="$AAAI_ROOT/evaluation/eval_qwen3vl_3reason.py"

    local comprehensive="$DATA_ROOT/comprehensive.jsonl"
    local l1="$DATA_ROOT/L1_discrim.jsonl"
    local l2="$DATA_ROOT/L2_evidence.jsonl"
    local l3="$DATA_ROOT/L3_dose.jsonl"
    local refcocog="$DATA_ROOT/refcocog.jsonl"
    local plantwild="$DATA_ROOT/plantwild.jsonl"

    local native_data="$comprehensive,$refcocog,$plantwild"
    local native_out="$EVAL_DIR/native/comprehensive,$EVAL_DIR/native/refcocog,$EVAL_DIR/native/plantwild"
    local native_tags="comprehensive,refcocog,plantwild"
    torchrun --nproc_per_node="$NPROC" --master_port=29633 "$evaluator" \
        --data "$native_data" \
        --base "$BASE_MODEL" \
        --adapter "$checkpoint" \
        --out "$native_out" \
        --tag "$native_tags" \
        --backend hf \
        --system-override decision \
        --max-new-tokens 384 \
        --max-pixels 262144 2>&1 | tee "$LOG_DIR/eval_native.log"

    local regate_data="$comprehensive,$l1,$l2,$l3,$refcocog,$plantwild"
    local regate_out="$EVAL_DIR/regate/comprehensive,$EVAL_DIR/regate/L1_discrim,$EVAL_DIR/regate/L2_evidence,$EVAL_DIR/regate/L3_dose,$EVAL_DIR/regate/refcocog,$EVAL_DIR/regate/plantwild"
    local regate_tags="comprehensive,L1_discrim,L2_evidence,L3_dose,refcocog,plantwild"
    torchrun --nproc_per_node="$NPROC" --master_port=29634 "$evaluator" \
        --data "$regate_data" \
        --base "$BASE_MODEL" \
        --adapter "$checkpoint" \
        --calibrated-head "$calibrated_head" \
        --out "$regate_out" \
        --tag "$regate_tags" \
        --backend hf \
        --dual-head \
        --faithful-dual-output \
        --dump-hidden \
        --dump-all-answer-candidates \
        --system-override decision \
        --max-new-tokens 384 \
        --max-pixels 262144 2>&1 | tee "$LOG_DIR/eval_regate.log"
}

reroute_linear() {
    local model="$LINEAR_DIR/linear_gate_best.npz"
    local datasets=(comprehensive L1_discrim L2_evidence L3_dose refcocog plantwild)
    for dataset in "${datasets[@]}"; do
        python "$SCRIPT_DIR/reroute_linear.py" \
            --model "$model" \
            --predictions "$EVAL_DIR/regate/$dataset/predictions.jsonl" \
            --dataset "$dataset" \
            --output-dir "$LINEAR_EVAL_DIR/$dataset" \
            2>&1 | tee "$LOG_DIR/reroute_linear_$dataset.log"
    done
}

require_inputs
record_environment
case "$STAGE" in
    train)
        train_model
        ;;
    select)
        select_checkpoint
        ;;
    calibrate)
        calibrate_head
        ;;
    linear)
        train_linear
        ;;
    eval)
        evaluate_models
        ;;
    reroute)
        reroute_linear
        ;;
    all)
        train_model
        select_checkpoint
        calibrate_head
        train_linear
        evaluate_models
        reroute_linear
        ;;
    *)
        echo "Usage: bash code/run_condition.sh {grouped|independent} {7|21|42} {train|select|calibrate|linear|eval|reroute|all}"
        exit 2
        ;;
esac

touch "$RUN_ROOT/.complete-$STAGE"
echo "[h27-complete] condition=$CONDITION seed=$RUN_SEED stage=$STAGE run_root=$RUN_ROOT"
