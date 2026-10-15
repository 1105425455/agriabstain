#!/bin/bash
# End-to-end independently sampled-negative grouping ablation.
# All generated files stay below experiments/grouping_ablation/artifacts.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAAI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$AAAI_ROOT/.." && pwd)"
ARTIFACT_ROOT="${GROUPING_ABLATION_ARTIFACT_ROOT:-$SCRIPT_DIR/artifacts}"
DATA_DIR="$ARTIFACT_ROOT/data"
TRAIN_DIR="$ARTIFACT_ROOT/checkpoints/independent_negative_seed42"
LOG_DIR="$ARTIFACT_ROOT/logs"
SELECTION_DIR="$ARTIFACT_ROOT/selection"
CALIB_DIR="$ARTIFACT_ROOT/calibration"
EVAL_DIR="$ARTIFACT_ROOT/evaluation"
SUMMARY_DIR="$ARTIFACT_ROOT/summary"

: "${BASE_MODEL:?Set BASE_MODEL to a local Qwen3-VL-8B-Instruct directory}"
DISCRIM_MUSD_PATH="${DISCRIM_MUSD_PATH:-$REPO_ROOT/final_story/outputs/discrim_musd.pt}"
TRAIN_DATA="$DATA_DIR/sft_train_independent_negatives.json"
DEV_SFT="$REPO_ROOT/new_reason/outputs/sft_3reason/sft_dev.json"
DEV_RAW="$REPO_ROOT/new_reason/outputs/splits/dev.jsonl"
NPROC="${NPROC:-8}"
SEED="${SEED:-42}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$AAAI_ROOT/training:$AAAI_ROOT:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export USE_TRI_HEAD=true
export WO_HEAD_WEIGHT=1.5
export NO_TARGET_LOSS_WEIGHT=1.0
export WRONG_OBJECT_LOSS_WEIGHT=1.0
export GROUPING_ABLATION_TRAIN_DATA="$TRAIN_DATA"
export DISCRIM_MUSD_PATH

mkdir -p "$DATA_DIR" "$TRAIN_DIR" "$LOG_DIR" "$SELECTION_DIR" \
    "$CALIB_DIR" "$EVAL_DIR" "$SUMMARY_DIR"

record_environment() {
    {
        date -Iseconds
        echo "BASE_MODEL=$BASE_MODEL"
        echo "DISCRIM_MUSD_PATH=$DISCRIM_MUSD_PATH"
        echo "NPROC=$NPROC"
        echo "SEED=$SEED"
        echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"
        python - <<'PY'
import platform
import torch
import transformers

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
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

build_data() {
    python "$SCRIPT_DIR/build_independent_negatives.py" \
        --output-dir "$DATA_DIR" \
        --seed "$SEED" 2>&1 | tee "$LOG_DIR/build_data.log"
}

train_model() {
    if [ ! -f "$TRAIN_DATA" ]; then
        echo "[train][FATAL] missing training data: $TRAIN_DATA"
        exit 1
    fi
    if [ ! -f "$DISCRIM_MUSD_PATH" ]; then
        echo "[train][FATAL] missing discrim mu/sd: $DISCRIM_MUSD_PATH"
        exit 1
    fi
    torchrun --nproc_per_node="$NPROC" --master_port=29631 \
        "$SCRIPT_DIR/train_qwen.py" \
        --deepspeed "$AAAI_ROOT/training/deepspeed/zero2.json" \
        --model_name_or_path "$BASE_MODEL" \
        --dataset_use grouping_ablation_independent_negatives \
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
        --run_name grouping_ablation_independent_negative_seed42 \
        --report_to none \
        --seed "$SEED" \
        --use_dual_head True \
        --head_lambda 0.4 \
        --head_warmup_steps 10 2>&1 | tee "$LOG_DIR/train.log"
}

select_checkpoint() {
    checkpoints=("$TRAIN_DIR"/checkpoint-*)
    if [ ! -d "${checkpoints[0]}" ]; then
        echo "[select][FATAL] no checkpoints under $TRAIN_DIR"
        exit 1
    fi

    pids=""
    gpu=0
    for checkpoint in "${checkpoints[@]}"; do
        step="${checkpoint##*-}"
        log_path="$LOG_DIR/dev_loss_checkpoint-$step.log"
        (
            CUDA_VISIBLE_DEVICES="$gpu" python \
                "$SCRIPT_DIR/eval_dev_loss.py" \
                --base "$BASE_MODEL" \
                --adapter "$checkpoint" \
                --data "$DEV_SFT" \
                --limit 100 \
                --max-pixels 262144 >"$log_path" 2>&1
        ) &
        pids="$pids $!"
        gpu=$((gpu + 1))
    done

    failed=0
    for pid in $pids; do
        wait "$pid" || failed=1
    done
    if [ "$failed" -ne 0 ]; then
        echo "[select][FATAL] at least one validation-loss worker failed"
        exit 1
    fi

    python "$SCRIPT_DIR/select_checkpoint.py" \
        --checkpoint-root "$TRAIN_DIR" \
        --dev-log-dir "$LOG_DIR" \
        --output-dir "$SELECTION_DIR" 2>&1 | tee "$LOG_DIR/select_checkpoint.log"
}

best_checkpoint() {
    python - "$SELECTION_DIR/checkpoint_selection.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["best_checkpoint"])
PY
}

calibrate_head() {
    checkpoint="$(best_checkpoint)"
    torchrun --nproc_per_node="$NPROC" --master_port=29632 \
        "$AAAI_ROOT/training/sft_final/calibrate_head.py" \
        --mode extract \
        --base "$BASE_MODEL" \
        --adapter "$checkpoint" \
        --sft-train "$TRAIN_DATA" \
        --dev "$DEV_RAW" \
        --dev-max 100 \
        --log-dir "$CALIB_DIR" 2>&1 | tee "$LOG_DIR/calibrate_extract.log"

    CUDA_VISIBLE_DEVICES=0 python "$AAAI_ROOT/training/sft_final/calibrate_head.py" \
        --mode train \
        --base "$BASE_MODEL" \
        --adapter "$checkpoint" \
        --sft-train "$TRAIN_DATA" \
        --dev "$DEV_RAW" \
        --dev-max 100 \
        --log-dir "$CALIB_DIR" 2>&1 | tee "$LOG_DIR/calibrate_train.log"

    if [ ! -f "$checkpoint/discrim_head_calibrated.pt" ]; then
        echo "[calibrate][FATAL] calibrated head was not written"
        exit 1
    fi
}

evaluate() {
    checkpoint="$(best_checkpoint)"
    calibrated_head="$checkpoint/discrim_head_calibrated.pt"
    evaluator="$AAAI_ROOT/evaluation/eval_qwen3vl_3reason.py"

    comprehensive="$REPO_ROOT/new_reason/outputs/sft_3reason/sft_test.jsonl"
    l1="$REPO_ROOT/new_reason/outputs/eval_probes/test_L1_discrim.jsonl"
    l2="$REPO_ROOT/new_reason/outputs/eval_probes/test_L2_evidence.jsonl"
    l3="$REPO_ROOT/new_reason/outputs/eval_probes/test_L3_dose.jsonl"
    refcocog="$REPO_ROOT/new_reason/outputs/eval_probes/ood_refcocog.jsonl"
    plantwild="$REPO_ROOT/new_reason/outputs/eval_probes/plantwild_real_closed12_v1/plantwild_real_closed12_paired_408.jsonl"

    native_data="$comprehensive,$refcocog,$plantwild"
    native_out="$EVAL_DIR/native/comprehensive,$EVAL_DIR/native/refcocog,$EVAL_DIR/native/plantwild"
    native_tags="comprehensive,refcocog,plantwild"
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

    regate_data="$comprehensive,$l1,$l2,$l3,$refcocog,$plantwild"
    regate_out="$EVAL_DIR/regate/comprehensive,$EVAL_DIR/regate/L1_discrim,$EVAL_DIR/regate/L2_evidence,$EVAL_DIR/regate/L3_dose,$EVAL_DIR/regate/refcocog,$EVAL_DIR/regate/plantwild"
    regate_tags="comprehensive,L1_discrim,L2_evidence,L3_dose,refcocog,plantwild"
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
        --system-override decision \
        --max-new-tokens 384 \
        --max-pixels 262144 2>&1 | tee "$LOG_DIR/eval_regate.log"

    python "$SCRIPT_DIR/summarize_results.py" \
        --eval-root "$EVAL_DIR" \
        --out "$SUMMARY_DIR/comparison.json" 2>&1 | tee "$LOG_DIR/summarize.log"
}

stage="${1:-all}"
record_environment
case "$stage" in
    build)
        build_data
        ;;
    train)
        train_model
        ;;
    select)
        select_checkpoint
        ;;
    calibrate)
        calibrate_head
        ;;
    eval)
        evaluate
        ;;
    all)
        build_data
        train_model
        select_checkpoint
        calibrate_head
        evaluate
        ;;
    *)
        echo "Usage: bash run_pipeline.sh {build|train|select|calibrate|eval|all}"
        exit 2
        ;;
esac

echo "[grouping-ablation-complete] stage=$stage artifact_root=$ARTIFACT_ROOT"
