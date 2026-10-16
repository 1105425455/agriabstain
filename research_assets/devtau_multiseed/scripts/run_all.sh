#!/usr/bin/env bash
# 多 seed 的 development-selected operating point 复跑。
# 对 h27experiment 三个 grouped checkpoint 复用论文原始协议：
#   dev(2408) 上跑 SFT-native 生成 + calibrated head 概率 -> 选 tau -> 冻结后应用到各自的 test(2646) 产物。
# 只读依赖：aaai_code、h27experiment/runs、new_reason/outputs/benchmarks。
# 只写：本目录 devtau_multiseed/ 下。
set -eo pipefail

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

REPO="/user/huangyongbin/newrepo/agriabstain"
AAAI_CODE="$REPO/aaai_code"
ROOT="$REPO/devtau_multiseed"
H27="$REPO/h27experiment/runs/grouped"
BASE_MODEL="/user/huangyongbin/models/modelscope/Qwen/Qwen3-VL-8B-Instruct"
DERIVED_ROOT="$REPO/new_reason/outputs/benchmarks/transformed_images"
DEV_INPUT="$ROOT/inputs/dev_resolved.jsonl"
NPROC="${NPROC:-8}"

SEEDS=(7 21 42)
declare -A CKPT_STEP=([7]=333 [21]=222 [42]=444)

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/pipeline.log") 2>&1

echo "=== multi-seed development-selected operating point ==="
echo "NPROC=$NPROC  DEV_INPUT=$DEV_INPUT"
date

if [ "$(wc -l < "$DEV_INPUT")" -ne 2408 ]; then
  echo "ERROR: dev input must contain exactly 2408 records" >&2
  exit 1
fi
sha256sum -c <<< "f4d5e3ab4de2a9bf3d6e9671d024b584c473a8d41bd60c5969125af8d64d05c3  $DEV_INPUT"

for seed in "${SEEDS[@]}"; do
  step="${CKPT_STEP[$seed]}"
  ckpt="$H27/seed-$seed/checkpoints/checkpoint-$step"
  for required in \
    "$ckpt" "$ckpt/discrim_head_calibrated.pt" \
    "$H27/seed-$seed/evaluation/native/comprehensive/predictions.jsonl" \
    "$H27/seed-$seed/evaluation/regate/comprehensive/predictions.jsonl"
  do
    if [ ! -e "$required" ]; then
      echo "ERROR: missing required path: $required" >&2
      exit 1
    fi
  done
done
echo "preflight OK: all 3 seeds have checkpoint, calibrated head and test predictions"

for seed in "${SEEDS[@]}"; do
  step="${CKPT_STEP[$seed]}"
  ckpt="$H27/seed-$seed/checkpoints/checkpoint-$step"
  head="$ckpt/discrim_head_calibrated.pt"
  out="$ROOT/runs/seed-$seed"
  mkdir -p "$out/logs" "$out/dev_native" "$out/dev_head" "$out/analysis"

  if [ -f "$out/analysis/result.json" ]; then
    echo "=== seed $seed already complete, skipping ==="
    continue
  fi

  echo ""
  echo "############ seed $seed  (checkpoint-$step) ############"
  date

  if [ ! -f "$out/dev_native/predictions.jsonl" ]; then
    echo "--- seed $seed stage 1/3: SFT-native generation on dev ---"
    torchrun --nproc_per_node="$NPROC" --master_port="$((21000 + seed))" \
      "$AAAI_CODE/evaluation/eval_qwen3vl_3reason.py" \
      --base "$BASE_MODEL" \
      --adapter "$ckpt" \
      --data "$DEV_INPUT" \
      --out "$out/dev_native" \
      --tag comprehensive_dev \
      --backend hf \
      --system-override decision \
      --max-new-tokens 384 \
      --max-pixels 262144 2>&1 | tee "$out/logs/dev_native.log"
  else
    echo "--- seed $seed stage 1/3 already done ---"
  fi

  if [ ! -f "$out/dev_head/head_probabilities.jsonl" ]; then
    echo "--- seed $seed stage 2/3: calibrated head probabilities on dev ---"
    torchrun --nproc_per_node="$NPROC" --master_port="$((22000 + seed))" \
      "$AAAI_CODE/experiments/evidence_judge/extract_head_probabilities.py" \
      --base "$BASE_MODEL" \
      --adapter "$ckpt" \
      --calibrated-head "$head" \
      --data "$DEV_INPUT" \
      --output "$out/dev_head" \
      --variant-set none \
      --batch-size 1 \
      --max-pixels 262144 \
      --derived-root "$DERIVED_ROOT" 2>&1 | tee "$out/logs/dev_head.log"
  else
    echo "--- seed $seed stage 2/3 already done ---"
  fi

  echo "--- seed $seed stage 3/3: dev selection + frozen test application ---"
  python "$ROOT/scripts/select_and_apply_threshold.py" \
    --aaai-code "$AAAI_CODE" \
    --dev-native "$out/dev_native/predictions.jsonl" \
    --dev-probabilities "$out/dev_head/head_probabilities.jsonl" \
    --test-native "$H27/seed-$seed/evaluation/native/comprehensive/predictions.jsonl" \
    --test-regate "$H27/seed-$seed/evaluation/regate/comprehensive/predictions.jsonl" \
    --output-dir "$out/analysis" 2>&1 | tee "$out/logs/select_threshold.log"

  echo "############ seed $seed done ############"
  date
done

echo ""
echo "=== aggregating ==="
python "$ROOT/scripts/aggregate.py" --root "$ROOT" --output "$ROOT/summary/dev_selected_multiseed.json"

echo "=== pipeline complete ==="
date
