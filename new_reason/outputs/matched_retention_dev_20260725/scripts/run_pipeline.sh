#!/usr/bin/env bash
# 独立补跑：dev 上选择 threshold，冻结后应用于已有 comprehensive test predictions。
set -eo pipefail

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

REPO="/user/huangyongbin/newrepo/agriabstain"
AAAI_CODE="$REPO/aaai_code"
OUTPUT_ROOT="$REPO/new_reason/outputs/matched_retention_dev_20260725"
EXISTING_TEST_ROOT="$REPO/new_reason/outputs/base_native_regate_20260720"
BASE_MODEL="/user/huangyongbin/models/modelscope/Qwen/Qwen3-VL-8B-Instruct"
CHECKPOINT="$REPO/new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444"
CALIBRATED_HEAD="$CHECKPOINT/discrim_head_calibrated.pt"
DERIVED_ROOT="$REPO/new_reason/outputs/benchmarks/transformed_images"
DEV_SPLIT="$AAAI_CODE/benchmark/annotations/splits/dev.jsonl"
TEST_SPLIT="$AAAI_CODE/benchmark/annotations/splits/test.jsonl"
SFT_DEV="$AAAI_CODE/benchmark/annotations/sft/sft_dev.json"
DEV_INPUT="$OUTPUT_ROOT/inputs/dev_resolved.jsonl"
DEV_NATIVE_OUT="$OUTPUT_ROOT/dev/native_sft"
DEV_HEAD_OUT="$OUTPUT_ROOT/dev/regate_head"
ANALYSIS_OUT="$OUTPUT_ROOT/analysis"
NPROC="${NPROC:-4}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/inputs" "$OUTPUT_ROOT/dev" "$ANALYSIS_OUT"
exec > >(tee -a "$OUTPUT_ROOT/logs/pipeline.log") 2>&1

echo "=== matched-retention dev-selection pipeline ==="
echo "BASE_MODEL=$BASE_MODEL"
echo "CHECKPOINT=$CHECKPOINT"
echo "CALIBRATED_HEAD=$CALIBRATED_HEAD"
echo "DEV_SPLIT=$DEV_SPLIT"
echo "TEST_SPLIT=$TEST_SPLIT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "NPROC=$NPROC"

for required in \
  "$BASE_MODEL" \
  "$CHECKPOINT" \
  "$CALIBRATED_HEAD" \
  "$DERIVED_ROOT" \
  "$DEV_SPLIT" \
  "$TEST_SPLIT" \
  "$SFT_DEV" \
  "$EXISTING_TEST_ROOT/native/sft/comprehensive/predictions.jsonl" \
  "$EXISTING_TEST_ROOT/regate/comprehensive/predictions.jsonl"
do
  if [ ! -e "$required" ]; then
    echo "ERROR: required path does not exist: $required" >&2
    exit 1
  fi
done

python "$OUTPUT_ROOT/scripts/prepare_dev_eval.py" \
  --benchmark-root "$AAAI_CODE/benchmark" \
  --derived-root "$DERIVED_ROOT" \
  --dev-split "$DEV_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --sft-dev "$SFT_DEV" \
  --output "$DEV_INPUT" \
  --manifest "$OUTPUT_ROOT/inputs/manifest.json"

if [ "$(wc -l < "$DEV_INPUT")" -ne 2408 ]; then
  echo "ERROR: prepared dev input must contain exactly 2408 records" >&2
  exit 1
fi
if [ "$(wc -l < "$EXISTING_TEST_ROOT/native/sft/comprehensive/predictions.jsonl")" -ne 2646 ]; then
  echo "ERROR: existing SFT-native comprehensive test predictions must contain 2646 records" >&2
  exit 1
fi
if [ "$(wc -l < "$EXISTING_TEST_ROOT/regate/comprehensive/predictions.jsonl")" -ne 2646 ]; then
  echo "ERROR: existing ReGate comprehensive test predictions must contain 2646 records" >&2
  exit 1
fi

echo "=== Stage 1/3: SFT-native generation on canonical dev ==="
torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$((20000 + RANDOM % 10000))" \
  "$AAAI_CODE/evaluation/eval_qwen3vl_3reason.py" \
  --base "$BASE_MODEL" \
  --adapter "$CHECKPOINT" \
  --data "$DEV_INPUT" \
  --out "$DEV_NATIVE_OUT" \
  --tag comprehensive_dev \
  --backend hf \
  --system-override decision \
  --max-new-tokens 384 \
  --max-pixels 262144

echo "=== Stage 2/3: calibrated ReGate head probabilities on canonical dev ==="
torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$((20000 + RANDOM % 10000))" \
  "$AAAI_CODE/experiments/evidence_judge/extract_head_probabilities.py" \
  --base "$BASE_MODEL" \
  --adapter "$CHECKPOINT" \
  --calibrated-head "$CALIBRATED_HEAD" \
  --data "$DEV_INPUT" \
  --output "$DEV_HEAD_OUT" \
  --variant-set none \
  --batch-size 1 \
  --max-pixels 262144 \
  --derived-root "$DERIVED_ROOT"

echo "=== Stage 3/3: dev selection and frozen test application ==="
python "$OUTPUT_ROOT/scripts/select_and_apply_threshold.py" \
  --aaai-code "$AAAI_CODE" \
  --dev-native "$DEV_NATIVE_OUT/predictions.jsonl" \
  --dev-probabilities "$DEV_HEAD_OUT/head_probabilities.jsonl" \
  --test-native "$EXISTING_TEST_ROOT/native/sft/comprehensive/predictions.jsonl" \
  --test-regate "$EXISTING_TEST_ROOT/regate/comprehensive/predictions.jsonl" \
  --output-dir "$ANALYSIS_OUT"

echo "=== pipeline complete ==="
echo "RESULT=$ANALYSIS_OUT/result.json"
