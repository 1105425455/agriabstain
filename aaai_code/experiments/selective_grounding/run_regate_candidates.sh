#!/usr/bin/env bash
set -euo pipefail

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to an external runtime output directory}"

BASE_MODEL="${BASE_MODEL:-/user/huangyongbin/models/modelscope/Qwen/Qwen3-VL-8B-Instruct}"
CHECKPOINT="${CHECKPOINT:-/user/huangyongbin/newrepo/agriabstain/new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444}"
DERIVED_IMAGE_ROOT="${DERIVED_IMAGE_ROOT:-/user/huangyongbin/newrepo/agriabstain/new_reason/outputs/benchmarks/transformed_images}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPREHENSIVE_DATA="${OUTPUT_ROOT}/inputs/sft_test_resolved.jsonl"

python "${CODE_ROOT}/experiments/selective_grounding/resolve_image_paths.py" \
  --input "${CODE_ROOT}/benchmark/annotations/sft/sft_test.jsonl" \
  --output "${COMPREHENSIVE_DATA}" \
  --benchmark-root "${CODE_ROOT}/benchmark" \
  --derived-root "${DERIVED_IMAGE_ROOT}"

python "${CODE_ROOT}/evaluation/eval_qwen3vl_3reason.py" \
  --data "${COMPREHENSIVE_DATA},${CODE_ROOT}/benchmark/annotations/eval/ood_refcocog.jsonl,${CODE_ROOT}/benchmark/annotations/eval/plantwild_real_closed12_paired_408.jsonl" \
  --out "${OUTPUT_ROOT}/comprehensive,${OUTPUT_ROOT}/refcocog,${OUTPUT_ROOT}/plantwild_closed12" \
  --tag "comprehensive,refcocog,plantwild_closed12" \
  --base "${BASE_MODEL}" \
  --adapter "${CHECKPOINT}" \
  --calibrated-head "${CHECKPOINT}/discrim_head_calibrated.pt" \
  --backend hf \
  --dual-head \
  --faithful-dual-output \
  --dump-all-answer-candidates \
  --system-override decision \
  "$@"
