#!/usr/bin/env bash
set -eo pipefail

# Derive AAAI_CODE from the script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AAAI_CODE="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUNNER="$AAAI_CODE/experiments/gpt56_sol_l3/run_gpt56_sol_l3.py"
ANALYZER="$AAAI_CODE/experiments/gpt56_sol_l3/analyze_gpt56_sol_l3.py"

# DATA_ROOT defaults to benchmark annotations in the standard location
DATA_ROOT="${DATA_ROOT:-$AAAI_CODE/benchmark}"
REGATE_ROOT="${REGATE_ROOT:?REGate_ROOT must be set via environment}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set via environment}"
WORKERS="${WORKERS:-4}"

if [[ -z "${MODEL_API_KEY:-}" ]]; then
    echo "[FATAL] MODEL_API_KEY is unset"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

PROBES=(
    "L1_discrim|$DATA_ROOT/annotations/splits/test_L1_discrim.jsonl|$REGATE_ROOT/L1_discrim"
    "L2_evidence|$DATA_ROOT/annotations/splits/test_L2_evidence.jsonl|$REGATE_ROOT/L2_evidence"
    "L3_dose|$DATA_ROOT/annotations/splits/test_L3_dose.jsonl|$REGATE_ROOT/L3_dose"
    "test_full|$DATA_ROOT/annotations/splits/test.jsonl|$REGATE_ROOT/test_full"
    "ood_refcocog|$DATA_ROOT/annotations/splits/ood_refcocog.jsonl|$REGATE_ROOT/ood_refcocog"
    "plantwild_real_closed12_v1|$DATA_ROOT/annotations/splits/plantwild_real_closed12_paired_408.jsonl|$REGATE_ROOT/plantwild_real_closed12_v1"
)

for probe in "${PROBES[@]}"; do
    IFS="|" read -r tag data regate_dir <<< "$probe"
    probe_dir="$OUTPUT_ROOT/$tag"
    mkdir -p "$probe_dir"

    python "$RUNNER" \
        --data "$data" \
        --output "$probe_dir/api_outputs.jsonl" \
        --image-detail original \
        --max-completion-tokens 384 \
        --workers "$WORKERS"

    python "$ANALYZER" \
        --data "$data" \
        --api-output "$probe_dir/api_outputs.jsonl" \
        --regate-predictions "$regate_dir/predictions.jsonl" \
        --regate-metrics "$regate_dir/metrics.json" \
        --output-dir "$probe_dir/analysis"
done

echo "[OK] GPT-5.6 Sol six-probe evaluation complete: $OUTPUT_ROOT"