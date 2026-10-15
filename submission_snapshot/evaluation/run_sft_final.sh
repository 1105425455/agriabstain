#!/bin/bash
# Evaluate a checkpoint on every probe: full test, L1, L2, L3, RefCOCOg and the 408-record PlantWild closed set.
#
# Usage:
#   BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct \
#   ADAPTER=/path/to/checkpoint-444 \
#   CALIBRATED_HEAD=/path/to/discrim_head_calibrated.pt \
#   bash evaluation/run_sft_final.sh /path/to/output_dir
#
# Every argument may also be passed positionally:
#   bash evaluation/run_sft_final.sh <base_model> <adapter> <calibrated_head> <output_dir>
#
# Uses 4 GPUs by default; override with NPROC.
# Data is read from benchmark/annotations/eval/ inside this package.
set -eo pipefail

# ==============================================================================
# Derive the package root from the location of this script
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AAAI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_qwen3vl_3reason.py"
BENCH_EVAL="$AAAI_ROOT/benchmark/annotations/eval"
BENCH_SFT="$AAAI_ROOT/benchmark/annotations/sft"

# ==============================================================================
# Arguments: environment variables take precedence over positional parameters
# ==============================================================================
BASE_MODEL="${BASE_MODEL:-${1:-}}"
ADAPTER="${ADAPTER:-${2:-}}"
CALIBRATED_HEAD="${CALIBRATED_HEAD:-${3:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${4:-}}"

if [ -z "$BASE_MODEL" ]; then
    echo "[FATAL] set BASE_MODEL, either as an environment variable or as argument 1"
    exit 1
fi
if [ -z "$ADAPTER" ]; then
    echo "[FATAL] set ADAPTER, either as an environment variable or as argument 2"
    exit 1
fi
if [ -z "$CALIBRATED_HEAD" ]; then
    echo "[FATAL] set CALIBRATED_HEAD, either as an environment variable or as argument 3, pointing at discrim_head_calibrated.pt"
    exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "[FATAL] set OUTPUT_DIR, either as an environment variable or as argument 4"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$AAAI_ROOT"

# Verify that every required file is present
for f in "$EVAL_SCRIPT" \
    "$BENCH_SFT/sft_test.jsonl" \
    "$BENCH_EVAL/test_L1_discrim.jsonl" \
    "$BENCH_EVAL/test_L2_evidence.jsonl" \
    "$BENCH_EVAL/test_L3_dose.jsonl" \
    "$BENCH_EVAL/ood_refcocog.jsonl" \
    "$BENCH_EVAL/plantwild_real_closed12_paired_408.jsonl"; do
    if [ ! -f "$f" ]; then
        echo "[FATAL] file not found: $f"
        exit 1
    fi
done

N_PLANTWILD=$(wc -l < "$BENCH_EVAL/plantwild_real_closed12_paired_408.jsonl")
if [ "$N_PLANTWILD" -ne 408 ]; then
    echo "[FATAL] the PlantWild closed set must hold 408 records, found $N_PLANTWILD"
    exit 1
fi

# Verify the calibrated head
if [ ! -f "$CALIBRATED_HEAD" ]; then
    echo "[FATAL] CALIBRATED_HEAD not found: $CALIBRATED_HEAD"
    exit 1
fi

NPROC=${NPROC:-4}

# Echo the resolved arguments
echo "============================================"
echo "[run_sft_final] evaluation"
echo "============================================"
echo "  BASE_MODEL=$BASE_MODEL"
echo "  ADAPTER=$ADAPTER"
echo "  CALIBRATED_HEAD=$CALIBRATED_HEAD"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  NPROC=$NPROC"
echo "============================================"

# ==============================================================================
# Probes, given as data_path:out_subdir:tag
# ==============================================================================
PROBES=(
    "$BENCH_SFT/sft_test.jsonl:full_test:full_test"
    "$BENCH_EVAL/test_L1_discrim.jsonl:L1:test_L1"
    "$BENCH_EVAL/test_L2_evidence.jsonl:L2:test_L2"
    "$BENCH_EVAL/test_L3_dose.jsonl:L3:test_L3"
    "$BENCH_EVAL/ood_refcocog.jsonl:RefCOCOg:refcocog"
    "$BENCH_EVAL/plantwild_real_closed12_paired_408.jsonl:PlantWild:plantwild_closed12"
)

for probe in "${PROBES[@]}"; do
    IFS=':' read -r data_path subdir tag <<< "$probe"
    OUT_PATH="$OUTPUT_DIR/$subdir"

    echo ""
    echo "--- evaluating: $tag ($data_path) -> $OUT_PATH ---"

    torchrun --nproc_per_node="$NPROC" --master_port=$(shuf -i 20001-29999 -n 1) \
        "$EVAL_SCRIPT" \
        --backend hf \
        --base "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --calibrated-head "$CALIBRATED_HEAD" \
        --data "$data_path" \
        --out "$OUT_PATH" \
        --tag "$tag" \
        --dual-head \
        --faithful-dual-output \
        --system-override decision \
        --max-new-tokens 384 \
        --max-pixels 262144

    echo "  done: $tag"
done

echo ""
echo "============================================"
echo "[run_sft_final] all probes complete"
echo "  output directory: $OUTPUT_DIR"
echo "============================================"
echo ""
echo "Summary:"
for probe in "${PROBES[@]}"; do
    IFS=':' read -r _data_path subdir tag <<< "$probe"
    SUMMARY="$OUTPUT_DIR/$subdir/summary.txt"
    if [ -f "$SUMMARY" ]; then
        echo "----- $tag -----"
        head -20 "$SUMMARY"
        echo ""
    fi
done
