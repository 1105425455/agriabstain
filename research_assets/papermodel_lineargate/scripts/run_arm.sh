#!/usr/bin/env bash
# Linear gate vs released MLP gate, measured on the paper's OWN Stage-1 models.
#
# Nothing here retrains Stage-1 and nothing refits the released MLP head: the adapters and
# their calibrated heads are read from byte-identical local copies under models/, and every
# artifact this script produces lands under papermodel_lineargate/.
#
# calibrate_head.py --mode train is deliberately never invoked: it writes a freshly fitted
# head back into the adapter directory, which would destroy the link to the published table.

set -eo pipefail

ARM="${1:?usage: run_arm.sh <grouped|independent> [stage]}"
STAGE="${2:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_ROOT/.." && pwd)"
AAAI_ROOT="$REPO_ROOT/aaai_code"

RUN_SEED=42
NPROC="${NPROC:-8}"
BASE_MODEL="${BASE_MODEL:-/user/huangyongbin/models/modelscope/Qwen/Qwen3-VL-8B-Instruct}"
DISCRIM_MUSD_PATH="$REPO_ROOT/final_story/outputs/discrim_musd.pt"

# Real-path data only. The aaai_code/benchmark tree is the anonymised release copy whose
# image fields were rewritten to placeholders like data/source/coco/images/... and must
# never be fed to a forward pass.
DEV_RAW="$REPO_ROOT/new_reason/outputs/splits/dev.jsonl"
TEST_DATA="$REPO_ROOT/new_reason/outputs/sft_3reason/sft_test.jsonl"
DATASET_TAG="comprehensive"

case "$ARM" in
grouped)
    ADAPTER="$EXP_ROOT/models/grouped"
    TRAIN_DATA="$REPO_ROOT/new_reason/outputs/sft_3reason/sft_train.json"
    TRAIN_SHA256="f6a0f922be69a5f5a585c89937ced60101a413c33bac7c6f98c6ff7967472663"
    # No cached feature set on disk reproduces this head's stored mu/sd, so the adapted
    # layer-27 features have to be extracted again.
    NEED_EXTRACT=1
    ;;
independent)
    ADAPTER="$EXP_ROOT/models/independent"
    TRAIN_DATA="$AAAI_ROOT/experiments/grouping_ablation/artifacts/data/sft_train_independent_negatives.json"
    TRAIN_SHA256="76709bd2012e0bd0e48f0f01eb05b50466000e49b028ccbc3d48113e301e8555"
    # The released calibration shards reproduce this head's stored mu/sd exactly, so they
    # are the very features the published head was fitted on; re-extraction would only add
    # bf16 noise. They are pre-staged into runs/independent/calibration/.
    NEED_EXTRACT=0
    ;;
*)
    echo "[FATAL] unknown arm: $ARM" >&2
    exit 2
    ;;
esac

# RUN_ROOT_OVERRIDE plus FORCE_EXTRACT lets a pipeline-fidelity check re-extract an arm into
# a scratch directory: extracting the independent arm and comparing against its released
# shards tells us whether this extraction reproduces the paper's, without ever touching the
# provenance-verified shards staged under runs/independent/.
RUN_ROOT="${RUN_ROOT_OVERRIDE:-$EXP_ROOT/runs/$ARM}"
[[ "$FORCE_EXTRACT" == "1" ]] && NEED_EXTRACT=1
CALIB_DIR="$RUN_ROOT/calibration"
LINEAR_DIR="$RUN_ROOT/linear_gate"
EVAL_DIR="$RUN_ROOT/evaluation/regate/$DATASET_TAG"
MLP_SCORE_DIR="$RUN_ROOT/mlp_evaluation/$DATASET_TAG"
LINEAR_EVAL_DIR="$RUN_ROOT/linear_evaluation/$DATASET_TAG"
LOG_DIR="$RUN_ROOT/logs"

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

fatal() {
    echo "[FATAL] $*" >&2
    exit 1
}

guard() {
    for path in "$BASE_MODEL" "$DISCRIM_MUSD_PATH" "$TRAIN_DATA" "$DEV_RAW" "$TEST_DATA"; do
        [[ -e "$path" ]] || fatal "missing input: $path"
    done

    # Refuse any data path from the anonymised release tree.
    for path in "$TRAIN_DATA" "$DEV_RAW" "$TEST_DATA"; do
        case "$path" in
        */aaai_code/benchmark/*) fatal "anonymised release data path: $path" ;;
        esac
    done

    # The adapter must be the isolated local copy, never the paper's checkpoint directory.
    case "$ADAPTER" in
    "$EXP_ROOT"/models/*) ;;
    *) fatal "adapter outside the isolated tree: $ADAPTER" ;;
    esac
    [[ -f "$ADAPTER/adapter_model.safetensors" ]] || fatal "no adapter weights in $ADAPTER"
    [[ -f "$ADAPTER/discrim_head_calibrated.pt" ]] || fatal "no calibrated head in $ADAPTER"

    # Every output directory must stay inside the isolated tree.
    for directory in "$CALIB_DIR" "$LINEAR_DIR" "$EVAL_DIR" "$MLP_SCORE_DIR" \
        "$LINEAR_EVAL_DIR" "$LOG_DIR"; do
        case "$directory" in
        "$EXP_ROOT"/*) ;;
        *) fatal "output directory escapes the isolated tree: $directory" ;;
        esac
    done

    local actual
    actual="$(sha256sum "$TRAIN_DATA" | cut -d' ' -f1)"
    [[ "$actual" == "$TRAIN_SHA256" ]] ||
        fatal "training data changed: $TRAIN_DATA has $actual, expected $TRAIN_SHA256"

    mkdir -p "$CALIB_DIR" "$LINEAR_DIR" "$EVAL_DIR" "$MLP_SCORE_DIR" \
        "$LINEAR_EVAL_DIR" "$LOG_DIR"
    echo "[guard] arm=$ARM adapter=$ADAPTER train=$(basename "$TRAIN_DATA") ok"
}

record_environment() {
    {
        date -Iseconds
        echo "arm=$ARM"
        echo "stage=$STAGE"
        echo "run_seed=$RUN_SEED"
        echo "adapter=$ADAPTER"
        echo "adapter_sha256=$(sha256sum "$ADAPTER/adapter_model.safetensors" | cut -d' ' -f1)"
        echo "head_sha256=$(sha256sum "$ADAPTER/discrim_head_calibrated.pt" | cut -d' ' -f1)"
        echo "train_data=$TRAIN_DATA"
        echo "train_data_sha256=$TRAIN_SHA256"
        echo "dev_raw_sha256=$(sha256sum "$DEV_RAW" | cut -d' ' -f1)"
        echo "test_data_sha256=$(sha256sum "$TEST_DATA" | cut -d' ' -f1)"
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
            --format=csv,noheader || echo "nvidia-smi unavailable"
    } | tee "$LOG_DIR/environment_$STAGE.log"

    python - <<'PY'
import subprocess

try:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception as error:  # noqa: BLE001 - reporting only, must not stop the run
    print(f"[gpu-start][WARN] cannot query GPUs: {error}", flush=True)
    rows = ""
for row in rows.splitlines():
    index, total, used = [int(value.strip()) for value in row.split(",")]
    ratio = used / total
    print(f"[gpu-start] gpu={index} memory={used}/{total} MiB ({ratio:.1%})", flush=True)
    if ratio > 0.90:
        print(f"[gpu-start][WARN] gpu={index} memory usage exceeds 90%", flush=True)
PY
}

# Cheap end-to-end plumbing check before the full run: a 16-record forward pass must
# produce the head_hidden and answer_candidate fields that the gate comparison needs.
smoke_test() {
    local smoke_dir="$RUN_ROOT/smoke/$DATASET_TAG"
    rm -rf "$smoke_dir"
    mkdir -p "$smoke_dir"
    torchrun --nproc_per_node="$NPROC" --master_port=29733 \
        "$AAAI_ROOT/evaluation/eval_qwen3vl_3reason.py" \
        --data "$TEST_DATA" \
        --base "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --calibrated-head "$ADAPTER/discrim_head_calibrated.pt" \
        --out "$smoke_dir" \
        --tag "$DATASET_TAG" \
        --backend hf \
        --dual-head \
        --faithful-dual-output \
        --dump-hidden \
        --dump-all-answer-candidates \
        --system-override decision \
        --limit 16 \
        --max-new-tokens 384 \
        --max-pixels 262144 2>&1 | tee "$LOG_DIR/smoke_eval.log"

    python - "$smoke_dir/predictions.jsonl" <<'PY'
import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
if not rows:
    raise SystemExit(f"[smoke][FATAL] no predictions in {path}")
first = rows[0]
missing = [
    field
    for field in ("head_hidden", "answer_candidate", "p_answer", "p_wo", "p_nt", "gt")
    if field not in first
]
if missing:
    raise SystemExit(f"[smoke][FATAL] predictions lack {missing}; fields={sorted(first)}")
width = len(first["head_hidden"])
if width != 4096:
    raise SystemExit(f"[smoke][FATAL] head_hidden width {width} != 4096")
print(f"[smoke] OK rows={len(rows)} head_hidden dim={width}", flush=True)
PY
}

extract_features() {
    if [[ "$NEED_EXTRACT" != "1" ]]; then
        echo "[extract] arm=$ARM reuses provenance-verified released shards; skipping"
        return 0
    fi
    torchrun --nproc_per_node="$NPROC" --master_port=29731 \
        "$SCRIPT_DIR/run_calibrate.py" \
        --run-seed "$RUN_SEED" \
        --mode extract \
        --base "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --sft-train "$TRAIN_DATA" \
        --dev "$DEV_RAW" \
        --dev-max 100 \
        --log-dir "$CALIB_DIR" 2>&1 | tee "$LOG_DIR/calibrate_extract.log"
}

verify_features() {
    python "$SCRIPT_DIR/check_features.py" \
        --calibration-dir "$CALIB_DIR" \
        --head "$ADAPTER/discrim_head_calibrated.pt" 2>&1 |
        tee "$LOG_DIR/check_features.log"
}

train_linear() {
    python "$SCRIPT_DIR/train_linear_gate.py" \
        --calibration-dir "$CALIB_DIR" \
        --output-dir "$LINEAR_DIR" \
        --run-seed "$RUN_SEED" 2>&1 | tee "$LOG_DIR/train_linear_gate.log"
}

evaluate_regate() {
    torchrun --nproc_per_node="$NPROC" --master_port=29732 \
        "$AAAI_ROOT/evaluation/eval_qwen3vl_3reason.py" \
        --data "$TEST_DATA" \
        --base "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --calibrated-head "$ADAPTER/discrim_head_calibrated.pt" \
        --out "$EVAL_DIR" \
        --tag "$DATASET_TAG" \
        --backend hf \
        --dual-head \
        --faithful-dual-output \
        --dump-hidden \
        --dump-all-answer-candidates \
        --system-override decision \
        --max-new-tokens 384 \
        --max-pixels 262144 2>&1 | tee "$LOG_DIR/eval_regate.log"
}

score_gates() {
    python "$SCRIPT_DIR/score_mlp.py" \
        --predictions "$EVAL_DIR/predictions.jsonl" \
        --dataset "$DATASET_TAG" \
        --output-dir "$MLP_SCORE_DIR" 2>&1 | tee "$LOG_DIR/score_mlp.log"
    python "$SCRIPT_DIR/reroute_linear.py" \
        --model "$LINEAR_DIR/linear_gate_best.npz" \
        --predictions "$EVAL_DIR/predictions.jsonl" \
        --dataset "$DATASET_TAG" \
        --output-dir "$LINEAR_EVAL_DIR" 2>&1 | tee "$LOG_DIR/reroute_linear.log"
}

guard
case "$STAGE" in
smoke) record_environment && smoke_test ;;
extract) record_environment && extract_features && verify_features ;;
verify) verify_features ;;
linear) train_linear ;;
eval) record_environment && evaluate_regate ;;
score) score_gates ;;
all)
    record_environment
    smoke_test
    extract_features
    verify_features
    train_linear
    evaluate_regate
    score_gates
    ;;
*)
    echo "Usage: run_arm.sh <grouped|independent>" \
        "{smoke|extract|verify|linear|eval|score|all}" >&2
    exit 2
    ;;
esac

touch "$RUN_ROOT/.complete-$STAGE"
echo "[complete] arm=$ARM stage=$STAGE run_root=$RUN_ROOT"
