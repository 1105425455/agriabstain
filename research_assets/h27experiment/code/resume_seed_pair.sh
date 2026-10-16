#!/bin/bash
# Resume after grouped Stage-1 training/checkpoint selection has completed.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEED="${1:-}"
if [[ "$SEED" != "7" && "$SEED" != "21" && "$SEED" != "42" ]]; then
    echo "Usage: bash code/resume_seed_pair.sh {7|21|42}"
    exit 2
fi

SELECTION="$EXPERIMENT_ROOT/runs/grouped/seed-$SEED/selection/checkpoint_selection.json"
if [[ ! -f "$SELECTION" ]]; then
    echo "[h27][FATAL] grouped checkpoint selection is missing: $SELECTION"
    exit 1
fi

echo "[h27-seed-pair-resume] seed=$SEED stage=grouped-calibration"
bash "$SCRIPT_DIR/run_condition.sh" grouped "$SEED" calibrate
bash "$SCRIPT_DIR/run_condition.sh" grouped "$SEED" linear
bash "$SCRIPT_DIR/run_condition.sh" grouped "$SEED" eval
bash "$SCRIPT_DIR/run_condition.sh" grouped "$SEED" reroute

echo "[h27-seed-pair-resume] seed=$SEED stage=independent-all"
bash "$SCRIPT_DIR/run_condition.sh" independent "$SEED" all
echo "[h27-seed-pair-complete] seed=$SEED"
