#!/bin/bash
# Resume after grouped MLP evaluation completed but linear rerouting failed.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-}"
if [[ "$SEED" != "7" && "$SEED" != "21" && "$SEED" != "42" ]]; then
    echo "Usage: bash code/resume_after_reroute.sh {7|21|42}"
    exit 2
fi

echo "[h27-seed-pair-resume] seed=$SEED stage=grouped-linear-reroute"
bash "$SCRIPT_DIR/run_condition.sh" grouped "$SEED" reroute

echo "[h27-seed-pair-resume] seed=$SEED stage=independent-all"
bash "$SCRIPT_DIR/run_condition.sh" independent "$SEED" all
echo "[h27-seed-pair-complete] seed=$SEED"
