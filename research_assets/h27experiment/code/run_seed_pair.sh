#!/bin/bash
# Run both fixed construction conditions sequentially for one full-pipeline seed.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-}"
if [[ "$SEED" != "7" && "$SEED" != "21" && "$SEED" != "42" ]]; then
    echo "Usage: bash code/run_seed_pair.sh {7|21|42}"
    exit 2
fi

echo "[h27-seed-pair-start] seed=$SEED"
bash "$SCRIPT_DIR/run_condition.sh" grouped "$SEED" all
bash "$SCRIPT_DIR/run_condition.sh" independent "$SEED" all
echo "[h27-seed-pair-complete] seed=$SEED"
