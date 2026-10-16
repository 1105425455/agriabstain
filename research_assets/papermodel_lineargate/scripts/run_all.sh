#!/usr/bin/env bash
# Entry point for the GPU job: linear gate vs released MLP gate on both paper arms.
#
# The independent arm runs first because its calibration features are already on disk and
# provenance-verified, so it reaches a complete result quickly and doubles as a check that
# the re-evaluation reproduces the published ablation row before the expensive grouped
# extraction starts.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

status_independent="not_run"
status_grouped="not_run"

if bash "$SCRIPT_DIR/run_arm.sh" independent all; then
    status_independent="ok"
else
    status_independent="failed"
    echo "[run-all][ERROR] independent arm failed" >&2
fi

if bash "$SCRIPT_DIR/run_arm.sh" grouped all; then
    status_grouped="ok"
else
    status_grouped="failed"
    echo "[run-all][ERROR] grouped arm failed" >&2
fi

python "$SCRIPT_DIR/summarize.py" \
    --runs-root "$EXP_ROOT/runs" \
    --output "$EXP_ROOT/summary/gate_comparison.json" || true

echo "[run-all] independent=$status_independent grouped=$status_grouped"
[[ "$status_independent" == "ok" && "$status_grouped" == "ok" ]]
