from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.base_native_regate.rescore_metrics import build_result, rescore_run


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_rescore_uses_fixed_gt_denominator(tmp_path: Path) -> None:
    supported_gt = {"decision": "answer", "label": "corn_rust_leaf", "box": [0, 0, 10, 10]}
    records = [
        {
            "pred": {
                "valid": True,
                "decision": "answer",
                "label": "corn rust",
                "box": [0, 0, 10, 10],
            },
            "gt": supported_gt,
        },
        {"pred": {"valid": False}, "gt": supported_gt},
        {
            "pred": {"valid": True, "decision": "abstain", "reason": "no_target"},
            "gt": {"decision": "abstain", "reason": "no_target"},
        },
    ]
    _write_jsonl(tmp_path / "predictions.jsonl", records)
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")

    result = build_result(tmp_path)
    assert result["metrics"]["safeguard"]["Supported_Retention"] == 0.5
    assert result["metrics"]["grounding_reason"]["E2E_mIoU"] == 0.5
    assert result["metrics"]["semantic"]["Strict_3"] == 2 / 3

    assert not rescore_run(tmp_path, check=False)
    assert rescore_run(tmp_path, check=True)
