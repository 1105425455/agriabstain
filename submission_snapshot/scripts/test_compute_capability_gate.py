"""Tests for scripts.compute_capability_gate.

Verifies that the per-label answer_rate, label_match_rate, correct_rate, and
mean_iou are computed correctly from the predictions JSONL schema written by
eval_qwen3vl_3reason.py, and that threshold-based exclusions work.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add aaai_code to path so that reason_abstain and scripts can be imported
_SCRIPT_DIR = Path(__file__).resolve().parent
_AAAI_CODE = _SCRIPT_DIR.parent  # scripts/.. = aaai_code
sys.path.insert(0, str(_AAAI_CODE))

# Explicitly import the module via its file path to avoid namespace package
# conflicts with the root-level scripts/ directory.
_MODULE_PATH = _SCRIPT_DIR / "compute_capability_gate.py"
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("compute_capability_gate", _MODULE_PATH)
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)
_summarize = _gate._summarize
_extract_label = _gate._extract_label
main = _gate.main


def _make_row(
    *,
    gt_decision: str = "answer",
    gt_label: str = "cat",
    gt_box: list[float] | None = None,
    gt_reason: str | None = None,
    pred_valid: bool = True,
    pred_decision: str = "answer",
    pred_label: str = "cat",
    pred_box: list[float] | None = None,
    pred_reason: str | None = None,
) -> dict:
    return {
        "gt": {
            "decision": gt_decision,
            "label": gt_label,
            "box": gt_box,
            "reason": gt_reason,
        },
        "pred": {
            "valid": pred_valid,
            "decision": pred_decision,
            "label": pred_label,
            "box": pred_box,
            "reason": pred_reason,
        },
    }


class TestExtractLabel:
    def test_from_gt(self):
        row = {"gt": {"label": "apple_rust"}}
        assert _extract_label(row) == "apple_rust"

    def test_fallback_flat(self):
        row = {"label": "cedar_rust"}
        assert _extract_label(row) == "cedar_rust"

    def test_unknown(self):
        row = {"gt": {}}
        assert _extract_label(row) == "UNKNOWN"


class TestSummarize:
    def test_empty(self):
        result = _summarize([], iou_threshold=0.5)
        assert result["n"] == 0
        assert result["answer_rate"] == 0.0

    def test_all_correct_answerable(self):
        rows = [
            _make_row(gt_decision="answer", gt_label="cat", gt_box=[0, 0, 100, 100],
                      pred_decision="answer", pred_label="cat", pred_box=[0, 0, 100, 100]),
            _make_row(gt_decision="answer", gt_label="dog", gt_box=[0, 0, 50, 50],
                      pred_decision="answer", pred_label="dog", pred_box=[0, 0, 50, 50]),
        ]
        result = _summarize(rows, iou_threshold=0.5)
        assert result["n"] == 2
        assert result["answer_rate"] == 1.0
        assert result["label_match_rate"] == 1.0
        assert result["correct_rate"] == 1.0
        assert result["mean_iou"] == 1.0

    def test_wrong_label(self):
        rows = [
            _make_row(gt_decision="answer", gt_label="cat", gt_box=[0, 0, 100, 100],
                      pred_decision="answer", pred_label="dog", pred_box=[0, 0, 100, 100]),
        ]
        result = _summarize(rows, iou_threshold=0.5)
        assert result["n"] == 1
        assert result["answer_rate"] == 1.0
        assert result["label_match_rate"] == 0.0
        assert result["correct_rate"] == 0.0
        assert result["mean_iou"] == 0.0

    def test_abstain_correct(self):
        rows = [
            _make_row(gt_decision="abstain", gt_reason="no_target",
                      pred_decision="abstain", pred_reason="no_target"),
        ]
        result = _summarize(rows, iou_threshold=0.5)
        assert result["n"] == 1
        assert result["answer_rate"] == 0.0
        assert result["correct_rate"] == 1.0

    def test_abstain_wrong_reason(self):
        rows = [
            _make_row(gt_decision="abstain", gt_reason="no_target",
                      pred_decision="abstain", pred_reason="wrong_object"),
        ]
        result = _summarize(rows, iou_threshold=0.5)
        assert result["n"] == 1
        assert result["correct_rate"] == 0.0

    def test_partial_iou(self):
        rows = [
            _make_row(gt_decision="answer", gt_label="cat", gt_box=[0, 0, 100, 100],
                      pred_decision="answer", pred_label="cat", pred_box=[0, 0, 50, 100]),
        ]
        result = _summarize(rows, iou_threshold=0.5)
        # IoU = (50*100)/(100*100) = 0.5
        assert result["mean_iou"] == pytest.approx(0.5)

    def test_invalid_pred(self):
        rows = [
            _make_row(gt_decision="answer", gt_label="cat",
                      pred_valid=False),
        ]
        result = _summarize(rows, iou_threshold=0.5)
        assert result["answer_rate"] == 0.0
        assert result["correct_rate"] == 0.0

    def test_mixed_correctness(self):
        """Mix of correct/wrong/abstain to verify all rates."""
        rows = [
            _make_row(gt_decision="answer", gt_label="cat", gt_box=[0, 0, 100, 100],
                      pred_decision="answer", pred_label="cat", pred_box=[0, 0, 100, 100]),
            _make_row(gt_decision="answer", gt_label="dog", gt_box=[0, 0, 100, 100],
                      pred_decision="answer", pred_label="wrong", pred_box=[0, 0, 100, 100]),
            _make_row(gt_decision="abstain", gt_reason="no_target",
                      pred_decision="abstain", pred_reason="no_target"),
            _make_row(gt_decision="abstain", gt_reason="no_target",
                      pred_decision="answer", pred_label="cat", pred_box=[0, 0, 100, 100]),
        ]
        result = _summarize(rows, iou_threshold=0.5)
        assert result["n"] == 4
        # answer_rate: 3/4 (rows 0, 1, 3)
        assert result["answer_rate"] == 0.75
        # label_match_rate: 3/4
        # row 0: cat==cat ✓, row 1: wrong!=dog ✗, row 2: no label (skipped),
        # row 3: cat==cat (default label matches) ✓
        assert result["label_match_rate"] == 0.75
        # correct_rate: 2/4 (row 0 correct, row 2 correct)
        assert result["correct_rate"] == 0.5
        # mean_iou: row 0 IoU=1.0, row 1 label wrong so no IoU added
        assert result["mean_iou"] == 1.0


def test_threshold_exclusion():
    """Verify that _summarize output can be used for threshold-based exclusion."""
    rows = [
        _make_row(gt_decision="answer", gt_label="good_class", gt_box=[0, 0, 100, 100],
                  pred_decision="answer", pred_label="good_class", pred_box=[0, 0, 100, 100]),
        _make_row(gt_decision="answer", gt_label="good_class", gt_box=[0, 0, 100, 100],
                  pred_decision="answer", pred_label="good_class", pred_box=[0, 0, 100, 100]),
        _make_row(gt_decision="answer", gt_label="bad_class", gt_box=[0, 0, 100, 100],
                  pred_decision="answer", pred_label="wrong", pred_box=[0, 0, 100, 100]),
        _make_row(gt_decision="answer", gt_label="bad_class", gt_box=[0, 0, 100, 100],
                  pred_decision="answer", pred_label="wrong", pred_box=[0, 0, 100, 100]),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f_pred:
        for row in rows:
            f_pred.write(json.dumps(row) + "\n")
        pred_path = f_pred.name

    out_path = Path(tempfile.mktemp(suffix=".json"))

    old_argv = sys.argv
    try:
        sys.argv = [
            "compute_capability_gate",
            "--predictions", pred_path,
            "--output", str(out_path),
            "--threshold", "0.5",
            "--iou-threshold", "0.5",
        ]
        main()
        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert result["num_samples"] == 4
        assert result["overall"]["correct_rate"] == 0.5
        assert "bad_class" in result["excluded_class_names"]
        assert "good_class" not in result["excluded_class_names"]
        assert result["by_label"]["good_class"]["correct_rate"] == 1.0
        assert result["by_label"]["bad_class"]["correct_rate"] == 0.0
    finally:
        sys.argv = old_argv
        Path(pred_path).unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])