"""Tests for reason_abstain.eval.metrics.

Covers the E2E_mIoU / E2E_mIoU_c keys and their backward-compatible aliases
A_Ground / A_Ground_c.
"""

from __future__ import annotations

import math

import pytest

from reason_abstain.eval.metrics import aggregate_metrics, metamorphic_test_satisfaction


def _sample(
    *,
    decision: str = "answer",
    label: str = "cat",
    box: list[float] | None = None,
    reason: str | None = None,
    valid: bool = True,
) -> dict:
    sample: dict = {
        "decision": decision,
        "label": label,
        "valid": valid,
    }
    if box is not None:
        sample["box"] = box
    if reason is not None:
        sample["reason"] = reason
    return sample


def _pred(
    *,
    decision: str = "answer",
    label: str = "cat",
    box: list[float] | None = None,
    reason: str | None = None,
    valid: bool = True,
) -> dict:
    pred: dict = {
        "decision": decision,
        "label": label,
        "valid": valid,
    }
    if box is not None:
        pred["box"] = box
    if reason is not None:
        pred["reason"] = reason
    return pred


class TestE2EMIoU:
    """Verify E2E_mIoU / E2E_mIoU_c equal A_Ground / A_Ground_c exactly."""

    def test_perfect_answerable(self):
        """All answerable answered correctly with perfect IoU."""
        pairs = [
            (_pred(label="cat", box=[0, 0, 100, 100]), _sample(label="cat", box=[0, 0, 100, 100])),
            (
                _pred(label="dog", box=[50, 50, 150, 150]),
                _sample(label="dog", box=[50, 50, 150, 150]),
            ),
        ]
        m = aggregate_metrics(pairs)
        gr = m["grounding_reason"]
        assert gr["E2E_mIoU"] == 1.0
        assert gr["E2E_mIoU_c"] == 1.0
        assert gr["E2E_mIoU"] == gr["A_Ground"]
        assert gr["E2E_mIoU_c"] == gr["A_Ground_c"]

    def test_half_iou_answerable(self):
        """Answerable with partial IoU."""
        pairs = [
            (_pred(label="cat", box=[0, 0, 100, 100]), _sample(label="cat", box=[0, 0, 100, 100])),
            (_pred(label="cat", box=[0, 0, 50, 100]), _sample(label="cat", box=[0, 0, 100, 100])),
        ]
        m = aggregate_metrics(pairs)
        gr = m["grounding_reason"]
        # box IOU of [0,0,50,100] vs [0,0,100,100] = (50*100)/(100*100) = 0.5
        assert gr["E2E_mIoU"] == pytest.approx(0.75)
        assert gr["E2E_mIoU_c"] == pytest.approx(0.75)
        assert gr["E2E_mIoU"] == gr["A_Ground"]
        assert gr["E2E_mIoU_c"] == gr["A_Ground_c"]

    def test_wrong_label_contributes_zero(self):
        """E2E_mIoU is label-gated; E2E_mIoU_c isolates answered-box quality."""
        pairs = [
            # label-correct, perfect box
            (_pred(label="cat", box=[0, 0, 100, 100]), _sample(label="cat", box=[0, 0, 100, 100])),
            # label-wrong, perfect box
            (_pred(label="dog", box=[0, 0, 100, 100]), _sample(label="cat", box=[0, 0, 100, 100])),
        ]
        m = aggregate_metrics(pairs)
        gr = m["grounding_reason"]
        # E2E_mIoU: (1.0 + 0.0) / 2 = 0.5
        # E2E_mIoU_c includes both answered boxes and intentionally ignores labels.
        assert gr["E2E_mIoU"] == pytest.approx(0.5)
        assert gr["E2E_mIoU_c"] == pytest.approx(1.0)
        assert gr["E2E_mIoU"] == gr["A_Ground"]
        assert gr["E2E_mIoU_c"] == gr["A_Ground_c"]

    def test_abstain_contributes_zero(self):
        """Abstaining on answerable contributes 0 to E2E_mIoU."""
        pairs = [
            (_pred(label="cat", box=[0, 0, 100, 100]), _sample(label="cat", box=[0, 0, 100, 100])),
            (
                _pred(decision="abstain", reason="no_target"),
                _sample(decision="answer", label="cat", box=[0, 0, 100, 100]),
            ),
        ]
        m = aggregate_metrics(pairs)
        gr = m["grounding_reason"]
        # E2E_mIoU = 1.0 / 2 = 0.5
        # E2E_mIoU_c = 1.0 / 1 = 1.0 (only the answered one counted)
        assert gr["E2E_mIoU"] == pytest.approx(0.5)
        assert gr["E2E_mIoU_c"] == pytest.approx(1.0)
        assert gr["E2E_mIoU"] == gr["A_Ground"]
        assert gr["E2E_mIoU_c"] == gr["A_Ground_c"]

    def test_no_box_label_only(self):
        """Label-only sample (no GT box) — decision+label correct counts as
        correct for E2E_mIoU but IoU contribution is 0."""
        pairs = [
            # No GT box → score_answerable returns True (decision+label match)
            (_pred(label="cat", box=[10, 10, 50, 50]), _sample(label="cat")),
            # Wrong label
            (_pred(label="dog", box=[10, 10, 50, 50]), _sample(label="cat")),
        ]
        m = aggregate_metrics(pairs)
        gr = m["grounding_reason"]
        # Both answered answerable. n_answerable=2, n_answered_on_answerable=2.
        # pair 1: no GT box → score_answerable passes (decision+label OK) → iou considered,
        #          but sample has no box so iou=0. label correct → both sums get 0.
        # pair 2: wrong label → iou_sum_all NOT incremented, iou_sum_answered gets iou=0
        #   iou_sum_all = 0, iou_sum_answered = 0
        #   E2E_mIoU = 0/2 = 0, E2E_mIoU_c = 0/2 = 0
        assert gr["E2E_mIoU"] == pytest.approx(0.0)
        assert gr["E2E_mIoU_c"] == pytest.approx(0.0)
        assert gr["E2E_mIoU"] == gr["A_Ground"]
        assert gr["E2E_mIoU_c"] == gr["A_Ground_c"]

    def test_mixed_answerable_unanswerable(self):
        """Mix of answerable and unanswerable samples; unanswerable do not
        affect grounding metrics."""
        pairs = [
            # answerable, correct
            (_pred(label="cat", box=[0, 0, 100, 100]), _sample(label="cat", box=[0, 0, 100, 100])),
            # unanswerable — doesn't affect grounding
            (
                _pred(decision="abstain", reason="no_target"),
                _sample(decision="abstain", reason="no_target"),
            ),
        ]
        m = aggregate_metrics(pairs)
        gr = m["grounding_reason"]
        assert gr["E2E_mIoU"] == 1.0
        assert gr["E2E_mIoU_c"] == 1.0
        assert gr["E2E_mIoU"] == gr["A_Ground"]
        assert gr["E2E_mIoU_c"] == gr["A_Ground_c"]

    def test_no_answerable_samples(self):
        """No answerable samples — E2E_mIoU should be 0 (safe_div with
        denominator 0)."""
        pairs = [
            (
                _pred(decision="abstain", reason="no_target"),
                _sample(decision="abstain", reason="no_target"),
            ),
        ]
        m = aggregate_metrics(pairs)
        gr = m["grounding_reason"]
        assert math.isnan(gr["E2E_mIoU"]) or gr["E2E_mIoU"] == 0.0
        assert math.isnan(gr["E2E_mIoU_c"]) or gr["E2E_mIoU_c"] == 0.0
        assert gr["E2E_mIoU"] == gr["A_Ground"]
        assert gr["E2E_mIoU_c"] == gr["A_Ground_c"]

    def test_all_keys_present(self):
        """All expected keys are present in grounding_reason."""
        pairs = [
            (_pred(label="cat", box=[0, 0, 100, 100]), _sample(label="cat", box=[0, 0, 100, 100])),
        ]
        m = aggregate_metrics(pairs)
        gr = m["grounding_reason"]
        expected_keys = {
            "E2E_mIoU",
            "E2E_mIoU_c",
            "A_Ground",
            "A_Ground_c",
            "U_Reason",
            "U_Reason_c",
        }
        assert set(gr.keys()) == expected_keys, f"Got keys: {set(gr.keys())}"


def test_invalid_outputs_use_fixed_gt_denominators():
    pairs = [
        (
            _pred(label="cat", box=[0, 0, 100, 100]),
            _sample(label="cat", box=[0, 0, 100, 100]),
        ),
        (
            _pred(valid=False),
            _sample(label="cat", box=[0, 0, 100, 100]),
        ),
        (
            _pred(decision="answer", label="cat", box=[0, 0, 100, 100]),
            _sample(decision="abstain", reason="no_target"),
        ),
        (
            _pred(valid=False),
            _sample(decision="abstain", reason="no_target"),
        ),
    ]

    metrics = aggregate_metrics(pairs)

    assert metrics["counts"] == {
        "total": 4,
        "answerable": 2,
        "unanswerable": 2,
        "invalid": 2,
        "invalid_answerable": 1,
        "invalid_unanswerable": 1,
    }
    assert metrics["headline"]["FCA"] == pytest.approx(0.5)
    assert metrics["safeguard"]["A_FU"] == pytest.approx(0.5)
    assert metrics["safeguard"]["Supported_Retention"] == pytest.approx(0.5)
    assert metrics["safeguard"]["U_Abs"] == pytest.approx(0.0)
    assert metrics["selection_confusion_matrix"] == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}
    assert metrics["safeguard"]["MCC"] == pytest.approx(0.0)
    assert metrics["semantic"]["Strict_3"] == pytest.approx(0.25)
    assert metrics["grounding_reason"]["E2E_mIoU"] == pytest.approx(0.5)


def test_mts_uses_sample_rates_then_relation_macro_average():
    def relation_sample(relation: str) -> dict:
        sample = _sample(label="cat", box=[0, 0, 100, 100])
        sample["metadata"] = {"relation": relation}
        return sample

    correct = _pred(label="cat", box=[0, 0, 100, 100])
    incorrect = _pred(decision="abstain", reason="no_target")
    pairs = [
        (correct, relation_sample("r1")),
        (correct, relation_sample("r1")),
        (incorrect, relation_sample("r1")),
        (incorrect, relation_sample("r2")),
        (incorrect, relation_sample("base")),
    ]

    mts = metamorphic_test_satisfaction(pairs)

    assert mts["per_relation"] == pytest.approx({"r1": 2 / 3, "r2": 0.0})
    assert mts["per_relation_counts"] == {
        "r1": {"pass": 2, "total": 3},
        "r2": {"pass": 0, "total": 1},
    }
    assert mts["MTS_macro"] == pytest.approx(1 / 3)
    assert mts["MTS_micro"] == pytest.approx(1 / 2)
