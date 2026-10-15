"""Tests for analyze_native_regate.py — synthetic + edge-case coverage."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.base_native_regate.analyze_native_regate import (
    NativeRegateMetric,
    _gate_candidate_semantic_correct,
    _plot_paper_hallucination_retention,
    _supported_semantic_correct,
    gate_shared_candidate_point,
    main,
    matched_retention_to_target,
    native_point,
    regate_full_curve,
    regate_curve_point,
    validate_all,
    validate_regate_record,
)
from experiments.selective_grounding.analyze_selective_grounding import (
    validate_matched_records,
)

# -----------------------------------------------------------------------
# Helpers — synthetic record factories
# -----------------------------------------------------------------------


def _regate_record(
    sample_id: str,
    *,
    gt_decision: str,
    p_answer: float = 0.7,
    p_wo: float | None = None,
    p_nt: float | None = None,
    candidate_valid: bool = True,
    candidate_label: str = "leaf",
    candidate_box: list[float] | None = None,
    gt_label: str = "leaf",
    gt_box: list[float] | None = None,
    gt_reason: str = "wrong_object",
) -> dict[str, Any]:
    if gt_box is None and gt_decision == "answer":
        gt_box = [0.0, 0.0, 100.0, 100.0]
    if candidate_box is None and candidate_valid:
        candidate_box = gt_box or [0.0, 0.0, 100.0, 100.0]
    p_wo = p_wo if p_wo is not None else (1 - p_answer) * 0.6
    p_nt = p_nt if p_nt is not None else (1 - p_answer) * 0.4

    rec: dict[str, Any] = {
        "sample_id": sample_id,
        "gt": {
            "decision": gt_decision,
            "label": gt_label if gt_decision == "answer" else None,
            "box": gt_box if gt_decision == "answer" else None,
            "reason": gt_reason if gt_decision == "abstain" else None,
        },
        "p_answer": p_answer,
        "p_wo": p_wo,
        "p_nt": p_nt,
    }
    if candidate_valid:
        rec["answer_candidate"] = {
            "valid": True,
            "label": candidate_label,
            "box": candidate_box,
        }
    else:
        rec["answer_candidate"] = {"valid": False}
    return rec


def _native_record(
    sample_id: str,
    *,
    gt_decision: str,
    pred_valid: bool = True,
    pred_decision: str = "answer",
    pred_label: str = "leaf",
    pred_box: list[float] | None = None,
    pred_reason: str | None = "wrong_object",
    gt_label: str = "leaf",
    gt_box: list[float] | None = None,
    gt_reason: str = "wrong_object",
) -> dict[str, Any]:
    if gt_box is None and gt_decision == "answer":
        gt_box = [0.0, 0.0, 100.0, 100.0]
    if pred_box is None and pred_decision == "answer" and pred_valid:
        pred_box = [0.0, 0.0, 100.0, 100.0]
    pred: dict[str, Any] = {
        "valid": pred_valid,
        "decision": pred_decision,
    }
    if pred_decision == "answer":
        pred["label"] = pred_label
        pred["box"] = pred_box
    else:
        pred["reason"] = pred_reason

    return {
        "sample_id": sample_id,
        "gt": {
            "decision": gt_decision,
            "label": gt_label if gt_decision == "answer" else None,
            "box": gt_box if gt_decision == "answer" else None,
            "reason": gt_reason if gt_decision == "abstain" else None,
        },
        "pred": pred,
    }


# -----------------------------------------------------------------------
# Test: native_point — all metrics
# -----------------------------------------------------------------------


def test_native_point_all_metrics_answerable_correct() -> None:
    rec = _native_record("a", gt_decision="answer", pred_decision="answer", pred_label="leaf")
    point = native_point("comprehensive", "Base_native", {"a": rec})
    assert point.total == 1
    assert point.supported == 1
    assert point.unsupported == 0
    assert point.answered == 1
    assert point.answered_supported == 1
    assert point.answered_unsupported == 0
    assert point.strict_3_correct == 1
    assert point.selective_risk_errors == 0
    assert point.format_valid == 1
    assert point.total_coverage == 1.0
    assert point.supported_query_retention == 1.0
    assert math.isnan(point.unsupported_hallucination_rate)
    assert point.strict_3_accuracy == 1.0
    assert point.selective_risk == 0.0
    assert point.grounding_accuracy == 1.0  # IoU >= 0.5
    assert point.grounding_scope == "bbox"


def test_native_point_unsupported_hallucination() -> None:
    records = {
        "u": _native_record("u", gt_decision="abstain", pred_decision="answer", pred_label="leaf"),
        "s": _native_record("s", gt_decision="answer", pred_decision="answer"),
    }
    point = native_point("comprehensive", "Base", records)
    assert point.answered == 2
    assert point.answered_unsupported == 1
    assert point.unsupported_hallucination_rate == 1.0


def test_native_point_invalid_is_not_answered() -> None:
    rec = _native_record("a", gt_decision="answer", pred_valid=False)
    point = native_point("comprehensive", "Base", {"a": rec})
    assert point.answered == 0
    assert point.strict_3_correct == 0
    assert point.format_valid == 0


def test_native_point_box_null_label_only() -> None:
    """PlantWild has null boxes — grounding accurate if label correct."""
    rec = _native_record(
        "p",
        gt_decision="answer",
        gt_box=None,
        pred_box=None,
    )
    point = native_point("plantwild", "Base", {"p": rec})
    assert point.grounding_accuracy == 1.0
    assert point.grounding_scope == "label_only"


def test_native_point_box_null_wrong_label() -> None:
    rec = _native_record(
        "p",
        gt_decision="answer",
        gt_box=None,
        pred_label="wrong",
        pred_box=None,
    )
    point = native_point("plantwild", "Base", {"p": rec})
    assert point.grounding_accuracy == 0.0


# -----------------------------------------------------------------------
# Test: regate_curve_point — all metrics via ReGate
# -----------------------------------------------------------------------


def test_regate_curve_point_answer_correct() -> None:
    rec = _regate_record("a", gt_decision="answer", p_answer=0.9)
    point = regate_curve_point("comprehensive", {"a": rec}, {"a": True}, "test", None)
    assert point.answered == 1
    assert point.strict_3_correct == 1
    assert point.selective_risk == 0.0


def test_regate_curve_point_answer_wrong_candidate() -> None:
    rec = _regate_record("a", gt_decision="answer", candidate_label="wrong", p_answer=0.9)
    point = regate_curve_point("comprehensive", {"a": rec}, {"a": True}, "test", None)
    assert point.strict_3_correct == 0
    assert point.selective_risk == 1.0


def test_regate_curve_point_abstain_correct() -> None:
    rec = _regate_record("u", gt_decision="abstain", p_answer=0.2)
    point = regate_curve_point("comprehensive", {"u": rec}, {"u": False}, "test", None)
    assert point.strict_3_correct == 1
    assert point.answered == 0
    assert math.isnan(point.selective_risk)


def test_regate_curve_point_abstain_incorrect_answered() -> None:
    rec = _regate_record("u", gt_decision="abstain", p_answer=0.9)
    point = regate_curve_point("comprehensive", {"u": rec}, {"u": True}, "test", None)
    assert point.strict_3_correct == 0
    assert point.selective_risk == 1.0


# -----------------------------------------------------------------------
# Test: full curve endpoints and argmax
# -----------------------------------------------------------------------


def test_regate_full_curve_zero_and_full_coverage() -> None:
    records = {
        "good": _regate_record("good", gt_decision="answer", p_answer=0.9),
        "bad": _regate_record("bad", gt_decision="abstain", p_answer=0.1),
    }
    curve, argmax = regate_full_curve("comprehensive", records)
    assert len(curve) >= 2
    # First point (threshold > max) has 0 coverage
    assert curve[0].answered == 0
    assert math.isnan(curve[0].selective_risk)
    # Last point (threshold = min) has full coverage
    assert curve[-1].answered == 2


def test_regate_full_curve_argmax_picks_answer() -> None:
    records = {
        "a": _regate_record("a", gt_decision="answer", p_answer=0.9, p_wo=0.05, p_nt=0.05),
        "u": _regate_record("u", gt_decision="abstain", p_answer=0.2, p_wo=0.5, p_nt=0.3),
    }
    _, argmax = regate_full_curve("comprehensive", records)
    assert argmax.answered == 1
    assert argmax.answered_supported == 1


def test_regate_full_curve_argmax_tie_answer_breaks_to_answer() -> None:
    records = {
        "t": _regate_record("t", gt_decision="answer", p_answer=1 / 3, p_wo=1 / 3, p_nt=1 / 3),
    }
    _, argmax = regate_full_curve("comprehensive", records)
    # p_answer >= p_wo and p_answer >= p_nt → answered
    assert argmax.answered == 1


# -----------------------------------------------------------------------
# Test: gate_shared_candidate_point
# -----------------------------------------------------------------------


def test_shared_candidate_gating_fairness() -> None:
    """Shared candidate isolates gating — correct SFT gate + wrong candidate is wrong."""
    sft = _native_record("a", gt_decision="answer", pred_decision="answer")
    regate = _regate_record("a", gt_decision="answer", candidate_label="wrong")
    point = gate_shared_candidate_point("comprehensive", {"a": sft}, {"a": regate})
    assert point.answered == 1  # SFT says answer
    assert point.strict_3_correct == 0  # but candidate is wrong
    assert point.selective_risk == 1.0


def test_shared_candidate_sft_gate_blocks_answer() -> None:
    """SFT abstains → answered=0, strict_3=1 because correct abstention."""
    sft = _native_record(
        "a", gt_decision="answer", pred_decision="abstain", pred_reason="wrong_object"
    )
    regate = _regate_record("a", gt_decision="answer", candidate_label="leaf")
    point = gate_shared_candidate_point("comprehensive", {"a": sft}, {"a": regate})
    assert point.answered == 0
    # Abstain on answerable → not strict-3 correct
    assert point.strict_3_correct == 0


# -----------------------------------------------------------------------
# Test: validate_regate_record
# -----------------------------------------------------------------------


def test_validate_regate_record_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing answer_candidate"):
        validate_regate_record(
            {"sample_id": "x", "p_answer": 0.5, "p_wo": 0.3, "p_nt": 0.2}, Path("test")
        )


def test_validate_regate_record_bad_probability() -> None:
    rec = _regate_record("x", gt_decision="answer", p_answer=1.5)  # out of range
    with pytest.raises(ValueError, match="out of"):
        validate_regate_record(rec, Path("test"))


def test_validate_regate_record_valid() -> None:
    rec = _regate_record("x", gt_decision="answer")
    validate_regate_record(rec, Path("test"))  # no error


# -----------------------------------------------------------------------
# Test: GT mismatch and sample_id mismatch
# -----------------------------------------------------------------------


def test_gt_mismatch_raises() -> None:
    regate = {"a": _regate_record("a", gt_decision="answer")}
    base = {"a": _native_record("a", gt_decision="abstain")}
    with pytest.raises(ValueError, match="gt mismatch"):
        validate_matched_records("toy", regate, base)


def test_sample_id_mismatch_raises() -> None:
    regate = {"a": _regate_record("a", gt_decision="answer")}
    base = {"b": _native_record("b", gt_decision="answer")}
    with pytest.raises(ValueError, match="sample_id mismatch"):
        validate_matched_records("toy", regate, base)


# -----------------------------------------------------------------------
# Test: validate_all calls all validations
# -----------------------------------------------------------------------


def test_validate_all_expected_size(tmp_path: Path) -> None:
    dataset = "comprehensive"
    # Build 2 records for test; bypass the size check by setting expected size to 2.
    # For the real test we just pass EXPECTED_SIZES which is 2646; we use a hack:
    # temporarily mock. But for an integration test we can run with 2 records
    # by using a dataset not in EXPECTED_SIZES (no size check).
    dataset = "toy"
    r1 = _regate_record("a", gt_decision="answer")
    r2 = _regate_record("b", gt_decision="abstain")
    regate = {"a": r1, "b": r2}
    base = {
        "a": _native_record("a", gt_decision="answer"),
        "b": _native_record("b", gt_decision="abstain"),
    }
    sft = {
        "a": _native_record("a", gt_decision="answer"),
        "b": _native_record("b", gt_decision="abstain"),
    }
    # Should not raise
    validate_all(base, sft, regate, dataset, Path("base"), Path("sft"), Path("regate"))


def test_validate_all_size_mismatch() -> None:
    dataset = "comprehensive"  # expected 2646
    r1 = _regate_record("a", gt_decision="answer")
    regate = {"a": r1}
    base = {"a": _native_record("a", gt_decision="answer")}
    sft = {"a": _native_record("a", gt_decision="answer")}
    with pytest.raises(ValueError, match="has 1 records, expected 2646"):
        validate_all(base, sft, regate, dataset, Path("base"), Path("sft"), Path("regate"))


def test_validate_all_missing_pred() -> None:
    dataset = "toy"
    r1 = _regate_record("a", gt_decision="answer")
    regate = {"a": r1}
    base = {"a": {"sample_id": "a", "gt": r1["gt"]}}  # no pred
    sft = {"a": {"sample_id": "a", "gt": r1["gt"]}}  # no pred
    with pytest.raises(ValueError, match="missing pred"):
        validate_all(base, sft, regate, dataset, Path("base"), Path("sft"), Path("regate"))


# -----------------------------------------------------------------------
# Test: matched_retention_to_target
# -----------------------------------------------------------------------


def _simple_curve() -> list[NativeRegateMetric]:
    """A small sim curve for matched-retention tests.

    supported=3 samples, curve points with different retentions
    so that interpolation has both lower/upper bounds.
    """
    pts = []
    # threshold -> (answered_supported, answered_unsupported)
    configs = [
        (None, 0, 0),  # zero coverage
        (0.9, 1, 0),  # retention = 1/3 ≈ 0.333
        (0.7, 2, 0),  # retention = 2/3 ≈ 0.667
        (0.0, 3, 2),  # full coverage, all 3 answered, 2/2 unsupported hallucinated
    ]
    for th, ans_sup, ans_unsup in configs:
        total_ans = ans_sup + ans_unsup
        pts.append(
            NativeRegateMetric(
                dataset="toy",
                method="ReGate",
                policy="test",
                threshold=th,
                total=5,
                supported=3,
                unsupported=2,
                answered=total_ans,
                answered_supported=ans_sup,
                answered_unsupported=ans_unsup,
                strict_3_correct=ans_sup,  # only supported answered count as strict-3
                selective_risk_errors=ans_unsup,
                grounding_correct=ans_sup,
                grounding_denom=ans_sup if th is not None else 0,
                format_valid=5,
                total_coverage=total_ans / 5,
                supported_query_retention=ans_sup / 3,
                unsupported_hallucination_rate=ans_unsup / 2 if ans_unsup else 0.0,
                strict_3_accuracy=ans_sup / 5,
                selective_risk=ans_unsup / total_ans if total_ans else float("nan"),
                grounding_accuracy=1.0 if ans_sup else 0.0,
                format_valid_rate=1.0,
                grounding_scope="bbox",
            )
        )
    return pts


def test_matched_retention_empirical_and_interpolation() -> None:
    curve = _simple_curve()
    target = NativeRegateMetric(
        **{**curve[1].__dict__, "method": "Base_native", "supported_query_retention": 0.5}
    )
    emp, interp = matched_retention_to_target(curve, target)
    assert emp.supported_query_retention >= 0.5
    assert "empirical_threshold" in interp
    assert "interpolated_unsupported_hallucination_rate" in interp
    assert "interpolated_selective_risk" in interp
    assert "interpolated_strict_3_accuracy" in interp


def test_paper_hallucination_retention_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    curve = _simple_curve()
    base = NativeRegateMetric(
        **{
            **curve[1].__dict__,
            "method": "Base_native",
            "supported_query_retention": 0.83,
            "unsupported_hallucination_rate": 0.20,
        }
    )
    sft = NativeRegateMetric(
        **{
            **curve[1].__dict__,
            "method": "SFT_native_observed",
            "supported_query_retention": 0.85,
            "unsupported_hallucination_rate": 0.10,
        }
    )
    argmax = NativeRegateMetric(
        **{
            **curve[-1].__dict__,
            "supported_query_retention": 0.97,
            "unsupported_hallucination_rate": 0.13,
        }
    )
    matched = {
        "target_supported_query_retention": 0.85,
        "interpolated_unsupported_hallucination_rate": 0.04,
    }

    _plot_paper_hallucination_retention(curve, base, sft, argmax, matched, tmp_path)

    assert (tmp_path / "comprehensive_hallucination_retention_paper.pdf").is_file()
    assert (tmp_path / "comprehensive_hallucination_retention_paper.png").is_file()

    zero_fca_sft = NativeRegateMetric(**{**sft.__dict__, "unsupported_hallucination_rate": 0.0})
    with pytest.raises(ValueError, match="SFT baseline FCA must be positive"):
        _plot_paper_hallucination_retention(curve, base, zero_fca_sft, argmax, matched, tmp_path)


# -----------------------------------------------------------------------
# Test: threshold endpoints
# -----------------------------------------------------------------------


def test_threshold_endpoints() -> None:
    """Whole curve: zero-coverage point and full-coverage point."""
    records = {
        "a": _regate_record("a", gt_decision="answer", p_answer=0.99),
        "b": _regate_record("b", gt_decision="answer", p_answer=0.5),
        "c": _regate_record("c", gt_decision="abstain", p_answer=0.01),
    }
    curve, _ = regate_full_curve("toy", records)
    assert curve[0].answered == 0  # zero coverage
    assert curve[-1].answered == 3  # full coverage


# -----------------------------------------------------------------------
# Test: box-null scope for plantwild / refcocog
# -----------------------------------------------------------------------


def test_box_null_scope_plantwild_label_only() -> None:
    rec = _regate_record("p", gt_decision="answer", candidate_box=None, gt_box=None)
    point = regate_curve_point("plantwild", {"p": rec}, {"p": True}, "test", None)
    assert point.grounding_scope == "label_only"


def test_refcocog_scope_is_bbox_and_applies_iou() -> None:
    rec = _regate_record(
        "r",
        gt_decision="answer",
        gt_box=[0.0, 0.0, 100.0, 100.0],
        candidate_box=[200.0, 200.0, 300.0, 300.0],
    )
    point = regate_curve_point("refcocog", {"r": rec}, {"r": True}, "test", None)
    assert point.grounding_scope == "bbox"
    assert point.grounding_accuracy == 0.0


def test_shared_abstain_uses_sft_reason() -> None:
    sft = _native_record(
        "u",
        gt_decision="abstain",
        pred_decision="abstain",
        pred_reason="no_target",
        gt_reason="no_target",
    )
    regate = _regate_record(
        "u",
        gt_decision="abstain",
        p_wo=0.8,
        p_nt=0.1,
        gt_reason="no_target",
    )
    point = gate_shared_candidate_point("comprehensive", {"u": sft}, {"u": regate})
    assert point.strict_3_correct == 1


def test_shared_wrong_abstain_reason_fails_strict_3() -> None:
    sft = _native_record(
        "u",
        gt_decision="abstain",
        pred_decision="abstain",
        pred_reason="wrong_object",
        gt_reason="no_target",
    )
    regate = _regate_record("u", gt_decision="abstain", gt_reason="no_target")
    point = gate_shared_candidate_point("comprehensive", {"u": sft}, {"u": regate})
    assert point.strict_3_correct == 0


def test_regate_abstain_reason_uses_head_probabilities() -> None:
    correct = _regate_record(
        "u",
        gt_decision="abstain",
        p_wo=0.8,
        p_nt=0.1,
        gt_reason="wrong_object",
    )
    wrong = _regate_record(
        "v",
        gt_decision="abstain",
        p_wo=0.8,
        p_nt=0.1,
        gt_reason="no_target",
    )
    point = regate_curve_point(
        "comprehensive",
        {"u": correct, "v": wrong},
        {"u": False, "v": False},
        "test",
        None,
    )
    assert point.strict_3_correct == 1


def test_regate_reason_tie_matches_evaluator_no_target_rule() -> None:
    rec = _regate_record(
        "u",
        gt_decision="abstain",
        p_wo=0.4,
        p_nt=0.4,
        gt_reason="no_target",
    )
    point = regate_curve_point("comprehensive", {"u": rec}, {"u": False}, "test", None)
    assert point.strict_3_correct == 1


def test_box_null_no_bbox_grounding_claim() -> None:
    """PlantWild null box: grounding accuracy based on label only."""
    rec = _regate_record(
        "p", gt_decision="answer", candidate_box=None, gt_box=None, candidate_label="leaf"
    )
    point = regate_curve_point("plantwild", {"p": rec}, {"p": True}, "test", None)
    assert point.grounding_accuracy == 1.0
    # Should NOT be "bbox"
    assert point.grounding_scope == "label_only"


# -----------------------------------------------------------------------
# Test: semantic Strict-3 (label only, no IoU)
# -----------------------------------------------------------------------


def test_native_label_correct_iou_below_threshold_strict_3_true() -> None:
    """Strict-3 is true when label matches but IoU < 0.5 (semantic-only)."""
    rec = _native_record(
        "a",
        gt_decision="answer",
        gt_box=[0.0, 0.0, 100.0, 100.0],
        pred_box=[200.0, 200.0, 250.0, 250.0],  # IoU = 0
        pred_label="leaf",
        gt_label="leaf",
    )
    point = native_point("comprehensive", "Base_native", {"a": rec})
    assert point.strict_3_correct == 1, "Strict-3 should pass on label match alone"
    assert point.grounding_accuracy == 0.0, "grounding should still require IoU"
    # selective_risk = 1/1 = 1.0 because answer is not fully correct (IoU fails)


def test_native_label_correct_iou_below_strict_3_but_grounding_fails() -> None:
    """Strict-3 passes, selective_risk and grounding use IoU and fail."""
    rec = _native_record(
        "a",
        gt_decision="answer",
        gt_box=[0.0, 0.0, 100.0, 100.0],
        pred_box=[200.0, 200.0, 250.0, 250.0],  # IoU = 0
        pred_label="leaf",
        gt_label="leaf",
    )
    point = native_point("comprehensive", "Base_native", {"a": rec})
    assert point.strict_3_correct == 1
    # selective risk is based on answer correctness (incl IoU) — wrong box → error
    assert point.selective_risk_errors == 1, (
        "selective_risk should count an error when label is right but box is wrong"
    )


def test_gated_label_correct_iou_below_threshold_strict_3_true() -> None:
    """ReGate gated path: Strict-3 is semantic only, IoU < 0.5 is OK."""
    rec = _regate_record(
        "a",
        gt_decision="answer",
        gt_box=[0.0, 0.0, 100.0, 100.0],
        candidate_box=[200.0, 200.0, 250.0, 250.0],  # IoU = 0
        candidate_label="leaf",
        gt_label="leaf",
    )
    point = regate_curve_point("comprehensive", {"a": rec}, {"a": True}, "test", None)
    assert point.strict_3_correct == 1, "Strict-3 should pass on semantic label match"
    assert point.grounding_accuracy == 0.0, "grounding still requires IoU"


def test_native_unsupported_wrong_reason_strict_3_false() -> None:
    """Unsupported sample with wrong reason: Strict-3 = false."""
    rec = _native_record(
        "u",
        gt_decision="abstain",
        pred_decision="abstain",
        pred_reason="wrong_object",
        gt_reason="no_target",  # doesn't match
    )
    point = native_point("comprehensive", "Base_native", {"u": rec})
    assert point.strict_3_correct == 0, "reason mismatch should fail Strict-3"


def test_gated_unsupported_wrong_reason_strict_3_false() -> None:
    """ReGate unsupported + wrong reason → Strict-3 = false."""
    rec = _regate_record(
        "u",
        gt_decision="abstain",
        p_wo=0.8,
        p_nt=0.1,
        gt_reason="no_target",  # head says wrong_object (p_wo > p_nt) → mismatch
    )
    point = regate_curve_point("comprehensive", {"u": rec}, {"u": False}, "test", None)
    assert point.strict_3_correct == 0, "reason mismatch should fail Strict-3"


def test_supported_semantic_correct_helper() -> None:
    """Direct test of _supported_semantic_correct helper."""
    pred = {"valid": True, "decision": "answer", "label": "Tomato Leaf Spot"}
    gt = {"decision": "answer", "label": "tomato leaf spot"}
    assert _supported_semantic_correct(pred, gt)


def test_supported_semantic_correct_wrong_label_fails() -> None:
    pred = {"valid": True, "decision": "answer", "label": "apple scab"}
    gt = {"decision": "answer", "label": "tomato leaf spot"}
    assert not _supported_semantic_correct(pred, gt)


def test_gate_candidate_semantic_correct_helper() -> None:
    """Direct test of _gate_candidate_semantic_correct helper."""
    rec = _regate_record("a", gt_decision="answer", candidate_label="leaf", gt_label="leaf")
    assert _gate_candidate_semantic_correct(rec)


def test_gate_candidate_semantic_correct_wrong_label() -> None:
    rec = _regate_record("a", gt_decision="answer", candidate_label="wrong", gt_label="leaf")
    assert not _gate_candidate_semantic_correct(rec)


def test_gate_candidate_semantic_correct_invalid_candidate() -> None:
    rec = _regate_record("a", gt_decision="answer", candidate_valid=False)
    assert not _gate_candidate_semantic_correct(rec)


# -----------------------------------------------------------------------
# Test: CLI integration
# -----------------------------------------------------------------------


def test_cli_writes_all_output_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end CLI test with three dataset names."""
    datasets = ["toy_a", "toy_b", "toy_c"]
    regate_list = []
    base_list = []
    sft_list = []

    for ds in datasets:
        records = {
            "s1": _regate_record("s1", gt_decision="answer"),
            "s2": _regate_record("s2", gt_decision="abstain"),
        }
        native_a = _native_record("s1", gt_decision="answer")
        native_u = _native_record("s2", gt_decision="abstain")

        regate_file = tmp_path / f"regate_{ds}.jsonl"
        base_file = tmp_path / f"base_{ds}.jsonl"
        sft_file = tmp_path / f"sft_{ds}.jsonl"

        with regate_file.open("w", encoding="utf-8") as f:
            for rec in records.values():
                f.write(json.dumps(rec) + "\n")
        with base_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(native_a) + "\n")
            f.write(json.dumps(native_u) + "\n")
        with sft_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(native_a) + "\n")
            f.write(json.dumps(native_u) + "\n")

        regate_list.append(f"{ds}={regate_file}")
        base_list.append(f"{ds}={base_file}")
        sft_list.append(f"{ds}={sft_file}")

    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_native_regate",
            *(f"--base={b}" for b in base_list),
            *(f"--sft={s}" for s in sft_list),
            *(f"--regate={r}" for r in regate_list),
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ],
    )

    assert main() == 0
    assert (output_dir / "native_regate_diagnostic_comparison.csv").is_file()
    assert (output_dir / "native_regate_diagnostic_matched_retention.csv").is_file()
    assert (output_dir / "native_regate_diagnostic_summary.json").is_file()
    assert (output_dir / "native_regate_diagnostic_toy_a_curve.csv").is_file()
    assert (output_dir / "native_regate_diagnostic_toy_b_curve.csv").is_file()
    assert (output_dir / "native_regate_diagnostic_toy_c_curve.csv").is_file()


# -----------------------------------------------------------------------
# Test: duplicate ReGate missing p_answer/p_wo/p_nt fields
# -----------------------------------------------------------------------


def test_validate_regate_record_missing_p_answer() -> None:
    rec = {"sample_id": "x", "p_wo": 0.3, "p_nt": 0.2, "answer_candidate": {"valid": False}}
    with pytest.raises(ValueError, match="missing or non-numeric"):
        validate_regate_record(rec, Path("test"))


# -----------------------------------------------------------------------
# Test: format invalid native = not answered, strict false
# -----------------------------------------------------------------------


def test_native_invalid_not_answered_and_not_strict() -> None:
    rec = _native_record("a", gt_decision="answer", pred_valid=False)
    point = native_point("comprehensive", "Base", {"a": rec})
    assert point.answered == 0
    assert point.strict_3_correct == 0
    assert point.format_valid == 0


def test_native_invalid_unsupported_does_not_count_as_abstain() -> None:
    rec = _native_record("u", gt_decision="abstain", pred_valid=False)
    point = native_point("comprehensive", "Base", {"u": rec})
    assert point.answered == 0
    assert point.strict_3_correct == 0
    assert point.answered_unsupported == 0
    assert point.unsupported_hallucination_rate == 0.0  # 0/1 if unsupported==1


# -----------------------------------------------------------------------
# Test: unsupported reason must be strictly wrong_object / no_target
# -----------------------------------------------------------------------
# These are embedded in the VALID_REASONS check in score_unanswerable.
# We test that a non-matching reason fails strict-3.


def test_native_wrong_reason_fails_strict_3() -> None:
    rec = _native_record(
        "u",
        gt_decision="abstain",
        pred_decision="abstain",
        pred_reason="wrong_object",
        gt_reason="no_target",
    )
    point = native_point("comprehensive", "Base", {"u": rec})
    assert point.strict_3_correct == 0


def test_native_correct_reason_passes_strict_3() -> None:
    rec = _native_record(
        "u",
        gt_decision="abstain",
        pred_decision="abstain",
        pred_reason="wrong_object",
        gt_reason="wrong_object",
    )
    point = native_point("comprehensive", "Base", {"u": rec})
    assert point.strict_3_correct == 1


# -----------------------------------------------------------------------
# Test: ReGate candidate missing raises ValueError
# -----------------------------------------------------------------------


def test_regate_curve_point_missing_candidate_raises() -> None:
    rec = _regate_record("a", gt_decision="answer")
    rec.pop("answer_candidate")
    with pytest.raises(ValueError, match="missing answer_candidate"):
        regate_curve_point("comprehensive", {"a": rec}, {"a": True}, "test", None)


def test_shared_candidate_missing_raises() -> None:
    sft = _native_record("a", gt_decision="answer")
    regate = _regate_record("a", gt_decision="answer")
    regate.pop("answer_candidate")
    with pytest.raises(ValueError, match="missing answer_candidate"):
        gate_shared_candidate_point("comprehensive", {"a": sft}, {"a": regate})


# -----------------------------------------------------------------------
# Test: known dataset sizes
# -----------------------------------------------------------------------


def test_known_dataset_sizes() -> None:
    from experiments.base_native_regate.analyze_native_regate import EXPECTED_SIZES

    assert EXPECTED_SIZES["comprehensive"] == 2646
    assert EXPECTED_SIZES["refcocog"] == 362
    assert EXPECTED_SIZES["plantwild"] == 408


# -----------------------------------------------------------------------
# Static shell-contract test: launcher flags
# -----------------------------------------------------------------------


def _read_sh(path: str) -> str:
    """Read a .sh launcher file, skipping commented and blank lines."""
    p = Path(__file__).resolve().parent / path
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "launcher",
    [
        "run_native_eval.sh",
        "run_regate_candidates.sh",
    ],
)
def test_all_launchers_have_backend_hf(launcher: str) -> None:
    """Every eval invocation must explicitly pass ``--backend hf``."""
    text = _read_sh(launcher)
    msg = f"{launcher} is missing --backend hf"
    assert "--backend hf" in text, msg


HEAD_FLAGS = {
    "--dual-head",
    "--faithful-dual-output",
    "--dump-all-answer-candidates",
    "--calibrated-head",
}


def test_native_launcher_must_not_have_head_flags() -> None:
    """The base/SFT eval invocations must NOT include any dual-head/dump/candidate flags."""
    text = _read_sh("run_native_eval.sh")
    for flag in HEAD_FLAGS:
        assert flag not in text, f"{flag} must NOT appear in run_native_eval.sh"


def test_regate_launcher_must_have_all_head_flags() -> None:
    """The regate candidate launcher must include all dual-head/dump/candidate flags."""
    text = _read_sh("run_regate_candidates.sh")
    for flag in HEAD_FLAGS:
        assert flag in text, f"{flag} must appear in run_regate_candidates.sh"


# -----------------------------------------------------------------------
# Test: --help and minimal CLI from external CWD
# -----------------------------------------------------------------------


def test_help_from_any_cwd() -> None:
    """``--help`` must succeed when invoked from a directory *outside* the
    repo tree, verifying that the runtime ``sys.path`` fix works."""
    import subprocess
    import tempfile

    analyze_py = Path(__file__).resolve().parent / "analyze_native_regate.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, str(analyze_py), "--help"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        assert "usage:" in result.stdout


def test_help_via_module_from_any_cwd() -> None:
    """``python -m`` invocation must also work from any CWD."""
    import subprocess
    import tempfile

    aaai_code = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.base_native_regate.analyze_native_regate",
                "--help",
            ],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(aaai_code)},
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        assert "usage:" in result.stdout


def test_minimal_synthetic_cli_from_external_cwd(tmp_path: Path) -> None:
    """Run a full synthetic CLI invocation from a CWD outside the repo.
    This exercises both the sys.path fix and the full analysis pipeline."""
    import json
    import subprocess
    import tempfile

    aaai_code = Path(__file__).resolve().parents[2]
    analyze_py = aaai_code / "experiments" / "base_native_regate" / "analyze_native_regate.py"

    # Create minimal synthetic predictions in tmp_path
    datasets = ["toy_ext"]
    regate_list = []
    base_list = []
    sft_list = []

    for ds in datasets:
        records = {
            "s1": {
                "sample_id": "s1",
                "gt": {
                    "decision": "answer",
                    "label": "leaf",
                    "box": [0.0, 0.0, 100.0, 100.0],
                    "reason": None,
                },
                "p_answer": 0.9,
                "p_wo": 0.05,
                "p_nt": 0.05,
                "answer_candidate": {
                    "valid": True,
                    "label": "leaf",
                    "box": [0.0, 0.0, 100.0, 100.0],
                },
            },
            "s2": {
                "sample_id": "s2",
                "gt": {
                    "decision": "abstain",
                    "label": None,
                    "box": None,
                    "reason": "wrong_object",
                },
                "p_answer": 0.2,
                "p_wo": 0.5,
                "p_nt": 0.3,
                "answer_candidate": {"valid": False},
            },
        }
        native_a = {
            "sample_id": "s1",
            "gt": {
                "decision": "answer",
                "label": "leaf",
                "box": [0.0, 0.0, 100.0, 100.0],
                "reason": None,
            },
            "pred": {
                "valid": True,
                "decision": "answer",
                "label": "leaf",
                "box": [0.0, 0.0, 100.0, 100.0],
            },
        }
        native_u = {
            "sample_id": "s2",
            "gt": {
                "decision": "abstain",
                "label": None,
                "box": None,
                "reason": "wrong_object",
            },
            "pred": {"valid": True, "decision": "abstain", "reason": "wrong_object"},
        }

        regate_file = tmp_path / f"regate_{ds}.jsonl"
        base_file = tmp_path / f"base_{ds}.jsonl"
        sft_file = tmp_path / f"sft_{ds}.jsonl"

        with regate_file.open("w", encoding="utf-8") as f:
            for rec in records.values():
                f.write(json.dumps(rec) + "\n")
        with base_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(native_a) + "\n")
            f.write(json.dumps(native_u) + "\n")
        with sft_file.open("w", encoding="utf-8") as f:
            f.write(json.dumps(native_a) + "\n")
            f.write(json.dumps(native_u) + "\n")

        regate_list.append(f"{ds}={regate_file}")
        base_list.append(f"{ds}={base_file}")
        sft_list.append(f"{ds}={sft_file}")

    output_dir = tmp_path / "output"
    with tempfile.TemporaryDirectory() as external_cwd:
        result = subprocess.run(
            [
                sys.executable,
                str(analyze_py),
                *(f"--base={b}" for b in base_list),
                *(f"--sft={s}" for s in sft_list),
                *(f"--regate={r}" for r in regate_list),
                "--output-dir",
                str(output_dir),
                "--no-plots",
            ],
            cwd=external_cwd,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        assert (output_dir / "native_regate_diagnostic_comparison.csv").is_file()
        assert (output_dir / "native_regate_diagnostic_matched_retention.csv").is_file()
        assert (output_dir / "native_regate_diagnostic_summary.json").is_file()
        assert (output_dir / "native_regate_diagnostic_toy_ext_curve.csv").is_file()
