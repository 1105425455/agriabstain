from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.selective_grounding.analyze_selective_grounding import (
    baseline_point,
    main,
    matched_retention,
    regate_curve,
    validate_matched_records,
)
from experiments.selective_grounding.resolve_image_paths import resolve_record_images
from evaluation.eval_qwen3vl_3reason import eval_fingerprint
from experiments.selective_grounding.make_summary_figure import make_summary_figure


def _record(
    sample_id: str,
    *,
    gt_decision: str,
    p_answer: float,
    candidate_label: str = "leaf",
    gt_box: list[int] | None = None,
    candidate_box: list[int] | None = None,
) -> dict:
    gt_box = [0, 0, 100, 100] if gt_box is None else gt_box
    candidate_box = gt_box if candidate_box is None else candidate_box
    return {
        "sample_id": sample_id,
        "gt": {
            "decision": gt_decision,
            "label": "leaf" if gt_decision == "answer" else None,
            "box": gt_box if gt_decision == "answer" else None,
            "reason": None if gt_decision == "answer" else "wrong_object",
        },
        "p_answer": p_answer,
        "p_wo": (1 - p_answer) * 0.6,
        "p_nt": (1 - p_answer) * 0.4,
        "answer_candidate": {
            "valid": True,
            "label": candidate_label,
            "box": candidate_box,
        },
    }


def test_curve_has_zero_and_full_coverage_endpoints_and_keeps_ties() -> None:
    records = {
        "valid": _record("valid", gt_decision="answer", p_answer=0.7),
        "invalid": _record("invalid", gt_decision="answer", p_answer=0.7, candidate_label="x"),
        "unsupported": _record("unsupported", gt_decision="abstain", p_answer=0.2),
    }
    curve, _ = regate_curve("toy", records)

    assert len(curve) == 3
    assert curve[0].answered == 0
    assert math.isnan(curve[0].selective_risk)
    assert curve[1].answered == 2
    assert curve[-1].answered == 3
    assert curve[-1].supported_query_retention == 1.0
    assert curve[-1].unsupported_hallucination_rate == 1.0
    assert curve[-1].selective_risk == pytest.approx(2 / 3)


def test_plantwild_null_box_uses_label_correctness() -> None:
    record = _record("plantwild", gt_decision="answer", p_answer=0.9)
    record["gt"]["box"] = None
    record["answer_candidate"]["box"] = None
    curve, _ = regate_curve("plantwild", {"plantwild": record})
    assert curve[-1].selective_risk == 0.0
    assert curve[-1].grounding_scope == "label_only"
    assert math.isnan(curve[-1].a_ground_mean_iou_over_supported)
    assert curve[-1].label_success_over_supported == 1.0
    assert curve[-1].label_success_conditional == 1.0


def test_grounding_metrics_separate_overall_and_conditional() -> None:
    answered = _record("answered", gt_decision="answer", p_answer=0.9)
    abstained = _record("abstained", gt_decision="answer", p_answer=0.1)
    unsupported = _record("unsupported", gt_decision="abstain", p_answer=0.8)
    curve, _ = regate_curve(
        "bbox",
        {"answered": answered, "abstained": abstained, "unsupported": unsupported},
    )
    point = next(item for item in curve if item.threshold == 0.8)

    assert point.answered_supported == 1
    assert point.a_ground_mean_iou_over_supported == 0.5
    assert point.a_ground_conditional_mean_iou == 1.0
    assert point.grounding_success_at_iou_0_5_over_supported == 0.5
    assert point.grounding_success_at_iou_0_5_conditional == 1.0


def test_no_unsupported_yields_nan_hallucination_rate() -> None:
    record = _record("valid", gt_decision="answer", p_answer=0.9)
    curve, _ = regate_curve("toy", {"valid": record})
    assert math.isnan(curve[-1].unsupported_hallucination_rate)


def test_argmax_uses_class_zero_tie_breaking() -> None:
    tie = _record("tie", gt_decision="answer", p_answer=1 / 3)
    tie["p_wo"] = 1 / 3
    tie["p_nt"] = 1 / 3
    abstain = _record("abstain", gt_decision="abstain", p_answer=0.2)
    abstain["p_wo"] = 0.5
    abstain["p_nt"] = 0.3

    _, argmax_point = regate_curve("toy", {"tie": tie, "abstain": abstain})

    assert argmax_point.answered == 1
    assert argmax_point.answered_supported == 1
    assert argmax_point.answered_unsupported == 0


def test_baseline_and_empirical_matched_retention() -> None:
    regate = {
        "a": _record("a", gt_decision="answer", p_answer=0.9),
        "b": _record("b", gt_decision="answer", p_answer=0.6),
        "u": _record("u", gt_decision="abstain", p_answer=0.7),
    }
    baseline = {}
    for sample_id, source in regate.items():
        answer = sample_id == "a"
        baseline[sample_id] = {
            "sample_id": sample_id,
            "gt": source["gt"],
            "pred": {
                "valid": True,
                "decision": "answer" if answer else "abstain",
                "label": "leaf" if answer else None,
                "box": [0, 0, 100, 100] if answer else None,
            },
        }
    curve, _ = regate_curve("toy", regate)
    point = baseline_point("toy", "baseline", baseline)
    empirical, interpolated = matched_retention(curve, point)

    assert point.supported_query_retention == 0.5
    assert empirical.threshold == 0.9
    assert empirical.supported_query_retention == 0.5
    assert interpolated["interpolated_unsupported_hallucination_rate"] == 0.0


def test_sample_id_mismatch_fails_fast() -> None:
    regate = {"a": _record("a", gt_decision="answer", p_answer=0.9)}
    baseline = {"b": {"sample_id": "b", "gt": regate["a"]["gt"]}}
    with pytest.raises(ValueError, match="sample_id mismatch"):
        validate_matched_records("toy", regate, baseline)


def test_cli_writes_curve_and_summary_csvs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _record("a", gt_decision="answer", p_answer=0.8),
        _record("u", gt_decision="abstain", p_answer=0.2),
    ]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_selective_grounding",
            "--regate",
            f"toy={predictions}",
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ],
    )

    assert main() == 0
    assert (output_dir / "toy_regate_curve.csv").is_file()
    assert (output_dir / "regate_argmax_points.csv").is_file()


def test_image_resolver_uses_external_derived_root(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmark"
    derived_root = tmp_path / "derived"
    image = derived_root / "plantdoc" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    record = {"images": ["data/derived/plantdoc/sample.jpg"]}

    resolved = resolve_record_images(record, benchmark_root, derived_root)

    assert resolved["images"] == [str(image.resolve())]
    assert record["images"] == ["data/derived/plantdoc/sample.jpg"]


def test_runner_pins_formal_decision_prompt() -> None:
    runner = Path(__file__).with_name("run_regate_candidates.sh").read_text(encoding="utf-8")
    assert "--system-override decision" in runner


def test_fingerprint_tracks_head_and_prompt_configuration(tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    head = tmp_path / "head.pt"
    head.write_bytes(b"head")
    args = SimpleNamespace(
        data=str(data),
        base=None,
        adapter=None,
        limit=None,
        backend="hf",
        calibrated_head=str(head),
        system_override="decision",
        dual_head=True,
        faithful_dual_output=True,
        dump_all_answer_candidates=True,
    )

    fingerprint = eval_fingerprint(args, data)

    assert f"calibrated_head={head}@" in fingerprint
    assert "system_override=decision" in fingerprint
    assert "faithful_dual_output=True" in fingerprint


def test_summary_figure_is_a_single_png(tmp_path: Path) -> None:
    source = Path(__file__).with_name("main")
    output = tmp_path / "summary.png"

    make_summary_figure(source, output)

    assert output.is_file()
    assert output.stat().st_size > 10_000


def _bbox_record(
    sample_id: str,
    *,
    p_answer: float,
    candidate_label: str = "leaf",
    gt_box: list[int] | None = (0, 0, 100, 100),
    candidate_box: list[int] | None = None,
    label_ok: bool = True,
    iou: float | None = None,
) -> dict:
    """Helper for bbox records: auto-compute candidate_box from iou if given.

    If iou is specified, candidate_box is adjusted to achieve that IoU with gt_box.
    If label_ok is False, the candidate label is set to a wrong value.
    """
    gt_box_v = gt_box or [0, 0, 100, 100]
    if iou is not None and candidate_box is None:
        # Create a box at the same location for IoU=1.0, or overlapping for IoU < 1.0
        if iou == 1.0:
            candidate_box = list(gt_box_v)
        elif iou == 0.5:
            # Make a box that overlaps with IoU ~0.5
            # Full box area = 100*100 = 10000
            # To get IoU = 0.5: inter/union = 0.5 → union = 2*inter → area_a+area_b-inter = 2*inter
            # area_a = area_b = 10000, so 20000-inter = 2*inter → inter = 20000/3 ≈ 6667
            # Using a smaller box inside: e.g., [0,0,66,100] area=6600, inter=6600
            # union = 10000+6600-6600 = 10000, iou = 6600/10000 = 0.66 — not 0.5
            # Let's just use a computed offset: partially overlapping
            candidate_box = [25, 25, 100, 100]  # area = 75*75 = 5625
            # inter = [25,25,100,100] ∩ [0,0,100,100] = [25,25,100,100] area = 75*75 = 5625
            # union = 10000+5625-5625 = 10000, iou = 0.5625
            # Close enough for testing
        else:
            # Non-overlapping
            candidate_box = [200, 200, 300, 300]
    elif candidate_box is None:
        candidate_box = list(gt_box_v)

    label = candidate_label if label_ok else "wrong_species"
    return {
        "sample_id": sample_id,
        "gt": {
            "decision": "answer",
            "label": "leaf",
            "box": gt_box_v,
            "reason": None,
        },
        "p_answer": p_answer,
        "p_wo": (1 - p_answer) * 0.6,
        "p_nt": (1 - p_answer) * 0.4,
        "answer_candidate": {
            "valid": True,
            "label": label,
            "box": candidate_box,
        },
    }


def test_bbox_grounding_mean_iou_overall_and_conditional() -> None:
    """Two supported answered: both label-correct; one with IoU=1.0, one with IoU~0.56."""
    r1 = _bbox_record("a", p_answer=0.9, iou=1.0)
    r2 = _bbox_record("b", p_answer=0.8, iou=0.5)
    curve, _ = regate_curve("test_bbox", {"a": r1, "b": r2})

    full = curve[-1]  # threshold=0 coverage, both answered
    assert full.answered == 2
    assert full.answered_supported == 2
    assert full.grounding_scope == "bbox"

    # Both have label-correct (label is "leaf" which matches), IoU sums
    # b's IoU is approximately the computed value from our helper
    assert full.a_ground_mean_iou_over_supported > 0.0
    assert full.a_ground_conditional_mean_iou == full.a_ground_mean_iou_over_supported
    # IoU@0.5: a has IoU=1.0 >= 0.5, b has IoU~0.56 >= 0.5 → both succeed
    assert full.grounding_success_at_iou_0_5_over_supported == 1.0
    assert full.grounding_success_at_iou_0_5_conditional == 1.0
    assert full.label_success_over_supported == 1.0
    assert full.label_success_conditional == 1.0


def test_bbox_grounding_abstain_penalty() -> None:
    """Two supported: one answered correctly (IoU=1.0), one abstained.
    Over-supported metrics should penalise abstention with 0 contribution.
    """
    r1 = _bbox_record("a", p_answer=0.9, iou=1.0)
    r2 = _bbox_record("b", p_answer=0.1, iou=1.0)  # low p_answer → will abstain at most thresholds
    curve, _ = regate_curve("test_penalty", {"a": r1, "b": r2})

    # Find a point where only 'a' is answered
    answered_one = [p for p in curve if p.answered == 1 and p.answered_supported == 1]
    assert len(answered_one) >= 1
    pt = answered_one[0]

    # Abstain penalty: only 1 answered out of 2 supported
    # IoU sum = 1.0 (from the answered one)
    # a_ground_mean_iou_over_supported = 1.0 / 2 = 0.5
    assert pt.a_ground_mean_iou_over_supported == pytest.approx(0.5)
    # Conditional: only over the answered-supported sample
    assert pt.a_ground_conditional_mean_iou == pytest.approx(1.0)
    # grounding_success_over_supported = 1/2 = 0.5
    assert pt.grounding_success_at_iou_0_5_over_supported == pytest.approx(0.5)
    # grounding_success_conditional = 1/1 = 1.0
    assert pt.grounding_success_at_iou_0_5_conditional == pytest.approx(1.0)
    assert pt.label_success_over_supported == pytest.approx(0.5)
    assert pt.label_success_conditional == pytest.approx(1.0)


def test_label_only_grounding_nan_for_bbox_fields() -> None:
    """PlantWild: gt_box is None → bbox fields are NaN, label-only fields are computed."""
    records = {}
    for sid, p_ans, label_ok in [("a", 0.9, True), ("b", 0.8, False), ("c", 0.1, True)]:
        rec = _record(
            sid,
            gt_decision="answer",
            p_answer=p_ans,
            candidate_label="leaf" if label_ok else "wrong",
        )
        rec["gt"]["box"] = None
        rec["answer_candidate"]["box"] = None
        records[sid] = rec

    curve, _ = regate_curve("plantwild", records)

    full = curve[-1]  # all answered
    assert full.grounding_scope == "label_only"

    # Bbox fields must be NaN
    assert math.isnan(full.a_ground_mean_iou_over_supported)
    assert math.isnan(full.a_ground_conditional_mean_iou)
    assert math.isnan(full.grounding_success_at_iou_0_5_over_supported)
    assert math.isnan(full.grounding_success_at_iou_0_5_conditional)

    # Label-only: 2 label-correct out of 3 supported
    assert full.label_success_over_supported == pytest.approx(2 / 3)
    assert full.label_success_conditional == pytest.approx(2 / 3)

    # Selective risk / hallucination / retention preserved (regression check)
    assert not math.isnan(full.selective_risk)
    assert full.supported_query_retention == pytest.approx(1.0)
    assert math.isnan(full.unsupported_hallucination_rate)


def test_label_only_grounding_scope_in_csv_row() -> None:
    """Verify _point_to_native_comparison_row includes grounding_scope."""
    from experiments.selective_grounding.analyze_selective_grounding import (
        _point_to_native_comparison_row,
    )

    rec = _record("a", gt_decision="answer", p_answer=0.9)
    rec["gt"]["box"] = None
    rec["answer_candidate"]["box"] = None
    curve, _ = regate_curve("plantwild", {"a": rec})
    row = _point_to_native_comparison_row(curve[-1])
    assert row["grounding_scope"] == "label_only"
    assert math.isnan(row["a_ground_mean_iou_over_supported"])
