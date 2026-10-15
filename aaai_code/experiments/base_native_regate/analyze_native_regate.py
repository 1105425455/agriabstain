"""Native base/SFT evaluation and ReGate analysis with answer-candidate sharing.

CLI usage::

    python -m experiments.base_native_regate.analyze_native_regate \\
        --base comprehensive=/path/to/base_comp.jsonl \\
        --base refcocog=/path/to/base_refcocog.jsonl \\
        --base plantwild=/path/to/base_plantwild.jsonl \\
        --sft comprehensive=/path/to/sft_comp.jsonl \\
        --sft refcocog=/path/to/sft_refcocog.jsonl \\
        --sft plantwild=/path/to/sft_plantwild.jsonl \\
        --regate comprehensive=/path/to/regate_comp.jsonl \\
        --regate refcocog=/path/to/regate_refcocog.jsonl \\
        --regate plantwild=/path/to/regate_plantwild.jsonl \\
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ── Resolve package root (aaai_code) so this module works when invoked
#    directly via ``python /abs/path/analyze_native_regate.py`` or via
#    ``python -m experiments.base_native_regate.analyze_native_regate``
#    without relying on a particular working directory. ──
_PKG_ROOT = Path(__file__).resolve().parents[2]  # → aaai_code/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from reason_abstain.eval.metrics import score_answerable, score_unanswerable  # noqa: E402
from reason_abstain.eval.parser import normalize_label  # noqa: E402
from experiments.selective_grounding.analyze_selective_grounding import (  # noqa: E402
    _ratio,
    _validate_probability,
    _write_csv,
    load_jsonl,
    validate_matched_records,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_SIZES: dict[str, int] = {
    "comprehensive": 2646,
    "refcocog": 362,
    "plantwild": 408,
}

GROUNDING_SCOPE: dict[str, str] = {
    "comprehensive": "bbox",
    "refcocog": "bbox",
    "plantwild": "label_only",
}


def _validate_reason_probability(value: Any, sample_id: str, name: str = "p_wo") -> float:
    """Validate a reason probability (p_wo / p_nt) is a finite number in [0, 1]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{sample_id}: {name} is not numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{sample_id}: invalid {name} {value!r}")
    return value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _native_answered(pred: dict[str, Any]) -> bool:
    """Whether a native (Base/SFT) prediction counts as 'answered'."""
    return bool(pred.get("valid")) and pred.get("decision") == "answer"


def _supported_semantic_correct(pred: dict[str, Any], gt: dict[str, Any]) -> bool:
    """Semantic-only correctness for supported (answerable) samples.

    Checks: valid + decision=answer + normalized label match.
    Does NOT check bbox/IoU.  This is the Strict-3 definition for supported
    cases per the paper.
    """
    if not bool(pred.get("valid")) or pred.get("decision") != "answer":
        return False
    if gt.get("decision") != "answer":
        return False
    return normalize_label(pred.get("label")) == normalize_label(gt.get("label"))


def _strict_3_native(pred: dict[str, Any], gt: dict[str, Any], iou_threshold: float) -> bool:
    """Strict-3 correctness for native (Base/SFT) predictions.

    Supported + answered : needs valid answer + correct label (semantic, no IoU)
    Unsupported + abstain: needs valid abstain + correct reason
    All other cases      : False
    """
    if not pred.get("valid"):
        return False
    if pred.get("decision") == "answer":
        return _supported_semantic_correct(pred, gt)
    # decision == "abstain"
    if gt.get("decision") != "abstain":
        return False
    return score_unanswerable(pred, gt)


def _strict_3_gated(
    answered: bool,
    gt: dict[str, Any],
    candidate_semantic_correct: bool,
    *,
    abstain_pred: dict[str, Any],
) -> bool:
    """Strict-3 for gated (ReGate / shared-candidate) decisions.

    Answered + supported : candidate must be semantically correct (no IoU)
    Not-answered + unsupported : uses *abstain_pred* via
                                 ``score_unanswerable`` to check reason correctness.
    All other cases      : False
    """
    if answered:
        if gt.get("decision") != "answer":
            return False
        return candidate_semantic_correct
    # abstain case
    if gt.get("decision") != "abstain":
        return False
    return score_unanswerable(abstain_pred, gt)


def _gate_candidate_semantic_correct(record: dict[str, Any]) -> bool:
    """Semantic-only correctness of the ReGate answer_candidate (no IoU).

    Checks: candidate valid + label matches GT label (normalized).
    This is used for Strict-3 in gated paths.
    """
    candidate = record.get("answer_candidate")
    if not isinstance(candidate, dict):
        return False
    if candidate.get("valid") is not True:
        return False
    gt = record["gt"]
    if gt.get("decision") != "answer":
        return False
    return normalize_label(candidate.get("label")) == normalize_label(gt.get("label"))


def _gate_candidate_correct(record: dict[str, Any], iou_threshold: float) -> bool:
    """Check whether the ReGate answer_candidate is a correct answer (label + IoU)."""
    candidate = record.get("answer_candidate")
    if not isinstance(candidate, dict):
        return False
    if candidate.get("valid") is not True:
        return False
    pred = {
        "valid": True,
        "decision": "answer",
        "label": candidate.get("label"),
        "box": candidate.get("box"),
    }
    return score_answerable(pred, record["gt"], iou_threshold)


def _compute_grounding_correct(
    pred: dict[str, Any], gt: dict[str, Any], iou_threshold: float
) -> bool:
    """Grounding correctness: label+IoU for bbox-scope, label-only for null-box."""
    return score_answerable(pred, gt, iou_threshold)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeRegateMetric:
    """Unified metrics record for any method in the native/regate analysis."""

    dataset: str
    method: str
    policy: str
    threshold: float | None
    total: int
    supported: int
    unsupported: int
    answered: int
    answered_supported: int
    answered_unsupported: int
    strict_3_correct: int
    selective_risk_errors: int
    grounding_correct: int
    grounding_denom: int
    format_valid: int
    total_coverage: float
    supported_query_retention: float
    unsupported_hallucination_rate: float
    strict_3_accuracy: float
    selective_risk: float
    grounding_accuracy: float
    format_valid_rate: float
    grounding_scope: str


# ---------------------------------------------------------------------------
# Native point (Base / SFT)
# ---------------------------------------------------------------------------


def native_point(
    dataset: str,
    method: str,
    records: dict[str, dict[str, Any]],
    iou_threshold: float = 0.5,
) -> NativeRegateMetric:
    """Compute all metrics for a fixed native (Base or SFT) model point."""
    supported_ids = {sid for sid, rec in records.items() if rec["gt"].get("decision") == "answer"}
    unsupported_ids = set(records) - supported_ids

    answered_set: set[str] = set()
    strict_3_correct = 0
    selective_errors = 0
    grounding_correct = 0
    grounding_denom = 0
    format_valid = 0

    for sid, rec in records.items():
        pred = rec.get("pred")
        if not isinstance(pred, dict):
            raise ValueError(f"{sid}: missing pred object")

        ans = _native_answered(pred)
        if ans:
            answered_set.add(sid)

        if bool(pred.get("valid")):
            format_valid += 1

        if _strict_3_native(pred, rec["gt"], iou_threshold):
            strict_3_correct += 1

        # selective risk: answered but answer is wrong
        if ans:
            if rec["gt"].get("decision") != "answer" or not score_answerable(
                pred, rec["gt"], iou_threshold
            ):
                selective_errors += 1

        # grounding among answered + supported
        if ans and sid in supported_ids:
            grounding_denom += 1
            if _compute_grounding_correct(pred, rec["gt"], iou_threshold):
                grounding_correct += 1

    answered_supported = answered_set & supported_ids
    answered_unsupported = answered_set & unsupported_ids
    scope = GROUNDING_SCOPE.get(dataset, "bbox")

    return NativeRegateMetric(
        dataset=dataset,
        method=method,
        policy="native_decision",
        threshold=None,
        total=len(records),
        supported=len(supported_ids),
        unsupported=len(unsupported_ids),
        answered=len(answered_set),
        answered_supported=len(answered_supported),
        answered_unsupported=len(answered_unsupported),
        strict_3_correct=strict_3_correct,
        selective_risk_errors=selective_errors,
        grounding_correct=grounding_correct,
        grounding_denom=grounding_denom,
        format_valid=format_valid,
        total_coverage=_ratio(len(answered_set), len(records)),
        supported_query_retention=_ratio(len(answered_supported), len(supported_ids)),
        unsupported_hallucination_rate=_ratio(len(answered_unsupported), len(unsupported_ids)),
        strict_3_accuracy=_ratio(strict_3_correct, len(records)),
        selective_risk=_ratio(selective_errors, len(answered_set)),
        grounding_accuracy=_ratio(grounding_correct, grounding_denom),
        format_valid_rate=_ratio(format_valid, len(records)),
        grounding_scope=scope,
    )


# ---------------------------------------------------------------------------
# SFT_native_gate_shared_candidate
# ---------------------------------------------------------------------------


def gate_shared_candidate_point(
    dataset: str,
    sft_records: dict[str, dict[str, Any]],
    regate_records: dict[str, dict[str, Any]],
    iou_threshold: float = 0.5,
) -> NativeRegateMetric:
    """SFT_native_gate_shared_candidate.

    Answered mask comes from the SFT native prediction (gating).
    Correctness and grounding use the ReGate answer_candidate.
    This isolates gating quality from generation quality.

    Strict-3 for abstain cases (answered=False, GT abstain) uses the SFT native
    pred's ``valid`` / ``decision`` / ``reason`` via ``score_unanswerable``,
    **not** the ReGate answer_candidate or the ReGate head reason.
    """
    supported_ids = {
        sid for sid, rec in regate_records.items() if rec["gt"].get("decision") == "answer"
    }
    unsupported_ids = set(regate_records) - supported_ids

    answered_set: set[str] = set()
    strict_3_correct = 0
    selective_errors = 0
    grounding_correct = 0
    grounding_denom = 0
    format_valid = 0

    for sid in regate_records:
        sft_pred = sft_records[sid].get("pred")
        if not isinstance(sft_pred, dict):
            raise ValueError(f"{sid}: SFT missing pred")
        ans = _native_answered(sft_pred)
        if ans:
            answered_set.add(sid)

        regate_rec = regate_records[sid]
        gt = regate_rec["gt"]
        candidate = regate_rec.get("answer_candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"{sid}: missing answer_candidate in ReGate record")
        candidate_valid = candidate.get("valid") is True
        if candidate_valid:
            format_valid += 1

        candidate_correct_val = (
            _gate_candidate_correct(regate_rec, iou_threshold) if candidate_valid else False
        )
        candidate_semantic_val = (
            _gate_candidate_semantic_correct(regate_rec) if candidate_valid else False
        )

        abstain_pred = {
            "valid": sft_pred.get("valid", False),
            "decision": sft_pred.get("decision"),
            "reason": sft_pred.get("reason"),
        }

        if _strict_3_gated(ans, gt, candidate_semantic_val, abstain_pred=abstain_pred):
            strict_3_correct += 1

        # selective risk
        if ans:
            if gt.get("decision") != "answer" or not candidate_correct_val:
                selective_errors += 1

        # grounding
        if ans and sid in supported_ids:
            grounding_denom += 1
            if candidate_correct_val:
                grounding_correct += 1

    answered_supported = answered_set & supported_ids
    answered_unsupported = answered_set & unsupported_ids
    scope = GROUNDING_SCOPE.get(dataset, "bbox")

    return NativeRegateMetric(
        dataset=dataset,
        method="SFT_native_gate_shared_candidate",
        policy="sft_gate_shared_candidate",
        threshold=None,
        total=len(regate_records),
        supported=len(supported_ids),
        unsupported=len(unsupported_ids),
        answered=len(answered_set),
        answered_supported=len(answered_supported),
        answered_unsupported=len(answered_unsupported),
        strict_3_correct=strict_3_correct,
        selective_risk_errors=selective_errors,
        grounding_correct=grounding_correct,
        grounding_denom=grounding_denom,
        format_valid=format_valid,
        total_coverage=_ratio(len(answered_set), len(regate_records)),
        supported_query_retention=_ratio(len(answered_supported), len(supported_ids)),
        unsupported_hallucination_rate=_ratio(len(answered_unsupported), len(unsupported_ids)),
        strict_3_accuracy=_ratio(strict_3_correct, len(regate_records)),
        selective_risk=_ratio(selective_errors, len(answered_set)),
        grounding_accuracy=_ratio(grounding_correct, grounding_denom),
        format_valid_rate=_ratio(format_valid, len(regate_records)),
        grounding_scope=scope,
    )


# ---------------------------------------------------------------------------
# ReGate curve point and full curve
# ---------------------------------------------------------------------------


def regate_curve_point(
    dataset: str,
    records: dict[str, dict[str, Any]],
    answered_map: dict[str, bool],
    policy: str,
    threshold: float | None,
    iou_threshold: float = 0.5,
) -> NativeRegateMetric:
    """Compute all metrics for a single ReGate operating point.

    For abstain samples (answered=False, GT abstain), the abstain reason is
    recovered from the head's p_wo/p_nt probabilities:
      - p_wo > p_nt    → reason = "wrong_object"
      - p_wo <= p_nt   → reason = "no_target" (matching the evaluator tie rule)

    The constructed ``abstain_pred`` is validated via ``score_unanswerable``.
    Answered samples use the recorded ``answer_candidate``.
    """
    supported_ids = {sid for sid, rec in records.items() if rec["gt"].get("decision") == "answer"}
    unsupported_ids = set(records) - supported_ids

    answered_set = {sid for sid, v in answered_map.items() if v}
    strict_3_correct = 0
    selective_errors = 0
    grounding_correct = 0
    grounding_denom = 0
    format_valid = 0

    for sid, rec in records.items():
        gt = rec["gt"]
        ans = sid in answered_set
        candidate = rec.get("answer_candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"{sid}: missing answer_candidate")
        candidate_valid = candidate.get("valid") is True
        if candidate_valid:
            format_valid += 1

        candidate_correct_val = (
            _gate_candidate_correct(rec, iou_threshold) if candidate_valid else False
        )
        candidate_semantic_val = _gate_candidate_semantic_correct(rec) if candidate_valid else False

        p_wo = _validate_reason_probability(rec.get("p_wo"), sid, "p_wo")
        p_nt = _validate_reason_probability(rec.get("p_nt"), sid, "p_nt")
        reason = "wrong_object" if p_wo > p_nt else "no_target"
        abstain_pred = {"valid": True, "decision": "abstain", "reason": reason}

        if _strict_3_gated(ans, gt, candidate_semantic_val, abstain_pred=abstain_pred):
            strict_3_correct += 1

        if ans:
            if gt.get("decision") != "answer" or not candidate_correct_val:
                selective_errors += 1

        # grounding among answered+supported
        if ans and sid in supported_ids:
            grounding_denom += 1
            if candidate_correct_val:
                grounding_correct += 1

    answered_supported = answered_set & supported_ids
    answered_unsupported = answered_set & unsupported_ids
    scope = GROUNDING_SCOPE.get(dataset, "bbox")

    return NativeRegateMetric(
        dataset=dataset,
        method="ReGate",
        policy=policy,
        threshold=threshold,
        total=len(records),
        supported=len(supported_ids),
        unsupported=len(unsupported_ids),
        answered=len(answered_set),
        answered_supported=len(answered_supported),
        answered_unsupported=len(answered_unsupported),
        strict_3_correct=strict_3_correct,
        selective_risk_errors=selective_errors,
        grounding_correct=grounding_correct,
        grounding_denom=grounding_denom,
        format_valid=format_valid,
        total_coverage=_ratio(len(answered_set), len(records)),
        supported_query_retention=_ratio(len(answered_supported), len(supported_ids)),
        unsupported_hallucination_rate=_ratio(len(answered_unsupported), len(unsupported_ids)),
        strict_3_accuracy=_ratio(strict_3_correct, len(records)),
        selective_risk=_ratio(selective_errors, len(answered_set)),
        grounding_accuracy=_ratio(grounding_correct, grounding_denom),
        format_valid_rate=_ratio(format_valid, len(records)),
        grounding_scope=scope,
    )


def regate_full_curve(
    dataset: str,
    records: dict[str, dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[list[NativeRegateMetric], NativeRegateMetric]:
    """Compute the full ReGate threshold curve and argmax operating point.

    Returns (curve_points, argmax_point).

    **Three-class argmax rule**:
    ``answered = p_answer >= p_wo AND p_answer >= p_nt`` (tie breaks to answer).
    When abstain is chosen, reason is recovered from p_wo vs p_nt:
      - p_wo >= p_nt  → ``wrong_object``
      - p_wo < p_nt   → ``no_target``
    """
    probabilities: dict[str, float] = {}
    for sid, rec in records.items():
        probabilities[sid] = _validate_probability(rec.get("p_answer"), sid)

    unique_thresholds = sorted(set(probabilities.values()), reverse=True)
    zero_cov_threshold = math.nextafter(max(unique_thresholds), math.inf)
    thresholds = [zero_cov_threshold, *unique_thresholds]

    curve = [
        regate_curve_point(
            dataset,
            records,
            {sid: p >= th for sid, p in probabilities.items()},
            "p_supported_threshold",
            th,
            iou_threshold,
        )
        for th in thresholds
    ]

    # Three-class argmax with tie-break: answer if p_answer >= p_wo and p_answer >= p_nt
    argmax_answered: dict[str, bool] = {}
    for sid, rec in records.items():
        p_a = probabilities[sid]
        p_wo = _validate_reason_probability(rec.get("p_wo"), sid, "p_wo")
        p_nt = _validate_reason_probability(rec.get("p_nt"), sid, "p_nt")
        argmax_answered[sid] = p_a >= p_wo and p_a >= p_nt

    argmax_point = regate_curve_point(
        dataset,
        records,
        argmax_answered,
        "three_class_argmax",
        None,
        iou_threshold,
    )
    return curve, argmax_point


# ---------------------------------------------------------------------------
# Matched retention (interpolation)
# ---------------------------------------------------------------------------


def matched_retention_to_target(
    curve: list[NativeRegateMetric],
    target_point: NativeRegateMetric,
) -> tuple[NativeRegateMetric, dict[str, float]]:
    """Match the ReGate curve to match the query retention of *target_point*.

    Returns (empirical point, interpolated_dict).
    interpolated_dict contains hallucination, risk, strict_3 at the exact target.

    **Tie handling**: when multiple curve points have the same retention, the one
    with the highest threshold is chosen (empirical). This is the conservative
    policy: among equally safe (same retention) operating points, pick the one
    with the tightest gate (highest threshold). Interpolation lower/upper bounds
    with identical retention tie-break by threshold descending, ensuring
    deterministic results even when curve input order varies.
    """
    target = target_point.supported_query_retention
    eligible = [
        pt for pt in curve if pt.supported_query_retention >= target and pt.threshold is not None
    ]
    if not eligible:
        raise ValueError(
            f"Cannot match retention {target} for {target_point.method} on {target_point.dataset}"
        )
    # Tie: same retention → highest threshold (conservative)
    empirical = max(eligible, key=lambda pt: (float(pt.threshold), pt.supported_query_retention))

    # Interpolated values at exact target
    interp_hallucination = _interpolate_at_retention(
        curve, target, "unsupported_hallucination_rate"
    )
    interp_risk = _interpolate_at_retention(curve, target, "selective_risk")
    interp_strict = _interpolate_at_retention(curve, target, "strict_3_accuracy")

    return empirical, {
        "dataset": target_point.dataset,
        "baseline_method": target_point.method,
        "target_supported_query_retention": target,
        "empirical_threshold": (
            float(empirical.threshold) if empirical.threshold is not None else math.nan
        ),
        "empirical_unsupported_hallucination_rate": empirical.unsupported_hallucination_rate,
        "empirical_selective_risk": empirical.selective_risk,
        "empirical_strict_3_accuracy": empirical.strict_3_accuracy,
        "interpolated_unsupported_hallucination_rate": interp_hallucination,
        "interpolated_selective_risk": interp_risk,
        "interpolated_strict_3_accuracy": interp_strict,
    }


def _interpolate_at_retention(curve: list[NativeRegateMetric], target: float, attr: str) -> float:
    """Linearly interpolate a metric attribute at the target retention.

    **Tie handling**: points with identical retention are sorted by threshold
    (descending) to ensure deterministic lower/upper bound selection even when
    curve input order varies.
    """
    # Sort by retention ascending; ties broken by threshold descending
    points = sorted(
        curve,
        key=lambda pt: (
            pt.supported_query_retention,
            -(pt.threshold if pt.threshold is not None else float("inf")),
        ),
    )
    for pt in points:
        if math.isclose(pt.supported_query_retention, target, abs_tol=1e-12):
            return getattr(pt, attr)
    lower = max(
        (pt for pt in points if pt.supported_query_retention < target),
        key=lambda pt: pt.supported_query_retention,
    )
    upper = min(
        (pt for pt in points if pt.supported_query_retention > target),
        key=lambda pt: pt.supported_query_retention,
    )
    weight = (target - lower.supported_query_retention) / (
        upper.supported_query_retention - lower.supported_query_retention
    )
    lower_val = getattr(lower, attr)
    upper_val = getattr(upper, attr)
    if math.isnan(lower_val) or math.isnan(upper_val):
        return math.nan
    return lower_val + weight * (upper_val - lower_val)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_regate_record(record: dict[str, Any], path: Path) -> None:
    """Check a single ReGate record has all required fields."""
    sid = record.get("sample_id", "?")
    for field in ("p_answer", "p_wo", "p_nt"):
        val = record.get(field)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"{path}: {sid}: missing or non-numeric {field}")
        if not 0.0 <= float(val) <= 1.0:
            raise ValueError(f"{path}: {sid}: {field} out of [0, 1]")
    candidate = record.get("answer_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{path}: {sid}: missing answer_candidate object")


def validate_all(
    base_records: dict[str, dict[str, Any]],
    sft_records: dict[str, dict[str, Any]],
    regate_records: dict[str, dict[str, Any]],
    dataset: str,
    base_path: Path,
    sft_path: Path,
    regate_path: Path,
) -> None:
    """Validate cross-file consistency for a single dataset."""
    expected = EXPECTED_SIZES.get(dataset)
    if expected is not None:
        for name, recs, p in [
            ("base", base_records, base_path),
            ("sft", sft_records, sft_path),
            ("regate", regate_records, regate_path),
        ]:
            if len(recs) != expected:
                raise ValueError(
                    f"{p}: {name} {dataset} has {len(recs)} records, expected {expected}"
                )

    # Cross-file consistency
    validate_matched_records(dataset, regate_records, base_records)
    validate_matched_records(dataset, regate_records, sft_records)

    # ReGate field validation
    for sid, rec in regate_records.items():
        validate_regate_record(rec, regate_path)

    scope = GROUNDING_SCOPE.get(dataset)
    if scope is not None:
        for sid, rec in regate_records.items():
            gt = rec["gt"]
            if gt.get("decision") != "answer":
                continue
            has_box = gt.get("box") is not None
            if scope == "bbox" and not has_box:
                raise ValueError(f"{regate_path}: {sid}: bbox grounding requires a GT box")
            if scope == "label_only" and has_box:
                raise ValueError(f"{regate_path}: {sid}: label-only grounding requires GT box=null")

    # Base/SFT must have pred
    for name, recs, p in [("base", base_records, base_path), ("sft", sft_records, sft_path)]:
        for sid, rec in recs.items():
            if not isinstance(rec.get("pred"), dict):
                raise ValueError(f"{p}: {sid}: missing pred in {name}")


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------


def _plot(
    dataset: str,
    curve: list[NativeRegateMetric],
    fixed_points: list[NativeRegateMetric],
    argmax: NativeRegateMetric,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figures = (
        ("risk_coverage", "total_coverage", "selective_risk", "Coverage", "Selective risk"),
        (
            "hallucination_retention",
            "supported_query_retention",
            "unsupported_hallucination_rate",
            "Supported-query retention",
            "Unsupported hallucination rate",
        ),
        (
            "strict3_coverage",
            "total_coverage",
            "strict_3_accuracy",
            "Coverage",
            "Strict-3 accuracy",
        ),
    )
    for suffix, x_key, y_key, x_label, y_label in figures:
        fig, axis = plt.subplots(figsize=(6.2, 4.6))
        axis.plot(
            [getattr(pt, x_key) for pt in curve],
            [getattr(pt, y_key) for pt in curve],
            label="ReGate threshold sweep",
            linewidth=2,
        )
        axis.scatter(
            [getattr(argmax, x_key)],
            [getattr(argmax, y_key)],
            marker="*",
            s=120,
            label="ReGate argmax",
            zorder=3,
        )
        for pt in fixed_points:
            axis.scatter(
                [getattr(pt, x_key)],
                [getattr(pt, y_key)],
                s=35,
                label=pt.method,
                zorder=3,
            )
        axis.set(xlabel=x_label, ylabel=y_label, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_{suffix}.pdf")
        fig.savefig(output_dir / f"{dataset}_{suffix}.png", dpi=180)
        plt.close(fig)


def _plot_paper_hallucination_retention(
    curve: list[NativeRegateMetric],
    base_point: NativeRegateMetric,
    sft_point: NativeRegateMetric,
    argmax: NativeRegateMetric,
    matched_sft: dict[str, float],
    output_dir: Path,
) -> None:
    """Plot the comprehensive operating region used by the matched-retention claim.

    Publication figure (AAAI double-column ``figure*``): the main axes zoom into
    the matched-retention operating region; a small inset shows the complete
    threshold sweep on [0, 1] x [0, 1] with the zoom window marked, so the crop
    is explicit rather than implicit.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    curve_x = [pt.supported_query_retention for pt in curve]
    curve_y = [pt.unsupported_hallucination_rate for pt in curve]
    matched_x = matched_sft["target_supported_query_retention"]
    matched_y = matched_sft["interpolated_unsupported_hallucination_rate"]
    baseline_fca = sft_point.unsupported_hallucination_rate
    if not math.isfinite(baseline_fca) or baseline_fca <= 0.0:
        raise ValueError(f"SFT baseline FCA must be positive, got {baseline_fca!r}")
    if not math.isfinite(matched_y) or matched_y < 0.0:
        raise ValueError(f"Matched ReGate FCA must be finite and non-negative, got {matched_y!r}")
    relative_reduction = 1.0 - matched_y / baseline_fca

    # Okabe-Ito colorblind-safe palette; marker shapes disambiguate in grayscale.
    color_sweep = "#0072B2"  # blue
    color_base = "#E69F00"  # orange
    color_sft = "#009E73"  # bluish green
    color_argmax = "#CC79A7"  # reddish purple

    # Zoomed operating region shown in the main axes.
    zoom_x = (0.780, 1.004)
    zoom_y = (0.0, 0.235)

    rc = {
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, axis = plt.subplots(figsize=(7.13, 3.55))

        # ---- Main axes: zoomed operating region ----
        (sweep_handle,) = axis.plot(
            curve_x,
            curve_y,
            color=color_sweep,
            linewidth=1.5,
            solid_capstyle="round",
            zorder=2,
            label="ReGate threshold sweep",
        )
        base_handle = axis.scatter(
            base_point.supported_query_retention,
            base_point.unsupported_hallucination_rate,
            marker="D",
            s=38,
            color=color_base,
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
            label="Base native",
        )
        sft_handle = axis.scatter(
            sft_point.supported_query_retention,
            sft_point.unsupported_hallucination_rate,
            marker="s",
            s=42,
            color=color_sft,
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
            label="SFT native",
        )
        argmax_handle = axis.scatter(
            argmax.supported_query_retention,
            argmax.unsupported_hallucination_rate,
            marker="*",
            s=120,
            color=color_argmax,
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
            label="ReGate argmax",
        )
        matched_handle = axis.scatter(
            matched_x,
            matched_y,
            marker="o",
            s=44,
            color=color_sweep,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="ReGate matched",
        )

        # ---- Single key annotation: FCA drop at matched retention ----
        axis.annotate(
            "",
            xy=(matched_x, matched_y + 0.007),
            xytext=(matched_x, sft_point.unsupported_hallucination_rate - 0.007),
            arrowprops={
                "arrowstyle": "->,head_width=0.18,head_length=0.32",
                "color": "0.25",
                "linewidth": 0.9,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=3,
        )
        axis.text(
            matched_x + 0.006,
            (matched_y + sft_point.unsupported_hallucination_rate) / 2.0,
            f"\u2212{relative_reduction:.1%} relative FCA",
            fontsize=8,
            color="0.2",
            va="center",
            ha="left",
        )

        # ---- Restrained corner cue for the preferred direction ----
        axis.text(
            0.99,
            0.03,
            "better \u2198",
            transform=axis.transAxes,
            fontsize=7,
            color="0.45",
            ha="right",
            va="bottom",
        )

        axis.set_xlim(*zoom_x)
        axis.set_ylim(*zoom_y)
        axis.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
        axis.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
        axis.set_xlabel("Supported-query retention")
        axis.set_ylabel("False commitment rate (FCA)")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

        axis.legend(
            handles=[base_handle, sft_handle, sweep_handle, matched_handle, argmax_handle],
            loc="upper left",
            bbox_to_anchor=(0.295, 1.0),
            frameon=False,
            borderaxespad=0.0,
            handletextpad=0.5,
            labelspacing=0.35,
        )

        # ---- Inset: full threshold sweep with the zoom window marked ----
        inset = axis.inset_axes((0.565, 0.50, 0.185, 0.46))
        inset.plot(curve_x, curve_y, color=color_sweep, linewidth=0.8, zorder=2)
        inset.add_patch(
            Rectangle(
                (zoom_x[0], zoom_y[0]),
                zoom_x[1] - zoom_x[0],
                zoom_y[1] - zoom_y[0],
                facecolor="0.55",
                alpha=0.18,
                edgecolor="0.35",
                linewidth=0.6,
                zorder=1,
            )
        )
        inset.set_xlim(-0.03, 1.03)
        inset.set_ylim(-0.03, 1.03)
        inset.set_xticks([0.0, 0.5, 1.0])
        inset.set_yticks([0.0, 0.5, 1.0])
        inset.tick_params(labelsize=6, length=2, width=0.6, pad=1.5)
        for spine in inset.spines.values():
            spine.set_linewidth(0.6)
        inset.text(
            0.06,
            0.94,
            "full sweep",
            transform=inset.transAxes,
            fontsize=6.5,
            color="0.35",
            va="top",
            ha="left",
        )

        fig.tight_layout(pad=0.4)
        stem = output_dir / "comprehensive_hallucination_retention_paper"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_named_path(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} expects NAME=PATH")
    name, raw = value.split("=", 1)
    if not name or not raw:
        raise argparse.ArgumentTypeError(f"{option} expects NAME=PATH")
    return name, Path(raw)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native Base/SFT evaluation and ReGate analysis")
    parser.add_argument(
        "--base",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="Base model predictions JSONL; repeat per dataset",
    )
    parser.add_argument(
        "--sft",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="SFT model predictions JSONL; repeat per dataset",
    )
    parser.add_argument(
        "--regate",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="ReGate candidate predictions JSONL; repeat per dataset",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    base_paths = dict(_parse_named_path(v, "--base") for v in args.base)
    sft_paths = dict(_parse_named_path(v, "--sft") for v in args.sft)
    regate_paths = dict(_parse_named_path(v, "--regate") for v in args.regate)

    all_datasets = set(base_paths) | set(sft_paths) | set(regate_paths)
    for d in all_datasets:
        if d not in base_paths or d not in sft_paths or d not in regate_paths:
            parser.error(f"Dataset {d!r} must have --base, --sft, and --regate")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_fixed: list[NativeRegateMetric] = []
    all_argmax: list[NativeRegateMetric] = []
    all_curves: dict[str, list[NativeRegateMetric]] = {}
    all_matched: list[dict[str, float]] = []
    summary: dict[str, Any] = {}

    for dataset in sorted(all_datasets):
        base_recs = load_jsonl(base_paths[dataset])
        sft_recs = load_jsonl(sft_paths[dataset])
        regate_recs = load_jsonl(regate_paths[dataset])

        validate_all(
            base_recs,
            sft_recs,
            regate_recs,
            dataset,
            base_paths[dataset],
            sft_paths[dataset],
            regate_paths[dataset],
        )

        # ---- Native fixed points ----
        base_point = native_point(dataset, "Base_native", base_recs, args.iou_threshold)
        sft_point = native_point(dataset, "SFT_native_observed", sft_recs, args.iou_threshold)

        # ---- SFT_native_gate_shared_candidate ----
        shared_point = gate_shared_candidate_point(
            dataset, sft_recs, regate_recs, args.iou_threshold
        )

        # ---- ReGate curve ----
        curve, argmax = regate_full_curve(dataset, regate_recs, args.iou_threshold)
        all_curves[dataset] = curve
        all_argmax.append(argmax)

        # Accumulate fixed points
        all_fixed.extend([base_point, sft_point, shared_point, argmax])

        # ---- Matched retention ----
        for target_pt in (base_point, sft_point):
            emp, interp = matched_retention_to_target(curve, target_pt)
            all_matched.append(
                {
                    "dataset": dataset,
                    "baseline_method": target_pt.method,
                    "baseline_supported_query_retention": target_pt.supported_query_retention,
                    "baseline_unsupported_hallucination_rate": target_pt.unsupported_hallucination_rate,
                    "baseline_selective_risk": target_pt.selective_risk,
                    "baseline_strict_3_accuracy": target_pt.strict_3_accuracy,
                    **interp,
                }
            )

        # ---- Write per-dataset curve CSV ----
        _write_csv(
            args.output_dir / f"native_regate_diagnostic_{dataset}_curve.csv",
            [asdict(pt) for pt in curve],
        )

    # ---- Write comparison CSV ----
    _write_csv(
        args.output_dir / "native_regate_diagnostic_comparison.csv",
        [asdict(pt) for pt in all_fixed],
    )

    # ---- Write matched retention CSV ----
    _write_csv(
        args.output_dir / "native_regate_diagnostic_matched_retention.csv",
        all_matched,
    )

    # ---- Write summary JSON ----
    summary = {
        "datasets": sorted(all_datasets),
        "iou_threshold": args.iou_threshold,
        "fixed_points": [asdict(pt) for pt in all_fixed],
        "matched_retention": all_matched,
    }
    (args.output_dir / "native_regate_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # ---- Optional plots ----
    if not args.no_plots:
        for dataset, curve in all_curves.items():
            fixed_ds = [pt for pt in all_fixed if pt.dataset == dataset]
            argmax_ds = next(pt for pt in all_argmax if pt.dataset == dataset)
            _plot(dataset, curve, fixed_ds, argmax_ds, args.output_dir)

        if "comprehensive" in all_curves:
            comprehensive_base = next(
                pt
                for pt in all_fixed
                if pt.dataset == "comprehensive" and pt.method == "Base_native"
            )
            comprehensive_sft = next(
                pt
                for pt in all_fixed
                if pt.dataset == "comprehensive" and pt.method == "SFT_native_observed"
            )
            comprehensive_argmax = next(pt for pt in all_argmax if pt.dataset == "comprehensive")
            comprehensive_matched_sft = next(
                row
                for row in all_matched
                if row["dataset"] == "comprehensive"
                and row["baseline_method"] == "SFT_native_observed"
            )
            _plot_paper_hallucination_retention(
                all_curves["comprehensive"],
                comprehensive_base,
                comprehensive_sft,
                comprehensive_argmax,
                comprehensive_matched_sft,
                args.output_dir,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
