from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reason_abstain.eval.metrics import box_iou, score_answerable
from reason_abstain.eval.parser import normalize_label


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
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
    errors: int  # selective risk errors

    # Coverage / retention / hallucination (no change)
    total_coverage: float
    supported_query_retention: float
    unsupported_hallucination_rate: float
    selective_risk: float

    # ---- Grounding / label fields for bbox datasets (comprehensive, refcocog) ----
    # A_Ground over ALL supported: mean IoU of label-correct answers (abstain → 0)
    a_ground_mean_iou_over_supported: float
    # Formal A_Ground_c: raw mean IoU conditional on answered-supported
    a_ground_conditional_mean_iou: float
    # Binary success (label + IoU >= 0.5) over ALL supported (abstain → 0)
    grounding_success_at_iou_0_5_over_supported: float
    # Binary success (label + IoU >= 0.5) conditional on answered-supported
    grounding_success_at_iou_0_5_conditional: float

    # ---- PlantWild (no bbox): label-only success ----
    label_success_over_supported: float  # label-correct / supported (abstain → 0)
    label_success_conditional: float  # label-correct / answered-supported

    # Scope: "bbox" or "label_only"
    grounding_scope: str

    # ---- Strict-3 ----
    strict_3_correct: int
    strict_3_accuracy: float
    format_valid: int  # count of valid predictions
    format_valid_rate: float


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else math.nan


# ---------------------------------------------------------------------------
# Helpers for building structured prediction dicts (ReGate candidates)
# ---------------------------------------------------------------------------


def _validate_probability(value: Any, sample_id: str, name: str = "p_answer") -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{sample_id}: {name} is not numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{sample_id}: invalid {name} {value!r}")
    return value


def _candidate_prediction(record: dict[str, Any]) -> dict[str, Any]:
    """Build a structured prediction dict from answer_candidate."""
    candidate = record.get("answer_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{record['sample_id']}: missing answer_candidate")
    if candidate.get("valid") is not True:
        return {"valid": False, "decision": None, "label": None, "box": None, "reason": None}
    return {
        "valid": True,
        "decision": "answer",
        "label": candidate.get("label"),
        "box": candidate.get("box"),
        "reason": None,
    }


def _abstain_prediction(record: dict[str, Any], reason: str) -> dict[str, Any]:
    """Build a structured prediction dict for abstain with a given reason."""
    return {"valid": True, "decision": "abstain", "label": None, "box": None, "reason": reason}


def _is_supported(sample: dict[str, Any]) -> bool:
    return sample.get("decision") == "answer"


# ---------------------------------------------------------------------------
# A_Ground computation (matches aggregate_metrics in reason_abstain.eval.metrics)
# ---------------------------------------------------------------------------


def _compute_a_ground(
    records: dict[str, dict[str, Any]],
    answered: dict[str, bool],
    answered_preds: dict[str, dict[str, Any]],
    iou_threshold: float,
    has_bbox: bool,
) -> tuple[float, float, float, float, float, float]:
    """Compute comprehensive grounding metrics.

    For bbox datasets (sample has GT box):
      - a_ground_mean_iou_over_supported: mean IoU (label-correct only, abstain→0) over ALL supported
      - a_ground_conditional_mean_iou: raw mean IoU over answered-supported only
      - grounding_success_at_iou_0_5_over_supported: binary success (label+IoU>=threshold) over ALL supported
      - grounding_success_at_iou_0_5_conditional: binary success over answered-supported only

    For label-only datasets (no GT box, e.g. PlantWild):
      - The four bbox fields are NaN (not applicable)
      - label_success_over_supported: label-correct / supported (abstain→0)
      - label_success_conditional: label-correct / answered-supported

    Returns (a_ground_mean_iou_over_supported, a_ground_conditional_mean_iou,
             grounding_success_at_iou_0_5_over_supported, grounding_success_at_iou_0_5_conditional,
             label_success_over_supported, label_success_conditional).
    """
    supported_ids = {sid for sid, rec in records.items() if _is_supported(rec["gt"])}
    n_supported = len(supported_ids)
    n_answered_supported = sum(1 for sid in supported_ids if answered.get(sid))

    # bbox metrics
    iou_sum_all = 0.0  # over all supported (abstain→0)
    iou_sum_answered = 0.0  # over answered-supported only
    n_answered_box = 0  # answered-supported samples that have a GT box
    correct_iou_binary = 0  # answered-supported with label+IoU>=threshold (among those with GT box)
    n_with_box = 0  # supported samples that have a GT box

    # label-only metrics
    correct_label_all = 0  # label-correct among ALL supported (abstain→0)
    correct_label_answered = 0  # label-correct among answered-supported

    for sid in supported_ids:
        rec = records[sid]
        gt = rec["gt"]
        gt_box = gt.get("box")
        has_box = bool(gt_box)

        # For label-only correctness, also check if GT has a box
        # PlantWild: gt_box is None, so it's label-only
        # comprehensive/refcocog: gt_box is not None

        if not answered.get(sid):
            # Abstained on supported: contributes 0 to all "over_supported" metrics
            continue

        pred = answered_preds[sid]
        pred_label = pred.get("label") if pred.get("valid") else None
        pred_box = pred.get("box") if pred.get("valid") else None

        label_ok = pred_label is not None and normalize_label(pred_label) == normalize_label(
            gt.get("label")
        )

        # ---- Label-only correctness (always applicable) ----
        if label_ok:
            correct_label_all += 1
            correct_label_answered += 1

        # ---- Bbox metrics ----
        if has_box:
            n_with_box += 1
            n_answered_box += 1
            if pred_box:
                iou = box_iou(pred_box, gt_box)
                iou_sum_answered += iou
                if label_ok:
                    iou_sum_all += iou
                if label_ok and iou >= iou_threshold:
                    correct_iou_binary += 1
            # else: label wrong or no pred_box → contributes 0 to IoU sums
            # For "over_supported" binary success: also 0
        # else: no GT box (PlantWild) → label-only correctness applies, bbox metrics remain 0

    # Bbox metrics (only meaningful if dataset has GT boxes)
    if has_bbox:
        a_ground_mean_iou = _ratio(iou_sum_all, n_supported)
        a_ground_cond_iou = _ratio(iou_sum_answered, n_answered_box) if n_answered_box else 0.0
        grounding_success_over = _ratio(correct_iou_binary, n_supported)
        grounding_success_cond = _ratio(correct_iou_binary, n_answered_supported)
    else:
        a_ground_mean_iou = math.nan
        a_ground_cond_iou = math.nan
        grounding_success_over = math.nan
        grounding_success_cond = math.nan

    # Label-only metrics (always meaningful)
    label_success_over = _ratio(correct_label_all, n_supported)
    label_success_cond = _ratio(correct_label_answered, n_answered_supported)

    return (
        a_ground_mean_iou,
        a_ground_cond_iou,
        grounding_success_over,
        grounding_success_cond,
        label_success_over,
        label_success_cond,
    )


# ---------------------------------------------------------------------------
# Strict-3 scoring
# ---------------------------------------------------------------------------


def _strict3_correct_count(
    records: dict[str, dict[str, Any]],
    answered: dict[str, bool],
    answered_preds: dict[str, dict[str, Any]],
    iou_threshold: float,
) -> int:
    """Strict-3: supported = correct answer (label correct, answer state).
    Unsupported = correct abstain reason. No bbox IoU involved in Strict-3."""
    correct_count = 0
    for sid, rec in records.items():
        gt = rec["gt"]
        pred = answered_preds.get(sid)
        if pred is None or not pred.get("valid"):
            continue
        if _is_supported(gt):
            # Supported: requires answer state + correct label only (no bbox check)
            if pred.get("decision") == "answer" and normalize_label(
                pred.get("label")
            ) == normalize_label(gt.get("label")):
                correct_count += 1
        else:
            # Unsupported: requires correct abstain reason
            if pred.get("decision") == "abstain" and pred.get("reason") == gt.get("reason"):
                correct_count += 1
    return correct_count


# ---------------------------------------------------------------------------
# Build answered_preds dict for a given decision policy
# ---------------------------------------------------------------------------


def _build_answered_preds_from_candidate(
    records: dict[str, dict[str, Any]],
    answered: dict[str, bool],
) -> dict[str, dict[str, Any]]:
    """For ReGate: use answer_candidate when answered, abstain pred when not answered."""
    preds = {}
    for sid, rec in records.items():
        if answered.get(sid):
            preds[sid] = _candidate_prediction(rec)
        else:
            # Determine which reason to use (argmax or threshold-based)
            # For threshold sweep: p_wo/p_nt argmax determines reason
            p_wo = _validate_probability(rec.get("p_wo", 0.0), sid, "p_wo")
            p_nt = _validate_probability(rec.get("p_nt", 0.0), sid, "p_nt")
            reason = "wrong_object" if p_wo >= p_nt else "no_target"
            preds[sid] = _abstain_prediction(rec, reason)
    return preds


def _build_answered_preds_from_native(
    records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """For native baselines: use the record's pred dict directly."""
    preds = {}
    for sid, rec in records.items():
        pred = rec.get("pred")
        if isinstance(pred, dict):
            preds[sid] = pred
        else:
            preds[sid] = {"valid": False}
    return preds


# ---------------------------------------------------------------------------
# _summarize
# ---------------------------------------------------------------------------


def _summarize(
    dataset: str,
    method: str,
    policy: str,
    threshold: float | None,
    records: dict[str, dict[str, Any]],
    answered: dict[str, bool],
    answered_preds: dict[str, dict[str, Any]],
    correct: dict[str, bool],
    iou_threshold: float,
) -> OperatingPoint:
    supported_ids = {sid for sid, rec in records.items() if _is_supported(rec["gt"])}
    unknown = [
        sid
        for sid, rec in records.items()
        if rec["gt"].get("decision") not in {"answer", "abstain"}
    ]
    if unknown:
        raise ValueError(f"{dataset}: unknown gt decision for {unknown[0]!r}")
    unsupported_ids = records.keys() - supported_ids
    answered_ids = {sid for sid, val in answered.items() if val}
    answered_supported = answered_ids & supported_ids
    answered_unsupported = answered_ids & unsupported_ids
    errors = sum(1 for sid in answered_ids if not correct.get(sid))

    format_valid = sum(pred.get("valid") is True for pred in answered_preds.values())

    # Determine grounding scope
    has_any_box = any(
        bool(rec["gt"].get("box")) for sid, rec in records.items() if _is_supported(rec["gt"])
    )
    scope = "bbox" if has_any_box else "label_only"

    # A_Ground and conditional grounding
    (
        a_ground_mean_iou,
        a_ground_cond_iou,
        grounding_success_over,
        grounding_success_cond,
        label_success_over,
        label_success_cond,
    ) = _compute_a_ground(records, answered, answered_preds, iou_threshold, has_any_box)

    # Strict-3
    strict_3_correct = _strict3_correct_count(records, answered, answered_preds, iou_threshold)

    return OperatingPoint(
        dataset=dataset,
        method=method,
        policy=policy,
        threshold=threshold,
        total=len(records),
        supported=len(supported_ids),
        unsupported=len(unsupported_ids),
        answered=len(answered_ids),
        answered_supported=len(answered_supported),
        answered_unsupported=len(answered_unsupported),
        errors=errors,
        total_coverage=_ratio(len(answered_ids), len(records)),
        supported_query_retention=_ratio(len(answered_supported), len(supported_ids)),
        unsupported_hallucination_rate=_ratio(len(answered_unsupported), len(unsupported_ids)),
        selective_risk=_ratio(errors, len(answered_ids)),
        a_ground_mean_iou_over_supported=a_ground_mean_iou,
        a_ground_conditional_mean_iou=a_ground_cond_iou,
        grounding_success_at_iou_0_5_over_supported=grounding_success_over,
        grounding_success_at_iou_0_5_conditional=grounding_success_cond,
        label_success_over_supported=label_success_over,
        label_success_conditional=label_success_cond,
        grounding_scope=scope,
        strict_3_correct=strict_3_correct,
        strict_3_accuracy=_ratio(strict_3_correct, len(records)),
        format_valid=format_valid,
        format_valid_rate=_ratio(format_valid, len(records)),
    )


# ---------------------------------------------------------------------------
# ReGate curve
# ---------------------------------------------------------------------------


def regate_curve(
    dataset: str,
    records: dict[str, dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[list[OperatingPoint], OperatingPoint]:
    probabilities: dict[str, float] = {}
    correct: dict[str, bool] = {}
    answered_preds: dict[str, dict[str, Any]] = {}

    for sid, rec in records.items():
        probabilities[sid] = _validate_probability(rec.get("p_answer"), sid)

    # Pre-compute correctness and IoU for ReGate candidate predictions
    for sid, rec in records.items():
        candidate_pred = _candidate_prediction(rec)
        gt = rec["gt"]
        correct[sid] = _is_supported(gt) and score_answerable(candidate_pred, gt, iou_threshold)

    unique_thresholds = sorted(set(probabilities.values()), reverse=True)
    zero_coverage_threshold = math.nextafter(max(unique_thresholds), math.inf)
    thresholds = [zero_coverage_threshold, *unique_thresholds]

    curve = []
    for threshold in thresholds:
        answered: dict[str, bool] = {sid: prob >= threshold for sid, prob in probabilities.items()}
        answered_preds = _build_answered_preds_from_candidate(records, answered)
        curve.append(
            _summarize(
                dataset,
                "ReGate",
                "p_supported_threshold",
                threshold,
                records,
                answered,
                answered_preds,
                correct,
                iou_threshold,
            )
        )

    # Argmax point
    argmax_answered: dict[str, bool] = {}
    for sid, rec in records.items():
        p_answer = probabilities[sid]
        p_wo = _validate_probability(rec.get("p_wo"), sid, "p_wo")
        p_nt = _validate_probability(rec.get("p_nt"), sid, "p_nt")
        argmax_answered[sid] = p_answer >= p_wo and p_answer >= p_nt

    argmax_preds = _build_answered_preds_from_candidate(records, argmax_answered)
    argmax_point = _summarize(
        dataset,
        "ReGate",
        "three_class_argmax",
        None,
        records,
        argmax_answered,
        argmax_preds,
        correct,
        iou_threshold,
    )
    return curve, argmax_point


# ---------------------------------------------------------------------------
# Baseline point
# ---------------------------------------------------------------------------


def baseline_point(
    dataset: str,
    method: str,
    records: dict[str, dict[str, Any]],
    iou_threshold: float = 0.5,
) -> OperatingPoint:
    answered: dict[str, bool] = {}
    correct: dict[str, bool] = {}
    for sid, rec in records.items():
        pred = rec.get("pred")
        if not isinstance(pred, dict):
            raise ValueError(f"{sid}: missing pred object")
        answered[sid] = pred.get("valid") is True and pred.get("decision") == "answer"
        correct[sid] = _is_supported(rec["gt"]) and score_answerable(pred, rec["gt"], iou_threshold)
    answered_preds = _build_answered_preds_from_native(records)
    return _summarize(
        dataset,
        method,
        "native_decision",
        None,
        records,
        answered,
        answered_preds,
        correct,
        iou_threshold,
    )


# ---------------------------------------------------------------------------
# Matched retention
# ---------------------------------------------------------------------------


def validate_matched_records(
    dataset: str,
    regate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> None:
    regate_ids = set(regate)
    baseline_ids = set(baseline)
    if regate_ids != baseline_ids:
        missing = sorted(regate_ids - baseline_ids)
        extra = sorted(baseline_ids - regate_ids)
        raise ValueError(
            f"{dataset}: baseline sample_id mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    for sid in regate_ids:
        regate_gt = regate[sid]["gt"]
        baseline_gt = baseline[sid]["gt"]
        for key in ("decision", "label", "box", "reason"):
            if regate_gt.get(key) != baseline_gt.get(key):
                raise ValueError(f"{dataset}/{sid}: gt mismatch in {key}")


def matched_retention(
    curve: list[OperatingPoint], baseline: OperatingPoint
) -> tuple[OperatingPoint, dict[str, float | str]]:
    target = baseline.supported_query_retention
    eligible = [
        point
        for point in curve
        if point.supported_query_retention >= target and point.threshold is not None
    ]
    if not eligible:
        raise ValueError(f"cannot match retention {target} for {baseline.method}")
    empirical = max(eligible, key=lambda point: float(point.threshold))
    interpolated = _interpolate_at_retention(curve, target)
    return empirical, {
        "dataset": baseline.dataset,
        "baseline_method": baseline.method,
        "baseline_supported_query_retention": target,
        "target_supported_query_retention": target,
        "regate_empirical_threshold": empirical.threshold,
        "regate_empirical_unsupported_hallucination_rate": empirical.unsupported_hallucination_rate,
        "regate_empirical_selective_risk": empirical.selective_risk,
        "regate_empirical_strict_3_accuracy": empirical.strict_3_accuracy,
        "interpolated_unsupported_hallucination_rate": interpolated[0],
        "interpolated_selective_risk": interpolated[1],
        "interpolated_strict_3_accuracy": interpolated[2],
    }


def _interpolate_at_retention(
    curve: list[OperatingPoint], target: float
) -> tuple[float, float, float]:
    points = sorted(curve, key=lambda point: point.supported_query_retention)
    for point in points:
        if math.isclose(point.supported_query_retention, target, abs_tol=1e-12):
            return (
                point.unsupported_hallucination_rate,
                point.selective_risk,
                point.strict_3_accuracy,
            )
    lower = max(
        (point for point in points if point.supported_query_retention < target),
        key=lambda point: point.supported_query_retention,
    )
    upper = min(
        (point for point in points if point.supported_query_retention > target),
        key=lambda point: point.supported_query_retention,
    )
    weight = (target - lower.supported_query_retention) / (
        upper.supported_query_retention - lower.supported_query_retention
    )
    return (
        _linear(lower.unsupported_hallucination_rate, upper.unsupported_hallucination_rate, weight),
        _linear(lower.selective_risk, upper.selective_risk, weight),
        _linear(lower.strict_3_accuracy, upper.strict_3_accuracy, weight),
    )


def _linear(start: float, end: float, weight: float) -> float:
    if math.isnan(start) or math.isnan(end):
        return math.nan
    return start + weight * (end - start)


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot(
    dataset: str,
    curve: list[OperatingPoint],
    baselines: list[OperatingPoint],
    argmax_point: OperatingPoint,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure_specs = (
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
    for suffix, x_key, y_key, x_label, y_label in figure_specs:
        fig, axis = plt.subplots(figsize=(6.2, 4.6))
        axis.plot(
            [getattr(point, x_key) for point in curve],
            [getattr(point, y_key) for point in curve],
            label="ReGate threshold sweep",
            linewidth=2,
        )
        axis.scatter(
            [getattr(argmax_point, x_key)],
            [getattr(argmax_point, y_key)],
            marker="*",
            s=120,
            label="ReGate argmax",
            zorder=3,
        )
        for point in baselines:
            axis.scatter(
                [getattr(point, x_key)],
                [getattr(point, y_key)],
                s=35,
                label=point.method,
                zorder=3,
            )
        axis.set(xlabel=x_label, ylabel=y_label, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_{suffix}.pdf")
        fig.savefig(output_dir / f"{dataset}_{suffix}.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_named_path(value: str, separator: str, option: str) -> tuple[str, Path]:
    if separator not in value:
        raise argparse.ArgumentTypeError(f"{option} expects NAME{separator}PATH")
    name, raw_path = value.split(separator, 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(f"{option} expects NAME{separator}PATH")
    return name, Path(raw_path)


def _parse_baseline(value: str) -> tuple[str, str, Path]:
    method_dataset, path = _parse_named_path(value, "=", "--baseline")
    if ":" not in method_dataset:
        raise argparse.ArgumentTypeError("--baseline expects METHOD:DATASET=PATH")
    method, dataset = method_dataset.split(":", 1)
    return method, dataset, path


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number}: missing sample_id")
            if sample_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
            if not isinstance(record.get("gt"), dict):
                raise ValueError(f"{path}:{line_number}: missing gt object")
            records[sample_id] = record
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _point_to_native_comparison_row(
    point: OperatingPoint,
    records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert an OperatingPoint to a row in native_comparison.csv format.

    Grounding scope is stored in the OperatingPoint itself.
    """
    scope = point.grounding_scope

    d = asdict(point)
    # Build output with new clear field names and no misleading grounding_accuracy
    return {
        "dataset": d["dataset"],
        "method": d["method"],
        "policy": d["policy"],
        "threshold": d["threshold"],
        "total": d["total"],
        "supported": d["supported"],
        "unsupported": d["unsupported"],
        "answered": d["answered"],
        "answered_supported": d["answered_supported"],
        "answered_unsupported": d["answered_unsupported"],
        "strict_3_correct": d["strict_3_correct"],
        "selective_risk_errors": d["errors"],
        "format_valid": d["format_valid"],
        "total_coverage": d["total_coverage"],
        "supported_query_retention": d["supported_query_retention"],
        "unsupported_hallucination_rate": d["unsupported_hallucination_rate"],
        "strict_3_accuracy": d["strict_3_accuracy"],
        "selective_risk": d["selective_risk"],
        # Grounding fields — unambiguous names
        "a_ground_mean_iou_over_supported": d["a_ground_mean_iou_over_supported"],
        "a_ground_conditional_mean_iou": d["a_ground_conditional_mean_iou"],
        "grounding_success_at_iou_0_5_over_supported": d[
            "grounding_success_at_iou_0_5_over_supported"
        ],
        "grounding_success_at_iou_0_5_conditional": d["grounding_success_at_iou_0_5_conditional"],
        "label_success_over_supported": d["label_success_over_supported"],
        "label_success_conditional": d["label_success_conditional"],
        "format_valid_rate": d["format_valid_rate"],
        "grounding_scope": scope,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regate",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="ReGate candidate prediction JSONL; repeat per dataset",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="METHOD:DATASET=PATH",
        help="Baseline prediction JSONL; repeat per method and dataset",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--summary-only", action="store_true", help="Re-generate summary.json from existing CSVs"
    )
    args = parser.parse_args()

    regate_paths = dict(_parse_named_path(value, "=", "--regate") for value in args.regate)
    if len(regate_paths) != len(args.regate):
        parser.error("duplicate --regate dataset")
    baseline_specs = [_parse_baseline(value) for value in args.baseline]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_baselines: list[OperatingPoint] = []
    all_argmax: list[OperatingPoint] = []
    matched_rows: list[dict[str, Any]] = []
    curves: dict[str, list[OperatingPoint]] = {}
    dataset_baselines: dict[str, list[OperatingPoint]] = {}

    regate_records = {dataset: load_jsonl(path) for dataset, path in regate_paths.items()}

    for dataset, records in regate_records.items():
        curve, argmax_point = regate_curve(dataset, records, args.iou_threshold)
        curves[dataset] = curve
        all_argmax.append(argmax_point)
        _write_csv(
            args.output_dir / f"{dataset}_regate_curve.csv",
            [_point_to_native_comparison_row(point, records) for point in curve],
        )

    seen_baselines: set[tuple[str, str]] = set()
    for method, dataset, path in baseline_specs:
        if dataset not in regate_records:
            parser.error(f"baseline dataset {dataset!r} has no matching --regate")
        key = (method, dataset)
        if key in seen_baselines:
            parser.error(f"duplicate baseline {method}:{dataset}")
        seen_baselines.add(key)
        records = load_jsonl(path)
        validate_matched_records(dataset, regate_records[dataset], records)
        point = baseline_point(dataset, method, records, args.iou_threshold)
        all_baselines.append(point)
        dataset_baselines.setdefault(dataset, []).append(point)
        empirical, interpolated = matched_retention(curves[dataset], point)
        matched_rows.append(
            {
                "dataset": dataset,
                "baseline_method": method,
                "baseline_supported_query_retention": point.supported_query_retention,
                "baseline_unsupported_hallucination_rate": point.unsupported_hallucination_rate,
                "baseline_selective_risk": point.selective_risk,
                "baseline_strict_3_accuracy": point.strict_3_accuracy,
                "target_supported_query_retention": point.supported_query_retention,
                "empirical_threshold": empirical.threshold,
                "empirical_unsupported_hallucination_rate": empirical.unsupported_hallucination_rate,
                "empirical_selective_risk": empirical.selective_risk,
                "empirical_strict_3_accuracy": empirical.strict_3_accuracy,
                **interpolated,
            }
        )

    # Build records-by-dataset for scope detection
    all_records: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, recs in regate_records.items():
        all_records[dataset] = recs
    for _, dataset, path in baseline_specs:
        if dataset not in all_records:
            all_records[dataset] = load_jsonl(path)

    _write_csv(
        args.output_dir / "baseline_points.csv",
        [_point_to_native_comparison_row(p, all_records.get(p.dataset)) for p in all_baselines],
    )
    _write_csv(
        args.output_dir / "regate_argmax_points.csv",
        [_point_to_native_comparison_row(p, all_records.get(p.dataset)) for p in all_argmax],
    )
    _write_csv(args.output_dir / "matched_retention.csv", matched_rows)

    # Write native_comparison.csv (all fixed points: baselines + argmax)
    native_comparison_rows = [
        _point_to_native_comparison_row(p, all_records.get(p.dataset))
        for p in all_baselines + all_argmax
    ]
    _write_csv(args.output_dir / "native_comparison.csv", native_comparison_rows)

    # Build and write summary.json
    summary = _build_summary(all_baselines, all_argmax, matched_rows, args.iou_threshold)
    _write_json(args.output_dir / "summary.json", summary)

    if not args.no_plots:
        for dataset, curve in curves.items():
            _plot(
                dataset,
                curve,
                dataset_baselines.get(dataset, []),
                next(point for point in all_argmax if point.dataset == dataset),
                args.output_dir,
            )
    return 0


def _build_summary(
    all_baselines: list[OperatingPoint],
    all_argmax: list[OperatingPoint],
    matched_rows: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    datasets: set[str] = set()
    fixed_points_dicts: list[dict[str, Any]] = []

    for point in all_baselines + all_argmax:
        datasets.add(point.dataset)
        d = asdict(point)
        fixed_points_dicts.append(d)

    # Deduplicate by (dataset, method) — keep last
    seen: set[tuple[str, str]] = set()
    deduped_points: list[dict[str, Any]] = []
    for d in fixed_points_dicts:
        key = (d["dataset"], d["method"])
        if key not in seen:
            seen.add(key)
            deduped_points.append(d)
        else:
            # Replace the earlier entry
            for i, existing in enumerate(deduped_points):
                if (existing["dataset"], existing["method"]) == key:
                    deduped_points[i] = d
                    break

    return {
        "datasets": sorted(datasets),
        "iou_threshold": iou_threshold,
        "fixed_points": deduped_points,
        "matched_retention": matched_rows,
    }


# ---------------------------------------------------------------------------
# Verification entry point
# ---------------------------------------------------------------------------


def verify_against_metrics_json(
    regate_pred_dir: Path,
    native_base_dir: Path,
    native_sft_dir: Path,
    iou_threshold: float = 0.5,
) -> None:
    """Compare analyzer-computed A_Ground against formal metrics.json values."""
    import sys

    problems: list[str] = []

    for dataset in ["comprehensive", "refcocog", "plantwild"]:
        # ReGate
        regate_pred_path = regate_pred_dir / dataset / "predictions.jsonl"
        regate_metrics_path = regate_pred_dir / dataset / "metrics.json"
        if regate_pred_path.exists() and regate_metrics_path.exists():
            records = load_jsonl(regate_pred_path)
            metrics_data = json.loads(regate_metrics_path.read_text(encoding="utf-8"))
            _, argmax = regate_curve(dataset, records, iou_threshold)

            formal_a_ground = metrics_data["metrics"]["grounding_reason"]["A_Ground"]
            formal_au = metrics_data["metrics"]["headline"]["FCA"]

            diff = abs(argmax.a_ground_mean_iou_over_supported - formal_a_ground)
            if dataset != "plantwild" and diff > 1e-10:
                problems.append(
                    f"ReGate {dataset}: A_Ground analyzer={argmax.a_ground_mean_iou_over_supported:.10f} "
                    f"formal={formal_a_ground:.10f} diff={diff:.2e}"
                )
            print(
                f"ReGate {dataset}: A_Ground={argmax.a_ground_mean_iou_over_supported:.6f} "
                f"(formal={formal_a_ground:.6f}), FCA={argmax.unsupported_hallucination_rate:.6f} "
                f"(formal={formal_au:.6f})",
                file=sys.stderr,
            )

        # Base native
        base_path = native_base_dir / dataset / "predictions.jsonl"
        base_metrics_path = native_base_dir / dataset / "metrics.json"
        if base_path.exists() and base_metrics_path.exists():
            records = load_jsonl(base_path)
            metrics_data = json.loads(base_metrics_path.read_text(encoding="utf-8"))
            point = baseline_point(dataset, "Base_native", records, iou_threshold)
            formal_a_ground = metrics_data["metrics"]["grounding_reason"]["A_Ground"]
            diff = abs(point.a_ground_mean_iou_over_supported - formal_a_ground)
            if dataset != "plantwild" and diff > 1e-10:
                problems.append(
                    f"Base_native {dataset}: A_Ground analyzer={point.a_ground_mean_iou_over_supported:.10f} "
                    f"formal={formal_a_ground:.10f} diff={diff:.2e}"
                )
            print(
                f"Base_native {dataset}: A_Ground={point.a_ground_mean_iou_over_supported:.6f} (formal={formal_a_ground:.6f})",
                file=sys.stderr,
            )

        # SFT native
        sft_path = native_sft_dir / dataset / "predictions.jsonl"
        sft_metrics_path = native_sft_dir / dataset / "metrics.json"
        if sft_path.exists() and sft_metrics_path.exists():
            records = load_jsonl(sft_path)
            metrics_data = json.loads(sft_metrics_path.read_text(encoding="utf-8"))
            point = baseline_point(dataset, "SFT_native_observed", records, iou_threshold)
            formal_a_ground = metrics_data["metrics"]["grounding_reason"]["A_Ground"]
            diff = abs(point.a_ground_mean_iou_over_supported - formal_a_ground)
            if dataset != "plantwild" and diff > 1e-10:
                problems.append(
                    f"SFT_native_observed {dataset}: A_Ground analyzer={point.a_ground_mean_iou_over_supported:.10f} "
                    f"formal={formal_a_ground:.10f} diff={diff:.2e}"
                )
            print(
                f"SFT_native_observed {dataset}: A_Ground={point.a_ground_mean_iou_over_supported:.6f} (formal={formal_a_ground:.6f})",
                file=sys.stderr,
            )

    if problems:
        print("\nVERIFICATION FAILED — discrepancies found:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        raise SystemExit(1)
    else:
        print("\nAll A_Ground values match formal metrics.json ✓", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
