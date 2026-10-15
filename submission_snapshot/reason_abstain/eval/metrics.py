from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .parser import normalize_label


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_answerable(
    prediction: dict[str, Any], sample: dict[str, Any], iou_threshold: float
) -> bool:
    if not prediction.get("valid") or prediction.get("decision") != "answer":
        return False
    if normalize_label(prediction.get("label")) != normalize_label(sample.get("label")):
        return False
    if not sample.get("box"):
        return True
    if not prediction.get("box"):
        return False
    return box_iou(prediction["box"], sample["box"]) >= iou_threshold


def score_unanswerable(prediction: dict[str, Any], sample: dict[str, Any]) -> bool:
    return (
        bool(prediction.get("valid"))
        and prediction.get("decision") == "abstain"
        and prediction.get("reason") == sample.get("reason")
    )


def false_confident_answer(prediction: dict[str, Any], sample: dict[str, Any]) -> bool:
    return sample.get("decision") == "abstain" and prediction.get("decision") == "answer"


def box_iou_pixel(
    box_a_rel: list[float],
    box_b_rel: list[float],
    image_width: int,
    image_height: int,
) -> float:
    """IoU in pixel space (for comparison with external benchmarks like AgroVG).

    Both inputs are in Qwen [0,1000] relative coordinates; they are scaled to the image's
    pixel dimensions before IoU. IoU is scale-invariant, so this equals ``box_iou`` for a
    single image; the function exists to make the pixel-space comparison explicit/auditable.
    """

    def to_pixels(box: list[float]) -> list[float]:
        return [
            box[0] / 1000.0 * image_width,
            box[1] / 1000.0 * image_height,
            box[2] / 1000.0 * image_width,
            box[3] / 1000.0 * image_height,
        ]

    return box_iou(to_pixels(box_a_rel), to_pixels(box_b_rel))


# --- 5-way confusion matrix (MM-AQA aligned) ---------------------------------------------
# For each (prediction, sample) pair, exactly one of these buckets is incremented.
# Mapping to the names used in the paper:
#   TP  : supported   & answered correctly (label + box@IoU)
#   FP  : supported   & answered with wrong label/box
#   FN  : supported   & abstained            == A-FU  (false unsupported)
#   TN  : unsupported & abstained correctly
#   AU  : unsupported & answered             == FCA   (unsupported hallucination)
# Invalid outputs are tracked separately (not folded into the matrix).

CONFUSION_KEYS = ("TP", "FP", "FN", "TN", "AU", "INVALID")


def classify_pair(
    prediction: dict[str, Any],
    sample: dict[str, Any],
    iou_threshold: float = 0.5,
) -> str:
    """Assign a (prediction, sample) pair to one 5-way confusion bucket (or INVALID)."""
    if not prediction.get("valid"):
        return "INVALID"

    gold_answerable = sample.get("decision") == "answer"
    pred_answer = prediction.get("decision") == "answer"

    if gold_answerable:
        if not pred_answer:
            return "FN"
        return "TP" if score_answerable(prediction, sample, iou_threshold) else "FP"
    # gold is unanswerable (abstain)
    return "AU" if pred_answer else "TN"


def confusion_matrix(
    pairs: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    counts = dict.fromkeys(CONFUSION_KEYS, 0)
    for prediction, sample in pairs:
        counts[classify_pair(prediction, sample, iou_threshold)] += 1
    return counts


def _is_high_occlusion(
    sample: dict[str, Any],
    visible_threshold: float,
    include_evidence_removed: bool,
) -> bool:
    """Membership test for the high-occlusion subset used by HOR-FCA.

    The subset holds the abstention samples whose evidence is most severely removed:
    - samples with `reason == "no_target"`, where the target is absent or cropped out, or
    - region-ladder samples whose achieved visible ratio is at or below `visible_threshold`,
      0.25 by default.

    Both criteria are discrete and domain independent. Degradation samples carry no reason
    label and only contribute to the dose-response curve, so they are outside this subset. The
    test applies only to samples whose reference decision is abstain, which makes HOR-FCA a
    subset of FCA.
    """
    if sample.get("decision") != "abstain":
        return False
    if sample.get("reason") == "no_target":
        return True
    metadata = sample.get("metadata") or {}
    vis = metadata.get("achieved_visible_ratio")
    if vis is None:
        vis = metadata.get("target_visible_ratio")
    if vis is not None and float(vis) <= visible_threshold:
        return True
    return False


def aggregate_metrics(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    iou_threshold: float = 0.5,
    hor_visible_threshold: float = 0.25,
    hor_include_evidence_removed: bool = True,
) -> dict[str, Any]:
    """Compute the headline, safeguard and comparison metrics from a list of pairs.

    Returns a dict with the confusion matrix and every derived metric: headline FCA and HOR-FCA,
    the safeguards A-FU, U-Abs, AAC and MCC, and AR as a diagnostic.

    `grounding_reason` output keys:
      - **E2E_mIoU** (canonical): end-to-end mean IoU over *all* supported samples. Label-gated,
        so only valid predictions carrying the correct label contribute box IoU, while invalid,
        abstaining and wrong-label predictions contribute 0. When no reference box exists, as in
        the label-only PlantWild probe, `score_answerable` considers decision and label
        correctness alone and the IoU term is 0 by construction.
      - **E2E_mIoU_c** (diagnostic): the localization-only variant, mean IoU over the supported
        samples the model answered, without label gating.
      - **A_Ground** / **A_Ground_c**: deprecated aliases of E2E_mIoU and E2E_mIoU_c, kept for
        backward compatibility. New code should prefer the `E2E_mIoU*` names.
      - **U_Reason**: fraction of unsupported samples where the model both abstained and gave the
        correct reason, over all unsupported samples. Abstaining with the wrong reason counts as
        a failure.
      - **U_Reason_c**: the same quantity restricted to the samples the model chose to abstain on.

    HOR-FCA is FCA restricted to the high-occlusion subset defined by `_is_high_occlusion`, whose
    boundary follows an objective evidence-removal criterion and is domain independent.
    """
    cm = confusion_matrix(pairs, iou_threshold)
    tp, fp, fn, tn, au, invalid = (cm[k] for k in CONFUSION_KEYS)
    total = len(pairs)
    answerable = sum(sample.get("decision") == "answer" for _prediction, sample in pairs)
    unanswerable = sum(sample.get("decision") == "abstain" for _prediction, sample in pairs)
    invalid_answerable = sum(
        sample.get("decision") == "answer" and not prediction.get("valid")
        for prediction, sample in pairs
    )
    invalid_unanswerable = sum(
        sample.get("decision") == "abstain" and not prediction.get("valid")
        for prediction, sample in pairs
    )

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    # --- grounding / reason differentiators (MM-AQA has neither) ---
    # E2E_mIoU (formerly A_Ground): label-gated mean IoU over all answerable samples
    # (unanswered and wrong-label predictions contribute zero). E2E_mIoU_c (formerly
    # A_Ground_c): localization-only diagnostic conditional on answering.
    iou_sum_all = 0.0
    iou_sum_answered = 0.0
    n_answerable = 0
    n_answered_on_answerable = 0
    # U-Reason: unanswerable samples where the model abstained AND gave the correct reason.
    # U-Reason_c: conditional, only over the subset the model abstained on.
    n_unanswerable = 0
    n_reason_correct = 0
    n_abstained_on_unanswerable = 0
    for prediction, sample in pairs:
        if sample.get("decision") == "answer":
            n_answerable += 1
            if not prediction.get("valid"):
                continue
            if prediction.get("decision") == "answer":
                n_answered_on_answerable += 1
                iou = (
                    box_iou(prediction["box"], sample["box"])
                    if prediction.get("box") and sample.get("box")
                    else 0.0
                )
                # Count IoU only when the predicted label is also correct (grounding must be
                # of the right object); otherwise it contributes 0 to A-Ground.
                if normalize_label(prediction.get("label")) == normalize_label(sample.get("label")):
                    iou_sum_all += iou
                    iou_sum_answered += iou
                else:
                    iou_sum_answered += iou
        else:  # unanswerable
            n_unanswerable += 1
            if not prediction.get("valid"):
                continue
            if prediction.get("decision") == "abstain":
                n_abstained_on_unanswerable += 1
                if prediction.get("reason") == sample.get("reason"):
                    n_reason_correct += 1

    a_ground = safe_div(iou_sum_all, n_answerable)
    a_ground_c = safe_div(iou_sum_answered, n_answered_on_answerable)
    u_reason = safe_div(n_reason_correct, n_unanswerable)
    u_reason_c = safe_div(n_reason_correct, n_abstained_on_unanswerable)

    fca = safe_div(au, unanswerable)

    hor_unanswerable = 0
    hor_au = 0
    for prediction, sample in pairs:
        if not _is_high_occlusion(sample, hor_visible_threshold, hor_include_evidence_removed):
            continue
        hor_unanswerable += 1
        if prediction.get("valid") and prediction.get("decision") == "answer":
            hor_au += 1
    hor_fca = safe_div(hor_au, hor_unanswerable)

    # Safeguards.
    a_fu = safe_div(fn + invalid_answerable, answerable)
    supported_retention = 1.0 - a_fu if answerable else 0.0
    u_abs = safe_div(tn, unanswerable)
    aac = safe_div(tp, answerable)
    uac = safe_div(tn, unanswerable)  # MM-AQA UAC == u_abs here
    ar = safe_div(fn + tn + invalid, total)  # non-answer rate (diagnostic only)

    # Binary selection MCC over the fixed GT universe. Invalid output is a non-answer:
    # it is a false rejection for supported GT and a non-commitment for unsupported GT.
    selection_cm = {
        "TP": tp + fp,
        "FP": au,
        "FN": fn + invalid_answerable,
        "TN": tn + invalid_unanswerable,
    }
    mcc = _binary_mcc(
        tp_=selection_cm["TP"],
        fp_=selection_cm["FP"],
        fn_=selection_cm["FN"],
        tn_=selection_cm["TN"],
    )
    strict_3_correct = sum(
        bool(prediction.get("valid"))
        and prediction.get("decision") == sample.get("decision")
        and (
            normalize_label(prediction.get("label")) == normalize_label(sample.get("label"))
            if sample.get("decision") == "answer"
            else prediction.get("reason") == sample.get("reason")
        )
        for prediction, sample in pairs
    )
    strict_3 = safe_div(strict_3_correct, total)

    return {
        "confusion_matrix": cm,
        "counts": {
            "total": total,
            "answerable": answerable,
            "unanswerable": unanswerable,
            "invalid": invalid,
            "invalid_answerable": invalid_answerable,
            "invalid_unanswerable": invalid_unanswerable,
        },
        "selection_confusion_matrix": selection_cm,
        "headline": {
            "FCA": fca,
            "HOR_FCA": hor_fca,
            "HOR_count": {
                "high_occlusion_au": hor_au,
                "high_occlusion_unanswerable": hor_unanswerable,
            },
        },
        "safeguard": {
            "A_FU": a_fu,
            "Supported_Retention": supported_retention,
            "U_Abs": u_abs,
            "AAC": aac,
            "MCC": mcc,
        },
        "semantic": {"Strict_3": strict_3},
        "grounding_reason": {
            # E2E_mIoU: end-to-end mean IoU over all GT answerable/supported samples.
            # Label-gated: only valid predictions with the correct label contribute box IoU;
            # invalid, wrong-label, and abstained predictions contribute 0.
            "E2E_mIoU": a_ground,
            "E2E_mIoU_c": a_ground_c,
            "A_Ground": a_ground,  # deprecated alias for E2E_mIoU
            "A_Ground_c": a_ground_c,  # deprecated alias for E2E_mIoU_c
            "U_Reason": u_reason,
            "U_Reason_c": u_reason_c,
        },
        "comparison": {"UAC": uac, "AR": ar},
        "invalid_rate": safe_div(invalid, total),
        "iou_threshold": iou_threshold,
        "denominator_policy": "fixed_gt_universe_invalid_is_incorrect",
    }


def _binary_mcc(tp_: int, fp_: int, fn_: int, tn_: int) -> float:
    """Matthews correlation on the answer-vs-abstain decision axis.

    Positive class = "model answered". tp_ = answered on answerable, fp_ = answered on
    unanswerable (== AU), fn_ = abstained on answerable (== A-FU), tn_ = abstained on
    unanswerable. Penalises both AU (FP) and A-FU (FN), giving a single hard-to-game scalar.
    """
    numerator = (tp_ * tn_) - (fp_ * fn_)
    denominator = math.sqrt((tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_))
    return numerator / denominator if denominator else 0.0


def metamorphic_test_satisfaction(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """MTS: fraction of metamorphic relations whose expected behaviour holds.

    Each sample carries ``metadata.relation``. The metamorphic expectation is fully determined
    by the gold decision (answer-preserving relations expect a correct answer; abstain relations
    expect an abstention with the right reason). MTS is the per-relation satisfaction rate plus
    macro- and micro-averages over the relation labels instantiated by that probe. The canonical
    paper metric is the relation macro-average. A sample is "satisfied" iff the prediction
    matches the gold metamorphic outcome.
    """
    per_relation_total: dict[str, int] = {}
    per_relation_pass: dict[str, int] = {}
    overall_total = 0
    overall_pass = 0

    for prediction, sample in pairs:
        metadata = sample.get("metadata") or {}
        relation = metadata.get("relation")
        if not relation or relation == "base":
            continue
        per_relation_total[relation] = per_relation_total.get(relation, 0) + 1
        overall_total += 1

        if sample.get("decision") == "answer":
            satisfied = score_answerable(prediction, sample, iou_threshold)
        else:
            satisfied = score_unanswerable(prediction, sample)

        if satisfied:
            per_relation_pass[relation] = per_relation_pass.get(relation, 0) + 1
            overall_pass += 1

    per_relation = {
        relation: (per_relation_pass.get(relation, 0) / total if total else 0.0)
        for relation, total in sorted(per_relation_total.items())
    }
    macro = sum(per_relation.values()) / len(per_relation) if per_relation else 0.0
    micro = overall_pass / overall_total if overall_total else 0.0
    return {
        "MTS_micro": micro,
        "MTS_macro": macro,
        "per_relation": per_relation,
        "per_relation_counts": {
            relation: {"pass": per_relation_pass.get(relation, 0), "total": total}
            for relation, total in sorted(per_relation_total.items())
        },
    }


def risk_coverage_curve(
    scored: list[tuple[float, float]],
) -> tuple[list[float], list[float], float]:
    """Risk-Coverage curve + AURC from (confidence, risk) pairs.

    ``scored`` is a list of (confidence, risk) where higher confidence == more willing to
    answer and risk in {0,1} (1 == unsafe/incorrect answer). Samples are sorted by descending
    confidence; coverage is the fraction answered, risk is the mean risk over the answered
    subset. Returns (coverages, risks, AURC). Lower AURC is better.
    """
    if not scored:
        return [], [], 0.0
    ordered = sorted(scored, key=lambda item: item[0], reverse=True)
    n = len(ordered)
    coverages: list[float] = []
    risks: list[float] = []
    cumulative_risk = 0.0
    for index, (_confidence, risk) in enumerate(ordered, start=1):
        cumulative_risk += risk
        coverages.append(index / n)
        risks.append(cumulative_risk / index)
    aurc = sum(risks) / len(risks)
    return coverages, risks, aurc
