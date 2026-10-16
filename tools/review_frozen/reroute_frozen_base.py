#!/usr/bin/env python3
"""Offline rerouting of a fitted frozen-base gate over the frozen-base NATIVE candidates.

Alignment: the frozen cache's test row i (sample_id s) is matched to the base-native
prediction with the same sample_id (strict string equality). Base-native predictions come
from new_reason/outputs/base_native_regate_20260720/native/base/comprehensive/ and are the
SAME base model's decode: no adapted candidate is ever used here.

Metrics written to --output-dir/metrics.json
  FCA / supported_query_retention / A_FU / strict_3_accuracy / E2E_mIoU / selective_risk
  (from the paper's regate_curve_point; Strict-3 and E2E use the base-native candidate)
  binary MCC (supported vs answered, three-class-argmax)      <-- the correct binary MCC
  state_accuracy, MCC_3class                                   <-- class metrics, NOT binary
  AUROC (supported vs p_answer)
  MTS_macro / MTS_micro                                       (metamorphic relations only)
  Base-native reference point (native_point on the same records) for context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from frozen_common import (
    AAAI_ROOT,
    BASE_NATIVE_PREDS,
    REPR_ROOT,
    build_gate_records,
    load_base_native_predictions,
    load_split,
    predict_with_model,
)

from experiments.base_native_regate.analyze_native_regate import native_point  # noqa: E402
from reason_abstain.eval.metrics import metamorphic_test_satisfaction  # noqa: E402
from reason_abstain.eval.parser import normalize_label  # noqa: E402


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def e2e_miou_from_gate(records: dict, answered_map: dict) -> float:
    """Paper E2E_mIoU convention: mean IoU over ALL supported records, where only gated-answer
    records whose shared candidate has the correct label contribute; others contribute 0."""
    iou_sum = 0.0
    n_supported = 0
    for key, rec in records.items():
        gt = rec["gt"]
        if gt.get("decision") != "answer":
            continue
        n_supported += 1
        if not answered_map[key]:
            continue
        cand = rec["answer_candidate"]
        if not cand.get("valid"):
            continue
        if normalize_label(cand.get("label")) != normalize_label(gt.get("label")):
            continue
        iou_sum += box_iou(cand.get("box"), gt.get("box")) if cand.get("box") and gt.get("box") else 0.0
    return iou_sum / n_supported if n_supported else 0.0


def gated_pred_for_mts(rec: dict, gate_answered: bool, p_wo: float, p_nt: float) -> dict:
    if gate_answered:
        cand = rec["answer_candidate"]
        return {
            "valid": True,
            "decision": "answer",
            "label": cand.get("label"),
            "box": cand.get("box"),
        }
    return {"valid": True, "decision": "abstain",
            "reason": "wrong_object" if p_wo > p_nt else "no_target"}


def sample_for_mts(rec: dict) -> dict:
    gt = rec["gt"]
    return {
        "decision": gt.get("decision"),
        "reason": gt.get("reason"),
        "label": gt.get("label"),
        "box": gt.get("box"),
        "relation": rec.get("relation"),
        "metadata": (gt.get("metadata") or {}) or {"relation": rec.get("relation")},
    }


def mts_from_records(records: dict, answered_map: dict) -> dict:
    pairs = []
    for key, rec in records.items():
        gated = gated_pred_for_mts(rec, answered_map[key], rec["p_wo"], rec["p_nt"])
        pairs.append((gated, sample_for_mts(rec)))
    return metamorphic_test_satisfaction(pairs)


def native_reference(base_rows: list[dict]) -> dict:
    records = {}
    for index, rec in enumerate(base_rows):
        sample_id = str(rec["sample_id"])
        key = f"{index}:{sample_id}"
        gt = rec.get("gt")
        if not isinstance(gt, dict) or gt.get("decision") not in {"answer", "abstain"}:
            continue
        records[key] = {
            "sample_id": sample_id,
            "relation": rec.get("relation"),
            "gt": {
                "decision": gt.get("decision"),
                "reason": gt.get("reason"),
                "label": gt.get("label"),
                "box": gt.get("box"),
                "metadata": (gt.get("metadata") or {}) or {"relation": rec.get("relation")},
            },
            "pred": rec.get("pred"),
            "raw": rec.get("raw", ""),
        }
    point = native_point("comprehensive", "frozen_base", records)
    out = asdict(point)
    recs = list(records.values())
    supported = np.asarray(
        [int(r["gt"]["decision"] == "answer") for r in recs], dtype=np.int64
    )
    answered = np.asarray(
        [int(bool(r["pred"].get("valid")) and r["pred"].get("decision") == "answer")
         for r in recs],
        dtype=np.int64,
    )
    pairs = [
        (r["pred"], {
            "decision": r["gt"]["decision"],
            "reason": r["gt"]["reason"],
            "label": r["gt"]["label"],
            "box": r["gt"]["box"],
            "relation": r["relation"],
            "metadata": (r["gt"]["metadata"] or {}) or {"relation": r["relation"]},
        })
        for r in recs
    ]
    out["binary_mcc"] = float(matthews_corrcoef(supported, answered))
    mts = metamorphic_test_satisfaction(pairs)
    out["MTS_macro"] = mts["MTS_macro"]
    out["MTS_micro"] = mts["MTS_micro"]
    out["E2E_mIoU_paper"] = e2e_miou_native(recs)
    return out


def e2e_miou_native(recs: list[dict]) -> float:
    """Paper E2E_mIoU over ALL supported records from the frozen-base native preds."""
    iou_sum = 0.0
    n_supported = 0
    for r in recs:
        gt = r["gt"]
        if gt.get("decision") != "answer":
            continue
        n_supported += 1
        pred = r["pred"]
        if not (isinstance(pred, dict) and pred.get("valid") and pred.get("decision") == "answer"):
            continue
        if normalize_label(pred.get("label")) != normalize_label(gt.get("label")):
            continue
        iou_sum += box_iou(pred.get("box"), gt.get("box")) if pred.get("box") and gt.get("box") else 0.0
    return iou_sum / n_supported if n_supported else 0.0


def base_native_published() -> dict:
    """Read the stored base-native metrics.json (the published base row)."""
    path = (
        REPR_ROOT.parents[1]
        / "base_native_regate_20260720"
        / "native"
        / "base"
        / "comprehensive"
        / "metrics.json"
    )
    if not path.exists():
        return {"missing": str(path)}
    raw = json.loads(path.read_text(encoding="utf-8"))
    body = raw.get("metrics") or raw
    head = body.get("headline", {}).get("FCA")
    saf = body.get("safeguard", {})
    sem = body.get("semantic", {})
    gr = body.get("grounding_reason", {})
    mts = raw.get("mts", {})
    return {
        "source": str(path.resolve()),
        "FCA": head,
        "supported_retention": saf.get("Supported_Retention"),
        "A_FU": saf.get("A_FU"),
        "MCC": saf.get("MCC"),
        "AAC": saf.get("AAC"),
        "Strict_3": sem.get("Strict_3"),
        "E2E_mIoU": gr.get("E2E_mIoU"),
        "E2E_mIoU_c": gr.get("E2E_mIoU_c"),
        "U_Reason": gr.get("U_Reason"),
        "MTS_macro": mts.get("MTS_macro"),
        "MTS_micro": mts.get("MTS_micro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=Path)
    parser.add_argument("--proba", default=None, type=Path,
                        help="optional (n,3) softmax probabilities npy (mutually exclusive with --model)")
    parser.add_argument("--tag", default="frozen-base linear", help="gate label")
    parser.add_argument("--repr-root", default=REPR_ROOT, type=Path)
    parser.add_argument("--predictions", default=BASE_NATIVE_PREDS, type=Path)
    parser.add_argument("--dataset", default="comprehensive")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.model is not None and args.proba is not None:
        raise ValueError("pass either --model or --proba, not both")
    if args.model is not None:
        with np.load(args.model) as loaded:
            model = {
                "coef": loaded["coef"],
                "intercept": loaded["intercept"],
                "classes": loaded["classes"],
                "mu": loaded["mu"],
                "sd": loaded["sd"],
                "selected_c": float(loaded["selected_c"]),
                "run_seed": int(loaded["run_seed"]),
            }
    else:
        model = {"tag": str(args.proba)}

    test = load_split(args.repr_root, "test", pool="last_token")
    base_rows = load_base_native_predictions(args.predictions)

    # strict sample-id alignment (not positional trust)
    by_sid = {str(r["sample_id"]): r for r in base_rows}
    if len(by_sid) != len(base_rows):
        raise ValueError("duplicate sample_id in base-native predictions")
    missing = set(test["sample_id"]) - set(by_sid)
    extra = set(by_sid) - set(test["sample_id"])
    if missing or extra:
        raise ValueError(
            f"sample-id misalignment: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
        )
    ordered = [by_sid[sid] for sid in test["sample_id"]]
    print(
        f"[reroute] n_test={len(ordered)} sample-id aligned, "
        f"base-native preds={len(base_rows)}",
        flush=True,
    )

    if args.model is not None:
        proba = predict_with_model(model, test["features"])
    else:
        proba = np.load(args.proba)
        if proba.shape != (len(ordered), 3):
            raise ValueError(f"proba shape {proba.shape} != {(len(ordered), 3)}")
    records, answered_map, supported, y_state, predicted_classes, record_keys = (
        build_gate_records(ordered, proba)
    )

    paper_point = regate_point(args.dataset, records, answered_map)
    answered_vals = np.asarray([1 if answered_map[k] else 0 for k in record_keys], dtype=np.int64)
    binary_mcc = float(matthews_corrcoef(supported, answered_vals))
    metrics = {
        "dataset": args.dataset,
        "gate": args.tag,
        "selected_C": model.get("selected_c") if args.model is not None else None,
        "run_seed": model.get("run_seed") if args.model is not None else None,
        "operating_point": "three_class_argmax",
        "candidate_source": "frozen-base native predictions (strict sample-id aligned)",
        "n_records": len(records),
        "n_supported": int(supported.sum()),
        "n_unsupported": int((supported == 0).sum()),
        "FCA": paper_point["FCA"],
        "supported_query_retention": paper_point["supported_query_retention"],
        "A_FU": paper_point["A_FU"],
        "strict_3_accuracy": paper_point["strict_3_accuracy"],
        "E2E_mIoU": e2e_miou_from_gate(records, answered_map),
        "E2E_conditional_binary_rate": paper_point["E2E_mIoU"],
        "selective_risk": paper_point["selective_risk"],
        "MCC_binary": binary_mcc,
        "state_accuracy": float((predicted_classes == y_state).mean()),
        "MCC_3class": float(matthews_corrcoef(y_state, predicted_classes)),
        "AUROC": float(roc_auc_score(supported, proba[:, 0])),
        "MTS_macro": mts_from_records(records, answered_map)["MTS_macro"],
        "MTS_micro": mts_from_records(records, answered_map)["MTS_micro"],
        "base_native_reference": native_reference(ordered),
        "base_native_published_metrics": base_native_published(),
        "alignment": {
            "mode": "sample_id strict string equality",
            "n_aligned": len(ordered),
            "missing": sorted(missing)[:10],
            "extra": sorted(extra)[:10],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "frozen_gate_probabilities.jsonl").open("w", encoding="utf-8") as h:
        for index, sid in enumerate(test["sample_id"]):
            h.write(
                json.dumps(
                    {
                        "record_index": index,
                        "sample_id": str(sid),
                        "p_answer": float(proba[index, 0]),
                        "p_wo": float(proba[index, 1]),
                        "p_nt": float(proba[index, 2]),
                        "predicted_class": int(predicted_classes[index]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(
        f"[reroute] FCA={metrics['FCA']:.4f} retention={metrics['supported_query_retention']:.4f} "
        f"MCC_binary={metrics['MCC_binary']:.4f} Strict3={metrics['strict_3_accuracy']:.4f} "
        f"E2E={metrics['E2E_mIoU']:.4f} MTS={metrics['MTS_macro']:.4f}",
        flush=True,
    )


def regate_point(dataset: str, records: dict, answered_map: dict) -> dict:
    from experiments.base_native_regate.analyze_native_regate import regate_curve_point
    point = regate_curve_point(dataset, records, answered_map, "three_class_argmax", None)
    return {
        "FCA": point.unsupported_hallucination_rate,
        "supported_query_retention": point.supported_query_retention,
        "A_FU": 1.0 - point.supported_query_retention,
        "strict_3_accuracy": point.strict_3_accuracy,
        "E2E_mIoU": point.grounding_accuracy,
        "selective_risk": point.selective_risk,
        "format_valid_rate": point.format_valid_rate,
    }


def answered_map_values(records: dict, answered_map: dict) -> np.ndarray:
    return np.asarray([1 if answered_map[k] else 0 for k in records], dtype=np.int64)


if __name__ == "__main__":
    main()