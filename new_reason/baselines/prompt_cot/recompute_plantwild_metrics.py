#!/usr/bin/env python3
"""Independent recomputation of PlantWild real metrics from per-sample predictions.

Uses the paper implementation in aaai_code/reason_abstain/eval/metrics.py.
No external dependencies beyond stdlib.
"""
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent  # agriabstain/
sys.path.insert(0, str(REPO / "aaai_code"))

# Import the authoritative metrics implementation
from reason_abstain.eval.metrics import (
    aggregate_metrics,
    metamorphic_test_satisfaction,
    confusion_matrix,
    classify_pair,
    _binary_mcc,
    _is_high_occlusion,
    box_iou,
    score_answerable,
    score_unanswerable,
    false_confident_answer,
    CONFUSION_KEYS,
)
from reason_abstain.eval.parser import normalize_label


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reconstruct_pairs(records: list[dict]) -> list[tuple[dict, dict]]:
    """Reconstruct (pred, gt) pairs from predictions.jsonl records.
    
    Filters out ambiguous samples (GT decision is None).
    """
    pairs = []
    n_ambiguous_skipped = 0
    for rec in records:
        pred = rec["pred"]
        g = rec["gt"]
        if g.get("decision") is None:
            n_ambiguous_skipped += 1
            continue
        gt_md = g.get("metadata") or {"relation": rec.get("relation")}
        gt = {**g, "relation": rec.get("relation") or gt_md.get("relation"), "metadata": gt_md}
        pairs.append((pred, gt))
    return pairs, n_ambiguous_skipped


def compute_metrics_from_pairs(pairs: list[tuple[dict, dict]]) -> dict:
    """Compute all metrics from scratch using the authoritative aggregate_metrics."""
    metrics = aggregate_metrics(pairs, iou_threshold=0.5)
    mts = metamorphic_test_satisfaction(pairs, iou_threshold=0.5)
    return metrics, mts


def cross_validate(variant_name: str, results_dir: Path) -> dict:
    """Cross-validate recomputed metrics against existing metrics.json."""
    pred_path = results_dir / "predictions.jsonl"
    metrics_path = results_dir / "metrics.json"
    
    if not pred_path.exists():
        return {"variant": variant_name, "error": f"predictions.jsonl not found at {pred_path}"}
    if not metrics_path.exists():
        return {"variant": variant_name, "error": f"metrics.json not found at {metrics_path}"}
    
    # Load existing metrics
    existing = json.loads(metrics_path.read_text())
    
    # Load predictions and reconstruct pairs
    records = load_predictions(pred_path)
    pairs, n_ambiguous_skipped = reconstruct_pairs(records)
    
    # Recompute from scratch
    recomputed_metrics, recomputed_mts = compute_metrics_from_pairs(pairs)
    
    # Compare
    diffs = {}
    
    # Compare confusion matrix
    cm_fields = ["TP", "FP", "FN", "TN", "AU", "INVALID"]
    for k in cm_fields:
        old = existing["metrics"]["confusion_matrix"].get(k)
        new = recomputed_metrics["confusion_matrix"].get(k)
        if old != new:
            diffs[f"confusion_matrix.{k}"] = {"existing": old, "recomputed": new}
    
    # Compare headline metrics
    for k in ["FCA", "HOR_FCA"]:
        old = existing["metrics"]["headline"].get(k)
        new = recomputed_metrics["headline"].get(k)
        if abs(old - new) > 1e-10:
            diffs[f"headline.{k}"] = {"existing": old, "recomputed": new}
    
    # Compare safeguard metrics
    for k in ["A_FU", "U_Abs", "AAC", "MCC"]:
        old = existing["metrics"]["safeguard"].get(k)
        new = recomputed_metrics["safeguard"].get(k)
        if abs(old - new) > 1e-10:
            diffs[f"safeguard.{k}"] = {"existing": old, "recomputed": new}
    
    # Compare grounding_reason metrics
    for k in ["A_Ground", "A_Ground_c", "U_Reason", "U_Reason_c"]:
        old = existing["metrics"]["grounding_reason"].get(k)
        new = recomputed_metrics["grounding_reason"].get(k)
        if abs(old - new) > 1e-10:
            diffs[f"grounding_reason.{k}"] = {"existing": old, "recomputed": new}
    
    # Compare counts
    for k in ["total", "answerable", "unanswerable", "invalid"]:
        old = existing["metrics"]["counts"].get(k)
        new = recomputed_metrics["counts"].get(k)
        if old != new:
            diffs[f"counts.{k}"] = {"existing": old, "recomputed": new}
    
    # Compare MTS
    for k in ["MTS_micro", "MTS_macro"]:
        old = existing["mts"].get(k)
        new = recomputed_mts.get(k)
        if abs(old - new) > 1e-10:
            diffs[f"mts.{k}"] = {"existing": old, "recomputed": new}
    
    # Compare HOR count
    for k in ["high_occlusion_au", "high_occlusion_unanswerable"]:
        old = existing["metrics"]["headline"]["HOR_count"].get(k)
        new = recomputed_metrics["headline"]["HOR_count"].get(k)
        if old != new:
            diffs[f"headline.HOR_count.{k}"] = {"existing": old, "recomputed": new}
    
    # Compare invalid_rate
    old = existing["metrics"]["invalid_rate"]
    new = recomputed_metrics["invalid_rate"]
    if abs(old - new) > 1e-10:
        diffs["invalid_rate"] = {"existing": old, "recomputed": new}
    
    return {
        "variant": variant_name,
        "n_samples": len(records),
        "n_ambiguous_skipped": n_ambiguous_skipped,
        "n_pairs": len(pairs),
        "existing_confusion_matrix": existing["metrics"]["confusion_matrix"],
        "recomputed_confusion_matrix": recomputed_metrics["confusion_matrix"],
        "existing_metrics": {
            "FCA": existing["metrics"]["headline"]["FCA"],
            "HOR_FCA": existing["metrics"]["headline"]["HOR_FCA"],
            "A_FU": existing["metrics"]["safeguard"]["A_FU"],
            "U_Abs": existing["metrics"]["safeguard"]["U_Abs"],
            "AAC": existing["metrics"]["safeguard"]["AAC"],
            "MCC": existing["metrics"]["safeguard"]["MCC"],
            "A_Ground": existing["metrics"]["grounding_reason"]["A_Ground"],
            "U_Reason": existing["metrics"]["grounding_reason"]["U_Reason"],
            "U_Reason_c": existing["metrics"]["grounding_reason"]["U_Reason_c"],
            "MTS_micro": existing["mts"]["MTS_micro"],
        },
        "recomputed_metrics": {
            "FCA": recomputed_metrics["headline"]["FCA"],
            "HOR_FCA": recomputed_metrics["headline"]["HOR_FCA"],
            "A_FU": recomputed_metrics["safeguard"]["A_FU"],
            "U_Abs": recomputed_metrics["safeguard"]["U_Abs"],
            "AAC": recomputed_metrics["safeguard"]["AAC"],
            "MCC": recomputed_metrics["safeguard"]["MCC"],
            "A_Ground": recomputed_metrics["grounding_reason"]["A_Ground"],
            "U_Reason": recomputed_metrics["grounding_reason"]["U_Reason"],
            "U_Reason_c": recomputed_metrics["grounding_reason"]["U_Reason_c"],
            "MTS_micro": recomputed_mts["MTS_micro"],
        },
        "differences": diffs,
        "all_match": len(diffs) == 0,
    }


def main():
    out_root = REPO / "new_reason" / "baselines" / "prompt_cot" / "outputs"
    variants = ["native_lm", "prompt_basic", "prompt_reward", "prompt_reward_norms", "cot_structured"]
    
    all_results = []
    for v in variants:
        result_dir = out_root / v / "plantwild_real_v1"
        print(f"\n{'='*60}")
        print(f"Cross-validating: {v}")
        print(f"{'='*60}")
        result = cross_validate(v, result_dir)
        all_results.append(result)
        
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        
        print(f"  n_samples={result['n_samples']}, n_pairs={result['n_pairs']}, n_ambiguous_skipped={result['n_ambiguous_skipped']}")
        print(f"  All match: {result['all_match']}")
        
        if not result['all_match']:
            print(f"  Differences ({len(result['differences'])}):")
            for k, v in result['differences'].items():
                print(f"    {k}: existing={v['existing']}, recomputed={v['recomputed']}")
        
        print(f"\n  Confusion matrix:")
        print(f"    Existing:   {result['existing_confusion_matrix']}")
        print(f"    Recomputed: {result['recomputed_confusion_matrix']}")
        
        print(f"\n  Key metrics comparison:")
        for k in ["FCA", "HOR_FCA", "A_FU", "U_Abs", "AAC", "MCC", "MTS_micro"]:
            old = result['existing_metrics'][k]
            new = result['recomputed_metrics'][k]
            match = "✓" if abs(old - new) < 1e-10 else "✗"
            print(f"    {k:15s}: existing={old:.10f}  recomputed={new:.10f}  {match}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = all(r.get("all_match", False) for r in all_results)
    if all_ok:
        print("✅ ALL 5 variants recomputed metrics match existing metrics.json exactly!")
    else:
        for r in all_results:
            if not r.get("all_match", True):
                print(f"❌ {r['variant']}: {len(r.get('differences', {}))} differences found")
    
    # Output detailed formulas
    print("\n\n=== FORMULA VERIFICATION ===")
    for r in all_results:
        if "error" in r:
            continue
        cm = r['recomputed_confusion_matrix']
        tp, fp, fn, tn, au = cm["TP"], cm["FP"], cm["FN"], cm["TN"], cm["AU"]
        ans = tp + fp + fn
        unans = tn + au
        print(f"\n--- {r['variant']} ---")
        print(f"  CM: TP={tp}, FP={fp}, FN={fn}, TN={tn}, AU={au}")
        print(f"  answerable={ans}, unanswerable={unans}")
        print(f"  FCA = AU/unanswerable = {au}/{unans} = {au/unans:.10f}")
        print(f"  A_FU = FN/answerable = {fn}/{ans} = {fn/ans:.10f}")
        print(f"  U_Abs = TN/unanswerable = {tn}/{unans} = {tn/unans:.10f}")
        print(f"  AAC = TP/(TP+FP+FN) = {tp}/{tp+fp+fn} = {tp/(tp+fp+fn):.10f}")
        mcc_num = (tp*tn) - ( (tp+fp) * fn )  # Wait, MCC formula is (TP*TN - FP*FN) / sqrt(...)
        # Actually the binary MCC in the code uses: tp_=TP+FP (all answered), fp_=AU, fn_=FN, tn_=TN
        mcc_num = ((tp+fp) * tn) - (au * fn)
        mcc_den = math.sqrt((tp+fp+au) * (tp+fp+fn) * (tn+au) * (tn+fn))
        print(f"  MCC = ((TP+FP)*TN - AU*FN) / sqrt((TP+FP+AU)*(TP+FP+FN)*(TN+AU)*(TN+FN))")
        print(f"      = ({tp+fp}*{tn} - {au}*{fn}) / sqrt({tp+fp+au}*{tp+fp+fn}*{tn+au}*{tn+fn})")
        print(f"      = {mcc_num} / sqrt({(tp+fp+au)*(tp+fp+fn)*(tn+au)*(tn+fn)})")
        print(f"      = {mcc_num} / {mcc_den} = {mcc_num/mcc_den if mcc_den else 0:.10f}")


if __name__ == "__main__":
    main()
