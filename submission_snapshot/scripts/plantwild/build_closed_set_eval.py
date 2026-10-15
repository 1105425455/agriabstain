#!/usr/bin/env python
# ruff: noqa: E402, E501
"""Build closed-set (12-class) PlantWild evaluation from the 13-class open-set (444) data.

Filtering rule (model-performance-agnostic):
  Exclude any pv13_label that does NOT appear as an answer label in SFT train
  (unique answer record ≥ 1 occurrence). This deterministically excludes
  ``tomato leaf yellow virus`` (SFT answer=0), dropping its 12 images × 3 probes
  (answer + no_target + wrong_object) = 36 records.

Output: 12 classes, 144 unique images, 408 paired records.
For all 6 methods (ours + 5 baselines), filtered predictions + metrics are written
to new ``plantwild_real_closed12_v1/`` subdirectories, without re-running inference.

IMPORTANT: This script requires the FULL source research repository (including old
``new_reason/outputs/eval_probes/plantwild_real_v1/`` and ``checkpoint-444``
predictions). It CANNOT be rebuilt from the data-only benchmark package alone.

Usage:
  python scripts/plantwild/build_closed_set_eval.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from reason_abstain.eval.metrics import aggregate_metrics, metamorphic_test_satisfaction
from reason_abstain.eval.parser import normalize_label

# ────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────
PAIRED_444 = (
    REPO / "new_reason/outputs/eval_probes/plantwild_real_v1/plantwild_real_paired_444.jsonl"
)
SFT_TRAIN = REPO / "new_reason/outputs/sft_3reason/sft_train.json"

OUT_CLOSED_DIR = REPO / "new_reason/outputs/eval_probes/plantwild_real_closed12_v1"
OUT_CLOSED_JSONL = OUT_CLOSED_DIR / "plantwild_real_closed12_paired_408.jsonl"
OUT_CLOSED_AUDIT = OUT_CLOSED_DIR / "BUILD_AUDIT.json"

# Ours
OURS_PRED = (
    REPO
    / "new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/eval_faithful_v1/plantwild_real_v1/predictions.jsonl"
)
OURS_DIR = (
    REPO
    / "new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/eval_faithful_v1/plantwild_real_closed12_v1"
)
OURS_METRICS_ORIG = (
    REPO
    / "new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444/eval_faithful_v1/plantwild_real_v1/metrics.json"
)

# Five baselines
BASELINE_NAMES = (
    "native_lm",
    "prompt_basic",
    "prompt_reward",
    "prompt_reward_norms",
    "cot_structured",
)
BASELINE_DIR = REPO / "new_reason/baselines/prompt_cot/outputs"

IOU_THRESHOLD = 0.5


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sample_id_from_row(row: dict) -> str:
    """Extract sample_id from a row (paired jsonl record or prediction record)."""
    # Try metadata first (for paired jsonl)
    md = row.get("metadata") or {}
    sid = md.get("sample_id")
    if sid:
        return sid
    # Try top-level (for predictions.jsonl)
    sid = row.get("sample_id")
    if sid:
        return sid
    # Try gt.metadata.sample_id (for predictions.jsonl with nested gt)
    gt = row.get("gt") or {}
    gtm = gt.get("metadata") or {}
    return gtm.get("sample_id") or ""


def image_key_from_sample_id(sample_id: str) -> str | None:
    """Extract the stable image key from a sample_id like
    ``pw_real_<class>_<filename>-<probe>`` → ``pw_real_<class>_<filename>``.
    """
    m = re.match(r"(pw_real_.+)-[^-]+$", sample_id)
    return m.group(1) if m else None


def build_pairs_from_predictions(predictions: list[dict]) -> list[tuple[dict, dict]]:
    """Rebuild (pred, gt) pairs from predictions records, matching ``merge_and_score``.

    Same logic as in eval_qwen3vl_swift_3reason.py merge_and_score().
    """
    pairs: list[tuple[dict, dict]] = []
    for rec in predictions:
        pred = rec["pred"]
        g = rec["gt"]
        if g.get("decision") is None:
            continue
        gt_md = g.get("metadata") or {"relation": rec.get("relation")}
        gt = {**g, "relation": rec.get("relation") or gt_md.get("relation"), "metadata": gt_md}
        pairs.append((pred, gt))
    return pairs


def compute_gate(predictions: list[dict]) -> dict[str, Any]:
    """Compute gate metrics (format_valid_rate, decision_accuracy) matching merge_and_score."""
    pairs = build_pairs_from_predictions(predictions)
    n = len(predictions)
    n_valid = 0
    n_dec_ok = 0
    for pred, gt in pairs:
        if pred.get("valid"):
            n_valid += 1
            if decision_correct(pred, gt):
                n_dec_ok += 1
    fmt_rate = n_valid / n if n else 0.0
    dec_acc = n_dec_ok / n if n else 0.0
    return {
        "format_valid_rate": round(fmt_rate, 4),
        "decision_accuracy": round(dec_acc, 4),
        "format_valid_min": 0.95,
        "decision_acc_min": 0.80,
        "passed": fmt_rate >= 0.95 and dec_acc >= 0.80,
    }


def decision_correct(pred: dict, gt: dict) -> bool:
    """A decision counts as correct under the same rule as the evaluation script."""
    if not pred.get("valid") or pred.get("decision") != gt.get("decision"):
        return False
    if gt.get("decision") == "answer":
        return normalize_label(pred.get("label")) == normalize_label(gt.get("label"))
    if gt.get("decision") == "abstain":
        return pred.get("reason") == gt.get("reason")
    return False


def make_summary_lines(
    method_name: str,
    predictions: list[dict],
    metrics: dict,
    mts: dict,
    gate: dict,
    n: int,
    n_ambiguous: int,
    n_unique_images: int,
    n_answerable_per_probe: dict,
    strict_tp: int,
    nt_decision_correct: int,
    nt_reason_correct: int,
    nt_au: int,
    wo_decision_correct: int,
    wo_reason_correct: int,
    wo_au: int,
    answer_coverage: int,
) -> list[str]:
    cm = metrics["confusion_matrix"]
    h = metrics["headline"]
    s = metrics["safeguard"]
    gr = metrics["grounding_reason"]
    c = metrics["counts"]
    lines = [
        f"=== ReGate eval [{method_name} / plantwild_real_closed12_v1] ===",
        f"n={n}  n_ambiguous_skipped={n_ambiguous}  unique_images={n_unique_images}",
        "",
        "--- Integer breakdown ---",
        f"TP={cm['TP']}  FP={cm['FP']}  FN={cm['FN']}  TN={cm['TN']}  AU={cm['AU']}  INVALID={cm['INVALID']}",
        f"answerable={c['answerable']}  unanswerable={c['unanswerable']}",
        f"answer_coverage = {answer_coverage}/144  (unique images with any answer-prediction)",
        f"strict_TP       = {strict_tp}/144  (TP among answerable-paired unique images)",
        "",
        "--- Probe-level breakdown ---",
        f"NT (no_target): decision_correct={nt_decision_correct}/144  reason_correct={nt_reason_correct}/144  AU={nt_au}/144",
        f"WO (wrong_object): decision_correct={wo_decision_correct}/120  reason_correct={wo_reason_correct}/120  AU={wo_au}/120",
        "",
        "--- GATE ---",
        f"format_valid_rate = {gate['format_valid_rate']:.4f}",
        f"decision_accuracy = {gate['decision_accuracy']:.4f}",
        f"GATE PASSED = {gate['passed']}",
        "",
        "--- Headline ---",
        f"FCA (lower better)       = {h['FCA']:.6f}",
        f"HOR_FCA (lower better)   = {h['HOR_FCA']:.6f}  "
        f"(n={h['HOR_count']['high_occlusion_unanswerable']})",
        f"MTS_micro           = {mts['MTS_micro']:.6f}",
        f"MTS_macro           = {mts['MTS_macro']:.6f}",
        "",
        "--- Safeguard ---",
        f"A_FU (lower better)      = {s['A_FU']:.6f}",
        f"U_Abs (higher better)    = {s['U_Abs']:.6f}",
        f"AAC (higher better)      = {s['AAC']:.6f}",
        f"MCC                      = {s['MCC']:.6f}",
        "",
        "--- Grounding / Reason ---",
        f"E2E_mIoU (IoU↑)     = {gr['E2E_mIoU']:.6f}",
        f"E2E_mIoU_c           = {gr['E2E_mIoU_c']:.6f}",
        f"A_Ground (depr.)     = {gr['A_Ground']:.6f}",
        f"A_Ground_c (depr.)   = {gr['A_Ground_c']:.6f}",
        f"U_Reason↑            = {gr['U_Reason']:.6f}",
        f"U_Reason_c           = {gr['U_Reason_c']:.6f}",
        "",
        f"invalid_rate        = {metrics['invalid_rate']:.6f}",
        f"confusion_matrix    = {cm}",
        "",
        "--- MTS per relation ---",
    ]
    for rel in sorted(mts.get("per_relation", {})):
        val = mts["per_relation"][rel]
        cnt = mts["per_relation_counts"].get(rel, {"pass": 0, "total": 0})
        lines.append(f"  {rel:20s} {val:.6f}  ({cnt['pass']}/{cnt['total']})")
    return lines


# ────────────────────────────────────────────────────────────────────────────
# Step 1: Determine excluded classes from SFT train
# ────────────────────────────────────────────────────────────────────────────


def get_sft_answer_labels() -> set[str]:
    """Return set of normalized answer labels that appear in SFT train records
    where decision_label == 1 (answer)."""
    data = load_json(SFT_TRAIN)
    labels: set[str] = set()
    for r in data:
        if r.get("decision_label") != 1:
            continue
        for conv in r.get("conversations", []):
            if conv.get("from") == "gpt":
                val = conv.get("value", "")
                try:
                    obj = json.loads(val)
                    label = obj.get("label", "")
                    if label:
                        labels.add(normalize_label(label))
                except (json.JSONDecodeError, TypeError):
                    pass
    return labels


def get_paired_rows_and_pv13() -> tuple[list[dict], dict[str, list[dict]]]:
    """Load paired 444 data and group by pv13_label."""
    rows = load_jsonl(PAIRED_444)
    by_pv13: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        sd = r.get("metadata", {}).get("_source_disjoint", {})
        pv13 = sd.get("pv13_label", "")
        by_pv13[pv13].append(r)
    return rows, by_pv13


def determine_excluded_classes(by_pv13: dict, sft_answer_labels: set[str]) -> set[str]:
    """Return set of pv13_labels to exclude (those NOT in SFT train answer labels)."""
    excluded: set[str] = set()
    for pv13 in by_pv13:
        if normalize_label(pv13) not in sft_answer_labels:
            excluded.add(pv13)
    return excluded


# ────────────────────────────────────────────────────────────────────────────
# Step 2: Filter paired data
# ────────────────────────────────────────────────────────────────────────────


def filter_paired_rows(
    all_rows: list[dict], excluded_pv13: set[str]
) -> tuple[list[dict], set[str]]:
    """Filter paired data, return (filtered_rows, kept_sample_ids)."""
    kept: list[dict] = []
    kept_sids: set[str] = set()
    for r in all_rows:
        sd = r.get("metadata", {}).get("_source_disjoint", {})
        pv13 = sd.get("pv13_label", "")
        if pv13 in excluded_pv13:
            continue
        kept.append(r)
        sid = sample_id_from_row(r)
        if sid:
            kept_sids.add(sid)
    return kept, kept_sids


# ────────────────────────────────────────────────────────────────────────────
# Step 3: Build audit
# ────────────────────────────────────────────────────────────────────────────


def build_audit(
    all_rows: list[dict],
    filtered_rows: list[dict],
    excluded_pv13: set[str],
    kept_sids: set[str],
    sft_answer_labels: set[str],
) -> dict:
    sft_data = load_json(SFT_TRAIN)
    sft_json_str = json.dumps(sft_data, sort_keys=True, ensure_ascii=False)

    # Per-class counts
    by_pv13_all: dict[str, int] = Counter()
    by_pv13_kept: dict[str, int] = Counter()
    for r in all_rows:
        sd = r.get("metadata", {}).get("_source_disjoint", {})
        pv13 = sd.get("pv13_label", "")
        by_pv13_all[pv13] += 1
    for r in filtered_rows:
        sd = r.get("metadata", {}).get("_source_disjoint", {})
        pv13 = sd.get("pv13_label", "")
        by_pv13_kept[pv13] += 1

    # Probe counts
    probe_counts_all: dict[str, int] = Counter()
    probe_counts_kept: dict[str, int] = Counter()
    for r in all_rows:
        md = r.get("metadata", {})
        probe = md.get("sample_id", "").rsplit("-", 1)[-1] if md.get("sample_id") else "unknown"
        probe_counts_all[probe] += 1
    for r in filtered_rows:
        md = r.get("metadata", {})
        probe = md.get("sample_id", "").rsplit("-", 1)[-1] if md.get("sample_id") else "unknown"
        probe_counts_kept[probe] += 1

    # Unique images
    unique_imgs_all: set[str] = set()
    unique_imgs_kept: set[str] = set()
    for r in all_rows:
        ik = image_key_from_sample_id(sample_id_from_row(r))
        if ik:
            unique_imgs_all.add(ik)
    for r in filtered_rows:
        ik = image_key_from_sample_id(sample_id_from_row(r))
        if ik:
            unique_imgs_kept.add(ik)

    return {
        "filter_rule": "Exclude pv13_label that does NOT appear as unique answer label in SFT train (≥1 occurrence). "
        "Model-performance-agnostic: no filtering by correct/incorrect predictions.",
        "sft_train_sha256": sha256_text(sft_json_str),
        "excluded_pv13_labels": sorted(excluded_pv13),
        "excluded_count": sum(by_pv13_all[pv13] for pv13 in excluded_pv13),
        "kept_pv13_labels": sorted(by_pv13_kept.keys()),
        "kept_class_count": len(by_pv13_kept),
        "per_class_counts_all": dict(by_pv13_all.most_common()),
        "per_class_counts_kept": dict(by_pv13_kept.most_common()),
        "probe_counts_all": dict(probe_counts_all.most_common()),
        "probe_counts_kept": dict(probe_counts_kept.most_common()),
        "unique_images_all": len(unique_imgs_all),
        "unique_images_kept": len(unique_imgs_kept),
        "sample_ids_kept": sorted(kept_sids),
        "all_sample_ids": sorted({sample_id_from_row(r) for r in all_rows}),
    }


# ────────────────────────────────────────────────────────────────────────────
# Step 4: Filter predictions and compute metrics for a method
# ────────────────────────────────────────────────────────────────────────────


def verify_closedness(predictions: list[dict], orig_metrics_path: Path) -> bool:
    """Verify that predictions[0:444] reproduce the original metrics.json confusion matrix."""
    pairs = build_pairs_from_predictions(predictions)
    cm = aggregate_metrics(pairs, iou_threshold=IOU_THRESHOLD)["confusion_matrix"]
    orig = load_json(orig_metrics_path)
    orig_cm = orig["metrics"]["confusion_matrix"]
    closed = cm == orig_cm
    if not closed:
        print(f"  ⚠️  CLOSEDNESS FAILED: got cm={cm}, expected={orig_cm}", flush=True)
    else:
        print(f"  ✓ Closedness verified: cm={cm}", flush=True)
    return closed


def process_method(
    method_name: str,
    predictions_path: Path,
    orig_metrics_path: Path | None,
    kept_sids: set[str],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Filter predictions for a method and write output files.

    Returns result summary dict, or None on failure.
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"Processing method: {method_name}", flush=True)
    print(f"  predictions: {predictions_path}", flush=True)

    predictions = load_jsonl(predictions_path)

    # Verify closedness against original metrics
    if orig_metrics_path and orig_metrics_path.exists():
        closed = verify_closedness(predictions, orig_metrics_path)
        if not closed:
            print(f"  ❌ Closedness verification FAILED for {method_name}", flush=True)
            return None
    else:
        print(
            f"  ⚠️  No original metrics.json for {method_name}, skipping closedness check",
            flush=True,
        )

    # Filter to kept sample_ids
    filtered: list[dict] = []
    filtered_sids: set[str] = set()
    for rec in predictions:
        sid = sample_id_from_row(rec)
        if sid in kept_sids:
            filtered.append(rec)
            filtered_sids.add(sid)

    print(f"  original predictions: {len(predictions)}", flush=True)
    print(f"  filtered predictions: {len(filtered)}", flush=True)

    if len(filtered) != len(kept_sids):
        print(
            f"  ⚠️  Mismatch: filtered={len(filtered)}, expected_sids={len(kept_sids)}", flush=True
        )
        print(f"  Missing SIDs: {kept_sids - filtered_sids}", flush=True)

    # Compute metrics
    pairs = build_pairs_from_predictions(filtered)
    metrics = aggregate_metrics(pairs, iou_threshold=IOU_THRESHOLD)
    mts = metamorphic_test_satisfaction(pairs, iou_threshold=IOU_THRESHOLD)
    gate = compute_gate(filtered)

    cm = metrics["confusion_matrix"]
    tp, fp, fn, tn, au, invalid = (cm[k] for k in ("TP", "FP", "FN", "TN", "AU", "INVALID"))

    # --- Per-probe breakdown ---
    nt_decision_correct = 0
    nt_reason_correct = 0
    nt_au = 0
    wo_decision_correct = 0
    wo_reason_correct = 0
    wo_au = 0
    strict_tp = 0  # TP among answerable-paired unique images
    answer_coverage = 0  # unique images with any answer-prediction (not just answerable)

    # Track which unique images have a TP for answerable
    tp_images: set[str] = set()
    # Track which unique images have any answerable prediction
    answerable_images: set[str] = set()
    # Answer coverage: unique images where model gave any answer on any probe
    images_with_answer_pred: set[str] = set()

    for rec in filtered:
        pred = rec["pred"]
        g = rec["gt"]
        sid = sample_id_from_row(rec)
        ik = image_key_from_sample_id(sid)
        if not pred.get("valid"):
            continue
        probe = sid.rsplit("-", 1)[-1] if sid else ""

        if probe == "no_target":
            if pred.get("decision") == "abstain":
                nt_decision_correct += 1
                if pred.get("reason") == "no_target":
                    nt_reason_correct += 1
            elif pred.get("decision") == "answer":
                nt_au += 1
            # Also count AU from the confusion matrix perspective
            if g.get("decision") == "abstain" and pred.get("decision") == "answer":
                pass  # already counted in cm["AU"]

        elif probe == "wrong_object":
            if pred.get("decision") == "abstain":
                wo_decision_correct += 1
                if pred.get("reason") == g.get("reason"):
                    wo_reason_correct += 1
            elif pred.get("decision") == "answer":
                wo_au += 1

        elif probe == "answer":
            if ik:
                answerable_images.add(ik)
            # Check if it's TP
            from reason_abstain.eval.metrics import score_answerable

            if pred.get("decision") == "answer" and score_answerable(pred, g, IOU_THRESHOLD):
                if ik:
                    tp_images.add(ik)
            # Check if model gave any answer on this image
            if ik and pred.get("decision") == "answer":
                images_with_answer_pred.add(ik)

    strict_tp = len(tp_images)
    answer_coverage = len(images_with_answer_pred)

    n_unique_images = len(answerable_images)

    # Build result for metrics.json (same structure as eval_qwen3vl_swift_3reason.py merge_and_score)
    result = {
        "tag": f"{method_name}/plantwild_real_closed12_v1",
        "data": str(OUT_CLOSED_JSONL),
        "adapter": "",
        "n_samples": len(filtered),
        "n_ambiguous_skipped": 0,
        "gate": gate,
        "metrics": metrics,
        "mts": mts,
        "within_image": None,
    }

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # predictions.jsonl
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in filtered) + "\n"
    )

    # metrics.json
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # summary.txt
    summary_lines = make_summary_lines(
        method_name=method_name,
        predictions=filtered,
        metrics=metrics,
        mts=mts,
        gate=gate,
        n=len(filtered),
        n_ambiguous=0,
        n_unique_images=n_unique_images,
        n_answerable_per_probe={},
        strict_tp=strict_tp,
        nt_decision_correct=nt_decision_correct,
        nt_reason_correct=nt_reason_correct,
        nt_au=nt_au,
        wo_decision_correct=wo_decision_correct,
        wo_reason_correct=wo_reason_correct,
        wo_au=wo_au,
        answer_coverage=answer_coverage,
    )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines))

    print(f"  ✓ Written: {output_dir}", flush=True)
    print(f"    cm={cm}", flush=True)
    print(f"    MCC={metrics['safeguard']['MCC']:.6f}", flush=True)
    print(f"    FCA={metrics['headline']['FCA']:.6f}", flush=True)

    return {
        "method": method_name,
        "n": len(filtered),
        "cm": cm,
        "gate": gate,
        "MCC": metrics["safeguard"]["MCC"],
        "FCA": metrics["headline"]["FCA"],
        "A_FU": metrics["safeguard"]["A_FU"],
        "U_Abs": metrics["safeguard"]["U_Abs"],
        "AAC": metrics["safeguard"]["AAC"],
        "MTS_micro": mts["MTS_micro"],
        "U_Reason": metrics["grounding_reason"]["U_Reason"],
        "U_Reason_c": metrics["grounding_reason"]["U_Reason_c"],
        "strict_TP": strict_tp,
        "answer_coverage": answer_coverage,
        "NT_decision": nt_decision_correct,
        "NT_reason": nt_reason_correct,
        "NT_AU": nt_au,
        "WO_decision": wo_decision_correct,
        "WO_reason": wo_reason_correct,
        "WO_AU": wo_au,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    missing = [p for p in (PAIRED_444, SFT_TRAIN, OURS_PRED) if not p.exists()]
    if missing:
        sys.exit(
            "This builder needs the full source research repository; the following inputs are "
            "absent from this archive:\n  " + "\n  ".join(str(p) for p in missing)
        )

    print("=" * 60, flush=True)
    print("Building PlantWild closed-set (12-class) evaluation", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Determine excluded classes
    print("\n--- Step 1: Determine excluded classes from SFT train ---", flush=True)
    sft_answer_labels = get_sft_answer_labels()
    print(f"  SFT unique answer labels: {len(sft_answer_labels)}", flush=True)

    all_rows, by_pv13 = get_paired_rows_and_pv13()
    print(f"  Paired 444 rows: {len(all_rows)}", flush=True)
    print(f"  pv13_labels in paired data: {len(by_pv13)}", flush=True)

    excluded_pv13 = determine_excluded_classes(by_pv13, sft_answer_labels)
    print(f"  Excluded pv13_labels: {excluded_pv13}", flush=True)

    # Step 2: Filter paired data
    print("\n--- Step 2: Filter paired data ---", flush=True)
    filtered_rows, kept_sids = filter_paired_rows(all_rows, excluded_pv13)
    print(f"  Filtered rows: {len(filtered_rows)}", flush=True)
    print(f"  Kept sample_ids: {len(kept_sids)}", flush=True)

    # Step 3: Write filtered paired data (A) and audit (B)
    print("\n--- Step 3: Write output files ---", flush=True)
    OUT_CLOSED_DIR.mkdir(parents=True, exist_ok=True)

    # A) Filtered paired data
    with open(OUT_CLOSED_JSONL, "w") as f:
        for row in filtered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Written: {OUT_CLOSED_JSONL}", flush=True)

    # B) BUILD_AUDIT
    audit = build_audit(all_rows, filtered_rows, excluded_pv13, kept_sids, sft_answer_labels)
    (OUT_CLOSED_DIR / "BUILD_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False)
    )
    print(f"  Written: {OUT_CLOSED_DIR / 'BUILD_AUDIT.json'}", flush=True)

    # Step 4: Process all 6 methods
    print("\n--- Step 4: Process all methods ---", flush=True)
    results: list[dict] = []

    # Ours
    ours_result = process_method(
        method_name="ours_fusion_3reason_v3_hl04",
        predictions_path=OURS_PRED,
        orig_metrics_path=OURS_METRICS_ORIG,
        kept_sids=kept_sids,
        output_dir=OURS_DIR,
    )
    if ours_result:
        results.append(ours_result)

    # Baselines
    for name in BASELINE_NAMES:
        pred_path = BASELINE_DIR / name / "plantwild_real_v1" / "predictions.jsonl"
        orig_metrics = BASELINE_DIR / name / "plantwild_real_v1" / "metrics.json"
        out_dir = BASELINE_DIR / name / "plantwild_real_closed12_v1"
        result = process_method(
            method_name=name,
            predictions_path=pred_path,
            orig_metrics_path=orig_metrics,
            kept_sids=kept_sids,
            output_dir=out_dir,
        )
        if result:
            results.append(result)

    # Step 5: Print final summary table
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY — PlantWild closed-set (12-class, 408 samples)", flush=True)
    print("=" * 80, flush=True)
    header = f"{'Method':30s} {'n':>4s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s} {'AU':>4s} {'MCC':>10s} {'FCA':>10s} {'A_FU':>8s} {'U_Abs':>8s} {'AAC':>8s} {'MTSμ':>8s} {'cov':>4s} {'sTP':>4s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        line = (
            f"{r['method']:30s} "
            f"{r['n']:4d} "
            f"{r['cm']['TP']:4d} "
            f"{r['cm']['FP']:4d} "
            f"{r['cm']['FN']:4d} "
            f"{r['cm']['TN']:4d} "
            f"{r['cm']['AU']:4d} "
            f"{r['MCC']:10.6f} "
            f"{r['FCA']:10.6f} "
            f"{r['A_FU']:8.4f} "
            f"{r['U_Abs']:8.4f} "
            f"{r['AAC']:8.4f} "
            f"{r['MTS_micro']:8.6f} "
            f"{r['answer_coverage']:4d} "
            f"{r['strict_TP']:4d}"
        )
        print(line, flush=True)
    print("-" * len(header), flush=True)

    # Also print per-probe breakdown
    print("\n--- Per-probe breakdown ---", flush=True)
    print(
        f"{'Method':30s} {'NTdec/144':>9s} {'NTrea/144':>9s} {'NTAU/144':>8s} {'WOdec/120':>9s} {'WOrea/120':>9s} {'WOAU/120':>8s}",
        flush=True,
    )
    print("-" * 85, flush=True)
    for r in results:
        line = (
            f"{r['method']:30s} "
            f"{r['NT_decision']:3d}/144  "
            f"{r['NT_reason']:3d}/144  "
            f"{r['NT_AU']:3d}/144  "
            f"{r['WO_decision']:3d}/120  "
            f"{r['WO_reason']:3d}/120  "
            f"{r['WO_AU']:3d}/120"
        )
        print(line, flush=True)
    print("-" * 85, flush=True)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
