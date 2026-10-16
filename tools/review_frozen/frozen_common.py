#!/usr/bin/env python3
"""Shared helpers for the frozen-base gate review (CPU only).

The frozen repr cache lives at
    <repo>/new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/repr/{train,dev,test}
and stores layer-27 4096-d features of the FROZEN base backbone (adapter=None) for the
paper's sft_train/dev split and comprehensive test split (7072 / 2408 / 2646 rows).

This module is a pure re-implementation used to *scoring* gated operating points from
frozen features while sharing the frozen-base native answer candidates. It imports the
paper's metric machinery (aaai_code/experiments/base_native_regate/analyze_native_regate.py)
so FCA / retention / Strict-3 / E2E are computed identically to the paper.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from sklearn.metrics import matthews_corrcoef, roc_auc_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
AAAI_ROOT = REPO_ROOT / "aaai_code"
AAAI_ROOT = AAAI_ROOT.resolve()
if str(AAAI_ROOT) not in sys.path:
    sys.path.insert(0, str(AAAI_ROOT))

from experiments.base_native_regate.analyze_native_regate import (  # noqa: E402
    regate_curve_point,
)
from reason_abstain.eval.metrics import metamorphic_test_satisfaction  # noqa: E402

REPR_ROOT = (
    REPO_ROOT
    / "new_reason"
    / "outputs"
    / "evidence_judge_full_4090_b1_v2_20260719"
    / "repr"
)
BASE_NATIVE_PREDS = (
    REPO_ROOT
    / "new_reason"
    / "outputs"
    / "base_native_regate_20260720"
    / "native"
    / "base"
    / "comprehensive"
    / "predictions.jsonl"
)

EXPECTED_ROWS = {"train": 7072, "dev": 2408, "test": 2646}


def load_split(repr_root: Path, split: str, pool: str = "last_token") -> dict:
    payload = np.load(repr_root / split / f"{split}.npz", allow_pickle=True)
    labels = np.asarray(payload["head_label"]).astype(np.int64)
    if labels.shape[0] != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_ROWS[split]} rows, found {labels.shape[0]}"
        )
    manifest = json.loads(str(payload["manifest_json"]))
    if manifest.get("adapter") is not None:
        raise ValueError(f"{split}: manifest reports adapter={manifest['adapter']!r}, not frozen")
    if int(manifest.get("layer", -1)) != 27:
        raise ValueError(f"{split}: manifest reports layer={manifest.get('layer')!r}, not 27")
    features = None
    for key in ("last_token", "visual_mean", "visual_max", "visual_query_attn"):
        if payload[key].ndim == 2 and payload[key].shape[1] == 4096:
            features = np.asarray(payload[pool]).astype(np.float64)
            break
    if features is None:
        raise ValueError("no 4096-d pooling feature found")
    return {
        "labels": labels,
        "sample_id": np.asarray([str(s) for s in payload["sample_id"]]),
        "features": features,
        "manifest": manifest,
    }


def softmax01(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_base_native_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not str(rec.get("sample_id", "")):
                raise ValueError(f"{path}: missing sample_id")
            rows.append(rec)
    return rows


def gt_class_from_gt(gt: dict) -> int:
    if gt.get("decision") == "answer":
        return 0
    if gt.get("reason") == "wrong_object":
        return 1
    if gt.get("reason") == "no_target":
        return 2
    raise ValueError(f"Unsupported ground truth: {gt}")


def build_base_native_candidate(pred: dict, raw: str = "") -> dict:
    """Shared-candidate convention: if the FROZEN base natively answered, its decode is the
    shared answer candidate; if it abstained there is no candidate (invalid)."""
    if isinstance(pred, dict) and pred.get("valid") is True and pred.get("decision") == "answer":
        return {
            "valid": True,
            "label": pred.get("label"),
            "box": pred.get("box"),
            "raw": raw,
        }
    return {"valid": False, "label": None, "box": None, "raw": raw}


def build_gate_records(
    base_rows: list[dict],
    proba: np.ndarray,
) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (records, answered_map, supported, y_state, predicted_classes, record_keys).

    proba: (n, 3) p_answer/p_wo/p_nt in the same row order as base_rows.
    Records are keyed by '{index}:{sample_id}' to allow duplicate-safe evaluation.
    The three arrays are aligned to the *scored* subset (keys in the same order).
    """
    records: dict[str, dict] = {}
    answered_map: dict[str, bool] = {}
    supported: list[int] = []
    y_state: list[int] = []
    predicted_classes: list[int] = []
    record_keys: list[str] = []
    for index, rec in enumerate(base_rows):
        sample_id = str(rec["sample_id"])
        key = f"{index}:{sample_id}"
        p_answer, p_wo, p_nt = (float(v) for v in proba[index])
        pred_class = int(np.argmax(proba[index]))
        gt = rec.get("gt")
        if not isinstance(gt, dict) or gt.get("decision") not in {"answer", "abstain"}:
            continue
        gt_fixed = {
            "decision": gt.get("decision"),
            "reason": gt.get("reason"),
            "label": gt.get("label"),
            "box": gt.get("box"),
            "metadata": (gt.get("metadata") or {}) or {"relation": rec.get("relation")},
        }
        supported.append(int(gt.get("decision") == "answer"))
        y_state.append(gt_class_from_gt(gt))
        predicted_classes.append(pred_class)
        records[key] = {
            "sample_id": sample_id,
            "relation": rec.get("relation"),
            "gt": gt_fixed,
            "p_answer": p_answer,
            "p_wo": p_wo,
            "p_nt": p_nt,
            "answer_candidate": build_base_native_candidate(
                rec.get("pred"), rec.get("raw", "")
            ),
        }
        answered_map[key] = bool(p_answer >= p_wo and p_answer >= p_nt)
        record_keys.append(key)
    return (
        records,
        answered_map,
        np.asarray(supported, dtype=np.int64),
        np.asarray(y_state, dtype=np.int64),
        np.asarray(predicted_classes, dtype=np.int64),
        record_keys,
    )


def predict_with_model(model: dict, x: np.ndarray) -> np.ndarray:
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    mu = np.asarray(model["mu"], dtype=np.float64)
    sd = np.asarray(model["sd"], dtype=np.float64)
    return softmax01(((x - mu) / sd) @ coef.T + intercept)


def probe_like_eval(x_norm: np.ndarray, y: np.ndarray, proba: np.ndarray) -> dict:
    """AUROC / argmax accuracy / 3-class MCC exactly as frozen_probe.py reported them."""
    pred = proba.argmax(axis=1)
    supported = (y == 0).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(supported, proba[:, 0])),
        "argmax_accuracy": float((pred == y).mean()),
        "mcc_3class": float(matthews_corrcoef(y, pred)),
    }


def gated_point_metrics(
    records: dict,
    answered_map: dict,
    dataset: str = "comprehensive",
) -> dict:
    point = regate_curve_point(dataset, records, answered_map, "three_class_argmax", None)
    out = {
        "FCA": point.unsupported_hallucination_rate,
        "supported_query_retention": point.supported_query_retention,
        "A_FU": 1.0 - point.supported_query_retention,
        "strict_3_accuracy": point.strict_3_accuracy,
        "E2E_mIoU": point.grounding_accuracy,
        "selective_risk": point.selective_risk,
        "answered": point.answered,
        "answered_supported": point.answered_supported,
        "answered_unsupported": point.answered_unsupported,
        "supported": point.supported,
        "unsupported": point.unsupported,
    }
    return out
