#!/usr/bin/env python3
"""scan_layers.py — Per-layer LogisticRegression scan for 36-layer diagnostic.

Given NPZ files from extract_all_layers.py, performs a seed-42 group-disjoint
internal train/dev split (by source_image_id), then for each layer fits
LogisticRegression with C∈{0.01,0.1,1.0} and per-layer normalisation.

Score = min(answer_accuracy, wrong_object_accuracy, no_target_accuracy).

Outputs:
  dev_scan.json        — full per-layer results
  best_layer.json      — best layer info
  min_score_curve.json — min_score vs layer depth
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# ── Constants ───────────────────────────────────────────────────────────────
EXPECTED_N_LAYERS = 36
DEV_SPLIT_SEED = 42
LOGISTIC_C_GRID = [0.01, 0.1, 1.0]


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════


class AllLayersDataset:
    """Load the multi-layer NPZ output from extract_all_layers.py."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path, allow_pickle=True)
        self.hidden = data["hidden"].astype(np.float32)  # [N, n_layers, D]
        self.head_label = data["head_label"].astype(np.int64)  # [N]
        self.original_idx = data.get("original_idx", np.arange(len(self.head_label))).astype(
            np.int64
        )
        self.n_layers = self.hidden.shape[1]
        self.n_samples = len(self.head_label)

        loaded_sid = data.get("source_image_id")
        self.source_image_id = (
            [str(x) for x in loaded_sid] if loaded_sid is not None else ["none"] * self.n_samples
        )

    def get_layer_data(self, layer_idx: int) -> np.ndarray:
        """Return features for layer layer_idx: [N, D]."""
        return np.ascontiguousarray(self.hidden[:, layer_idx, :])


# ═════════════════════════════════════════════════════════════════════════════
# Group-disjoint split (seed 42, by source_image_id)
# ═════════════════════════════════════════════════════════════════════════════


def group_disjoint_split(
    source_image_ids: list[str],
    head_label: np.ndarray,
    *,
    dev_target: int = 100,
    seed: int = DEV_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices by source_image_id groups, seed 42, disjoint.

    Greedily picks source groups until dev_target samples are reached
    with at least 25 samples per class.

    Returns:
        (train_indices, dev_indices) as 1D int arrays.
    """
    source_to_indices: dict[str, list[int]] = defaultdict(list)
    source_to_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for i in range(len(source_image_ids)):
        sid = source_image_ids[i]
        source_to_indices[sid].append(i)
        lbl = int(head_label[i])
        source_to_counts[sid][lbl] += 1

    all_sources = list(source_to_indices.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(all_sources)

    dev_indices_set: set[int] = set()
    dev_counts = [0, 0, 0]

    for sid in all_sources:
        if sum(dev_counts) >= dev_target and all(c >= 25 for c in dev_counts):
            break
        for idx in source_to_indices[sid]:
            dev_indices_set.add(idx)
            lbl = int(head_label[idx])
            dev_counts[lbl] += 1

    dev_indices = np.array(sorted(dev_indices_set), dtype=np.int64)
    train_indices = np.array(
        sorted(set(range(len(source_image_ids))) - dev_indices_set), dtype=np.int64
    )

    # Assert disjoint
    dev_sources = {source_image_ids[i] for i in dev_indices}
    train_sources = {source_image_ids[i] for i in train_indices}
    assert dev_sources.isdisjoint(train_sources), "Dev and train sources must be disjoint"

    return train_indices, dev_indices


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════


def evaluate_predictions(
    preds: np.ndarray,
    head_label: np.ndarray,
) -> dict:
    """Three-class metrics.

    - wo_recall:  (preds[WO]==1).mean()
    - answer_fa:  (preds[answer]!=0).mean()
    - nt_accuracy: (preds[nt]==2).mean()
    """
    wo_mask = head_label == 1
    wo_recall = float((preds[wo_mask] == 1).mean()) if wo_mask.sum() > 0 else 0.0

    ans_mask = head_label == 0
    answer_fa = float((preds[ans_mask] != 0).mean()) if ans_mask.sum() > 0 else 0.0

    nt_mask = head_label == 2
    nt_acc = float((preds[nt_mask] == 2).mean()) if nt_mask.sum() > 0 else 0.0

    confusion = np.zeros((3, 3), dtype=np.int64)
    for t in range(3):
        for p in range(3):
            confusion[t, p] = int(((head_label == t) & (preds == p)).sum())

    return {
        "wo_recall": round(wo_recall, 4),
        "answer_fa": round(answer_fa, 4),
        "nt_accuracy": round(nt_acc, 4),
        "n_wo": int(wo_mask.sum()),
        "n_answer": int(ans_mask.sum()),
        "n_nt": int(nt_mask.sum()),
        "confusion_matrix": confusion.tolist(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Per-layer normalisation
# ═════════════════════════════════════════════════════════════════════════════


def normalize_layer(
    train_feat: np.ndarray,
    dev_feat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Normalise using train statistics (mu/sd), apply to dev.

    Returns:
        (train_norm, dev_norm, mu, sd)
    """
    mu = np.mean(train_feat, axis=0, keepdims=True)
    sd = np.std(train_feat, axis=0, keepdims=True) + 1e-8
    train_norm = (train_feat - mu) / sd
    dev_norm = (dev_feat - mu) / sd if dev_feat is not None else None
    return train_norm, dev_norm, mu.flatten(), sd.flatten()


# ═════════════════════════════════════════════════════════════════════════════
# LogisticRegression with C grid
# ═════════════════════════════════════════════════════════════════════════════


def fit_logistic_cv(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    c_values: list[float] = LOGISTIC_C_GRID,
) -> tuple[dict, float, LogisticRegression]:
    """Fit LogisticRegression on train for each C, pick best on dev.

    Best C maximises min(wo_recall, 1-answer_fa, nt_accuracy).
    """
    best_score = -1.0
    best_metrics = None
    best_c = None
    best_model = None

    for C in c_values:
        clf = LogisticRegression(
            C=C,
            max_iter=1000,
            tol=1e-3,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(train_x, train_y)
        preds = clf.predict(dev_x)
        metrics = evaluate_predictions(preds, dev_y)
        score = min(metrics["wo_recall"], 1.0 - metrics["answer_fa"], metrics["nt_accuracy"])
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_c = C
            best_model = clf

    return best_metrics, best_c, best_model


# ═════════════════════════════════════════════════════════════════════════════
# Scan one layer
# ═════════════════════════════════════════════════════════════════════════════


def scan_one_layer(
    layer: int,
    layer_feat: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    head_label: np.ndarray,
) -> dict:
    """Scan a single layer: normalise, fit LogisticRegression with C grid.

    Returns a result dict.
    """
    t_layer = time.time()

    train_x = layer_feat[train_idx]
    train_y = head_label[train_idx]
    dev_x = layer_feat[dev_idx]
    dev_y = head_label[dev_idx]

    train_x_norm, dev_x_norm, _, _ = normalize_layer(train_x, dev_feat=dev_x)

    log_metrics, best_c, _ = fit_logistic_cv(train_x_norm, train_y, dev_x_norm, dev_y)
    log_score = min(
        log_metrics["wo_recall"], 1.0 - log_metrics["answer_fa"], log_metrics["nt_accuracy"]
    )

    entry = {
        "layer": layer,
        "npz_index": layer,
        "transformer_layer": layer + 1,
        "classifier": "logistic",
        "best_C": best_c,
        **log_metrics,
        "min_score": round(log_score, 4),
        "n_train": len(train_x),
        "n_dev": len(dev_x),
    }

    t_elapsed = time.time() - t_layer
    print(
        f"  [layer {layer}] Logistic: WO_recall={log_metrics['wo_recall'] * 100:.1f}% "
        f"answer_FA={log_metrics['answer_fa'] * 100:.1f}% "
        f"NT_acc={log_metrics['nt_accuracy'] * 100:.1f}% "
        f"min={log_score:.4f} (C={best_c}) [{t_elapsed:.1f}s]",
        flush=True,
    )
    return entry


# ═════════════════════════════════════════════════════════════════════════════
# Dev scan
# ═════════════════════════════════════════════════════════════════════════════


def run_dev_scan(args: argparse.Namespace) -> int:
    print(f"[scan_layers] Loading features: {args.train_npz}", flush=True)
    train_all = AllLayersDataset(args.train_npz)

    n_layers = train_all.n_layers
    if args.limit_layers is not None:
        n_layers = min(n_layers, args.limit_layers)
        print(
            f"[scan_layers] --limit-layers={args.limit_layers}, scanning {n_layers} layers",
            flush=True,
        )

    print(f"[scan_layers] train: {train_all.n_samples} samples, {n_layers} layers", flush=True)

    if n_layers != EXPECTED_N_LAYERS and args.limit_layers is None:
        print(
            f"[scan_layers] WARNING: n_layers={n_layers} != expected {EXPECTED_N_LAYERS}",
            flush=True,
        )

    # Group-disjoint split (seed 42)
    t_split = time.time()
    train_idx, dev_idx = group_disjoint_split(
        train_all.source_image_id,
        train_all.head_label,
        seed=DEV_SPLIT_SEED,
    )
    t_split = time.time() - t_split
    print(
        f"  [split] seed={DEV_SPLIT_SEED}, train={len(train_idx)} dev={len(dev_idx)} [{t_split:.1f}s]",
        flush=True,
    )

    # Preload layer features
    print(f"  [scan_layers] Preloading {n_layers} layer features ...", flush=True)
    t_preload = time.time()
    layer_feats = [train_all.get_layer_data(li) for li in range(n_layers)]
    t_preload = time.time() - t_preload
    print(f"  [scan_layers] Preload done [{t_preload:.1f}s]", flush=True)

    # Scan layers
    dev_results = []
    for layer in range(n_layers):
        entry = scan_one_layer(layer, layer_feats[layer], train_idx, dev_idx, train_all.head_label)
        dev_results.append(entry)

    # Best layer: highest min_score, tiebreak shallower layer first
    logistic_results = [r for r in dev_results if r["classifier"] == "logistic"]
    logistic_results.sort(key=lambda r: (-r["min_score"], r["layer"]))

    best_entry = logistic_results[0] if logistic_results else None
    best_layer = best_entry["layer"] if best_entry else 0

    print(f"\n{'=' * 60}", flush=True)
    print("[scan_layers] Dev scan complete", flush=True)
    print(
        f"  Best layer: npz_index={best_layer}, transformer_layer={best_layer + 1} "
        f"(min_score={best_entry['min_score']:.4f})",
        flush=True,
    )
    print(
        f"  WO_recall={best_entry['wo_recall'] * 100:.1f}% "
        f"answer_FA={best_entry['answer_fa'] * 100:.1f}% "
        f"NT_acc={best_entry['nt_accuracy'] * 100:.1f}%",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dev_scan.json
    dev_scan = {
        "baseline_note": "baseline layer 27 = npz index 26 (npz_index + 1 = transformer_layer)",
        "n_layers": n_layers,
        "dev_split_seed": DEV_SPLIT_SEED,
        "results": dev_results,
    }
    dev_scan_path = out_dir / "dev_scan.json"
    dev_scan_path.write_text(json.dumps(dev_scan, ensure_ascii=False, indent=2) + "\n")
    print(f"[scan_layers] Dev scan: {dev_scan_path}", flush=True)

    # best_layer.json
    best_layer_info = {
        "best_layer": best_layer,
        "npz_index": best_layer,
        "transformer_layer": best_layer + 1,
        "note": "baseline layer 27 = npz index 26 (npz_index + 1 = transformer_layer)",
        "best_layer_entry": best_entry,
    }
    best_layer_path = out_dir / "best_layer.json"
    best_layer_path.write_text(json.dumps(best_layer_info, ensure_ascii=False, indent=2) + "\n")
    print(f"[scan_layers] Best layer: {best_layer_path}", flush=True)

    # min_score_curve.json
    curve = []
    for r in logistic_results:
        curve.append(
            {
                "layer": r["layer"],
                "npz_index": r["layer"],
                "transformer_layer": r["layer"] + 1,
                "min_score": r["min_score"],
                "wo_recall": r["wo_recall"],
                "answer_fa": r["answer_fa"],
                "nt_accuracy": r["nt_accuracy"],
            }
        )
    curve_path = out_dir / "min_score_curve.json"
    curve_path.write_text(json.dumps(curve, ensure_ascii=False, indent=2) + "\n")
    print(f"[scan_layers] Min-score curve: {curve_path}", flush=True)

    return 0


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-layer LogisticRegression scan for 36-layer diagnostic",
    )
    ap.add_argument("--train-npz", required=True, help="Path to hidden_train.npz")
    ap.add_argument("--holdout-npz", required=True, help="Path to hidden_holdout.npz")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--limit-layers", type=int, default=None, help="Only scan first N layers (debug)"
    )
    args = ap.parse_args()

    return run_dev_scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
