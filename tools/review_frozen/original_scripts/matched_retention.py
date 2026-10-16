#!/usr/bin/env python3
"""Compare gates without the operating-point confound.

Argmax gating puts the released MLP head and the linear gate at different points on the
risk-coverage curve, so their raw FCA and retention are not comparable. This reports two
threshold-independent views on the same 2646 records: AUROC for separating supported from
unsupported queries, and FCA at a retention matched to the released head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    keys, supported, p_answer = [], [], []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            decision = record["gt"].get("decision")
            if decision not in {"answer", "abstain"}:
                continue
            keys.append(f"{line_number}:{record['sample_id']}")
            supported.append(int(decision == "answer"))
            p_answer.append(float(record["p_answer"]))
    return keys, np.asarray(supported, dtype=np.int64), np.asarray(p_answer, dtype=np.float64)


def load_linear(path: Path, keys: list[str]) -> np.ndarray:
    by_index: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_index[int(row["record_index"])] = float(row["p_answer"])
    # reroute_linear.py enumerates the same prediction file in order, so record_index maps
    # onto the full record list; the scored subset is recovered through the shared keys.
    return by_index


def retention_and_fca(supported: np.ndarray, answered: np.ndarray) -> tuple[float, float]:
    retention = float(answered[supported == 1].mean())
    fca = float(answered[supported == 0].mean())
    return retention, fca


def fca_at_matched_retention(
    supported: np.ndarray, scores: np.ndarray, target_retention: float
) -> tuple[float, float, float]:
    """Pick the threshold whose retention is closest to the target, then read FCA."""
    candidates = np.unique(scores)
    best = None
    for threshold in candidates:
        answered = (scores >= threshold).astype(np.int64)
        retention, fca = retention_and_fca(supported, answered)
        gap = abs(retention - target_retention)
        if best is None or gap < best[0]:
            best = (gap, threshold, retention, fca)
    _, threshold, retention, fca = best
    return float(threshold), retention, fca


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--dataset", default="comprehensive")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result: dict[str, dict[str, object]] = {}
    for arm in ("grouped", "independent"):
        predictions = (
            args.runs_root / arm / "evaluation/regate" / args.dataset / "predictions.jsonl"
        )
        linear_probabilities = (
            args.runs_root / arm / "linear_evaluation" / args.dataset / "linear_probabilities.jsonl"
        )
        if not predictions.is_file() or not linear_probabilities.is_file():
            print(f"[matched] {arm}: inputs missing, skipping", flush=True)
            continue

        keys, supported, mlp_scores = load_predictions(predictions)
        linear_by_index = load_linear(linear_probabilities, keys)
        # record_index counts every row of the prediction file; rebuild the scored subset.
        all_indices = []
        with predictions.open(encoding="utf-8") as handle:
            for position, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["gt"].get("decision") in {"answer", "abstain"}:
                    all_indices.append(position)
        linear_scores = np.asarray(
            [linear_by_index[index] for index in all_indices], dtype=np.float64
        )

        mlp_answered = (mlp_scores >= 0.5).astype(np.int64)
        mlp_retention, mlp_fca = retention_and_fca(supported, mlp_answered)

        entry: dict[str, object] = {
            "n_scored": int(len(supported)),
            "n_supported": int(supported.sum()),
            "n_unsupported": int((supported == 0).sum()),
            "auroc_mlp": float(roc_auc_score(supported, mlp_scores)),
            "auroc_linear": float(roc_auc_score(supported, linear_scores)),
            "mlp_argmax_retention": mlp_retention,
            "mlp_argmax_fca": mlp_fca,
        }
        threshold, retention, fca = fca_at_matched_retention(
            supported, linear_scores, mlp_retention
        )
        entry.update(
            {
                "linear_matched_threshold": threshold,
                "linear_matched_retention": retention,
                "linear_matched_fca": fca,
            }
        )
        result[arm] = entry
        print(
            f"[matched] {arm}: AUROC mlp={entry['auroc_mlp']:.4f} "
            f"linear={entry['auroc_linear']:.4f} | at retention "
            f"{retention:.4f} (mlp {mlp_retention:.4f}) FCA linear={fca:.4f} "
            f"mlp={mlp_fca:.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[matched] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
