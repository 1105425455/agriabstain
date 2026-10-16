#!/usr/bin/env python3
"""CPU fit of a multinomial linear gate on the FROZEN base layer-27 features.

Mirrors papermodel_lineargate/scripts/frozen_probe.py (last-token pool, C grid selected by
full-dev loss on the 2408-record dev split of the frozen cache). No GPU, no backbone pass.

Notes on development split:
  * The paper's Stage-2 heads (adapted arms) select on a 100-record stratified dev subset
    of new_reason/outputs/splits/dev.jsonl. The frozen cache's dev rows carry NO semantic
    sample id (sft json row indices only) and its image identity does not map onto the
    dev.jsonl sample ids (different derived-image build), so the paper's exact 100-record
    subset CANNOT be reconstructed inside this cache. Selection therefore uses the full
    2408-row frozen dev, which strictly uses *more* dev signal than the adapted arms.

Outputs under --out:
  metrics.jsonl          per-C dev/test probe metrics
  linear_gate_best.npz   best-C model (same format as train_linear_gate.save_model)
  summary.json           selected-C record + manifest
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from frozen_common import load_split, predict_with_model, probe_like_eval

C_GRID = (0.01, 0.1, 1.0, 10.0)
POOL = "last_token"


def save_model(path: Path, model, mu, sd, run_seed, selected_c) -> None:
    np.savez_compressed(
        path,
        coef=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float32),
        classes=model.classes_.astype(np.int64),
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
        run_seed=np.asarray(run_seed, dtype=np.int64),
        selected_c=np.asarray(selected_c, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pool", default=POOL)
    parser.add_argument("--run-seed", default=42, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {
        split: load_split(args.repr_root, split, pool=args.pool)
        for split in ("train", "dev", "test")
    }
    print(
        f"[frozen-linear] base={splits['test']['manifest']['base']} "
        f"adapter=None layer=27 pool={args.pool}",
        flush=True,
    )
    x_train, y_train = splits["train"]["features"], splits["train"]["labels"]
    x_dev, y_dev = splits["dev"]["features"], splits["dev"]["labels"]
    x_test, y_test = splits["test"]["features"], splits["test"]["labels"]

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-6
    x_train_n = (x_train - mu) / sd
    x_dev_n = (x_dev - mu) / sd
    x_test_n = (x_test - mu) / sd

    rows: list[dict] = []
    fitted: dict[float, LogisticRegression] = {}
    for c_value in C_GRID:
        model = LogisticRegression(
            C=c_value, solver="lbfgs", max_iter=2000, random_state=args.run_seed
        )
        model.fit(x_train_n, y_train)
        if model.classes_.tolist() != [0, 1, 2]:
            raise ValueError(f"Unexpected classes: {model.classes_.tolist()}")
        fitted[c_value] = model
        dev_proba = model.predict_proba(x_dev_n)
        row = {
            "pool": args.pool,
            "C": c_value,
            "dev_loss": float(log_loss(y_dev, dev_proba, labels=[0, 1, 2])),
            "dev_auroc": float(probe_like_eval(x_dev_n, y_dev, dev_proba)["auroc"]),
            "dev_mcc_3class": float(probe_like_eval(x_dev_n, y_dev, dev_proba)["mcc_3class"]),
            "dev_argmax_accuracy": float(probe_like_eval(x_dev_n, y_dev, dev_proba)["argmax_accuracy"]),
            "iterations": int(model.n_iter_.max()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        test_proba = model.predict_proba(x_test_n)
        row.update(
            {
                "test_auroc": float(roc_for(y_test, test_proba)),
                "test_argmax_accuracy": float((test_proba.argmax(axis=1) == y_test).mean()),
                "test_mcc_3class": float(mcc_3class(y_test, test_proba)),
            }
        )
        rows.append(row)
        print(
            f"[frozen-linear] C={c_value:g} dev_loss={row['dev_loss']:.4f} "
            f"test_auroc={row['test_auroc']:.4f} test_mcc_3class={row['test_mcc_3class']:.4f}",
            flush=True,
        )

    best = min(rows, key=lambda r: (r["dev_loss"], r["C"]))
    save_model(
        args.out / "linear_gate_best.npz",
        fitted[best["C"]],
        mu,
        sd,
        args.run_seed,
        float(best["C"]),
    )

    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "classifier": "multinomial LogisticRegression",
        "backbone": splits["test"]["manifest"]["base"],
        "adapter": None,
        "layer": 27,
        "pool": args.pool,
        "c_grid": list(C_GRID),
        "selection": "min dev_loss on FULL frozen dev (2408) - cached dev has no identity-matched "
                     "paper 100-record stratified subset",
        "run_seed": args.run_seed,
        "best": best,
        "all": rows,
        "frozen_repr_manifest_train": splits["train"]["manifest"],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[frozen-linear] selected C={best['C']:g} dev_loss={best['dev_loss']:.4f} "
        f"test_auroc={best['test_auroc']:.4f}",
        flush=True,
    )


def roc_for(y, proba) -> float:
    from sklearn.metrics import roc_auc_score
    supported = (y == 0).astype(np.int64)
    return float(roc_auc_score(supported, proba[:, 0]))


def mcc_3class(y, proba) -> float:
    from sklearn.metrics import matthews_corrcoef
    return float(matthews_corrcoef(y, proba.argmax(axis=1)))


if __name__ == "__main__":
    main()