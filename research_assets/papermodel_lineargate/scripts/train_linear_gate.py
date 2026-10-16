#!/usr/bin/env python3
"""Fit a development-selected multinomial linear gate on cached H27 features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, matthews_corrcoef
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0)


def load_shards(root: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(root.glob(f"{prefix}_rank*.npz"))
    if not paths:
        raise FileNotFoundError(f"No {prefix} shards under {root}")
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for path in paths:
        with np.load(path) as shard:
            features.append(shard["feats"].astype(np.float32))
            labels.append(shard["labels"].astype(np.int64))
    return np.concatenate(features), np.concatenate(labels)


def save_model(
    path: Path,
    model: LogisticRegression,
    mu: np.ndarray,
    sd: np.ndarray,
    run_seed: int,
    selected_c: float,
) -> None:
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


def gpu_start_report() -> None:
    if not torch.cuda.is_available():
        print("[gpu-start] CUDA unavailable; LogisticRegression runs on CPU", flush=True)
        return
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        used = total - free
        ratio = used / total
        print(
            f"[gpu-start] gpu={index} memory={used}/{total} bytes ({ratio:.1%})",
            flush=True,
        )
        if ratio > 0.90:
            print(f"[gpu-start][WARN] gpu={index} memory usage exceeds 90%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-seed", required=True, type=int)
    args = parser.parse_args()

    gpu_start_report()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_shards(args.calibration_dir, "calib_train")
    x_dev, y_dev = load_shards(args.calibration_dir, "calib_dev")
    if x_train.shape != (7072, 4096):
        raise ValueError(f"Expected train shape (7072, 4096), found {x_train.shape}")
    if x_dev.shape != (100, 4096):
        raise ValueError(f"Expected dev shape (100, 4096), found {x_dev.shape}")

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mu) / sd
    x_dev = (x_dev - mu) / sd

    rows: list[dict[str, Any]] = []
    fitted: dict[float, LogisticRegression] = {}
    for c_value in tqdm(C_GRID, desc="linear-gate C grid", unit="candidate"):
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=2000,
            random_state=args.run_seed,
        )
        model.fit(x_train, y_train)
        if model.classes_.tolist() != [0, 1, 2]:
            raise ValueError(f"Unexpected classes: {model.classes_.tolist()}")
        dev_prob = model.predict_proba(x_dev)
        dev_pred = model.classes_[dev_prob.argmax(axis=1)]
        row = {
            "C": c_value,
            "dev_loss": float(log_loss(y_dev, dev_prob, labels=[0, 1, 2])),
            "dev_mcc": float(matthews_corrcoef(y_dev, dev_pred)),
            "iterations": int(model.n_iter_.max()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        fitted[c_value] = model
        save_model(
            candidate_dir / f"linear_gate_C{c_value:g}.npz",
            model,
            mu,
            sd,
            args.run_seed,
            c_value,
        )
        tqdm.write(f"C={c_value:g} dev_loss={row['dev_loss']:.6f} dev_mcc={row['dev_mcc']:.4f}")

    best_row = min(rows, key=lambda row: (row["dev_loss"], row["C"]))
    best_c = float(best_row["C"])
    best_model = fitted[best_c]
    save_model(
        args.output_dir / "linear_gate_best.npz",
        best_model,
        mu,
        sd,
        args.run_seed,
        best_c,
    )
    final_c = float(C_GRID[-1])
    save_model(
        args.output_dir / "linear_gate_final.npz",
        fitted[final_c],
        mu,
        sd,
        args.run_seed,
        final_c,
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    plt.figure(figsize=(6.5, 4.2))
    plt.semilogx(
        [row["C"] for row in rows],
        [row["dev_loss"] for row in rows],
        marker="o",
        label="development loss",
    )
    plt.xlabel("C")
    plt.ylabel("Three-class cross-entropy")
    plt.title("Linear gate development selection")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    curve_path = args.output_dir / "loss_curves.png"
    plt.savefig(curve_path, dpi=180)
    plt.close()

    boundary = best_c in {min(C_GRID), max(C_GRID)}
    analysis = (
        "The selected regularization lies on the search-grid boundary; a wider grid "
        "would be needed before interpreting regularization sensitivity."
        if boundary
        else "The selected regularization is interior to the frozen search grid."
    )
    final_metrics = {
        "classifier": "multinomial LogisticRegression",
        "run_seed": args.run_seed,
        "train_records": int(len(y_train)),
        "dev_records": int(len(y_dev)),
        "c_grid": list(C_GRID),
        "best_C": best_c,
        "best_dev_loss": best_row["dev_loss"],
        "best_dev_mcc": best_row["dev_mcc"],
        "best_model": str((args.output_dir / "linear_gate_best.npz").resolve()),
        "final_model": str((args.output_dir / "linear_gate_final.npz").resolve()),
        "loss_curve": str(curve_path.resolve()),
        "analysis": analysis,
    }
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "process.md").write_text(
        "\n".join(
            [
                "# Linear gate 训练记录",
                "",
                f"- run seed: `{args.run_seed}`",
                f"- train/dev records: `{len(y_train)}/{len(y_dev)}`",
                f"- selected C: `{best_c:g}`",
                f"- best dev loss: `{best_row['dev_loss']:.6f}`",
                f"- best dev MCC: `{best_row['dev_mcc']:.4f}`",
                f"- loss curve: `{curve_path}`",
                f"- 分析：{analysis}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"[linear-gate] best_C={best_c:g} dev_loss={best_row['dev_loss']:.6f} "
        f"dev_mcc={best_row['dev_mcc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
