#!/usr/bin/env python3
"""Fit alternative gate heads on an arm's calibration features.

The MLP variant replicates the published Stage-2 procedure exactly (mu/sd from the
calibration training split, 4096-512-256-3 with GELU and dropout 0.1, full-batch Adam at
lr 1e-3 / weight decay 1e-3, dev-loss early stopping with patience 5 evaluated every 50
epochs). Refitting it under our control separates "what the MLP architecture achieves" from
"what the specific released head achieves", which matters for the grouped arm because the
released head's stored statistics do not come from this checkpoint's features.

Nothing here touches the released heads; fitted gates are written to --output-dir.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn

EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE_LIMIT = 5


def load_shards(calibration_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(glob.glob(str(calibration_dir / f"{prefix}_rank*.npz")))
    if not paths:
        raise FileNotFoundError(f"no {prefix}_rank*.npz under {calibration_dir}")
    features, labels = [], []
    for path in paths:
        payload = np.load(path)
        features.append(payload["feats"].astype(np.float32))
        labels.append(payload["labels"].astype(np.int64))
    return np.concatenate(features), np.concatenate(labels)


def build_mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4096, 512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.run_seed)
    np.random.seed(args.run_seed)

    x_train, y_train = load_shards(args.calibration_dir, "calib_train")
    x_dev, y_dev = load_shards(args.calibration_dir, "calib_dev")
    print(f"[fit] train={x_train.shape} dev={x_dev.shape}", flush=True)

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-6
    x_train_std = torch.tensor((x_train - mu) / sd, dtype=torch.float32)
    x_dev_std = torch.tensor((x_dev - mu) / sd, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_dev_t = torch.tensor(y_dev, dtype=torch.long)

    model = build_mlp()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_dev_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history = []

    model.train()
    for epoch in range(EPOCHS):
        optimiser.zero_grad()
        loss = criterion(model(x_train_std), y_train_t)
        loss.backward()
        optimiser.step()

        if (epoch + 1) % EVAL_EVERY:
            continue

        model.eval()
        with torch.inference_mode():
            dev_logits = model(x_dev_std)
            dev_loss = criterion(dev_logits, y_dev_t).item()
            dev_predictions = dev_logits.argmax(-1).numpy()
        dev_mcc = float(matthews_corrcoef(y_dev, dev_predictions))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(loss.item()),
                "dev_loss": dev_loss,
                "dev_mcc": dev_mcc,
            }
        )
        print(
            f"[fit] epoch {epoch + 1}/{EPOCHS} train_loss={loss.item():.4f} "
            f"dev_loss={dev_loss:.4f} dev_mcc={dev_mcc:.4f}",
            flush=True,
        )
        model.train()

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience = 0
            print(f"[fit]   new best dev_loss={best_dev_loss:.4f}", flush=True)
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                print(f"[fit] early stop at epoch {epoch + 1}", flush=True)
                break

    if best_state is None:
        raise SystemExit("[fit][FATAL] no evaluation completed")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        train_predictions = model(x_train_std).argmax(-1).numpy()
        dev_predictions = model(x_dev_std).argmax(-1).numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state": {key: value for key, value in best_state.items()},
            "mu": torch.tensor(mu),
            "sd": torch.tensor(sd),
        },
        args.output_dir / "mlp_ours.pt",
    )
    manifest = {
        "gate": "mlp_ours",
        "architecture": "4096-512-256-3, GELU, dropout 0.1",
        "optimiser": "Adam lr=1e-3 weight_decay=1e-3, full batch",
        "selection": "lowest dev loss, patience 5 evaluated every 50 epochs",
        "run_seed": args.run_seed,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "train_rows": int(x_train.shape[0]),
        "dev_rows": int(x_dev.shape[0]),
        "train_accuracy": float((train_predictions == y_train).mean()),
        "train_mcc": float(matthews_corrcoef(y_train, train_predictions)),
        "dev_accuracy": float((dev_predictions == y_dev).mean()),
        "dev_mcc": float(matthews_corrcoef(y_dev, dev_predictions)),
        "calibration_dir": str(args.calibration_dir.resolve()),
        "history": history,
    }
    (args.output_dir / "mlp_ours_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[fit] done best_epoch={best_epoch} best_dev_loss={best_dev_loss:.4f} "
        f"train_mcc={manifest['train_mcc']:.4f} dev_mcc={manifest['dev_mcc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
