#!/usr/bin/env python3
"""CPU Stage-2 MLP gate trained on the FROZEN base layer-27 features.

Architecture and hyper-parameters replicate the paper's Stage-2 head exactly:
  4096-512-256-3, GELU, dropout 0.1, full-batch Adam lr=1e-3 wd=1e-3,
  dev-loss early stopping evaluated every 50 epochs with patience 5, seed 42.

Development split
  默认使用完整dev；--dev-indices使用经original_index与SFT记录唯一匹配的
  论文100条dev子集，映射由match_exact_dev.py生成。

Outputs under --out:
  mlp_ours.pt            best-state head (head_state/mu/sd), same layout as fit_gates.py
  mlp_ours_manifest.json history + best epoch / dev loss
  metrics.json           probe metrics on dev/test
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from frozen_common import load_split
from sklearn.metrics import matthews_corrcoef, roc_auc_score

EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE_LIMIT = 5
POOL = "last_token"


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


def probe(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    supported = (y == 0).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(supported, proba[:, 0])),
        "argmax_accuracy": float((pred == y).mean()),
        "mcc_3class": float(matthews_corrcoef(y, pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-seed", default=42, type=int)
    parser.add_argument("--dev-indices", type=Path, help="缓存dev中的精确子集行号JSON")
    args = parser.parse_args()

    torch.manual_seed(args.run_seed)
    np.random.seed(args.run_seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train = load_split(args.repr_root, "train", pool=POOL)
    dev = load_split(args.repr_root, "dev", pool=POOL)
    test = load_split(args.repr_root, "test", pool=POOL)

    x_train, y_train = train["features"].astype(np.float32), train["labels"]
    x_dev, y_dev = dev["features"].astype(np.float32), dev["labels"]
    if args.dev_indices:
        indices = json.loads(args.dev_indices.read_text())["cache_indices"]
        x_dev, y_dev = x_dev[indices], y_dev[indices]
    x_test, y_test = test["features"].astype(np.float32), test["labels"]

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-6
    x_train_s = torch.tensor((x_train - mu) / sd, dtype=torch.float32)
    x_dev_s = torch.tensor((x_dev - mu) / sd, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_dev_t = torch.tensor(y_dev, dtype=torch.long)

    model = build_mlp()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(
        f"[frozen-mlp] train={x_train.shape} dev={x_dev.shape} test={x_test.shape} "
        f"pool={POOL} seed={args.run_seed}",
        flush=True,
    )
    best_dev_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history: list[dict] = []

    model.train()
    for epoch in range(EPOCHS):
        optimiser.zero_grad()
        loss = criterion(model(x_train_s), y_train_t)
        loss.backward()
        optimiser.step()

        if (epoch + 1) % EVAL_EVERY:
            continue
        model.eval()
        with torch.inference_mode():
            dev_logits = model(x_dev_s)
            dev_loss = criterion(dev_logits, y_dev_t).item()
            dev_pred = dev_logits.argmax(-1).numpy()
        dev_mcc = float(matthews_corrcoef(y_dev, dev_pred))
        history.append(
            {"epoch": epoch + 1, "train_loss": float(loss.item()), "dev_loss": dev_loss,
             "dev_mcc_3class": dev_mcc}
        )
        print(
            f"[frozen-mlp] epoch {epoch + 1} train_loss={loss.item():.4f} "
            f"dev_loss={dev_loss:.4f} dev_mcc={dev_mcc:.4f}",
            flush=True,
        )
        model.train()
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience = 0
            print(f"[frozen-mlp]   new best dev_loss={best_dev_loss:.4f}", flush=True)
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                print(f"[frozen-mlp] early stop at epoch {epoch + 1}", flush=True)
                break

    if best_state is None:
        raise SystemExit("[frozen-mlp][FATAL] no evaluation completed")
    model.load_state_dict(best_state)
    model.eval()

    def proba_of(x: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            return torch.softmax(model(x), dim=-1).numpy()

    train_proba = proba_of(x_train_s)
    dev_proba = proba_of(x_dev_s)
    test_proba = proba_of(torch.tensor((x_test - mu) / sd, dtype=torch.float32))

    torch.save(
        {
            "head_state": {k: v for k, v in best_state.items()},
            "mu": torch.tensor(mu),
            "sd": torch.tensor(sd),
        },
        args.out / "mlp_ours.pt",
    )
    np.save(args.out / "test_proba.npy", test_proba)
    manifest = {
        "gate": "frozen_base_mlp",
        "architecture": "4096-512-256-3, GELU, dropout 0.1",
        "optimiser": "Adam lr=1e-3 weight_decay=1e-3, full batch",
        "selection": f"lowest dev loss on {len(y_dev)} frozen dev records, patience 5, eval every 50",
        "dev_indices_source": str(args.dev_indices) if args.dev_indices else None,
        "run_seed": args.run_seed,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "train_rows": int(x_train.shape[0]),
        "dev_rows": int(x_dev.shape[0]),
        "repr_root": str(args.repr_root.resolve()),
        "pool": POOL,
        "history": history,
        "train_probe": probe(train_proba, y_train),
        "dev_probe": probe(dev_proba, y_dev),
        "test_probe": probe(test_proba, y_test),
    }
    (args.out / "mlp_ours_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[frozen-mlp] done best_epoch={best_epoch} best_dev_loss={best_dev_loss:.4f} "
        f"test_auroc={manifest['test_probe']['auroc']:.4f} "
        f"test_mcc_3class={manifest['test_probe']['mcc_3class']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
