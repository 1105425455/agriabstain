"""在冻结基座(无 LoRA、无辅助头)的 layer-27 表征上训练线性门。

对照实验目的:回答"为什么用 MLP 头"这一质疑。若冻结表征上的线性探针已经
接近适配后的线性门,则 Stage-1 的判别头是多余的;若明显落后,则说明可分性
是 Stage-1 联合训练塑造出来的,而不是基座自带的。

为了让基线尽可能占优:
  * 四种 pooling 变体全部扫描,取 test AUROC 最高者作为冻结基线;
  * C 网格在完整 dev(2408 条)上选,而适配门只用 100 条标定 dev。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from tqdm import tqdm

C_GRID = (0.01, 0.1, 1.0, 10.0)
POOLS = ("last_token", "visual_mean", "visual_max", "visual_query_attn")
EXPECTED_ROWS = {"train": 7072, "dev": 2408, "test": 2646}


def load_split(repr_root: Path, split: str, pools: tuple[str, ...]) -> dict[str, Any]:
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
    return {
        "labels": labels,
        "features": {pool: np.asarray(payload[pool]).astype(np.float64) for pool in pools},
        "manifest": manifest,
    }


def evaluate(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    prob = model.predict_proba(x)
    pred = model.classes_[prob.argmax(axis=1)]
    supported = (y == 0).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(supported, prob[:, 0])),
        "argmax_accuracy": float((pred == y).mean()),
        "mcc": float(matthews_corrcoef(y, pred)),
        "loss": float(log_loss(y, prob, labels=[0, 1, 2])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-seed", default=42, type=int)
    parser.add_argument("--pools", default=",".join(POOLS))
    parser.add_argument("--c-values", default=",".join(f"{c:g}" for c in C_GRID))
    parser.add_argument("--tag", default="all")
    args = parser.parse_args()

    pools = tuple(p.strip() for p in args.pools.split(",") if p.strip())
    c_values = tuple(float(c) for c in args.c_values.split(",") if c.strip())
    unknown = [p for p in pools if p not in POOLS]
    if unknown:
        raise ValueError(f"Unknown pooling variants: {unknown}")

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / f"metrics_{args.tag}.jsonl"
    metrics_path.unlink(missing_ok=True)

    splits = {
        split: load_split(args.repr_root, split, pools) for split in ("train", "dev", "test")
    }
    print(
        f"[frozen] base={splits['test']['manifest']['base']} adapter=None layer=27",
        flush=True,
    )
    for split, payload in splits.items():
        print(
            f"[frozen] {split}: n={payload['labels'].shape[0]} "
            f"labels={np.bincount(payload['labels'], minlength=3).tolist()}",
            flush=True,
        )

    y_train = splits["train"]["labels"]
    y_dev = splits["dev"]["labels"]
    y_test = splits["test"]["labels"]

    for pool in tqdm(pools, desc="pooling variants", unit="pool"):
        x_train = splits["train"]["features"][pool]
        mu = x_train.mean(axis=0)
        sd = x_train.std(axis=0) + 1e-6
        x_train_n = (x_train - mu) / sd
        x_dev_n = (splits["dev"]["features"][pool] - mu) / sd
        x_test_n = (splits["test"]["features"][pool] - mu) / sd

        for c_value in tqdm(c_values, desc=f"{pool} C grid", unit="candidate", leave=False):
            model = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=2000,
                random_state=args.run_seed,
            )
            model.fit(x_train_n, y_train)
            if model.classes_.tolist() != [0, 1, 2]:
                raise ValueError(f"Unexpected classes: {model.classes_.tolist()}")
            dev = evaluate(model, x_dev_n, y_dev)
            test = evaluate(model, x_test_n, y_test)
            record = {
                "pool": pool,
                "C": c_value,
                "dev_loss": dev["loss"],
                "dev_auroc": dev["auroc"],
                "dev_mcc": dev["mcc"],
                "test_auroc": test["auroc"],
                "test_argmax_accuracy": test["argmax_accuracy"],
                "test_mcc": test["mcc"],
                "iterations": int(model.n_iter_.max()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"[frozen] {pool} C={c_value:g} dev_loss={dev['loss']:.4f} "
                f"test_auroc={test['auroc']:.4f} iters={record['iterations']}",
                flush=True,
            )

    print(f"[frozen] shard {args.tag} done", flush=True)


if __name__ == "__main__":
    main()
