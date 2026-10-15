#!/usr/bin/env python3
"""Train a light probe on each fixed representation, so the four read-outs are compared like for like.

The four representations come from `extract_representations.py`:
  - last_token
  - visual_mean
  - visual_max
  - visual_query_attn

Each is trained independently under identical settings:
  - per-dimension float32 mu and sd computed on the train split, with sd clamped at 1e-6, so
    no representation reuses the statistics of another
  - the same probe architecture, 4096 -> 512 -> 256 -> 3 with GELU and dropout 0.1
  - the same optimiser, Adam with lr 1e-3 and weight decay 1e-3
  - early stopping on dev loss
  - the test split is evaluated once, at the end

Usage:
  python experiments/evidence_judge/train_probes.py \
      --train outputs/train.npz --dev outputs/dev.npz --test outputs/test.npz
  python experiments/evidence_judge/train_probes.py --train ... --dev ... --test ... \
      --seeds 13 42 --methods last_token visual_mean
  python experiments/evidence_judge/train_probes.py --train ... --dev ... --test ... --save-probes
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENTS = _THIS_DIR.parent
_AAAI_CODE = _EXPERIMENTS.parent
_TRAINING = _AAAI_CODE / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

NUM_CLASSES = 3
CLASS_NAMES = ["answer", "wrong_object", "no_target"]
DEFAULT_CLASS_WEIGHTS = [1.0, 1.5, 1.0]

ALL_METHODS = ["last_token", "visual_mean", "visual_max", "visual_query_attn"]
DEFAULT_SEEDS = [13, 42, 123, 2024, 7]


def log(msg: str) -> None:
    print(f"[train_probes] {msg}", flush=True)


def compute_hash(data_paths: list[str]) -> str:
    h = hashlib.md5()
    for p in data_paths:
        h.update(p.encode())
        if os.path.exists(p):
            h.update(str(os.path.getmtime(p)).encode())
            h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:12]


class ProbeMLP(nn.Module):
    """Probe network: 4096 -> 512 -> 256 -> 3, with GELU and dropout 0.1."""

    def __init__(self, hidden_size: int = 4096, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


_PROBE_PARAM_COUNT = sum(p.numel() for p in ProbeMLP().parameters())


def multiclass_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, recall_score

    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    recall_per = recall_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)))
    return {f"f1_{CLASS_NAMES[i]}": float(f1_per[i]) for i in range(NUM_CLASSES)} | {
        f"recall_{CLASS_NAMES[i]}": float(recall_per[i]) for i in range(NUM_CLASSES)
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    from sklearn.metrics import confusion_matrix as cm

    return cm(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    seed: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 64,
    patience: int = 5,
    eval_every: int = 50,
    max_epochs: int = 1000,
    class_weights: Optional[list[float]] = None,
) -> dict:
    """Train one probe and return its metrics at the best dev loss.

    The final epoch is also evaluated on dev, so `best_state` is never None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu = X_train.mean(axis=0).astype(np.float32)
    sd = np.clip(X_train.std(axis=0).astype(np.float32), 1e-6, None)

    X_train_std = ((X_train.astype(np.float32) - mu) / sd).astype(np.float32)
    X_dev_std = ((X_dev.astype(np.float32) - mu) / sd).astype(np.float32)

    X_train_t = torch.from_numpy(X_train_std).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_dev_t = torch.from_numpy(X_dev_std).to(device)
    y_dev_t = torch.from_numpy(y_dev.astype(np.int64)).to(device)

    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        cw = None

    torch.manual_seed(seed)
    model = ProbeMLP(hidden_size=X_train.shape[1], num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    n_train = X_train_t.shape[0]

    best_dev_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = -1

    model.train()
    for epoch in range(max_epochs):
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            bx = X_train_t[idx]
            by = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        should_eval = (epoch + 1) % eval_every == 0 or epoch == max_epochs - 1
        if should_eval:
            model.eval()
            with torch.inference_mode():
                dev_logits = model(X_dev_t)
                dev_loss = loss_fn(dev_logits, y_dev_t).item()
                dev_preds = dev_logits.argmax(-1).cpu().numpy()
            dev_mcc = multiclass_mcc(y_dev, dev_preds)
            log(
                f"  epoch {epoch + 1}/{max_epochs}, "
                f"train_loss={epoch_loss / n_batches:.4f}, "
                f"dev_loss={dev_loss:.4f}, dev_MCC={dev_mcc:.4f}"
            )
            model.train()

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                best_epoch = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log(
                        f"  早停于 epoch {epoch + 1}, 最佳 dev loss: {best_dev_loss:.4f} (epoch {best_epoch})"
                    )
                    break

    if best_state is None:
        log("  WARNING: best_state 为 None，使用当前模型权重")
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = max_epochs

    model.load_state_dict(best_state)
    model.eval()

    with torch.inference_mode():
        train_preds = model(X_train_t).argmax(-1).cpu().numpy()
        dev_preds = model(X_dev_t).argmax(-1).cpu().numpy()

    train_acc = (train_preds == y_train).mean()
    dev_acc = (dev_preds == y_dev).mean()

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "train_accuracy": float(train_acc),
        "dev_accuracy": float(dev_acc),
        "dev_mcc": float(multiclass_mcc(y_dev, dev_preds)),
        "dev_macro_f1": float(macro_f1(y_dev, dev_preds)),
        "dev_per_class": per_class_metrics(y_dev, dev_preds),
        "dev_confusion": confusion_matrix(y_dev, dev_preds),
        "mu_sd_source": "train",
        "model_state": best_state,
        "mu": mu,
        "sd": sd,
    }


def evaluate_test(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> dict:
    """Evaluate once on the test split."""
    device = next(model.parameters()).device
    X_test_std = (
        (X_test.astype(np.float32) - mu.astype(np.float32)) / sd.astype(np.float32)
    ).astype(np.float32)
    X_test_t = torch.from_numpy(X_test_std).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(X_test_t)
        preds = logits.argmax(-1).cpu().numpy()

    acc = float((preds == y_test).mean())
    mcc_val = multiclass_mcc(y_test, preds)
    f1 = macro_f1(y_test, preds)
    per_class = per_class_metrics(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return {
        "test_accuracy": acc,
        "test_mcc": mcc_val,
        "test_macro_f1": f1,
        "test_per_class": per_class,
        "test_confusion": cm,
    }, preds


def paired_bootstrap_mcc_diff(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap 95% confidence interval for the MCC difference between two prediction sets.

    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        mcc_a = multiclass_mcc(y_true[idx], preds_a[idx])
        mcc_b = multiclass_mcc(y_true[idx], preds_b[idx])
        diffs.append(mcc_a - mcc_b)
    diffs = np.array(diffs)
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    mean_diff = float(np.mean(diffs))
    return {
        "delta_mcc_mean": mean_diff,
        "delta_mcc_95ci_low": ci_low,
        "delta_mcc_95ci_high": ci_high,
        "n_bootstrap": n_bootstrap,
    }


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    result = {}
    for key in data.keys():
        result[key] = data[key]
    return result


def validate_manifest_consistency(manifests: list[dict], names: list[str]) -> str | None:
    """Check that the three splits were extracted with the same base and adapter.

    Returns None when consistent, otherwise a message describing the mismatch.
    """
    base = manifests[0].get("base")
    adapter = manifests[0].get("adapter")
    is_diag = manifests[0].get("is_adapter_diagnostic", False)
    for i, (m, name) in enumerate(zip(manifests, names)):
        if m.get("base") != base:
            return f"{name}.base ({m.get('base')}) != {names[0]}.base ({base})"
        if m.get("adapter") != adapter:
            return f"{name}.adapter ({m.get('adapter')}) != {names[0]}.adapter ({adapter})"
        if m.get("is_adapter_diagnostic", False) != is_diag:
            return f"{name}.is_adapter_diagnostic 不一致"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train probes on fixed representations to fairly compare the four representation methods",
    )
    ap.add_argument("--train", required=True, help="Path to the train split npz")
    ap.add_argument("--dev", required=True, help="Path to the dev split npz")
    ap.add_argument("--test", required=True, help="Path to the test split npz")
    ap.add_argument(
        "--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS, help="Methods to compare"
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="List of random seeds")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    ap.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    ap.add_argument("--eval-every", type=int, default=50, help="Evaluate every N epochs (default: 50)")
    ap.add_argument("--max-epochs", type=int, default=1000, help="Maximum number of epochs (default: 1000)")
    ap.add_argument(
        "--class-weights",
        nargs=3,
        type=float,
        default=DEFAULT_CLASS_WEIGHTS,
        help="Class weights (default: 1 1.5 1)",
    )
    ap.add_argument(
        "--n-bootstrap", type=int, default=1000, help="Number of paired bootstrap resamples (default: 1000)"
    )
    ap.add_argument("--save-probes", default=None, help="Directory in which to save probe weights (not saved by default)")
    ap.add_argument("--output", default=None, help="Output path for the result JSON (default: stdout)")
    args = ap.parse_args()

    methods = list(args.methods)
    if "last_token" not in methods:
        log("last_token 必须存在于 methods 中（作为 bootstrap baseline）— 自动添加到首位")
        methods.insert(0, "last_token")

    log("加载数据...")
    train_data = load_npz(args.train)
    dev_data = load_npz(args.dev)
    test_data = load_npz(args.test)

    train_manifest = json.loads(str(train_data.get("manifest_json", "{}")))
    dev_manifest = json.loads(str(dev_data.get("manifest_json", "{}")))
    test_manifest = json.loads(str(test_data.get("manifest_json", "{}")))

    err = validate_manifest_consistency(
        [train_manifest, dev_manifest, test_manifest],
        ["train", "dev", "test"],
    )
    if err is not None:
        log(f"FATAL: manifest 不一致: {err}")
        return 1

    input_hash = compute_hash([args.train, args.dev, args.test])

    for name, d, npz_path in [
        ("train", train_data, args.train),
        ("dev", dev_data, args.dev),
        ("test", test_data, args.test),
    ]:
        n_labels = len(np.asarray(d["head_label"]))
        for m in methods:
            feat = np.asarray(d[m])
            if feat.ndim != 2:
                log(f"FATAL: {name}.{m} 维度异常: {feat.shape} (from {npz_path})")
                return 1
            if feat.shape[0] != n_labels:
                log(
                    f"FATAL: {name}.{m} 行数 ({feat.shape[0]}) != head_label 行数 ({n_labels}) (from {npz_path})"
                )
                return 1
            if feat.shape[1] != 4096:
                log(f"FATAL: {name}.{m} 特征维度 ({feat.shape[1]}) != 4096 (from {npz_path})")
                return 1

    log(
        f"train: {train_data['last_token'].shape[0]} 条, "
        f"dev: {dev_data['last_token'].shape[0]} 条, "
        f"test: {test_data['last_token'].shape[0]} 条"
    )

    for name, d in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        hl = d["head_label"]
        uniq, cnts = np.unique(hl, return_counts=True)
        log(f"  {name} head_label: {dict(zip(uniq.tolist(), cnts.tolist()))}")

    baseline_preds_by_seed: dict[int, np.ndarray] = {}

    results: dict = {}
    param_counts: dict = {}
    saved_probes: dict[str, dict[int, dict]] = {}

    for method in methods:
        log(f"\n{'=' * 60}")
        log(f"方法: {method}")
        log(f"{'=' * 60}")

        X_train = np.asarray(train_data[method]).astype(np.float32)
        X_dev = np.asarray(dev_data[method]).astype(np.float32)
        X_test = np.asarray(test_data[method]).astype(np.float32)
        y_train = np.asarray(train_data["head_label"]).astype(np.int64)
        y_dev = np.asarray(dev_data["head_label"]).astype(np.int64)
        y_test = np.asarray(test_data["head_label"]).astype(np.int64)

        method_results = {"seeds": []}

        for seed_idx, seed in enumerate(args.seeds):
            log(f"\n  seed={seed} ({seed_idx + 1}/{len(args.seeds)})")

            t0 = time.time()
            train_out = train_probe(
                X_train,
                y_train,
                X_dev,
                y_dev,
                seed=seed,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                patience=args.patience,
                eval_every=args.eval_every,
                max_epochs=args.max_epochs,
                class_weights=args.class_weights,
            )

            if method not in param_counts:
                param_counts[method] = _PROBE_PARAM_COUNT
                log(f"  {method} 参数量: {_PROBE_PARAM_COUNT}")

            mu = train_out["mu"]
            sd = train_out["sd"]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_model = ProbeMLP(hidden_size=X_test.shape[1], num_classes=NUM_CLASSES).to(device)
            test_model.load_state_dict(train_out["model_state"])
            test_out, test_preds = evaluate_test(test_model, X_test, y_test, mu, sd)

            if method == "last_token":
                baseline_preds_by_seed[seed] = test_preds.copy()

            bootstrap_result = None
            if method != "last_token" and seed in baseline_preds_by_seed:
                bs = paired_bootstrap_mcc_diff(
                    y_test,
                    test_preds,
                    baseline_preds_by_seed[seed],
                    n_bootstrap=args.n_bootstrap,
                    seed=seed,
                )
                bootstrap_result = bs

            seed_entry = {
                "seed": seed,
                "best_epoch": train_out["best_epoch"],
                "best_dev_loss": train_out["best_dev_loss"],
                "train_accuracy": train_out["train_accuracy"],
                "dev_accuracy": train_out["dev_accuracy"],
                "dev_mcc": train_out["dev_mcc"],
                "dev_macro_f1": train_out["dev_macro_f1"],
                "dev_per_class": train_out["dev_per_class"],
                "dev_confusion": train_out["dev_confusion"],
                **test_out,
                "test_preds": test_preds.tolist(),
            }
            if bootstrap_result is not None:
                seed_entry["bootstrap_vs_last_token"] = bootstrap_result

            method_results["seeds"].append(seed_entry)

            saved_probes.setdefault(method, {})[seed] = {
                "model_state": {k: v.cpu() for k, v in train_out["model_state"].items()},
                "mu": train_out["mu"].copy(),
                "sd": train_out["sd"].copy(),
            }

            dt = time.time() - t0
            log(f"  seed {seed} 完成, {dt:.1f}s")

        metrics_keys = [
            "train_accuracy",
            "dev_accuracy",
            "dev_mcc",
            "dev_macro_f1",
            "test_accuracy",
            "test_mcc",
            "test_macro_f1",
        ]
        per_class_keys = [f"f1_{n}" for n in CLASS_NAMES] + [f"recall_{n}" for n in CLASS_NAMES]

        agg = {}
        for key in metrics_keys:
            vals = [s[key] for s in method_results["seeds"]]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

        for key in per_class_keys:
            vals = [s["test_per_class"][key] for s in method_results["seeds"]]
            agg[f"test_{key}_mean"] = float(np.mean(vals))
            agg[f"test_{key}_std"] = float(np.std(vals))

        method_results["aggregate"] = agg

        if method != "last_token":
            bs_keys = ["delta_mcc_mean", "delta_mcc_95ci_low", "delta_mcc_95ci_high"]
            bootstrap_agg = {}
            for k in bs_keys:
                vals = [
                    s["bootstrap_vs_last_token"][k]
                    for s in method_results["seeds"]
                    if "bootstrap_vs_last_token" in s
                ]
                if vals:
                    bootstrap_agg[f"{k}_across_seeds_mean"] = float(np.mean(vals))
                    bootstrap_agg[f"{k}_across_seeds_std"] = float(np.std(vals))
            if bootstrap_agg:
                bootstrap_agg["n_bootstrap_per_seed"] = args.n_bootstrap
                bootstrap_agg["n_seeds"] = len(
                    [s for s in method_results["seeds"] if "bootstrap_vs_last_token" in s]
                )
                method_results["bootstrap_vs_last_token_summary"] = bootstrap_agg

        results[method] = method_results

    is_adapter = (
        train_manifest.get("is_adapter_diagnostic", False)
        or dev_manifest.get("is_adapter_diagnostic", False)
        or test_manifest.get("is_adapter_diagnostic", False)
    )

    output = {
        "input_hash": input_hash,
        "input_manifest": {
            "train": train_manifest,
            "dev": dev_manifest,
            "test": test_manifest,
        },
        "parameters": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "eval_every": args.eval_every,
            "max_epochs": args.max_epochs,
            "class_weights": args.class_weights,
            "seeds": args.seeds,
            "n_bootstrap": args.n_bootstrap,
            "probe_architecture": "MLP(4096→512→256→3) GELU+Dropout0.1",
        },
        "param_counts": param_counts,
        "claim_scope": "adapter_diagnostic" if is_adapter else "frozen_base_representation_probe",
        "results": results,
        "note": "test_preds 保留于各 seed.test_preds 用于可核查",
    }

    output_json = json.dumps(output, indent=2, ensure_ascii=False, default=str)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json)
        log(f"结果已保存: {out_path}")
    else:
        print(output_json)

    if args.save_probes:
        save_dir = Path(args.save_probes)
        save_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            if method not in saved_probes:
                continue
            for seed in args.seeds:
                if seed not in saved_probes[method]:
                    continue
                probe_data = saved_probes[method][seed]
                save_path = save_dir / f"probe_{method}_seed{seed}.pt"
                torch.save(
                    {
                        "model_state": probe_data["model_state"],
                        "mu": probe_data["mu"],
                        "sd": probe_data["sd"],
                        "method": method,
                        "seed": seed,
                    },
                    str(save_path),
                )
                log(f"保存 probe: {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
