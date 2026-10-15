#!/usr/bin/env python3
"""Repeat the post-adaptation layer scan across group-disjoint split seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from experiments.layer_selection.scan_layers import (  # noqa: E402
    AllLayersDataset,
    group_disjoint_split,
    scan_one_layer,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def run_scan(
    *,
    train_npz: Path,
    output: Path,
    seeds: list[int],
    limit_layers: int | None,
) -> dict[str, Any]:
    started_at = time.time()
    dataset = AllLayersDataset(train_npz)
    n_layers = dataset.n_layers if limit_layers is None else min(dataset.n_layers, limit_layers)
    payload: dict[str, Any] = {
        "protocol": "post_adaptation_group_disjoint_split_seed_stability_v1",
        "input": str(train_npz.resolve()),
        "input_sha256": file_sha256(train_npz),
        "n_samples": dataset.n_samples,
        "n_layers": n_layers,
        "seeds_requested": seeds,
        "completed": [],
    }
    atomic_write_json(output, payload)

    for seed in seeds:
        seed_started_at = time.time()
        train_indices, dev_indices = group_disjoint_split(
            dataset.source_image_id,
            dataset.head_label,
            seed=seed,
        )
        results = []
        print(
            f"[seed {seed}] train={len(train_indices)} dev={len(dev_indices)} layers={n_layers}",
            flush=True,
        )
        for layer in range(n_layers):
            results.append(
                scan_one_layer(
                    layer,
                    dataset.get_layer_data(layer),
                    train_indices,
                    dev_indices,
                    dataset.head_label,
                )
            )

        ranked = sorted(results, key=lambda result: (-result["min_score"], result["layer"]))
        best = ranked[0]
        seed_result = {
            "seed": seed,
            "n_train": len(train_indices),
            "n_dev": len(dev_indices),
            "best_layer": best["layer"],
            "best_transformer_layer": best["transformer_layer"],
            "best_min_score": best["min_score"],
            "top5": ranked[:5],
            "results": sorted(results, key=lambda result: result["layer"]),
            "elapsed_seconds": round(time.time() - seed_started_at, 3),
        }
        payload["completed"].append(seed_result)
        payload["elapsed_seconds"] = round(time.time() - started_at, 3)
        atomic_write_json(output, payload)
        print(
            f"[seed {seed}] best transformer layer={best['transformer_layer']} "
            f"min_score={best['min_score']:.4f}",
            flush=True,
        )

    payload["best_transformer_layers"] = [
        result["best_transformer_layer"] for result in payload["completed"]
    ]
    payload["elapsed_seconds"] = round(time.time() - started_at, 3)
    atomic_write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", default="7,21,42,87,123", type=parse_seeds)
    parser.add_argument("--limit-layers", type=int)
    args = parser.parse_args()

    run_scan(
        train_npz=args.train_npz,
        output=args.output,
        seeds=args.seeds,
        limit_layers=args.limit_layers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
