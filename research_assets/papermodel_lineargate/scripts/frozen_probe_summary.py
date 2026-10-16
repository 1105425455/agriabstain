"""汇总冻结基座线性探针的分片结果,并与适配后的门做对比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

POOLS = ("last_token", "visual_mean", "visual_max", "visual_query_attn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--adapted", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for shard in sorted(args.run_dir.glob("metrics_*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no shard metrics under {args.run_dir}")

    per_pool: dict[str, Any] = {}
    for pool in POOLS:
        candidates = [row for row in rows if row["pool"] == pool]
        if not candidates:
            continue
        per_pool[pool] = {
            "n_candidates": len(candidates),
            "selected": min(candidates, key=lambda row: row["dev_loss"]),
            "all": sorted(candidates, key=lambda row: row["C"]),
        }

    adapted = json.loads(args.adapted.read_text(encoding="utf-8"))
    comparable = per_pool.get("last_token", {}).get("selected")
    best_any = max(
        (payload["selected"] for payload in per_pool.values()),
        key=lambda row: row["test_auroc"],
    )
    summary = {
        "per_pool": per_pool,
        "comparable_pool": "last_token",
        "comparable_test_auroc": comparable["test_auroc"] if comparable else None,
        "best_pool": best_any["pool"],
        "best_test_auroc": best_any["test_auroc"],
        "adapted_reference": {
            arm: {
                "auroc_linear": adapted[arm]["auroc_linear"],
                "auroc_mlp": adapted[arm]["auroc_mlp"],
            }
            for arm in ("grouped", "independent")
        },
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"{'configuration':44s}{'C':>7s}{'dev_loss':>11s}{'test_AUROC':>12s}{'test_MCC':>10s}")
    print("-" * 84)
    for pool, payload in per_pool.items():
        sel = payload["selected"]
        mark = "  <- head-input analogue" if pool == "last_token" else ""
        print(
            f"{'frozen base / ' + pool:44s}{sel['C']:>7g}{sel['dev_loss']:>11.4f}"
            f"{sel['test_auroc']:>12.4f}{sel['test_mcc']:>10.4f}{mark}"
        )
    print("-" * 84)
    for arm in ("grouped", "independent"):
        for gate in ("linear", "mlp"):
            print(
                f"{'stage-1 adapted / ' + arm + ' / ' + gate:44s}{'-':>7s}{'-':>11s}"
                f"{adapted[arm][f'auroc_{gate}']:>12.4f}{'-':>10s}"
            )
    if comparable:
        delta = adapted["grouped"]["auroc_linear"] - comparable["test_auroc"]
        print(
            "\nAUROC gain from stage-1 adaptation "
            f"(grouped linear - frozen last_token): {delta:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
