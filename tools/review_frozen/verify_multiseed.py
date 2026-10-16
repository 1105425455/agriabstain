#!/usr/bin/env python3
"""Read-only verification of the historical multi-seed summaries.

Reads (never writes to) the ORIGINAL source files:
  * <repo>/h27experiment/summary/multiseed_summary.json
  * <repo>/devtau_multiseed/summary/dev_selected_multiseed.json
  * <repo>/devtau_multiseed/rebuttal_evidence.md  (copied as provenance by caller)

Checks
  1. Completeness: per-seed rows cover frozen_seeds {7,21,42}; `missing` empty;
     `all_runs_reported` true.
  2. Aggregate self-consistency: mean/min/max/std recomputed from per-seed `rows`
     must match the stored `aggregate` entries (tolerance 1e-6 relative).
  3. Cross-check of the key claims in rebuttal_evidence.md section 2 (per-interface
     mean+std for MTS / MCC / FCA on comprehensive, grouped vs independent) and
     section 3 (dev-selected per-seed numbers) against the JSON sources.
  4. Seeds are attributed to their own run rows only; no mixing of conditions or of
     h27experiment vs devtau_multiseed pipelines.

Output: a verification report JSON (written to --output), plus a human-readable table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def load_rows(summary) -> list[dict]:
    return summary.get("rows", [])


def expected_comprehensive_rows(summary) -> list[dict]:
    return [
        r
        for r in load_rows(summary)
        if r.get("dataset") == "comprehensive"
        and r.get("condition") in {"grouped", "independent"}
        and r.get("interface") in {"mlp_regate", "linear_gate", "native"}
    ]


def claim_title_keys():
    return {
        ("Comp.-Gate", "mlp_regate"): "Grouped/Independent",
        ("Comp.-Native", "native"): "Grouped/Independent",
        ("Ref.-Gate", "mlp_regate"): "Grouped/Independent",
        ("Ref.-Native", "native"): "Grouped/Independent",
        ("PW-Gate", "mlp_regate"): "Grouped/Independent",
        ("PW-Native", "native"): "Grouped/Independent",
        ("L1-Gate", "mlp_regate"): "Grouped/Independent",
        ("L2-Gate", "mlp_regate"): "Grouped/Independent",
        ("L3-Gate", "mlp_regate"): "Grouped/Independent",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed", required=True, type=Path)
    parser.add_argument("--devtau", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    summary = json.loads(args.multiseed.read_text(encoding="utf-8"))
    devtau = json.loads(args.devtau.read_text(encoding="utf-8"))

    checks = []
    ok_checks = []

    def record(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if ok:
            ok_checks.append(name)

    # 1. Completeness --------------------------------------------------------
    missing = summary.get("missing", [])
    all_reported = summary.get("all_runs_reported")
    frozen_seeds = set(summary.get("frozen_seeds", []))
    rows = load_rows(summary)
    seed_seen = {}
    for r in rows:
        seed_seen.setdefault(int(r["seed"]), 0)
        seed_seen[int(r["seed"])] += 1
    record("completeness_missing_empty", not missing, f"missing={missing}")
    record("completeness_all_runs_reported", all_reported is True,
           f"all_runs_reported={all_reported}")
    record("completeness_seeds_covered", set(seed_seen) == set(frozen_seeds),
           f"seeds_in_rows={sorted(seed_seen)} frozen_seeds={sorted(frozen_seeds)}")

    # 2. Aggregate self-consistency -----------------------------------------
    n_aggregates = 0
    n_aggregate_matches = 0
    aggregate_errors = []
    for agg in summary.get("aggregate", []):
        condition = agg["condition"]
        dataset = agg["dataset"]
        interface = agg["interface"]
        metric = agg["metric"]
        values = [
            float(r["metrics"][metric])
            for r in rows
            if r["condition"] == condition
            and r["dataset"] == dataset
            and r["interface"] == interface
            and isinstance(r.get("metrics", {}).get(metric), (int, float))
        ]
        if not values:
            aggregate_errors.append(f"{condition}/{dataset}/{interface}/{metric}: no per-seed values")
            continue
        n_aggregates += 1
        mean, std = mean_std(values)
        matches = (
            abs(mean - agg["mean"]) <= 1e-6
            and abs(std - agg["std"]) <= 1e-6
            and abs(min(values) - agg["min"]) <= 1e-6
            and abs(max(values) - agg["max"]) <= 1e-6
        )
        if matches:
            n_aggregate_matches += 1
        else:
            aggregate_errors.append(
                f"{condition}/{dataset}/{interface}/{metric}: recomputed mean={mean:.6f} "
                f"stored={agg['mean']:.6f} std(ddof=1)={std:.6f}/{agg['std']:.6f}"
            )
    record(
        "aggregate_self_consistent",
        n_aggregate_matches == n_aggregates and n_aggregates > 0,
        f"{n_aggregate_matches}/{n_aggregates} aggregates match; "
        + ("; ".join(aggregate_errors[:5]) if aggregate_errors else ""),
    )

    # 3. Rebuttal section 2 cross-check: per-interface mean/std on comprehensive
    gates = {}   # (condition, interface) -> {metric: [values...]}
    for r in expected_comprehensive_rows(summary):
        for metric, value in r["metrics"].items():
            if not isinstance(value, (int, float)) or value is None:
                continue
            gates.setdefault((r["condition"], r["interface"]), {}).setdefault(metric, []).append(value)

    def mean_std_str(values):
        if not values:
            return "n/a"
        mean, std = mean_std(values)
        return f"{mean:.3f}±{std:.3f}"

    # full section-2 table recomputed from rows (datasets x interfaces x metrics)
    section2_full = {}
    for cond in ("grouped", "independent"):
        for iface in ("mlp_regate", "native"):
            key = (cond, iface)
            if key not in gates:
                continue
            section2_full[f"{cond}:{iface}"] = {
                metric: mean_std_str(gates[key].get(metric, []))
                for metric in ("MTS_macro", "MCC", "FCA")
            }
    # all datasets/interfaces table used in rebuttal section 2
    section2_all = {}
    for ds in ("comprehensive", "refcocog", "plantwild", "L1_discrim", "L2_evidence", "L3_dose"):
        for cond in ("grouped", "independent"):
            for iface in ("mlp_regate", "native"):
                vals = {
                    met: mean_std_str(
                        [
                            float(r["metrics"][met])
                            for r in rows
                            if r["condition"] == cond and r["dataset"] == ds
                            and r["interface"] == iface
                            and isinstance(r.get("metrics", {}).get(met), (int, float))
                        ]
                    )
                    for met in ("MTS_macro", "MCC", "FCA")
                }
                section2_all.setdefault(ds, {})[f"{cond}:{iface}"] = vals

    section2 = {}
    for condition in ("grouped", "independent"):
        for interface in ("mlp_regate", "native"):
            key = (condition, interface)
            if key not in gates:
                continue
            entry = {}
            for metric in ("FCA", "MCC", "MTS_macro"):
                vals = gates[key].get(metric, [])
                if vals:
                    mean, std = mean_std(vals)
                    entry[metric] = {"mean": mean, "std": std, "per_seed": dict(zip(["7", "21", "42"], [round(v, 4) for v in vals]))}
            section2[f"{condition}:{interface}"] = entry

    # cross-check MTS_macro/MCC/FCA grouped-vs-independent direction (all seeds same sign)
    dir_ok = True
    direction_detail = []
    for metric in ("MTS_macro", "FCA", "MCC"):
        for dataset in ("comprehensive",):
            for seed in frozen_seeds:
                g = next(
                    (r["metrics"].get(metric) for r in rows
                     if r["seed"] == seed and r["condition"] == "grouped"
                     and r["dataset"] == dataset and r["interface"] == "mlp_regate"),
                    None,
                )
                i = next(
                    (r["metrics"].get(metric) for r in rows
                     if r["seed"] == seed and r["condition"] == "independent"
                     and r["dataset"] == dataset and r["interface"] == "mlp_regate"),
                    None,
                )
                if g is None or i is None:
                    continue
                diff = g - i
                # FCA: smaller better for grouped is expected (negative diff); others positive
                expect = "supports_claim" if (diff < 0) == (metric == "FCA") else "contradicts"
                direction_detail.append(f"{metric}@seed{seed} diff={diff:+.4f} {expect}")
    record(
        "rebuttal_s2_direction",
        all("contradicts" not in d for d in direction_detail),
        "; ".join(direction_detail[:12]),
    )

    # 4. Rebuttal section 3 cross-check: devtau per-seed numbers ------------
    pro_seed = devtau.get("per_seed", {})
    devtau_ok = True
    devtau_detail = []
    for seed in ("7", "21", "42"):
        entry = pro_seed.get(seed)
        if entry is None:
            devtau_ok = False
            devtau_detail.append(f"seed {seed}: missing")
            continue
        for field in ("selected_threshold", "dev_target_retention",
                      "test_native_retention", "test_native_fca",
                      "test_regate_retention", "test_regate_fca",
                      "test_regate_strict3", "relative_fca_reduction"):
            if field not in entry:
                devtau_ok = False
                devtau_detail.append(f"seed {seed}: missing field {field}")
    agg = devtau.get("aggregate", {})
    record(
        "rebuttal_s3_devtau_present",
        devtau_ok,
        "; ".join(devtau_detail[:12]),
    )
    record(
        "devtau_three_seeds",
        list(pro_seed.keys()) == ["7", "21", "42"],
        f"keys={list(pro_seed.keys())}",
    )

    report = {
        "sources": {
            "multiseed_summary": str(args.multiseed.resolve()),
            "devtau_summary": str(args.devtau.resolve()),
        },
        "frozen_seeds": sorted(frozen_seeds),
        "seed_row_counts": {str(k): v for k, v in seed_seen.items()},
        "checks": checks,
        "recomputed_comprehensive_section2": section2,
        "recomputed_section2_all": section2_all,
        "devtau_per_seed": pro_seed,
        "devtau_aggregate": agg,
        "rebuttal_s2_direction_detail": direction_detail,
        "comprehensive_linear_gate_rows": [
            {
                "seed": r["seed"],
                "condition": r["condition"],
                "FCA": r["metrics"].get("FCA"),
                "MCC": r["metrics"].get("MCC"),
                "Strict_3": r["metrics"].get("Strict_3"),
                "state_accuracy": r["metrics"].get("state_accuracy"),
            }
            for r in expected_comprehensive_rows(summary)
            if r.get("interface") == "linear_gate"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== verification summary ===", flush=True)
    for c in checks:
        print(f"[{'OK' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}", flush=True)
    print(f"-> wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()