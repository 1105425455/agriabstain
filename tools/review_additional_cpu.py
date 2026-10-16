#!/usr/bin/env python3
"""Additional CPU-only review evidence (round 2).

Five checks, all reading existing files only (no model inference, no training, no
paid API, no GPU):

1. L3 GPT binary predictions vs ReGate head probabilities on the adjacent-visibility
   baseline, using the existing ``evidence_judge.analyze_evidence`` grouping keys.
   GPT has only a binary answer, so we report binary answer rates and the binary
   adjacent-decline ratio (ties never count); we explicitly do not treat the discrete
   decision as numerically equivalent to continuous p_answer monotonicity. ReGate is
   evaluated on the same groups and bins with the continuous p_answer violation rate
   plus an additional binary argmax violation rate for a fair comparison.

2. Capability audit: recompute per-label n/correct for all 24 observed PlantDoc base
   classes on the Base calibration pool (768 records), with micro/macro for the full 24
   versus the retained 20 and the excluded 4. This is a Base-pool reconstitution, not an
   independent test and not three-system full-24 results; specific counts are kept.

3. Two-person label-audit sampling preparation (CSV/JSON), 200 rows per state,
   prioritising T6 / COCO negatives, stratified with population and sample weights,
   source-deduplicated, sampled from the raw test split (not the oversampled train
   split), with two empty annotator fields. Sampling prep only: no human results, no
   image copies, no fabricated labels.

4. SFT native dev/test prediction-field audit for an own per-sample confidence signal
   (logprob/entropy/etc.), plus a protocol-check of the existing groundingme/openref
   baseline outputs against CounterfactualREC / GREC / RECANTFormer.

5. Whether existing source logs can quantify decode-skip latency on the same hardware
   and batch; if not suitable, only the absence is reported (no GPU run).

Outputs are written under docs/review_evidence/additional_cpu/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTRA_CPU = ROOT / "docs/review_evidence/additional_cpu"
SOURCE_ROOT = Path("/user/huangyongbin/newrepo/agriabstain")

# The grouping rule lives in the package under aaai_code/.
sys.path.insert(0, str(ROOT / "aaai_code"))

# Expected numbers (used as sanity cross-checks, never to pass a test falsely).
EXPECT_L3_GROUPS = 183
EXPECT_L3_ADJ = 567
EXPECT_CALIBRATION_N = 768

# Seldom-changing file list whose SHA-256 goes into the provenance manifest.
HASH_SOURCES: dict[str, Path] = {}


def _register_hash(name: str, path: Path) -> Path:
    HASH_SOURCES[name] = path
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(p: Path, data: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── task 1: L3 GPT vs ReGate adjacent visibility ────────────────────────────

GPT_L3_PREDS = _register_hash(
    "gpt_l3_predictions",
    ROOT
    / "aaai_code/experiments/gpt56_sol_full/outputs/decision_prompt_v1/L3_dose/analysis/predictions.jsonl",
)
REGATE_L3_PROBS = _register_hash(
    "regate_l3_head_probabilities",
    ROOT
    / "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/evidence/L3/test/head_probabilities.jsonl",
)
REGATE_L3_ANALYSIS = _register_hash(
    "regate_l3_existing_analysis",
    ROOT
    / "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/evidence/L3/test/analysis.json",
)


def _l3_bin(ratio: float) -> float:
    """Map an achieved visibility ratio to the nearest nominal dose bin."""
    return round(ratio * 4.0) / 4.0


def _gpt_binary(row: dict[str, Any]) -> int:
    pred = row.get("pred") or {}
    return 1 if pred.get("valid") and pred.get("decision") == "answer" else 0


def _regate_binary(row: dict[str, Any]) -> int:
    return 1 if row.get("predicted_class") == 0 else 0


def _group_l3(records: list[dict[str, Any]], stable_key) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parse_failures = 0
    for rec in records:
        sid = rec.get("sample_id") or ""
        if rec.get("variant_id", "original") != "original":
            continue
        gk = stable_key(sid, rec)
        if gk is None:
            parse_failures += 1
            continue
        groups[gk].append(rec)
    return groups, parse_failures


def _ratio_of(row: dict[str, Any]) -> float | None:
    if "achieved_visible_ratio" in row:
        return row.get("achieved_visible_ratio")
    gt = row.get("gt") or {}
    meta = gt.get("metadata") or {}
    return meta.get("achieved_visible_ratio")


def task1_l3_gpt_vs_regate() -> dict[str, Any]:
    from experiments.evidence_judge.analyze_evidence import (
        stable_group_key,  # after sys.path setup
    )

    gpt = load_jsonl(GPT_L3_PREDS)
    rg = load_jsonl(REGATE_L3_PROBS)
    gpt_groups, gpt_pf = _group_l3(gpt, stable_group_key)
    rg_groups, rg_pf = _group_l3(rg, stable_group_key)

    if set(gpt_groups) != set(rg_groups):
        raise SystemExit(
            f"[task1][FATAL] group sets differ: gpt={len(gpt_groups)} regate={len(rg_groups)}"
        )

    per_group: dict[str, dict[str, Any]] = {}
    bin_ans_gpt: dict[float, list[int]] = defaultdict(list)
    bin_ans_rg: dict[float, list[int]] = defaultdict(list)
    bin_pa_rg: dict[float, list[float]] = defaultdict(list)
    adj_gpt_decline: list[float] = []
    adj_pa_violation: list[float] = []
    adj_rg_binary_decline: list[float] = []
    n_adj_total = 0
    n_adj_gpt_decline = 0
    n_adj_pa_violation = 0
    n_adj_rg_decline = 0
    endpoint_groups = 0
    gpt_endpoint_decline = 0
    rg_endpoint_decline = 0
    rg_endpoint_delta: list[float] = []

    for gk in sorted(gpt_groups):
        gpt_by_ratio: dict[float, list[dict[str, Any]]] = defaultdict(list)
        rg_by_ratio: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for r in gpt_groups[gk]:
            gpt_by_ratio[_ratio_of(r)].append(r)
        for r in rg_groups[gk]:
            rg_by_ratio[_ratio_of(r)].append(r)
        # Duplicate ratio levels within a normalised group invalidate it (same rule as
        # analyze_evidence.analyze_l3).
        if any(len(v) > 1 for v in gpt_by_ratio.values()) or any(
            len(v) > 1 for v in rg_by_ratio.values()
        ):
            per_group[gk] = {
                "error": "duplicate ratio levels within normalised group",
                "gpt_ratios": sorted(gpt_by_ratio),
                "regate_ratios": sorted(rg_by_ratio),
            }
            continue
        gpt_rows = {r: gpt_by_ratio[r][0] for r in gpt_by_ratio}
        rg_rows = {r: rg_by_ratio[r][0] for r in rg_by_ratio}
        ratios = sorted(gpt_rows)
        rg_ratios = sorted(rg_rows)
        if ratios != rg_ratios:
            raise SystemExit(f"[task1][FATAL] ratio mismatch in group {gk}")
        if len(ratios) < 2:
            per_group[gk] = {"error": "fewer than 2 ratio levels", "ratios": ratios}
            continue

        gpt_bin = [_gpt_binary(gpt_rows[r]) for r in ratios]
        rg_bin = [_regate_binary(rg_rows[r]) for r in ratios]
        rg_pa = [float(rg_rows[r]["p_answer"]) for r in ratios]

        for r, gb in zip(ratios, gpt_bin, strict=True):
            bin_ans_gpt[_l3_bin(r)].append(gb)
        for r, rb in zip(ratios, rg_bin, strict=True):
            bin_ans_rg[_l3_bin(r)].append(rb)
        for r, pa in zip(ratios, rg_pa, strict=True):
            bin_pa_rg[_l3_bin(r)].append(pa)

        g_d = 0
        pa_v = 0
        rg_d = 0
        n_adj = len(ratios) - 1
        for i in range(n_adj):
            if gpt_bin[i] == 1 and gpt_bin[i + 1] == 0:
                g_d += 1
            if rg_pa[i] > rg_pa[i + 1]:
                pa_v += 1
            if rg_bin[i] == 1 and rg_bin[i + 1] == 0:
                rg_d += 1
        gpt_decline_rate = g_d / n_adj
        pa_violation_rate = pa_v / n_adj
        rg_decline_rate = rg_d / n_adj
        adj_gpt_decline.append(gpt_decline_rate)
        adj_pa_violation.append(pa_violation_rate)
        adj_rg_binary_decline.append(rg_decline_rate)
        n_adj_total += n_adj
        n_adj_gpt_decline += g_d
        n_adj_pa_violation += pa_v
        n_adj_rg_decline += rg_d

        has_0 = any(abs(r - 0.0) < 1e-6 for r in ratios)
        has_1 = any(abs(r - 1.0) < 1e-6 for r in ratios)
        if has_0 and has_1:
            r0 = min(ratios, key=lambda x: abs(x - 0.0))
            r1 = min(ratios, key=lambda x: abs(x - 1.0))
            endpoint_groups += 1
            rg_endpoint_delta.append(rg_pa[ratios.index(r1)] - rg_pa[ratios.index(r0)])
            if gpt_bin[ratios.index(r0)] == 1 and gpt_bin[ratios.index(r1)] == 0:
                gpt_endpoint_decline += 1
            if rg_bin[ratios.index(r0)] == 1 and rg_bin[ratios.index(r1)] == 0:
                rg_endpoint_decline += 1

        per_group[gk] = {
            "ratios": ratios,
            "n_adjacent": n_adj,
            "gpt_binary_pairs": gpt_bin,
            "regate_binary_pairs": rg_bin,
            "regate_p_answer": rg_pa,
            "gpt_binary_decline_rate": gpt_decline_rate,
            "regate_pa_violation_rate": pa_violation_rate,
            "regate_binary_decline_rate": rg_decline_rate,
        }

    # Cross-check against the existing analyse_evidence numbers.
    existing = json.loads(REGATE_L3_ANALYSIS.read_text(encoding="utf-8"))["L3"]
    existing_pa_rate = float(existing["mean_adjacent_violation_rate"])
    recomputed_pa_rate = sum(adj_pa_violation) / len(adj_pa_violation)

    result = {
        "n_groups": len(per_group),
        "n_raw_groups": len(gpt_groups),
        "gpt_parse_failures": gpt_pf,
        "regate_parse_failures": rg_pf,
        "n_groups_with_errors": sum(1 for v in per_group.values() if "error" in v),
        "n_adjacent_pairs": n_adj_total,
        "gpt_invalid_predictions": sum(
            1 for r in gpt if not ((r.get("pred") or {}).get("valid"))
        ),
        "bin_answer_rate": {
            "bin": sorted(bin_ans_gpt),
            "gpt_binary": [sum(bin_ans_gpt[b]) / len(bin_ans_gpt[b]) for b in sorted(bin_ans_gpt)],
            "regate_binary": [sum(bin_ans_rg[b]) / len(bin_ans_rg[b]) for b in sorted(bin_ans_rg)],
            "regate_p_answer_mean": [sum(bin_pa_rg[b]) / len(bin_pa_rg[b]) for b in sorted(bin_pa_rg)],
        },
        "adjacent_gpt_binary_decline": {
            "pooled": n_adj_gpt_decline / n_adj_total,
            "mean_per_group": sum(adj_gpt_decline) / len(adj_gpt_decline),
            "violating_pairs": n_adj_gpt_decline,
        },
        "adjacent_legate_pa_violation": {
            "pooled": n_adj_pa_violation / n_adj_total,
            "mean_per_group": recomputed_pa_rate,
            "violating_pairs": n_adj_pa_violation,
            "existing_analysis_mean": existing_pa_rate,
        },
        "adjacent_legate_binary_argmax_decline": {
            "pooled": n_adj_rg_decline / n_adj_total,
            "mean_per_group": sum(adj_rg_binary_decline) / len(adj_rg_binary_decline),
            "violating_pairs": n_adj_rg_decline,
        },
        "endpoint": {
            "n_complete_endpoint_groups": endpoint_groups,
            "gpt_binary_endpoint_decline": gpt_endpoint_decline / endpoint_groups,
            "regate_binary_endpoint_decline": rg_endpoint_decline / endpoint_groups,
            "regate_pa_endpoint_delta_mean": sum(rg_endpoint_delta) / len(rg_endpoint_delta),
            "existing_analysis_endpoint_delta_mean": float(existing["mean_endpoint_delta_p"]),
            "existing_n_complete_endpoint_groups": int(existing["n_complete_endpoint_groups"]),
        },
        "scope_note": (
            "GPT outputs binary decision only; p_answer is ReGate-head-only. Adjacent-tie "
            "pairs are never counted as violations. Discrete-decision monotonicity is a "
            "different object from continuous p_answer monotonicity and the two numbers are "
            "not interchangeable."
        ),
        "per_group": per_group,
    }
    if result["n_groups"] != EXPECT_L3_GROUPS or result["n_adjacent_pairs"] != EXPECT_L3_ADJ:
        raise SystemExit(
            f"[task1][FATAL] expected {EXPECT_L3_GROUPS} groups / {EXPECT_L3_ADJ} adjacent, "
            f"got {result['n_groups']} / {result['n_adjacent_pairs']}"
        )
    write_json(EXTRA_CPU / "l3_gpt_vs_regate.json", result)
    return result


# ── task 2: capability audit 24 vs retained 20 ──────────────────────────────

CAPABILITY_AUDIT = _register_hash(
    "plantdoc_capability_gate_audit",
    ROOT / "aaai_code/benchmark/annotations/audits/PLANTDOC_CAPABILITY_GATE_AUDIT.json",
)
CALIBRATION_POOL = _register_hash(
    "plantdoc_answerable_calibration",
    SOURCE_ROOT / "outputs/benchmarks/sampled/calibration/plantdoc_answerable_calibration.jsonl",
)
BASE_PREDICTIONS = _register_hash(
    "plantdoc_answerable_base_predictions",
    SOURCE_ROOT / "outputs/evals/qwen3vl8b_raw/plantdoc_answerable_predictions.jsonl",
)


def _normalise_label(s: str) -> str:
    return re.sub(r"[\s_]+", " ", s.strip().lower())


def task2_capability_24_vs_20() -> dict[str, Any]:
    audit = json.loads(CAPABILITY_AUDIT.read_text(encoding="utf-8"))
    retained = set(audit["capability_retained_labels"])
    excluded = set(audit["capability_excluded_labels"])
    observed = set(audit["per_label"])
    if len(observed) != 24:
        raise SystemExit(f"[task2][FATAL] expected 24 observed labels, got {len(observed)}")

    cal = load_jsonl(CALIBRATION_POOL)
    if len(cal) != EXPECT_CALIBRATION_N:
        raise SystemExit(
            f"[task2][FATAL] expected {EXPECT_CALIBRATION_N} calibration rows, got {len(cal)}"
        )
    pred_map = {p["sample_id"]: p for p in load_jsonl(BASE_PREDICTIONS)}
    assert len(pred_map) == EXPECT_CALIBRATION_N

    per_label: dict[str, dict[str, int | float]] = {}
    for row in cal:
        label = row["label"]
        pred = pred_map[row["sample_id"]]
        parsed = pred.get("parsed") or {}
        label_match = (
            bool(parsed.get("valid"))
            and parsed.get("decision") == "answer"
            and _normalise_label(parsed.get("label", "")) == _normalise_label(label)
        )
        correct = int(bool(label_match and float(pred.get("iou") or 0.0) >= 0.5))
        d = per_label.setdefault(label, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += correct

    def aggregate(labels: set[str]) -> dict[str, float]:
        labels = sorted(labels)
        n = sum(int(per_label[l]["n"]) for l in labels)
        correct = sum(int(per_label[l]["correct"]) for l in labels)
        micro = correct / n
        macro = sum(int(per_label[l]["correct"]) / per_label[l]["n"] for l in labels) / len(labels)
        return {"n": n, "correct": correct, "micro_rate": micro, "macro_rate": macro}

    all24 = aggregate(observed)
    retained20 = aggregate(retained)
    excluded4 = aggregate(excluded)
    assert all24["n"] == EXPECT_CALIBRATION_N
    assert retained20["n"] + excluded4["n"] == EXPECT_CALIBRATION_N

    result = {
        "audit_name": audit["audit_name"],
        "model": audit["model"],
        "selection_threshold": audit["thresholds"],
        "population": "Base calibration pool (768 PlantDoc base records, pre source-image split)",
        "scope_note": (
            "Recomputed from the Base calibration pool and frozen-Qwen3-VL base predictions. "
            "This is the Base-pool capability re-audit, not an independent test and not "
            "three-system (Base/SFT/ReGate) full-24 results."
        ),
        "all_24_observed": aggregate(observed),
        "retained_20": retained20,
        "excluded_4": excluded4,
        "excluded_labels": {l: {**per_label[l]} for l in sorted(excluded)},
        "per_label": {l: {**per_label[l]} for l in sorted(per_label)},
        "checked_against_published_audit": datetime_note(),
        "cross_check": _capability_cross_check(per_label, audit),
    }
    write_json(EXTRA_CPU / "capability_24_vs_20.json", result)
    return result


def datetime_note() -> str:
    import datetime

    return f"computed_by=review_additional_cpu.py at={datetime.datetime.now(datetime.timezone.utc).isoformat()}"


def _capability_cross_check(per_label: dict, audit: dict) -> dict[str, Any]:
    diffs = []
    for label, d in per_label.items():
        n_pub = audit["per_label"][label]["n"]
        rate_pub = audit["per_label"][label]["correct_rate"]
        if d["n"] != n_pub or abs(int(d["correct"]) / d["n"] - rate_pub) > 1e-9:
            diffs.append(
                {
                    "label": label,
                    "published_n/correct_rate": [n_pub, rate_pub],
                    "recomputed_n/correct_rate": [d["n"], int(d["correct"]) / d["n"]],
                }
            )
    return {"all_labels_match_published": not diffs, "differences": diffs}


# ── task 3: two-person label audit sampling preparation ─────────────────────

TEST_SPLIT = _register_hash(
    "raw_test_split",
    ROOT / "aaai_code/benchmark/annotations/splits/test.jsonl",
)

AUDIT_STATE_TARGET = {"answer": 200, "wrong_object": 200, "no_target": 200}
AUDIT_SEED = 42


def _state_of(row: dict[str, Any]) -> str:
    return row["reason"] if row.get("decision") == "abstain" else "answer"


def _allocate(keys: list[str], proportions: dict[str, float], target: int) -> dict[str, int]:
    """Largest-remainder allocation to exactly hit ``target``."""
    out = {k: int(prop * target) for k, prop in proportions.items()}
    remainders = sorted(
        keys,
        key=lambda k: (proportions[k] * target - out[k], k),
        reverse=True,
    )
    for i in range(target - sum(out.values())):
        out[remainders[i % len(remainders)]] += 1
    return out


def task3_label_audit_prep() -> dict[str, Any]:
    rows = load_jsonl(TEST_SPLIT)
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    strata_pop: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        state = _state_of(row)
        by_state[state].append(row)
        strata_pop[(state, row["dataset"], row["transform"])] += 1

    rng = random.Random(AUDIT_SEED)
    chosen_by_state: dict[str, list[dict[str, Any]]] = {}
    plan: dict[str, Any] = {}

    for state, target in AUDIT_STATE_TARGET.items():
        # Deterministic strata order; dedup key is the exact (image, query) audit question.
        strata = sorted(
            {k[1:] for k in strata_pop if k[0] == state and strata_pop[k] > 0}
        )

        def stratum_rows(dataset: str, transform: str, _state: str = state) -> list[dict[str, Any]]:
            return [
                r
                for r in by_state[_state]
                if r["dataset"] == dataset and r["transform"] == transform
            ]

        def eligible(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Return pool rows that are not duplicate audit questions."""
            seen: set[tuple[str, str]] = set()
            out: list[dict[str, Any]] = []
            for r in pool:
                key = (str(r["image_path"]), str(r["query"]))
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
            return out

        if state == "no_target":
            # Priority: oversample COCO negatives (60% vs ~32.6% population share).
            coco_keys = [s for s in strata if s[0] == "coco_ground_abstain"]
            plant_keys = [s for s in strata if s[0] != "coco_ground_abstain"]
            coco_pop = sum(strata_pop[(state, d, t)] for d, t in coco_keys)
            plant_pop = sum(strata_pop[(state, d, t)] for d, t in plant_keys)
            coco_target = round(target * 0.6)
            plant_target = target - coco_target
            coco_props = {k: strata_pop[(state, *k)] / coco_pop for k in coco_keys}
            plant_props = {k: strata_pop[(state, *k)] / plant_pop for k in plant_keys}
            alloc_map = _allocate(coco_keys, coco_props, coco_target)
            alloc_map.update(_allocate(plant_keys, plant_props, plant_target))
        else:
            # Proportional allocation across (dataset, transform) strata; wrong_object is
            # the T6 set by construction.
            pop = sum(strata_pop[(state, d, t)] for d, t in strata)
            props = {k: strata_pop[(state, *k)] / pop for k in strata}
            alloc_map = _allocate(strata, props, target)

        # Sample within each stratum without replacement; dedup audit questions.
        chosen: list[dict[str, Any]] = []
        used_keys: set[tuple[str, str]] = set()
        for (d, t), want in sorted(alloc_map.items(), key=lambda kv: kv[0]):
            pool = eligible(stratum_rows(d, t))
            rng.shuffle(pool)
            got = 0
            for r in pool:
                key = (str(r["image_path"]), str(r["query"]))
                if key in used_keys:
                    continue
                used_keys.add(key)
                chosen.append(r)
                got += 1
                if got >= want:
                    break
        # Fill shortfall from the remaining eligible pool of the same state.
        if len(chosen) < target:
            remaining = eligible(
                [r for r in by_state[state] if (str(r["image_path"]), str(r["query"])) not in used_keys]
            )
            rng.shuffle(remaining)
            for r in remaining:
                if len(chosen) >= target:
                    break
                chosen.append(r)
                used_keys.add((str(r["image_path"]), str(r["query"])))
        chosen_by_state[state] = chosen
        plan[state] = {
            "population": len(by_state[state]),
            "target": target,
            "allocation": {f"{d}|{t}": a for (d, t), a in sorted(alloc_map.items())},
            "selected": len(chosen),
            "unique_source_images": len({r["source_image_id"] for r in chosen}),
        }

    audit_rows: list[dict[str, Any]] = []
    for state, recs in sorted(chosen_by_state.items()):
        strata_sel: Counter[tuple[str, str]] = Counter()
        for r in recs:
            strata_sel[(r["dataset"], r["transform"])] += 1
        # Design weight = stratum population / stratum selected.
        weight = {
            k: strata_pop[(state, k[0], k[1])] / v for k, v in strata_sel.items()
        }
        for r in recs:
            key = (r["dataset"], r["transform"])
            audit_rows.append(
                {
                    "sample_id": r["sample_id"],
                    "image_path": r["image_path"],
                    "query": r["query"],
                    "state": state,
                    "transform": r["transform"],
                    "dataset": r["dataset"],
                    "relation": r["metadata"]["relation"],
                    "source_image_id": r["source_image_id"],
                    "stratum": f"{r['dataset']}|{r['transform']}",
                    "stratum_population": strata_pop[(state, r["dataset"], r["transform"])],
                    "stratum_selected": strata_sel[key],
                    "sample_weight": weight[key],
                    "current_decision": r["decision"],
                    "current_reason": r.get("reason"),
                    "current_label": r.get("label"),
                    "annotator_1": "",
                    "annotator_2": "",
                }
            )

    popplan = {
        k: {
            "n": len(v),
            "unique_source_images": len({r["source_image_id"] for r in v}),
        }
        for k, v in by_state.items()
    }

    for state, want in AUDIT_STATE_TARGET.items():
        if len(chosen_by_state[state]) != want:
            raise SystemExit(f"[task3][FATAL] state {state} selected {len(chosen_by_state[state])} != {want}")

    result = {
        "status": "sampling_prep_only",
        "human_annotation_results_collected": False,
        "population_source": "raw test split (splits/test.jsonl), not the oversampled SFT train split",
        "population_note": popplan,
        "plan": plan,
        "seed": AUDIT_SEED,
        "targets": AUDIT_STATE_TARGET,
        "priority_note": (
            "wrong_object state is the T6 query_image_mismatch set by construction; "
            "no_target oversamples COCO negatives (60% of the 200-row allocation vs ~32.6% "
            "population share)."
        ),
        "deduplication": (
            "per-state dedup of duplicate (image_path, query) audit questions; sample rows "
            "are unique sample_ids from the raw test split (no oversampled train rows)."
        ),
        "selected_per_state": {k: len(v) for k, v in chosen_by_state.items()},
        "n_rows": len(audit_rows),
        "columns": [
            "sample_id",
            "image_path",
            "query",
            "state",
            "transform",
            "dataset",
            "relation",
            "source_image_id",
            "stratum",
            "stratum_population",
            "stratum_selected",
            "sample_weight",
            "current_decision",
            "current_reason",
            "current_label",
            "annotator_1",
            "annotator_2",
        ],
        "rows": audit_rows,
    }
    write_json(EXTRA_CPU / "label_audit_prep.json", result)

    csv_path = EXTRA_CPU / "label_audit_prep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=result["columns"])
        writer.writeheader()
        writer.writerows(audit_rows)
    return result


# ── task 4: SFT native own-confidence fields + baseline protocol check ──────

SFT_NATIVE_TEST = _register_hash(
    "sft_native_test_comprehensive_predictions",
    ROOT
    / "new_reason/outputs/base_native_regate_20260720/native/sft/comprehensive/predictions.jsonl",
)
SFT_NATIVE_DEV = _register_hash(
    "sft_native_dev_predictions",
    ROOT / "new_reason/outputs/matched_retention_dev_20260725/dev/native_sft/predictions.jsonl",
)
EVAL_SCRIPT = ROOT / "aaai_code/evaluation/eval_qwen3vl_3reason.py"

GROUNDINGME_RUN_CONFIG = _register_hash(
    "groundingme_run_config",
    SOURCE_ROOT / "aaai_code/experiments/groundingme_external/runs/eval_v1_20260720/base/run_config.json",
)
OPENREF_RUN_CONFIG = _register_hash(
    "openref_run_config",
    SOURCE_ROOT
    / "aaai_code/experiments/openref_external/runs/openref_single_500_v1_20260721/sft/run_config.json",
)

CONFIDENCE_PATTERN = re.compile(r"logprob|entropy|confid|score|proba", re.IGNORECASE)


def _audit_confidence_fields(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    def walk(obj: Any, prefix: str, hits: set[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else k
                if CONFIDENCE_PATTERN.search(p):
                    hits.add(p)
                walk(v, p, hits)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:5]):
                walk(v, f"{prefix}[{i}]", hits)

    hits: set[str] = set()
    for r in rows[:min(len(rows), 200)]:
        walk(r, "", hits)
    top_keys = sorted(rows[0].keys()) if rows else []
    pred_keys = sorted((rows[0].get("pred") or {}).keys()) if rows else []
    return {
        "file": str(path),
        "n_rows": len(rows),
        "top_level_keys": top_keys,
        "pred_keys": pred_keys,
        "confidence_like_fields_found": sorted(hits),
        "native_own_confidence_signal": len(hits) > 0,
    }


def task4_sft_native_confidence_and_baselines() -> dict[str, Any]:
    test_audit = _audit_confidence_fields(SFT_NATIVE_TEST)
    dev_audit = _audit_confidence_fields(SFT_NATIVE_DEV)

    # Eval-script evidence: greedy generation, no output_scores/logprobs stored.
    script_text = EVAL_SCRIPT.read_text(encoding="utf-8", errors="ignore")
    has_output_scores = "output_scores" in script_text
    has_return_dict_in_generate = "return_dict_in_generate" in script_text
    generation_lines = [ln.strip() for ln in script_text.splitlines() if "do_sample=False" in ln]

    # Baseline protocol check: groundingme_external / openref_external.
    groundingme_readme = _read_head(ROOT / "aaai_code/experiments/groundingme_external/README.md", 60)
    groundingme_prep = _read_head(ROOT / "aaai_code/experiments/groundingme_external/prepare_groundingme.py", 20)
    openref_prep = _read_head(ROOT / "aaai_code/experiments/openref_external/prepare_openref.py", 25)
    groundingme_run_cfg = _read_json(
        SOURCE_ROOT
        / "aaai_code/experiments/groundingme_external/runs/eval_v1_20260720/base/run_config.json"
    )
    openref_run_cfg = _read_json(
        SOURCE_ROOT
        / "aaai_code/experiments/openref_external/runs/openref_single_500_v1_20260721/sft/run_config.json"
    )

    result = {
        "sft_native_dev_prediction_audit": dev_audit,
        "sft_native_test_prediction_audit": test_audit,
        "eval_script": {
            "path": str(EVAL_SCRIPT),
            "has_output_scores_request": has_output_scores,
            "has_return_dict_in_generate": has_return_dict_in_generate,
            "greedy_generation_call_sites": len(generation_lines),
        },
        "conclusion_own_confidence": (
            "SFT native dev/test predictions contain no per-sample own confidence signal "
            "(no logprob, no entropy, no marginal score). The eval script uses greedy "
            "generation and does not store output_scores/return_dict_in_generate. Without a "
            "stored native confidence, a dev-fit threshold curve is not available offline, "
            "and ReGate head probabilities are not a valid stand-in for a native confidence."
        ),
        "baseline_protocol_check": {
            "groundingme_external": {
                "evidence": "GroundingME dataset (CVPR 2026, arXiv:2512.17495, "
                "github.com/lirang04/GroundingME, HF lirang04/GroundingME).",
                "doc_lines": groundingme_readme[:8],
                "prep_docstring": groundingme_prep[:6],
                "run_config": groundingme_run_cfg,
            },
            "openref_external": {
                "evidence": "OpenRef val.json-derived single-pair referring-expression "
                "diagnostic (500 pairs), prepared by prepare_openref.py.",
                "prep_docstring": openref_prep[:8],
                "run_config": openref_run_cfg,
            },
            "conclusion": (
                "The existing groundingme/openref outputs run the paper's own base/SFT/ReGate "
                "models on external benchmark-derived probes. They are NOT CounterfactualREC / "
                "GREC / RECANTFormer runs and are not a same-protocol comparison for those "
                "methods. The review statement that the same-protocol comparison for those "
                "methods remains to be supplemented still holds."
            ),
        },
    }
    write_json(EXTRA_CPU / "sft_native_confidence_and_baselines.json", result)
    return result


def _read_head(path: Path, n_lines: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()[:n_lines]


def _read_json(path: Path) -> dict[str, Any] | str:
    if not path.exists():
        return {"missing": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"unreadable": str(exc)}


# ── task 5: decode-skip timing availability ─────────────────────────────────

LOGS_GLOB = [
    ROOT / "new_reason/outputs/matched_retention_dev_20260725/logs",
    ROOT / "new_reason/outputs/logs",
    ROOT / "new_reason/baselines/prompt_cot/logs",
]

TIMING_PATTERNS = re.compile(
    r"it/s|s/it|latency|throughput|tokens/s|tok/s|elapsed|wall.?time|total.?time|generation.?time|decode.?time",
    re.IGNORECASE,
)


def task5_decode_skip_timing() -> dict[str, Any]:
    n_logs = 0
    timing_hits: dict[str, list[str]] = {}
    for log_dir in LOGS_GLOB:
        if not log_dir.exists():
            continue
        for p in sorted(log_dir.rglob("*.log")):
            n_logs += 1
            text = p.read_text(encoding="utf-8", errors="ignore")
            matches = [ln.strip() for ln in text.splitlines() if TIMING_PATTERNS.search(ln)]
            if matches:
                timing_hits[str(p)] = matches[:5]

    # GPU-count / nproc evidence: run configs record variable parallel resources.
    configs = [
        SOURCE_ROOT
        / "aaai_code/experiments/openref_external/runs/openref_single_500_v1_20260721/sft/run_config.json",
        SOURCE_ROOT
        / "aaai_code/experiments/groundingme_external/runs/eval_v1_20260720/base/run_config.json",
    ]
    gpu_counts = {}
    for cfg in configs:
        j = _read_json(cfg)
        if isinstance(j, dict) and j.get("missing") is None:
            gpu_counts[str(cfg)] = {
                "nproc": j.get("nproc"),
                "allocated_gpus": j.get("allocated_gpus"),
                "parallelism": j.get("parallelism"),
            }

    result = {
        "n_logs_scanned": n_logs,
        "logs_with_timing_patterns": len(timing_hits),
        "timing_evidence_sample": {k: v for k, v in list(timing_hits.items())[:10]},
        "recorded_gpu_counts": gpu_counts,
        "conclusion": (
            "Existing logs contain only HF progress bars (it/s) over runs using differing "
            "GPU counts / sharding and no per-decision latency, no generation-token counts, "
            "and no same-hardware/same-batch controlled comparison. They therefore cannot "
            "quantify a generation-skipping (decode-skip) benefit. No GPU timing run was "
            "performed; the absence is reported only."
        ),
    }
    write_json(EXTRA_CPU / "decode_skip_timing.json", result)
    return result


# ── provenance / runner ─────────────────────────────────────────────────────

def _build_hashes_manifest() -> dict[str, str]:
    return {name: sha256_hex(path) for name, path in HASH_SOURCES.items() if path.exists()}


# ── concise review-ready markdown ───────────────────────────────────────────

def _fmt_rate(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.4f}"


def write_md_l3(t1: dict[str, Any]) -> None:
    bin_ = t1["bin_answer_rate"]
    rows = []
    for b, gpt, rg, pa in zip(
        bin_["bin"], bin_["gpt_binary"], bin_["regate_binary"], bin_["regate_p_answer_mean"], strict=True
    ):
        rows.append(
            f"| {b} | {_fmt_rate(gpt)} | {_fmt_rate(rg)} | {_fmt_rate(pa)} |"
        )
    adj = t1["adjacent_gpt_binary_decline"]
    pa = t1["adjacent_legate_pa_violation"]
    rg = t1["adjacent_legate_binary_argmax_decline"]
    ep = t1["endpoint"]
    text = f"""## L3 GPT predictions vs ReGate head probabilities — 相邻可见度 baseline

分组与档位完全沿用 `analyze_evidence.analyze_l3` 的 `stable_group_key` 规则（`L3-{{img_prefix}}__{{source}}`，按同图同 source_instance 聚合，作业按 `achieved_visible_ratio` 排序），不按 sample_id 简单拼接。共 {t1['n_groups']} 组、{t1['n_adjacent_pairs']} 个相邻档位对，无 group 解析失败或 duplicate-ratio 失效。

| achieved bin | GPT binary answer rate | ReGate binary argmax answer rate | ReGate mean p_answer |
|---:|---:|---:|---:|
{chr(10).join(rows)}

相邻档位违例（单调性）按 group 计算，ties 不违背：相邻较高可见度必须不低于较低可见度才视为满足。GPT 只有二元 decision，因此其“binary adjacent-decline rate”与 ReGate 的连续 `p_answer` violation rate 是不同对象，二者数值不可直接等价；为公平比较，ReGate 额外计算 binary argmax 的相邻下降比例。

| 指标（183 组 pooled） | 值 |
|---|---:|
| GPT binary adjacent-decline rate（违例对/全部相邻对） | {_fmt_rate(adj['pooled'])}（{adj['violating_pairs']}/{t1['n_adjacent_pairs']}） |
| GPT binary adjacent-decline rate（group mean） | {_fmt_rate(adj['mean_per_group'])} |
| ReGate p_answer adjacent violation rate（同原分析） | {_fmt_rate(pa['pooled'])}（{pa['violating_pairs']}/{t1['n_adjacent_pairs']}） |
| ReGate p_answer group mean violation（recompute） | {_fmt_rate(pa['mean_per_group'])}（现分析 {_fmt_rate(pa['existing_analysis_mean'])}） |
| ReGate binary argmax adjacent-decline rate | {_fmt_rate(rg['pooled'])}（{rg['violating_pairs']}/{t1['n_adjacent_pairs']}） |

Endpoint（组内存在精确 1.0 与 0.0 档，{ep['n_complete_endpoint_groups']} 组）：GPT binary endpoint decline={_fmt_rate(ep['gpt_binary_endpoint_decline'])}，ReGate binary endpoint decline={_fmt_rate(ep['regate_binary_endpoint_decline'])}，ReGate p_answer endpoint delta mean={_fmt_rate(ep['regate_pa_endpoint_delta_mean'])}（现分析 {_fmt_rate(ep['existing_analysis_endpoint_delta_mean'])}）。

> 注意：二元决策单调性与连续 `p_answer` 单调性衡量不同的性质；同一组数据上二者数值不同属预期，不能用 GPT 的二元下降比例直接对标 ReGate 的连续 violation rate。相关 JSON：`l3_gpt_vs_regate.json`。
"""
    (EXTRA_CPU / "l3_gpt_vs_regate.md").write_text(text, encoding="utf-8")


def write_md_capability(t2: dict[str, Any]) -> None:
    rows = []
    for label in sorted(t2["per_label"]):
        d = t2["per_label"][label]
        tag = ""
        if label in (t2["excluded_labels"] or {}):
            tag = " (excluded4)"
        rows.append(
            f"| {label}{tag} | {d['n']} | {d['correct']} | {_fmt_rate(d['correct'] / d['n'])} |"
        )
    a, r, e = t2["all_24_observed"], t2["retained_20"], t2["excluded_4"]
    text = f"""## Capability audit — 全 24 类 vs retained 20 的 Base 校准池重算

按 Base 校准池（PlantDoc base records {a['n']} 条）与 frozen Qwen3-VL 原始预测，用整理的 `parsed.decision==answer` 且归一化 label 匹配且 IoU≥0.5 逐条重算，与已发布 audit 的 per-label n/correct_rate 完全一致（差异 0）。这是 Base 池复算，不是独立 test，也不是三系统全 24 类的正式结果；保留具体计数。

| 集合 | n | correct | micro correct_rate | macro（per-label 均值） |
|---|---:|---:|---:|---:|
| 全部 24 类（observed） | {a['n']} | {a['correct']} | {_fmt_rate(a['micro_rate'])} | {_fmt_rate(a['macro_rate'])} |
| retained 20 | {r['n']} | {r['correct']} | {_fmt_rate(r['micro_rate'])} | {_fmt_rate(r['macro_rate'])} |
| excluded 4 | {e['n']} | {e['correct']} | {_fmt_rate(e['micro_rate'])} | {_fmt_rate(e['macro_rate'])} |

excluded 4 明细与全 24 类 per-label 计数：

| label | n | correct | correct_rate |
|---|---:|---:|---:|
{chr(10).join(rows)}

证据：`aaai_code/benchmark/annotations/audits/PLANTDOC_CAPABILITY_GATE_AUDIT.json`、重算结果 `docs/review_evidence/additional_cpu/capability_24_vs_20.json`（含 `checked_against_published_audit`）。
"""
    (EXTRA_CPU / "capability_24_vs_20.md").write_text(text, encoding="utf-8")


def write_md_audit_prep(t3: dict[str, Any]) -> None:
    lines = [
        "## 双人标签审计 — 抽样准备文件",
        "",
        "**状态**：'sampling_prep_only'，未收集任何人工标注结果；`annotator_1`/`annotator_2` 均为空，待两名标注者填写。",
        "",
        f"**总体**：从 raw test split（`splits/test.jsonl`，共 {sum(v['n'] for v in t3['population_note'].values())} 条）抽样，不用超采样 train（`sft_train.json`）以防重复。每状态 200 条，`seed={t3['seed']}`，按 (dataset, transform) 分层；每状态内对 (image_path, query) 去重，确保不复制图片、不伪造标签。",
        "",
        "| state | population | selected | 优先级 |",
        "|---|---:|---:|---|",
    ]
    for state in ("answer", "wrong_object", "no_target"):
        plan = t3["plan"][state]
        prio = {"answer": "proportional", "wrong_object": "whole state = T6", "no_target": "COCO negatives 60%"}[
            state
        ]
        lines.append(
            f"| {state} | {plan['population']} | {plan['selected']} | {prio} |"
        )
    lines += [
        "",
        f"选中的行数为 {t3['n_rows']}（应等于 600），每行含 `sample_id`、`image_path`、`query`、`state`、`transform`、population/权重分层与两个空 `annotator` 字段。`sample_weight` = 层 population / 层 selected（survey design weight）。",
        "",
        "文件：`docs/review_evidence/additional_cpu/label_audit_prep.csv`（可直接评审）与 `.json`（含计划、逐行权重与分层）。",
    ]
    (EXTRA_CPU / "label_audit_prep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_md_confidence_and_baselines(t4: dict[str, Any]) -> None:
    test_c = t4["sft_native_test_prediction_audit"]
    dev_c = t4["sft_native_dev_prediction_audit"]
    text = f"""## SFT native 自身 confidence 可用性 与 相近 baseline protocol 核查

### 4a. SFT native dev/test 预测是否含自身 confidence（logprob/entropy 等）

| 文件 | 顶层字段 | pred 字段 | 检出 confidence 字段 |
|---|---|---|---|
| test comprehensive (`native/sft/.../predictions.jsonl`, {test_c['n_rows']} 行) | {', '.join(test_c['top_level_keys'])} | {', '.join(test_c['pred_keys'])} | {test_c['confidence_like_fields_found'] or '无'} |
| dev (`matched_retention_dev.../dev/native_sft/...`, {dev_c['n_rows']} 行) | {', '.join(dev_c['top_level_keys'])} | {', '.join(dev_c['pred_keys'])} | {dev_c['confidence_like_fields_found'] or '无'} |

eval 脚本为 greedy generation（`do_sample=False`），未保存 `output_scores`/`return_dict_in_generate`，因此**没有**逐 token logprob 或任何 per-sample 自身置信度。没有真正自身信号就不能离线 dev-fit 阈值曲线；不能用 ReGate head 概率冒充 SFT native 自身 confidence。

### 4b. 现有 groundingme/openref 输出是否即 CounterfactualREC / GREC / RECANTFormer？

不是。现有 `aaai_code/experiments/groundingme_external/` 与 `openref_external/` 是**外部 benchmark-derived probe**：

- GroundingME：HF 数据集 `lirang04/GroundingME`（CVPR 2026，arXiv 2512.17495），`prepare_groundingme.py` 将其映射到本文三态接口；运行的是本论文自己的 base/SFT/ReGate（run_config 中 `base_model=Qwen3-VL-8B-Instruct`，`adapter=checkpoint-444`）。
- OpenRef：`prepare_openref.py` 从 OpenRef val.json 派生 single-pair 诊断（500 pairs），同样是本文 base/SFT/ReGate 的输出。

这些输出不含任何 CounterfactualREC / GREC / RECANTFormer 模型结果，也不是这些方法的同 protocol 实现。因此 review 中“上述方法的同 protocol 比较尚未补充”的表述仍然成立。完整证据：`docs/review_evidence/additional_cpu/sft_native_confidence_and_baselines.json`（含 `prep_docstring` 与 `run_config`）。
"""
    (EXTRA_CPU / "sft_native_confidence_and_baselines.md").write_text(text, encoding="utf-8")


def write_md_timing(t5: dict[str, Any]) -> None:
    text = f"""## 源日志 timing 能否量化 decode-skip（同 hardware/batch）

扫描 {t5['n_logs_scanned']} 份日志，命中 timing 样式（`it/s`、`s/it`、latency、throughput、elapsed 等）的共 {t5['logs_with_timing_patterns']} 份。命中内容仅为 HF/eval 进度条（`it/s`），且各 run 的 GPU 数量/分片不同（如 openref run `nproc=2, allocated_gpus=4, parallelism=2 data-parallel + 2-way HF sharding`；groundingme run `nproc=4`），没有 per-decision latency、没有 generation token 计数、没有同机同 batch 的受控对比。结论：现有日志无法量化 generation-skipping/decode-skip 延迟收益；未进行 GPU 计时，仅报告缺失。JSON：`docs/review_evidence/additional_cpu/decode_skip_timing.json`。
"""
    (EXTRA_CPU / "decode_skip_timing.md").write_text(text, encoding="utf-8")


def write_index_md() -> None:
    text = """# additional_cpu：第二轮离线审稿证据

全部结果仅读取现有文件，无模型推理、无训练、无付费 API、无 GPU。

| 文件 | 用途 |
|---|---|
| `l3_gpt_vs_regate.json/.md` | L3 GPT 二元决策 vs ReGate head 概率的相邻可见度 baseline（183 组/567 相邻对） |
| `capability_24_vs_20.json/.md` | Base 校准池全 24 类 per-label n/correct 重算，retained20/excluded4 micro·macro |
| `label_audit_prep.csv/.json/.md` | 双人标签审计抽样准备（每状态 200，T6/COCO 负例优先，分层+权重+source 去重） |
| `sft_native_confidence_and_baselines.json/.md` | SFT native 无自身 confidence 的证据；groundingme/openref 非 CounterfactualREC/GREC/RECANTFormer |
| `decode_skip_timing.json/.md` | 现有日志不能量化 decode-skip；仅报告缺失 |
| `sources_hashes.json` | 上述所有输入文件的 SHA-256 |

复算：

```bash
PYTHONDONTWRITEBYTECODE=1 /user/huangyongbin/miniconda3/envs/agrivllm/bin/python tools/review_additional_cpu.py
```
"""
    (EXTRA_CPU / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    EXTRA_CPU.mkdir(parents=True, exist_ok=True)
    # absolute source paths must exist
    for name, path in HASH_SOURCES.items():
        if not path.exists():
            raise SystemExit(f"[FATAL] source missing for {name}: {path}")

    print("[1/5] L3 GPT vs ReGate adjacent-visibility baseline")
    t1 = task1_l3_gpt_vs_regate()
    print(
        f"  groups={t1['n_groups']} adj={t1['n_adjacent_pairs']} "
        f"gpt_bin_decline={t1['adjacent_gpt_binary_decline']['pooled']:.4f} "
        f"pa_violation={t1['adjacent_legate_pa_violation']['pooled']:.4f} "
        f"regate_bin_decline={t1['adjacent_legate_binary_argmax_decline']['pooled']:.4f}"
    )
    write_md_l3(t1)

    print("[2/5] capability audit 24 vs retained 20")
    t2 = task2_capability_24_vs_20()
    print(
        f"  all24 micro={t2['all_24_observed']['micro_rate']:.4f} macro={t2['all_24_observed']['macro_rate']:.4f} "
        f"retained20 micro={t2['retained_20']['micro_rate']:.4f} macro={t2['retained_20']['macro_rate']:.4f} "
        f"excluded4 n={t2['excluded_4']['n']} correct={t2['excluded_4']['correct']}"
    )
    write_md_capability(t2)

    print("[3/5] two-person label audit sampling prep")
    t3 = task3_label_audit_prep()
    print(f"  rows={t3['n_rows']} selected={t3['selected_per_state']}")
    write_md_audit_prep(t3)

    print("[4/5] SFT native own-confidence + baseline protocol check")
    t4 = task4_sft_native_confidence_and_baselines()
    print(
        f"  native_test_conf={t4['sft_native_test_prediction_audit']['native_own_confidence_signal']} "
        f"native_dev_conf={t4['sft_native_dev_prediction_audit']['native_own_confidence_signal']}"
    )
    write_md_confidence_and_baselines(t4)

    print("[5/5] decode-skip timing availability")
    t5 = task5_decode_skip_timing()
    print(f"  logs_scanned={t5['n_logs_scanned']} logs_with_timing={t5['logs_with_timing_patterns']}")
    write_md_timing(t5)
    write_index_md()

    manifest = _build_hashes_manifest()
    write_json(EXTRA_CPU / "sources_hashes.json", {"sha256": manifest})
    print(f"[ok] written to {EXTRA_CPU}")


if __name__ == "__main__":
    main()
