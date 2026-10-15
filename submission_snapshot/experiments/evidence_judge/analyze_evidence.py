#!/usr/bin/env python
"""Analyze head probability JSONL from extract_head_probabilities.py.

Computes:
  - L2 counterfactual (keep vs cropout): delta_p_answer, correct direction ratio,
    mean/median, bootstrap 95% CI.
  - L3 dose-response: per-group, per achieved_visible_ratio aggregation, Spearman rho,
    Kendall tau, tau>0 ratio, adjacent non-decreasing violation rate, endpoint delta
    (only when exact ratio 0.0 and 1.0 present), cross-group bootstrap CI, shuffled-ratio
    baseline.
  - Input equivariance consistency: per-sample query/image variants vs original,
    p_answer std/range, argmax decision flip, absolute delta; aggregated mean/median/bootstrap CI.

Output: JSON report and optional group-level CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── scipy check ─────────────────────────────────────────────────────────────

_SCIPY_AVAILABLE: bool = False
try:
    from scipy.stats import spearmanr, kendalltau, bootstrap  # noqa: F401

    _SCIPY_AVAILABLE = True
except ImportError:
    pass


def _require_scipy() -> None:
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for L3 dose-response analysis (Spearman rho, Kendall tau, "
            "bootstrap).  Install with: pip install scipy"
        )


# ── group key helpers ───────────────────────────────────────────────────────


def _strip_ladder_suffix(s: str) -> str:
    """Strip known '-ladder-\\d{3}' suffix from a string."""
    import re

    return re.sub(r"-ladder-\d{3}$", "", s)


def _group_key_l2(sample_id: str) -> str | None:
    """L2 group key = image prefix + normalized source sample ID.

    Format:
      L2-{image_id}-keep-{source_sample_id}-answer
      L2-{image_id}-cropout-{source_sample_id}-ladder-000

    Group key = f"{image_prefix}__{source}" where source is the full
    source_sample_id (e.g. plantdoc-train-<hash>-0 or coco-train-<coco_id>-<ann_id>).
    This prevents merging records with different annotation IDs on the same image.
    """
    for marker in ("-keep-", "-cropout-"):
        idx = sample_id.find(marker)
        if idx != -1:
            image_prefix = sample_id[:idx]
            suffix = sample_id[idx + len(marker) :]
            # Strip known terminal suffixes
            if suffix.endswith("-answer"):
                source = suffix[: -len("-answer")]
            else:
                # Strip -ladder-XXX
                source = _strip_ladder_suffix(suffix)
            return f"{image_prefix}__{source}"
    return None


def _group_key_l3(sample_id: str) -> str | None:
    """L3 group key = image prefix + normalized source sample ID.

    Format:
      L3-{image_id}-{ratio:.3f}-{source_sample_id}-ladder-{dose}

    Group key combines ratio-before image prefix with ratio-after source
    (with -ladder-XXX stripped).  This groups same source instance across
    all ratio levels.  COCO annotation IDs are preserved in source, so
    different objects on the same image do not collide.
    """

    if not sample_id.startswith("L3-"):
        return None
    rest = sample_id[3:]  # after "L3-"
    parts = rest.split("-")
    # Find the first part that looks like a ratio (contains a dot and is numeric in [0,1])
    ratio_idx = None
    for i, p in enumerate(parts):
        try:
            val = float(p)
            if 0.0 <= val <= 1.0 and "." in p:
                ratio_idx = i
                break
        except ValueError:
            continue
    if ratio_idx is None:
        return None
    # Image prefix = parts before the ratio segment
    img_prefix = "-".join(parts[:ratio_idx])
    # After ratio: the source_sample_id + -ladder-XXX
    after_ratio = "-".join(parts[ratio_idx + 1 :])
    # Strip -ladder-\d{3} suffix
    source = _strip_ladder_suffix(after_ratio)
    return f"L3-{img_prefix}__{source}"


def stable_group_key(sample_id: str, _row: dict | None = None) -> str | None:
    """Determine stable group key from sample_id.

    For L2: {image_prefix}__{source_sample_id}
    For L3: L3-{image_prefix}__{source_sample_id}
    """
    if sample_id.startswith("L2-"):
        return _group_key_l2(sample_id)
    elif sample_id.startswith("L3-"):
        return _group_key_l3(sample_id)
    return None


def normalize_target_label(label: str | None) -> str:
    """Normalize target_label for grouping: strip whitespace, casefold.

    Returns empty string if label is None or empty.
    """
    if not label:
        return ""
    return label.strip().casefold()


def pairing_key_l2(sample_id: str, target_label: str | None) -> str | None:
    """L2 pairing key = stable_group_key + normalized target_label.

    Returns None if stable_group_key cannot be determined.
    """
    gk = stable_group_key(sample_id)
    if gk is None:
        return None
    tl = normalize_target_label(target_label)
    return f"{gk}__tl:{tl}"


# ── median helpers ──────────────────────────────────────────────────────────


def _true_median(values: list[float]) -> float:
    """True even-sample median (average of two middle values)."""
    sv = sorted(values)
    n = len(sv)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return sv[mid]
    return (sv[mid - 1] + sv[mid]) / 2.0


# ── bootstrap CI ────────────────────────────────────────────────────────────


def _bootstrap_ci(
    values: list[float],
    statistic: str = "mean",
    n_resamples: int = 9999,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CI for a statistic.

    Returns {low, high, method}.  Uses scipy BCa if available, falls back to
    manual percentile bootstrap.
    """
    if not values:
        return {"low": None, "high": None, "method": "no_data"}

    n = len(values)

    def _stat(vals):
        if statistic == "mean":
            return float(sum(vals) / len(vals))
        elif statistic == "median":
            return _true_median(vals)
        else:
            return float(sum(vals) / len(vals))

    if _SCIPY_AVAILABLE:
        from scipy.stats import bootstrap as _bootstrap

        try:
            res = _bootstrap(
                (values,),
                _stat,
                n_resamples=n_resamples,
                confidence_level=ci,
                random_state=seed,
                method="BCa",
            )
            return {
                "low": float(res.confidence_interval.low),
                "high": float(res.confidence_interval.high),
                "method": "BCa",
            }
        except Exception:
            pass

    # Fallback: manual percentile bootstrap
    rng = random.Random(seed)
    boot_stats: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(n)]
        boot_stats.append(_stat(sample))
    boot_stats.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1.0 - alpha) * n_resamples)
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return {
        "low": float(boot_stats[lo_idx]),
        "high": float(boot_stats[hi_idx]),
        "method": "percentile",
    }


# ── L2 analysis ─────────────────────────────────────────────────────────────


def analyze_l2(
    records: list[dict],
    n_bootstrap: int = 9999,
    bootstrap_seed: int = 42,
) -> dict:
    """Analyze L2 counterfactual (keep vs cropout).

    Groups by (image prefix + source sample ID).  Performs semantic deduplication:
    records with identical (sample_id, relation, image, target_label,
    achieved_visible_ratio, original_gt_decision) and matching probability
    fields (p_answer/p_wo/p_nt/predicted_class within 1e-7) are folded into one.

    Conflicts:
      - Same semantic identity but conflicting probabilities → group invalidated.
      - Same group+relation with different semantic identities after dedup → group invalidated.
    """
    parse_failures = 0
    incomplete_pairs = 0

    # Build raw groups: list of records per pairing key (source_instance + target_label)
    raw_groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        relation = rec.get("relation", "")
        if "L2" not in relation:
            continue
        sid = rec.get("sample_id") or ""
        target_label = rec.get("target_label", "")
        if not target_label:
            parse_failures += 1
            continue
        pk = pairing_key_l2(sid, target_label)
        if pk is None:
            parse_failures += 1
            continue
        vid = rec.get("variant_id", "original")
        if vid != "original":
            continue
        raw_groups[pk].append(rec)

    delta_ps: list[float] = []
    correct_dir: list[bool] = []
    deduplicated_records = 0
    groups_with_duplicates = 0
    conflicting_duplicate_groups = 0
    invalid_groups = 0

    for gk, recs in raw_groups.items():
        # ── Step 1: group by semantic identity ──────────────────────────────
        id_groups: dict[tuple, list[dict]] = defaultdict(list)
        for rec in recs:
            identity = (
                rec.get("sample_id", ""),
                rec.get("relation", ""),
                rec.get("image", ""),
                rec.get("target_label", ""),
                rec.get("achieved_visible_ratio"),
                rec.get("original_gt_decision"),
            )
            id_groups[identity].append(rec)

        # ── Step 2: deduplicate or detect conflicts per identity ────────────
        deduped: list[dict] = []
        has_conflict = False
        for identity, id_recs in id_groups.items():
            n_id = len(id_recs)
            if n_id == 1:
                deduped.append(id_recs[0])
                continue
            # Multiple records with same identity — check probability consistency
            p_answers = {r.get("p_answer") for r in id_recs}
            p_wos = {r.get("p_wo") for r in id_recs}
            p_nts = {r.get("p_nt") for r in id_recs}
            preds = {r.get("predicted_class") for r in id_recs}
            # Check if all probability fields are consistent (within 1e-7)
            probs_consistent = (
                len(p_answers) == 1 and len(p_wos) == 1 and len(p_nts) == 1 and len(preds) == 1
            )
            if probs_consistent:
                # Identical duplicates — fold into one record
                deduplicated_records += n_id - 1
                deduped.append(id_recs[0])
                groups_with_duplicates += 1
            else:
                # Same identity, conflicting probabilities
                has_conflict = True
                break

        if has_conflict:
            conflicting_duplicate_groups += 1
            invalid_groups += 1
            incomplete_pairs += 1
            continue

        # ── Step 3: separate into keep/cropout after dedup ──────────────────
        keeps = [r for r in deduped if "keep" in r.get("relation", "")]
        cropouts = [r for r in deduped if "cropout" in r.get("relation", "")]

        # Check for different semantic identities within same relation
        if len(keeps) > 1 or len(cropouts) > 1:
            conflicting_duplicate_groups += 1
            invalid_groups += 1
            incomplete_pairs += 1
            continue

        if len(keeps) == 0 or len(cropouts) == 0:
            incomplete_pairs += 1
            continue

        p_keep = keeps[0].get("p_answer")
        p_cropout = cropouts[0].get("p_answer")
        if p_keep is None or p_cropout is None:
            incomplete_pairs += 1
            continue

        delta = p_keep - p_cropout
        delta_ps.append(delta)
        correct_dir.append(delta > 0)

    n_pairs = len(delta_ps)
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "parse_failures": parse_failures,
            "incomplete_pairs": incomplete_pairs,
            "deduplicated_records": deduplicated_records,
            "groups_with_duplicates": groups_with_duplicates,
            "conflicting_duplicate_groups": conflicting_duplicate_groups,
            "invalid_groups": invalid_groups,
            "n_total_groups": len(raw_groups),
            "error": "no valid keep-cropout pairs",
        }

    correct_ratio = sum(correct_dir) / n_pairs if n_pairs else 0.0
    mean_delta = sum(delta_ps) / n_pairs
    median_delta = _true_median(delta_ps)

    ci = _bootstrap_ci(delta_ps, statistic="mean", n_resamples=n_bootstrap, seed=bootstrap_seed)

    return {
        "n_pairs": n_pairs,
        "n_total_groups": len(raw_groups),
        "parse_failures": parse_failures,
        "incomplete_pairs": incomplete_pairs,
        "deduplicated_records": deduplicated_records,
        "groups_with_duplicates": groups_with_duplicates,
        "conflicting_duplicate_groups": conflicting_duplicate_groups,
        "invalid_groups": invalid_groups,
        "delta_p_answer_mean": mean_delta,
        "delta_p_answer_median": median_delta,
        "correct_direction_ratio": correct_ratio,
        "bootstrap_95ci": ci,
        "pairing_key": "source_instance+target_label",
    }


# ── L3 analysis ─────────────────────────────────────────────────────────────


def analyze_l3(
    records: list[dict],
    bootstrap_seed: int = 42,
    shuffle_rounds: int = 1000,
    n_bootstrap: int = 9999,
) -> dict:
    """Analyze L3 dose-response.

    Groups by sample-level key, then aggregates by achieved_visible_ratio bins.
    Computes monotonicity statistics.  Endpoint delta only computed when both
    exact ratio 0.0 and 1.0 are present in the group.
    """
    _require_scipy()

    groups: dict[str, list[dict]] = defaultdict(list)
    parse_failures = 0
    for rec in records:
        sid = rec.get("sample_id") or ""
        vid = rec.get("variant_id", "original")
        if vid != "original":
            continue
        gk = stable_group_key(sid, rec)
        if gk is None:
            parse_failures += 1
            continue
        groups[gk].append(rec)

    per_group_results: dict[str, dict] = {}
    all_rhos: list[float] = []
    all_taus: list[float] = []
    all_tau_positive: list[bool] = []
    all_violation_rates: list[float] = []
    all_endpoint_deltas: list[float] = []
    n_complete_endpoint_groups = 0
    n_duplicate_ratio_groups = 0
    n_invalid_groups = 0

    # Collect all (group_key, ratio, p_answer) pairs for shuffle baseline
    all_group_pairs: dict[str, list[tuple[float, float]]] = {}

    for gk, recs in groups.items():
        ratio_groups: dict[float, list[float]] = defaultdict(list)
        had_duplicate_ratio = False
        for rec in recs:
            r = rec.get("achieved_visible_ratio")
            pa = rec.get("p_answer")
            if r is None or pa is None:
                continue
            r_key = round(r, 4)
            # Check for duplicate ratios within this normalized group
            if r_key in ratio_groups:
                had_duplicate_ratio = True
            ratio_groups[r_key].append(float(pa))

        # If duplicate ratios were found, invalidate this group
        if had_duplicate_ratio:
            n_duplicate_ratio_groups += 1
            n_invalid_groups += 1
            per_group_results[gk] = {
                "n_ratio_levels": len(ratio_groups),
                "ratio_levels": sorted(ratio_groups.keys()),
                "error": "duplicate ratio levels within normalized group",
                "invalid": True,
            }
            continue

        if len(ratio_groups) < 2:
            continue

        all_group_pairs[gk] = [(r, p) for r, plist in ratio_groups.items() for p in plist]

        ratios_sorted = sorted(ratio_groups.keys())
        ratio_means = {r: sum(ratio_groups[r]) / len(ratio_groups[r]) for r in ratios_sorted}
        ratio_vals = [ratio_means[r] for r in ratios_sorted]
        ratio_nums = list(ratios_sorted)

        # Spearman rho / Kendall tau
        from scipy.stats import kendalltau, spearmanr

        sr = spearmanr(ratio_nums, ratio_vals)
        rho = float(sr.correlation) if not math.isnan(sr.correlation) else None
        kt = kendalltau(ratio_nums, ratio_vals)
        tau = float(kt.correlation) if not math.isnan(kt.correlation) else None
        tau_pos = tau is not None and tau > 0

        # Adjacent non-decreasing violation rate
        violations = 0
        n_adj = len(ratio_vals) - 1
        for i in range(n_adj):
            if ratio_vals[i + 1] < ratio_vals[i]:
                violations += 1
        violation_rate = violations / n_adj if n_adj > 0 else 0.0

        # Endpoint delta: only when both exact 0.0 and 1.0 exist
        endpoint_delta = None
        has_ratio_0 = any(abs(r - 0.0) < 1e-6 for r in ratio_nums)
        has_ratio_1 = any(abs(r - 1.0) < 1e-6 for r in ratio_nums)
        if has_ratio_0 and has_ratio_1:
            r0 = min(ratio_nums, key=lambda x: abs(x - 0.0))
            r1 = min(ratio_nums, key=lambda x: abs(x - 1.0))
            endpoint_delta = ratio_means[r1] - ratio_means[r0]
            all_endpoint_deltas.append(endpoint_delta)
            n_complete_endpoint_groups += 1

        per_group_results[gk] = {
            "n_ratio_levels": len(ratio_nums),
            "ratio_levels": ratio_nums,
            "p_answer_means": ratio_vals,
            "spearman_rho": rho,
            "kendall_tau": tau,
            "tau_positive": tau_pos,
            "adjacent_violation_rate": violation_rate,
            "endpoint_delta_p": endpoint_delta,
            "has_complete_endpoints": has_ratio_0 and has_ratio_1,
        }

        if rho is not None:
            all_rhos.append(rho)
        if tau is not None:
            all_taus.append(tau)
            all_tau_positive.append(tau > 0)
        all_violation_rates.append(violation_rate)

    # Filter out invalid groups from n_groups count; keep them in per_group_results
    # for debugging but do not include in aggregate statistics.
    n_valid_groups = len(per_group_results) - n_invalid_groups
    n_groups = n_valid_groups
    if n_groups == 0:
        return {
            "n_groups": 0,
            "n_total_raw_groups": len(groups),
            "parse_failures": parse_failures,
            "n_duplicate_ratio_groups": n_duplicate_ratio_groups,
            "n_invalid_groups": n_invalid_groups,
            "per_group": per_group_results,
            "error": "no valid groups (all invalid or <2 ratio levels)",
        }

    # Aggregate statistics
    mean_rho = sum(all_rhos) / len(all_rhos) if all_rhos else None
    mean_tau = sum(all_taus) / len(all_taus) if all_taus else None
    tau_pos_ratio = sum(all_tau_positive) / len(all_tau_positive) if all_tau_positive else None
    mean_violation = sum(all_violation_rates) / len(all_violation_rates)
    mean_endpoint_delta = (
        sum(all_endpoint_deltas) / len(all_endpoint_deltas) if all_endpoint_deltas else None
    )

    rho_ci = (
        _bootstrap_ci(all_rhos, statistic="mean", n_resamples=n_bootstrap, seed=bootstrap_seed)
        if all_rhos
        else None
    )
    tau_ci = (
        _bootstrap_ci(all_taus, statistic="mean", n_resamples=n_bootstrap, seed=bootstrap_seed)
        if all_taus
        else None
    )
    violation_ci = _bootstrap_ci(
        all_violation_rates, statistic="mean", n_resamples=n_bootstrap, seed=bootstrap_seed
    )
    endpoint_ci = (
        _bootstrap_ci(
            all_endpoint_deltas, statistic="mean", n_resamples=n_bootstrap, seed=bootstrap_seed
        )
        if all_endpoint_deltas
        else None
    )

    # Shuffled ratio baseline: shuffle p_answer values across ratios within each group
    shuffle_results: dict = {}
    if len(all_group_pairs) >= 1:
        rng = random.Random(bootstrap_seed)
        shuffle_rhos: list[float] = []
        for _ in range(shuffle_rounds):
            round_rhos: list[float] = []
            for gk, pairs in all_group_pairs.items():
                ratios = [p[0] for p in pairs]
                p_vals = [p[1] for p in pairs]
                rng.shuffle(p_vals)
                # Aggregate per ratio
                shuf_ratio: dict[float, list[float]] = defaultdict(list)
                for rv, pv in zip(ratios, p_vals, strict=True):
                    shuf_ratio[round(rv, 4)].append(pv)
                rs = sorted(shuf_ratio.keys())
                rmeans = [sum(shuf_ratio[r]) / len(shuf_ratio[r]) for r in rs]
                if len(rs) >= 2:
                    sr = spearmanr(rs, rmeans)
                    sr_val = float(sr.correlation)
                    if not math.isnan(sr_val):
                        round_rhos.append(sr_val)
            if round_rhos:
                shuffle_rhos.append(sum(round_rhos) / len(round_rhos))
        if shuffle_rhos:
            shuffle_results = {
                "n_rounds": shuffle_rounds,
                "mean_shuffled_rho": sum(shuffle_rhos) / len(shuffle_rhos),
                "ci": _bootstrap_ci(
                    shuffle_rhos, statistic="mean", n_resamples=n_bootstrap, seed=bootstrap_seed
                ),
            }

    return {
        "n_groups": n_groups,
        "parse_failures": parse_failures,
        "mean_spearman_rho": mean_rho,
        "mean_kendall_tau": mean_tau,
        "tau_positive_ratio": tau_pos_ratio,
        "mean_adjacent_violation_rate": mean_violation,
        "mean_endpoint_delta_p": mean_endpoint_delta,
        "n_complete_endpoint_groups": n_complete_endpoint_groups,
        "n_duplicate_ratio_groups": n_duplicate_ratio_groups,
        "n_invalid_groups": n_invalid_groups,
        "spearman_rho_bootstrap_95ci": rho_ci,
        "kendall_tau_bootstrap_95ci": tau_ci,
        "violation_rate_bootstrap_95ci": violation_ci,
        "endpoint_delta_bootstrap_95ci": endpoint_ci,
        "shuffled_ratio_baseline": shuffle_results,
        "per_group": per_group_results,
    }


# ── consistency analysis ────────────────────────────────────────────────────


def analyze_input_equivariance_consistency(
    records: list[dict],
    n_bootstrap: int = 9999,
    bootstrap_seed: int = 42,
) -> dict:
    """Analyze input equivariance consistency across query/image variants.

    For each sample, compute:
      - p_answer std and range across variants (variants-only std)
      - all_equivalents_std: std including the original (within-variant-group std)
      - argmax decision flip count
      - absolute delta from original
    Then aggregate across all samples, broken down by variant_type and relation.
    """
    samples: dict[int, dict] = {}
    for rec in records:
        oi = rec.get("original_index")
        if oi is None:
            continue
        if oi not in samples:
            samples[oi] = {"records": [], "original": None}
        samples[oi]["records"].append(rec)
        if rec.get("variant_id") == "original":
            samples[oi]["original"] = rec

    per_sample_metrics: list[dict] = []

    for oi, data in samples.items():
        orig = data["original"]
        if orig is None:
            continue

        orig_pa = orig.get("p_answer")
        orig_pc = orig.get("predicted_class")
        if orig_pa is None or orig_pc is None:
            continue

        for vtype in ("query", "image"):
            variants = [
                r
                for r in data["records"]
                if r.get("variant_type") == vtype and r.get("variant_id") != "original"
            ]
            if not variants:
                continue

            p_answers = []
            decisions = []
            for v in variants:
                pa = v.get("p_answer")
                if pa is None:
                    continue
                p_answers.append(float(pa))
                decisions.append(v.get("predicted_class"))

            if not p_answers:
                continue

            n = len(p_answers)
            mean_pa = sum(p_answers) / n
            variance = sum((x - mean_pa) ** 2 for x in p_answers) / n
            std_pa = math.sqrt(variance)
            range_pa = max(p_answers) - min(p_answers)
            abs_deltas = [abs(x - float(orig_pa)) for x in p_answers]
            mean_abs_delta = sum(abs_deltas) / n
            flip_count = sum(1 for d in decisions if d != orig_pc)

            # all_equivalents_std: std including original
            all_vals = [float(orig_pa)] + p_answers
            all_mean = sum(all_vals) / len(all_vals)
            all_var = sum((x - all_mean) ** 2 for x in all_vals) / len(all_vals)
            all_std = math.sqrt(all_var)

            per_sample_metrics.append(
                {
                    "original_index": oi,
                    "variant_type": vtype,
                    "relation": orig.get("relation"),
                    "n_variants": n,
                    "p_answer_std": std_pa,
                    "p_answer_range": range_pa,
                    "mean_abs_delta_from_original": mean_abs_delta,
                    "argmax_decision_flip_count": flip_count,
                    "all_equivalents_std": all_std,
                }
            )

    if not per_sample_metrics:
        return {"input_equivariance_consistency": {"error": "no valid variant samples"}}

    by_type: dict[str, list[dict]] = defaultdict(list)
    by_relation: dict[str, list[dict]] = defaultdict(list)
    for m in per_sample_metrics:
        by_type[m["variant_type"]].append(m)
        rel = m.get("relation") or "unknown"
        by_relation[rel].append(m)

    def _aggregate_metrics(metrics: list[dict]) -> dict:
        stds = [m["p_answer_std"] for m in metrics]
        ranges = [m["p_answer_range"] for m in metrics]
        deltas = [m["mean_abs_delta_from_original"] for m in metrics]
        flips = [m["argmax_decision_flip_count"] for m in metrics]
        all_stds = [m["all_equivalents_std"] for m in metrics]
        n = len(metrics)

        # Flip rate: total flips / total variants across samples
        total_variants = sum(m["n_variants"] for m in metrics)
        flip_rate = sum(flips) / total_variants if total_variants > 0 else 0.0

        return {
            "n_samples": n,
            "p_answer_std_mean": sum(stds) / n,
            "p_answer_std_median": _true_median(stds),
            "p_answer_range_mean": sum(ranges) / n,
            "p_answer_range_median": _true_median(ranges),
            "mean_abs_delta_mean": sum(deltas) / n,
            "mean_abs_delta_median": _true_median(deltas),
            "argmax_decision_flip_rate": flip_rate,
            "all_equivalents_std_mean": sum(all_stds) / n,
            "all_equivalents_std_median": _true_median(all_stds),
            "bootstrap_95ci_p_answer_std": _bootstrap_ci(
                stds, statistic="mean", n_resamples=n_bootstrap, seed=bootstrap_seed
            ),
            "bootstrap_95ci_mean_abs_delta": _bootstrap_ci(
                deltas, statistic="mean", n_resamples=n_bootstrap, seed=bootstrap_seed
            ),
        }

    result: dict = {}
    for vtype in ("query", "image"):
        if by_type[vtype]:
            result[f"by_variant_type_{vtype}"] = _aggregate_metrics(by_type[vtype])

    result["by_relation"] = {}
    for rel, metrics in by_relation.items():
        result["by_relation"][rel] = _aggregate_metrics(metrics)

    return {"input_equivariance_consistency": result}


# ── self-test ───────────────────────────────────────────────────────────────


def _generate_synthetic_l2_l3(
    num_samples: int = 10, seed: int = 42, include_missing_endpoint: bool = True
) -> list[dict]:
    """Generate synthetic JSONL for CPU self-test.

    If include_missing_endpoint is True, some L3 groups will lack ratio 0.0 or 1.0
    to test the endpoint-delta-only-on-exact-match logic.
    """
    rng = random.Random(seed)
    records: list[dict] = []

    # L2 samples: proper format L2-{image_id}-{marker}-{source_sample_id}-{terminal}
    for i in range(num_samples):
        src = f"plantdoc-train-{hash(i):016x}-0"
        for idx, (rel, marker, terminal, pa_range) in enumerate(
            [
                ("L2_keep", "keep", "answer", (0.7, 1.0)),
                ("L2_cropout", "cropout", "ladder-000", (0.0, 0.3)),
            ]
        ):
            sid = f"L2-img_{i:04d}-{marker}-{src}-{terminal}"
            pa = rng.uniform(*pa_range)
            records.append(
                {
                    "sample_id": sid,
                    "relation": rel,
                    "image": f"data/source/img_{i}.jpg",
                    "target_label": "test disease",
                    "achieved_visible_ratio": None if rel == "L2_keep" else 0.0,
                    "visible_ratio_bin": None,
                    "original_gt_decision": "answer",
                    "variant_type": "original",
                    "variant_id": "original",
                    "p_answer": pa,
                    "p_wo": rng.uniform(0.0, 0.3),
                    "p_nt": rng.uniform(0.0, 0.3),
                    "predicted_class": 0 if pa > 0.5 else (1 if rng.random() > 0.5 else 2),
                    "decision": "answer" if pa > 0.5 else "abstain",
                    "rank": 0,
                    "original_index": i * 2 + idx,
                }
            )

    # Multi-instance L2: two different objects on same image
    # COCO-style: same annotation ID but different target_labels
    multi_srcs = [
        ("plantdoc-train-multi_aabb-0", "L2_keep"),
        ("plantdoc-train-multi_aabb-0", "L2_cropout"),
        ("coco-train-999-100", "L2_keep"),
        ("coco-train-999-100", "L2_cropout"),
    ]
    base_oi = num_samples * 2
    for j, (src, rel) in enumerate(multi_srcs):
        marker = "keep" if "keep" in rel else "cropout"
        terminal = "answer" if "keep" in rel else "ladder-000"
        sid = f"L2-img_multi-{marker}-{src}-{terminal}"
        pa = 0.8 if "keep" in rel else 0.1
        records.append(
            {
                "sample_id": sid,
                "relation": rel,
                "image": "data/source/img_multi.jpg",
                "target_label": "test disease",
                "achieved_visible_ratio": None if "keep" in rel else 0.0,
                "visible_ratio_bin": None,
                "original_gt_decision": "answer",
                "variant_type": "original",
                "variant_id": "original",
                "p_answer": pa,
                "p_wo": rng.uniform(0.0, 0.3),
                "p_nt": rng.uniform(0.0, 0.3),
                "predicted_class": 0 if pa > 0.5 else 1,
                "decision": "answer" if pa > 0.5 else "abstain",
                "rank": 0,
                "original_index": base_oi + j,
            }
        )

    # Multi-label COCO-style: same annotation, three cropout labels + one keep
    # Simulates: source annotation 156 has hot dog / cup / dining table; keep only hot dog
    coco_multi_label_base = base_oi + len(multi_srcs)
    coco_src = "coco-train-558840-156"
    keep_label = "hot dog"
    cropout_labels = ["hot dog", "cup", "dining table"]
    # Keep record
    records.append(
        {
            "sample_id": f"L2-558840-keep-{coco_src}-answer",
            "relation": "L2_keep",
            "image": "data/source/558840.jpg",
            "target_label": keep_label,
            "achieved_visible_ratio": None,
            "visible_ratio_bin": None,
            "original_gt_decision": "answer",
            "variant_type": "original",
            "variant_id": "original",
            "p_answer": 0.85,
            "p_wo": 0.10,
            "p_nt": 0.05,
            "predicted_class": 0,
            "decision": "answer",
            "rank": 0,
            "original_index": coco_multi_label_base,
        }
    )
    # Cropout records: one per label
    for li, cl in enumerate(cropout_labels):
        records.append(
            {
                "sample_id": f"L2-558840-cropout-{coco_src}-ladder-000",
                "relation": "L2_cropout",
                "image": "data/source/558840.jpg",
                "target_label": cl,
                "achieved_visible_ratio": 0.0,
                "visible_ratio_bin": None,
                "original_gt_decision": "answer",
                "variant_type": "original",
                "variant_id": "original",
                "p_answer": 0.15 if cl == keep_label else 0.70,
                "p_wo": 0.10,
                "p_nt": 0.05,
                "predicted_class": 0,
                "decision": "answer",
                "rank": 0,
                "original_index": coco_multi_label_base + 1 + li,
            }
        )

    # L3 samples
    base_offset = base_oi + len(multi_srcs)
    for i in range(num_samples):
        src = f"plantdoc-train-{hash(i + 100):016x}-0"
        if include_missing_endpoint and i % 3 == 0:
            ratios = (0.25, 0.5, 0.75)
        elif include_missing_endpoint and i % 3 == 1:
            ratios = (0.25, 0.5, 0.75, 1.0)
        else:
            ratios = (0.0, 0.25, 0.5, 0.75, 1.0)

        for ratio in ratios:
            pa = ratio * 0.5 + 0.2 + rng.uniform(-0.05, 0.05)
            pa = max(0.0, min(1.0, pa))
            sid = f"L3-img_{i:04d}-{ratio:.3f}-{src}-ladder-{int(ratio * 100):03d}"
            records.append(
                {
                    "sample_id": sid,
                    "relation": "L3_ladder",
                    "image": f"data/derived/img_{i}_ladder_{int(ratio * 100):03d}.jpg",
                    "target_label": "test disease",
                    "achieved_visible_ratio": ratio,
                    "visible_ratio_bin": f"{ratio:.1f}",
                    "original_gt_decision": "answer" if ratio >= 0.5 else "abstain",
                    "variant_type": "original",
                    "variant_id": "original",
                    "p_answer": pa,
                    "p_wo": rng.uniform(0.0, 0.3),
                    "p_nt": rng.uniform(0.0, 0.3),
                    "predicted_class": 0 if pa > 0.5 else (1 if rng.random() > 0.5 else 2),
                    "decision": "answer" if pa > 0.5 else "abstain",
                    "rank": 0,
                    "original_index": base_offset + i * len(ratios) + int(ratio * 4)
                    if ratio > 0
                    else i * 5,
                }
            )

    # Multi-instance L3: same image, different source IDs
    l3_multi_srcs = [
        ("plantdoc-train-multi_l3_a-0", [0.0, 0.5, 1.0]),
        ("coco-train-888-200", [0.0, 0.5, 1.0]),
    ]
    l3_offset = base_offset + num_samples * 5
    for j, (src, ratios) in enumerate(l3_multi_srcs):
        for k, ratio in enumerate(ratios):
            pa = ratio * 0.5 + 0.2
            sid = f"L3-img_multi_l3-{ratio:.3f}-{src}-ladder-{int(ratio * 100):03d}"
            records.append(
                {
                    "sample_id": sid,
                    "relation": "L3_ladder",
                    "image": "data/derived/img_multi_l3.jpg",
                    "target_label": "test disease",
                    "achieved_visible_ratio": ratio,
                    "visible_ratio_bin": f"{ratio:.1f}",
                    "original_gt_decision": "answer" if ratio >= 0.5 else "abstain",
                    "variant_type": "original",
                    "variant_id": "original",
                    "p_answer": pa,
                    "p_wo": rng.uniform(0.0, 0.3),
                    "p_nt": rng.uniform(0.0, 0.3),
                    "predicted_class": 0 if pa > 0.5 else 1,
                    "decision": "answer" if pa > 0.5 else "abstain",
                    "rank": 0,
                    "original_index": l3_offset + j * len(ratios) + k,
                }
            )

    # Add variant records for consistency analysis (mixed types per sample)
    for i in range(min(3, num_samples)):
        oi = i * 2
        pa_orig = 0.8
        var_specs = [
            ("query", "query_0"),
            ("query", "query_1"),
            ("image", "image_brightness_0.9"),
        ]
        for vtype, vid in var_specs:
            records.append(
                {
                    "sample_id": f"L2-img_{i:04d}-keep-plantdoc-train-xxx-0-answer",
                    "relation": "L2_keep",
                    "image": "data/source/img.jpg",
                    "target_label": "test disease",
                    "achieved_visible_ratio": None,
                    "visible_ratio_bin": None,
                    "original_gt_decision": "answer",
                    "variant_type": vtype,
                    "variant_id": vid,
                    "p_answer": pa_orig + rng.uniform(-0.1, 0.1),
                    "p_wo": rng.uniform(0.0, 0.2),
                    "p_nt": rng.uniform(0.0, 0.2),
                    "predicted_class": 0,
                    "decision": "answer",
                    "rank": 0,
                    "original_index": oi,
                }
            )

    return records


def _self_test(args: argparse.Namespace) -> dict:
    """Run a self-test with synthetic data on CPU."""
    print("[self-test] generating synthetic data...", flush=True)
    records = _generate_synthetic_l2_l3(num_samples=10, seed=42, include_missing_endpoint=True)
    print(f"[self-test] {len(records)} records generated", flush=True)

    n_bootstrap = args.n_bootstrap
    boot_seed = args.bootstrap_seed

    # L2 analysis
    print("[self-test] running L2 analysis...", flush=True)
    l2_result = analyze_l2(records, n_bootstrap=n_bootstrap, bootstrap_seed=boot_seed)
    print(
        f"  L2: n_pairs={l2_result.get('n_pairs', 0)} "
        f"parse_failures={l2_result.get('parse_failures', 0)} "
        f"incomplete_pairs={l2_result.get('incomplete_pairs', 0)} "
        f"conflicting={l2_result.get('conflicting_duplicate_groups', 0)} "
        f"invalid={l2_result.get('invalid_groups', 0)}",
        flush=True,
    )
    # Check COCO multi-label scenario: 1 pair (hot dog keep+cropout), 2 incomplete (cup, dining table)
    # Stable group key for coco_src = L2-558840__coco-train-558840-156
    # With pairing_key = that + target_label, we get:
    #   hot dog: keep + cropout → 1 pair
    #   cup: cropout only → incomplete
    #   dining table: cropout only → incomplete
    # No conflicts because each pairing key has exactly one label
    coco_keep = [
        r for r in records if "558840" in r.get("sample_id", "") and "keep" in r.get("relation", "")
    ]
    coco_cropouts = [
        r
        for r in records
        if "558840" in r.get("sample_id", "") and "cropout" in r.get("relation", "")
    ]
    coco_labels_keep = set(r.get("target_label", "") for r in coco_keep)
    coco_labels_crop = set(r.get("target_label", "") for r in coco_cropouts)
    print(
        f"  COCO multi-label: keep_labels={coco_labels_keep} cropout_labels={coco_labels_crop}",
        flush=True,
    )

    # L3 analysis
    print("[self-test] running L3 analysis...", flush=True)
    from scipy.stats import spearmanr, kendalltau  # noqa: F401

    l3_result = analyze_l3(
        records,
        bootstrap_seed=boot_seed,
        shuffle_rounds=args.shuffle_rounds,
        n_bootstrap=n_bootstrap,
    )
    n_groups = l3_result.get("n_groups", 0)
    n_complete = l3_result.get("n_complete_endpoint_groups", 0)
    print(f"  L3: n_groups={n_groups} n_complete_endpoint={n_complete}", flush=True)

    # Consistency
    print("[self-test] running consistency analysis...", flush=True)
    consistency_result = analyze_input_equivariance_consistency(
        records, n_bootstrap=n_bootstrap, bootstrap_seed=boot_seed
    )
    # Check flip rate denominator
    iec = consistency_result.get("input_equivariance_consistency", {})
    for k in ("by_variant_type_query", "by_variant_type_image", "by_relation"):
        sub = iec.get(k, {})
        if isinstance(sub, dict) and "argmax_decision_flip_rate" in sub:
            print(f"  {k}: flip_rate={sub['argmax_decision_flip_rate']:.4f}", flush=True)

    return {"L2": l2_result, "L3": l3_result, "consistency": consistency_result}


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analyze head probability JSONL from extract_head_probabilities.py"
    )
    ap.add_argument("--input", required=False, help="Input JSONL (extract output)")
    ap.add_argument("--output", required=False, help="Output JSON report path")
    ap.add_argument("--group-csv", default=None, help="Optional CSV output for per-group details")
    ap.add_argument("--bootstrap-seed", type=int, default=42, help="Random seed for bootstrap")
    ap.add_argument(
        "--n-bootstrap",
        type=int,
        default=9999,
        help="Number of bootstrap resamples (default: 9999)",
    )
    ap.add_argument(
        "--shuffle-rounds",
        type=int,
        default=1000,
        help="Number of shuffle rounds for ratio baseline",
    )
    ap.add_argument(
        "--self-test", action="store_true", help="Run CPU self-test with synthetic data"
    )
    args = ap.parse_args()

    if args.self_test:
        result = _self_test(args)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"[self-test] results written to {args.output}")
        return 0

    if not args.input or not args.output:
        ap.error("--input and --output required (or --self-test)")

    input_path = Path(args.input)
    if not input_path.exists():
        ap.error(f"input not found: {input_path}")

    records: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("No records found", flush=True)
        return 1

    print(f"Loaded {len(records)} records from {input_path}", flush=True)

    relations = {r.get("relation") for r in records if r.get("relation")}
    has_l2 = any("L2" in (r or "") for r in relations)
    has_l3 = any("L3" in (r or "") for r in relations)

    result: dict[str, Any] = {
        "metadata": {
            "input": str(input_path),
            "n_records": len(records),
            "relations": list(relations),
            "n_samples_original": len([r for r in records if r.get("variant_id") == "original"]),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
        }
    }

    if has_l2:
        print("Running L2 counterfactual analysis...", flush=True)
        result["L2"] = analyze_l2(
            records, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed
        )
        print(f"  L2: n_pairs={result['L2'].get('n_pairs', 0)}", flush=True)

    if has_l3:
        print("Running L3 dose-response analysis...", flush=True)
        l3_result = analyze_l3(
            records,
            bootstrap_seed=args.bootstrap_seed,
            shuffle_rounds=args.shuffle_rounds,
            n_bootstrap=args.n_bootstrap,
        )
        result["L3"] = l3_result
        n_groups = l3_result.get("n_groups", 0)
        print(f"  L3: n_groups={n_groups}", flush=True)

    # Consistency
    print("Running input equivariance consistency analysis...", flush=True)
    result.update(
        analyze_input_equivariance_consistency(
            records, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed
        )
    )
    print("  consistency done", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Report written to {output_path}", flush=True)

    # Optional group CSV (L3 per-group)
    if args.group_csv and has_l3:
        csv_path = Path(args.group_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        per_group = result.get("L3", {}).get("per_group", {})
        if per_group:
            with csv_path.open("w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(
                    [
                        "group_key",
                        "n_ratio_levels",
                        "spearman_rho",
                        "kendall_tau",
                        "tau_positive",
                        "adjacent_violation_rate",
                        "endpoint_delta_p",
                        "has_complete_endpoints",
                    ]
                )
                for gk, gd in sorted(per_group.items()):
                    writer.writerow(
                        [
                            gk,
                            gd.get("n_ratio_levels"),
                            gd.get("spearman_rho"),
                            gd.get("kendall_tau"),
                            gd.get("tau_positive"),
                            gd.get("adjacent_violation_rate"),
                            gd.get("endpoint_delta_p"),
                            gd.get("has_complete_endpoints"),
                        ]
                    )
            print(f"Group CSV written to {csv_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
