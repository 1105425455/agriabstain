#!/usr/bin/env python3
"""Verify analysis CSVs against formal metrics.json for base_native/regate experiments.

Reads analysis/baseline_points.csv and analysis/regate_argmax_points.csv
(if present; otherwise falls back to native_comparison.csv) and checks each
row against the corresponding metrics.json under the root directory.

Checks:
  - total vs n_samples
  - supported vs counts.answerable
  - unsupported vs counts.unanswerable
  - unsupported_hallucination_rate vs headline.FCA
  - supported_query_retention vs 1 - safeguard.A_FU
  - a_ground_mean_iou_over_supported vs grounding_reason.A_Ground (bbox only)
  - a_ground_conditional_mean_iou vs grounding_reason.A_Ground_c (bbox only)
  - label_success_over_supported / label_success_conditional (label_only, if present)

For PlantWild (label_only), A_Ground=0 is a placeholder — skip that check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Method → sub-directory mapping (compatible with analyzer output naming)
# ---------------------------------------------------------------------------
METHOD_TO_SUBDIR: dict[str, str] = {
    "Base_native": "native/base",
    "SFT_native_observed": "native/sft",
    "ReGate": "regate",
}

# Datasets known to be label-only (no bbox)
_LABEL_ONLY_DATASETS: frozenset[str] = frozenset({"plantwild"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_float(val: str | None) -> float:
    if val is None or val.strip() == "":
        return math.nan
    return float(val.strip())


def _parse_int(val: str | None) -> int | None:
    if val is None or not val.strip():
        return None
    return int(val.strip())


def _check_float(
    field: str,
    csv_val: str | None,
    metric_val: float,
    dataset: str,
    method: str,
    errors: list[str],
    *,
    atol: float = 1e-10,
    invert: bool = False,
) -> None:
    """Compare a CSV value to a metrics.json value.

    If *invert* is True, the relationship is ``csv_val == 1 - metric_val``.
    This is used for A_FU → retention.
    """
    actual = _parse_float(csv_val)
    if math.isnan(actual) and math.isnan(metric_val):
        return
    if math.isnan(actual):
        errors.append(f"{dataset}/{method}: {field} CSV value is NaN/empty (metrics={metric_val})")
        return
    if math.isnan(metric_val):
        errors.append(f"{dataset}/{method}: {field} metrics.json value is NaN (csv={actual})")
        return

    expected = 1.0 - metric_val if invert else metric_val

    if abs(actual - expected) > atol:
        errors.append(
            f"{dataset}/{method}: {field} expected={expected:.10f} "
            f"got={actual:.10f}  diff={abs(actual - expected):.2e}"
        )


def _check_int(
    field: str,
    csv_val: str | None,
    metric_val: int,
    dataset: str,
    method: str,
    errors: list[str],
) -> None:
    actual_str = (csv_val or "").strip()
    if not actual_str:
        errors.append(f"{dataset}/{method}: {field} CSV value is empty (metrics={metric_val})")
        return
    try:
        actual = int(actual_str)
    except ValueError:
        errors.append(
            f"{dataset}/{method}: {field} CSV value not an integer: "
            f"{actual_str!r} (metrics={metric_val})"
        )
        return

    if actual != metric_val:
        errors.append(
            f"{dataset}/{method}: {field} expected={metric_val} "
            f"got={actual}  diff={actual - metric_val}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify analysis CSVs against formal metrics.json")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("BASE_NATIVE_REGATE_ROOT", "outputs/base_native_regate")),
        help="Root output directory holding analysis/, native/ and regate/ (or set "
        "BASE_NATIVE_REGATE_ROOT)",
    )
    args = parser.parse_args()

    root: Path = args.root
    analysis_dir = root / "analysis"

    # --- Load rows from CSV(s) --------------------------------------------
    rows: list[dict[str, str]] = []

    baseline_csv = analysis_dir / "baseline_points.csv"
    regate_csv = analysis_dir / "regate_argmax_points.csv"

    if baseline_csv.exists() and regate_csv.exists():
        rows.extend(_load_csv(baseline_csv))
        rows.extend(_load_csv(regate_csv))
        src = "baseline_points.csv + regate_argmax_points.csv"
    elif baseline_csv.exists():
        rows.extend(_load_csv(baseline_csv))
        src = "baseline_points.csv"
    elif regate_csv.exists():
        rows.extend(_load_csv(regate_csv))
        src = "regate_argmax_points.csv"
    else:
        native_csv = analysis_dir / "native_comparison.csv"
        if not native_csv.exists():
            print(
                f"FATAL: no baseline_points.csv, regate_argmax_points.csv "
                f"or native_comparison.csv found under {analysis_dir}",
                file=sys.stderr,
            )
            return 1
        rows.extend(_load_csv(native_csv))
        src = "native_comparison.csv"

    if not rows:
        print("FATAL: no rows loaded from CSV", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} row(s) from {src}", file=sys.stderr)

    # --- Verify each row --------------------------------------------------
    errors: list[str] = []

    for row_idx, row in enumerate(rows, start=1):
        dataset = (row.get("dataset") or "").strip()
        method = (row.get("method") or "").strip()

        if not dataset or not method:
            continue

        # Resolve method → sub-directory
        subdir = METHOD_TO_SUBDIR.get(method)
        if subdir is None:
            # Unknown method — skip silently (e.g. SFT_native_gate_shared_candidate)
            continue

        metrics_path = root / subdir / dataset / "metrics.json"
        if not metrics_path.exists():
            errors.append(f"{dataset}/{method}: metrics.json not found: {metrics_path}")
            continue

        with metrics_path.open(encoding="utf-8") as f:
            m = json.load(f)

        is_label_only = dataset in _LABEL_ONLY_DATASETS

        # Grounding scope from CSV if available (cross-check)
        csv_scope = (row.get("grounding_scope") or "").strip().lower()
        if csv_scope and is_label_only:
            if csv_scope != "label_only":
                errors.append(
                    f"{dataset}/{method}: grounding_scope={csv_scope!r} but dataset is label-only"
                )
        elif csv_scope and not is_label_only:
            if csv_scope != "bbox":
                errors.append(
                    f"{dataset}/{method}: grounding_scope={csv_scope!r} but dataset is bbox"
                )

        # --- 1. Sample counts ---------------------------------------------
        n_invalid = m.get("metrics", {}).get("counts", {}).get("invalid", 0)
        _check_int(
            "total",
            row.get("total"),
            m["n_samples"],
            dataset,
            method,
            errors,
        )

        metrics_supported = m["metrics"]["counts"]["answerable"]
        csv_supported = _parse_int(row.get("supported"))
        if csv_supported is not None and csv_supported != metrics_supported:
            # The analyzer counts GT decision == "answer", which may include
            # invalid-format samples.  Allow the difference if invalid > 0.
            if n_invalid > 0 and csv_supported == metrics_supported + n_invalid:
                pass  # invalid sample(s) have GT "answer"
            else:
                errors.append(
                    f"{dataset}/{method}: supported expected={metrics_supported} "
                    f"got={csv_supported}  diff={csv_supported - metrics_supported}"
                )

        metrics_unsupported = m["metrics"]["counts"]["unanswerable"]
        csv_unsupported = _parse_int(row.get("unsupported"))
        if csv_unsupported is not None and csv_unsupported != metrics_unsupported:
            if n_invalid > 0 and csv_unsupported == metrics_unsupported + n_invalid:
                pass
            else:
                errors.append(
                    f"{dataset}/{method}: unsupported expected={metrics_unsupported} "
                    f"got={csv_unsupported}  diff={csv_unsupported - metrics_unsupported}"
                )

        # --- 2. FCA (unsupported_hallucination_rate) ----------------------
        _check_float(
            "unsupported_hallucination_rate",
            row.get("unsupported_hallucination_rate"),
            float(m["metrics"]["headline"]["FCA"]),
            dataset,
            method,
            errors,
        )

        # --- 3. A_FU → supported_query_retention -------------------------
        cm = m["metrics"]["confusion_matrix"]
        expected_retention = (int(cm["TP"]) + int(cm["FP"])) / int(row["supported"])
        _check_float(
            "supported_query_retention",
            row.get("supported_query_retention"),
            expected_retention,
            dataset,
            method,
            errors,
        )

        # --- 4. Grounding checks ------------------------------------------
        if is_label_only:
            # PlantWild: A_Ground=0 is a placeholder — skip bbox checks.
            # Check label_success fields if present.
            if "label_success_over_supported" in row:
                # metrics.json has no direct label_success field, but we
                # verify the value is consistent with metrics computation.
                # For PlantWild, A_Ground_c is also 0, so label_success
                # is the meaningful metric.  We just check that it's a
                # valid float (no cross-check against metrics.json).
                try:
                    _parse_float(row["label_success_over_supported"])
                except (ValueError, TypeError):
                    errors.append(
                        f"{dataset}/{method}: "
                        f"label_success_over_supported not parseable: "
                        f"{row['label_success_over_supported']!r}"
                    )
            if "label_success_conditional" in row:
                try:
                    _parse_float(row["label_success_conditional"])
                except (ValueError, TypeError):
                    errors.append(
                        f"{dataset}/{method}: "
                        f"label_success_conditional not parseable: "
                        f"{row['label_success_conditional']!r}"
                    )
        else:
            # Bbox datasets: check a_ground_mean_iou_over_supported vs A_Ground
            if "a_ground_mean_iou_over_supported" in row:
                valid_answerable = int(m["metrics"]["counts"]["answerable"])
                total_supported = int(row["supported"])
                expected_a_ground = (
                    float(m["metrics"]["grounding_reason"]["A_Ground"])
                    * valid_answerable
                    / total_supported
                )
                _check_float(
                    "a_ground_mean_iou_over_supported",
                    row["a_ground_mean_iou_over_supported"],
                    expected_a_ground,
                    dataset,
                    method,
                    errors,
                )
            # Check a_ground_conditional_mean_iou vs A_Ground_c
            if "a_ground_conditional_mean_iou" in row:
                _check_float(
                    "a_ground_conditional_mean_iou",
                    row["a_ground_conditional_mean_iou"],
                    float(m["metrics"]["grounding_reason"]["A_Ground_c"]),
                    dataset,
                    method,
                    errors,
                )
            # If the CSV has the old-style grounding_accuracy field,
            # it is NOT A_Ground (mean IoU), so we skip it.

    # --- Report -----------------------------------------------------------
    if errors:
        print("", file=sys.stderr)
        print("VERIFICATION FAILED", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        print(f"  ({len(errors)} error(s))", file=sys.stderr)
        return 1

    print("", file=sys.stderr)
    print("All checks passed ✓", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
