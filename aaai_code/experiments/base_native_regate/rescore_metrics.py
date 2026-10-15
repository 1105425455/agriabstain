"""Recompute evaluation artifacts from saved predictions using current metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from reason_abstain.eval.metrics import aggregate_metrics, metamorphic_test_satisfaction  # noqa: E402
from reason_abstain.eval.parser import normalize_label  # noqa: E402


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def _decision_correct(pred: dict[str, Any], gt: dict[str, Any]) -> bool:
    if not pred.get("valid") or pred.get("decision") != gt.get("decision"):
        return False
    if gt.get("decision") == "answer":
        return normalize_label(pred.get("label")) == normalize_label(gt.get("label"))
    return gt.get("decision") == "abstain" and pred.get("reason") == gt.get("reason")


def _image_key(sample_id: str | None) -> str | None:
    if not sample_id or not sample_id.startswith("L1-"):
        return None
    value = sample_id[3:]
    for marker in ("-real-", "-mismatch-"):
        if marker in value:
            return value.split(marker)[0]
    return None


def build_result(run_dir: Path) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    previous = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    )
    records = _load_records(predictions_path)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    ambiguous = 0
    valid = 0
    decision_correct = 0

    for record in records:
        pred = record["pred"]
        raw_gt = record["gt"]
        if raw_gt.get("decision") is None:
            ambiguous += 1
            continue
        metadata = raw_gt.get("metadata") or {"relation": record.get("relation")}
        gt = {
            **raw_gt,
            "relation": record.get("relation") or metadata.get("relation"),
            "metadata": metadata,
        }
        pairs.append((pred, gt))
        if pred.get("valid"):
            valid += 1
            decision_correct += int(_decision_correct(pred, gt))

    metrics = aggregate_metrics(pairs)
    mts = metamorphic_test_satisfaction(pairs)
    n_records = len(records)
    gate_previous = previous.get("gate", {})
    gate = {
        "format_valid_rate": round(valid / n_records, 4) if n_records else 0.0,
        "decision_accuracy": round(decision_correct / n_records, 4) if n_records else 0.0,
        "format_valid_min": gate_previous.get("format_valid_min", 0.95),
        "decision_acc_min": gate_previous.get("decision_acc_min", 0.8),
    }
    gate["passed"] = (
        gate["format_valid_rate"] >= gate["format_valid_min"]
        and gate["decision_accuracy"] >= gate["decision_acc_min"]
    )

    by_image: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pred, gt in pairs:
        key = _image_key((gt.get("metadata") or {}).get("sample_id"))
        if key is not None:
            by_image.setdefault(key, []).append((pred, gt))
    paired = [
        pair
        for group in by_image.values()
        if {gt.get("decision") for _, gt in group} >= {"answer", "abstain"}
        for pair in group
    ]
    within_image = None
    if paired:
        within_metrics = aggregate_metrics(paired)
        within_image = {
            "n_paired_images": sum(
                {gt.get("decision") for _, gt in group} >= {"answer", "abstain"}
                for group in by_image.values()
            ),
            "n_samples": len(paired),
            "MCC": within_metrics["safeguard"]["MCC"],
            "AAC": within_metrics["safeguard"]["AAC"],
            "FCA": within_metrics["headline"]["FCA"],
            "confusion_matrix": within_metrics["confusion_matrix"],
        }

    return {
        "tag": previous.get("tag", run_dir.name),
        "data": previous.get("data"),
        "adapter": previous.get("adapter"),
        "n_samples": n_records,
        "n_ambiguous_skipped": ambiguous,
        "gate": gate,
        "metrics": metrics,
        "mts": mts,
        "within_image": within_image,
    }


def rescore_run(run_dir: Path, *, check: bool) -> bool:
    expected = build_result(run_dir)
    metrics_path = run_dir / "metrics.json"
    current = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    )
    matches = current == expected
    if check:
        return matches
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(metrics_path)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    stale: list[Path] = []
    for run_dir in args.run_dirs:
        if not rescore_run(run_dir, check=args.check):
            stale.append(run_dir)
    if args.check and stale:
        for path in stale:
            print(f"stale: {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
