"""逐个历史重训run复算comprehensive结果并核对原汇总。"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/user/huangyongbin/newrepo/agriabstain")
sys.path.insert(0, str(ROOT / "aaai_code"))
from experiments.base_native_regate.rescore_metrics import build_result  # noqa: E402


def main() -> None:
    saved = json.loads(
        (SOURCE / "h27experiment/summary/multiseed_summary.json").read_text()
    )
    rows = []
    for condition in ("grouped", "independent"):
        for seed in (7, 21, 42):
            for interface in ("native", "regate"):
                path = (
                    SOURCE
                    / f"h27experiment/runs/{condition}/seed-{seed}/evaluation/{interface}/comprehensive"
                )
                result = build_result(path)
                m = result["metrics"]
                values = {
                    "FCA": m["headline"]["FCA"],
                    "retention": 1 - m["safeguard"]["A_FU"],
                    "MCC": m["safeguard"]["MCC"],
                    "Strict3": m["semantic"]["Strict_3"],
                    "E2E_mIoU": m["grounding_reason"]["E2E_mIoU"],
                    "MTS_macro": result["mts"]["MTS_macro"],
                }
                original = next(
                    r
                    for r in saved["rows"]
                    if r["condition"] == condition
                    and r["seed"] == seed
                    and r["dataset"] == "comprehensive"
                    and r["interface"]
                    == ("mlp_regate" if interface == "regate" else "native")
                )
                for key in ("FCA", "MCC", "E2E_mIoU", "MTS_macro"):
                    assert abs(values[key] - original["metrics"][key]) < 1e-12, (
                        condition,
                        seed,
                        interface,
                        key,
                    )
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "interface": interface,
                        **values,
                        "predictions": str(path / "predictions.jsonl"),
                        "sha256": hashlib.sha256(
                            (path / "predictions.jsonl").read_bytes()
                        ).hexdigest(),
                    }
                )
    summary = {}
    for condition in ("grouped", "independent"):
        for interface in ("native", "regate"):
            group = [
                r
                for r in rows
                if r["condition"] == condition and r["interface"] == interface
            ]
            summary[f"{condition}/{interface}"] = {
                key: {
                    "mean": statistics.mean(r[key] for r in group),
                    "sample_sd": statistics.stdev(r[key] for r in group),
                }
                for key in (
                    "FCA",
                    "retention",
                    "MCC",
                    "Strict3",
                    "E2E_mIoU",
                    "MTS_macro",
                )
            }
    output = ROOT / "docs/review_evidence/frozen_and_prior/prior_raw_verification.json"
    output.write_text(
        json.dumps(
            {"rows": rows, "summary": summary, "summary_matches_original": True},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
