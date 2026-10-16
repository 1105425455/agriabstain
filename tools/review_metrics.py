"""复用正式指标，离线生成统一表；不写入原实验目录。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aaai_code"))
from experiments.selective_grounding.analyze_selective_grounding import (  # noqa: E402
    _candidate_prediction,
)
from reason_abstain.eval.metrics import (  # noqa: E402
    aggregate_metrics,
    metamorphic_test_satisfaction,
    score_answerable,
    score_unanswerable,
)


def paired_selection(base: Path, dataset: str) -> dict:
    native = {
        r["sample_id"]: r
        for r in map(
            json.loads,
            (base / "native/sft" / dataset / "predictions.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    regate = {
        r["sample_id"]: r
        for r in map(
            json.loads,
            (base / "regate" / dataset / "predictions.jsonl").read_text().splitlines(),
        )
    }
    assert native.keys() == regate.keys()
    groups = {
        key: {
            "n": 0,
            "supported": 0,
            "regate_correct_supported": 0,
            "native_correct_supported": 0,
            "invalid_candidate": 0,
        }
        for key in ("both", "native_only", "regate_only", "neither")
    }
    for key, row in regate.items():
        n = native[key]
        assert n["gt"] == row["gt"]
        na = bool(n["pred"].get("valid") and n["pred"].get("decision") == "answer")
        ra = row["p_answer"] >= max(row["p_wo"], row["p_nt"])
        group = groups[
            "both"
            if na and ra
            else "native_only"
            if na
            else "regate_only"
            if ra
            else "neither"
        ]
        group["n"] += 1
        group["invalid_candidate"] += not row["answer_candidate"].get("valid", False)
        if row["gt"]["decision"] == "answer":
            group["supported"] += 1
            group["regate_correct_supported"] += ra and score_answerable(
                _candidate_prediction(row), row["gt"], 0.5
            )
            group["native_correct_supported"] += na and score_answerable(
                n["pred"], n["gt"], 0.5
            )
    return groups


def summarize(
    path: Path,
    dataset: str,
    method: str,
    tau: float | None = None,
    routed: bool = False,
) -> dict:
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    pairs = []
    tp = fp = ns = nu = strict = 0
    for row in records:
        gt = row["gt"]
        if gt.get("decision") not in {"answer", "abstain"}:
            continue
        gt = {
            **gt,
            "relation": row.get("relation", gt.get("metadata", {}).get("relation")),
        }
        if routed:
            selected = row["p_answer"] >= (
                tau if tau is not None else max(row["p_wo"], row["p_nt"])
            )
            pred = (
                _candidate_prediction(row)
                if selected
                else {
                    "valid": True,
                    "decision": "abstain",
                    "reason": "wrong_object"
                    if row["p_wo"] >= row["p_nt"]
                    else "no_target",
                }
            )
        else:
            pred = row["pred"]
            selected = bool(pred.get("valid") and pred.get("decision") == "answer")
        supported = gt["decision"] == "answer"
        ns += supported
        nu += not supported
        tp += supported and selected
        fp += not supported and selected
        strict += bool(
            score_answerable(pred, gt, 0.5)
            if supported
            else score_unanswerable(pred, gt)
        )
        pairs.append((pred, gt))
    metrics = aggregate_metrics(pairs)
    mts = metamorphic_test_satisfaction(pairs)
    fn, tn = ns - tp, nu - fp
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    e2e = (
        metrics["grounding_reason"]["E2E_mIoU"]
        if any(gt.get("box") for _, gt in pairs)
        else None
    )
    return {
        "dataset": dataset,
        "method": method,
        "n": len(pairs),
        "supported": ns,
        "unsupported": nu,
        "answered_supported": tp,
        "answered_unsupported": fp,
        "FCA": fp / nu,
        "retention": tp / ns,
        "MCC": (tp * tn - fp * fn) / denominator if denominator else 0,
        "Strict3": metrics["semantic"]["Strict_3"],
        "E2E_mIoU": e2e,
        "MTS_macro": mts["MTS_macro"],
        "FullStrict": strict / len(pairs),
        "supported_answer_conditioned_IoU": e2e / (tp / ns)
        if e2e is not None and tp
        else None,
        "tau": tau,
        "source": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    output = ROOT / "docs/review_evidence"
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "new_reason/outputs/base_native_regate_20260720"
    rows = []
    paired = {}
    for dataset in ("comprehensive", "refcocog", "plantwild"):
        paired[dataset] = paired_selection(base, dataset)
        for method, relative in (
            ("Base", "native/base"),
            ("SFT_native", "native/sft"),
            ("ReGate_argmax", "regate"),
        ):
            rows.append(
                summarize(
                    base / relative / dataset / "predictions.jsonl",
                    dataset,
                    method,
                    routed=method.startswith("ReGate"),
                )
            )
    threshold_path = (
        ROOT / "new_reason/outputs/matched_retention_dev_20260725/analysis/result.json"
    )
    threshold_record = json.loads(threshold_path.read_text())
    assert threshold_record["integrity"]["n_dev"] == 2408
    (output / "threshold_provenance.json").write_text(
        json.dumps(threshold_record, ensure_ascii=False, indent=2) + "\n"
    )
    tau = threshold_record["development_selection"]["selected_threshold"]
    rows.append(
        summarize(
            base / "regate/comprehensive/predictions.jsonl",
            "comprehensive",
            "ReGate_dev_tau",
            tau,
            True,
        )
    )
    rows.append(
        summarize(
            base / "regate/comprehensive/predictions.jsonl",
            "comprehensive",
            "ReGate_tau_0.5_diagnostic",
            0.5,
            True,
        )
    )
    gpt_base = ROOT / "aaai_code/experiments/gpt56_sol_full/outputs/decision_prompt_v1"
    for directory in sorted(gpt_base.iterdir()):
        p = directory / "analysis/predictions.jsonl"
        if p.exists():
            rows.append(summarize(p, directory.name, "GPT_reference"))
    prompt_base = ROOT / "new_reason/baselines/prompt_cot/outputs"
    for method in (
        "native_lm",
        "prompt_basic",
        "prompt_reward",
        "prompt_reward_norms",
        "cot_structured",
    ):
        for dataset in ("test_full", "ood_refcocog", "plantwild_real_closed12_v1"):
            p = prompt_base / method / dataset / "predictions.jsonl"
            if p.exists():
                rows.append(summarize(p, dataset, f"Prompt_{method}"))
    grouping_base = (
        ROOT / "aaai_code/experiments/grouping_ablation/artifacts/evaluation"
    )
    for p in sorted(grouping_base.glob("*/*/predictions.jsonl")):
        rows.append(summarize(p, p.parent.name, f"Independent_{p.parent.parent.name}"))
    # L1/L2/L3 的正式 grouped ReGate 输出保存于最终模型评测目录。
    checkpoint = (
        ROOT
        / "new_reason/outputs/sft_runs/fusion_3reason/fusion_3reason_v3_hl04/checkpoint-444"
    )
    for dataset in ("L1_discrim", "L2_evidence", "L3_dose"):
        p = checkpoint / "eval_faithful_v1" / dataset / "predictions.jsonl"
        rows.append(summarize(p, dataset, "Grouped_saved_ReGate"))
    (output / "unified_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "paired_selection.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2) + "\n"
    )
    with (output / "unified_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            row["dataset"],
            row["method"],
            {
                k: row[k]
                for k in (
                    "FCA",
                    "retention",
                    "MCC",
                    "Strict3",
                    "E2E_mIoU",
                    "MTS_macro",
                    "FullStrict",
                )
            },
        )


if __name__ == "__main__":
    main()
