from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from experiments.selective_grounding.rescore_prompt_interfaces import (
    build_artifact as build_prompt_interface_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = (
    REPO_ROOT
    / "paper"
    / "Learning_to_Locate_or_Abstain__A_Discriminative_Head_Architecture_for_Reason_Attributed_Selective_Visual_Grounding__1_"
)
ANALYSIS_DIR = REPO_ROOT / "new_reason" / "outputs" / "base_native_regate_20260720" / "analysis"
GPT_DIR = (
    REPO_ROOT / "aaai_code" / "experiments" / "gpt56_sol_full" / "outputs" / "decision_prompt_v1"
)
COPY_TEX = PAPER_DIR / "main.tex"
SUPPLEMENT_TEX = PAPER_DIR / "technical_supplement.tex"
EVIDENCE_DIR = REPO_ROOT / "new_reason" / "outputs" / "evidence_judge_full_4090_b1_v2_20260719"
LAYER_RESULT = (
    REPO_ROOT / "new_reason" / "outputs" / "head_boundary_refit" / "layer_scan" / "best_layer.json"
)
MARGIN_ROOT = REPO_ROOT / "new_reason" / "outputs" / "sft_runs" / "paired_retrain"
PROMPT_ROOT = REPO_ROOT / "new_reason" / "baselines" / "prompt_cot" / "outputs"
REGATE_ROOT = REPO_ROOT / "new_reason" / "outputs" / "base_native_regate_20260720" / "regate"
PROMPT_INTERFACE_ARTIFACT = (
    REPO_ROOT
    / "aaai_code"
    / "experiments"
    / "selective_grounding"
    / "main"
    / "prompt_interface_metrics_fixed_gt.json"
)
GROUPING_ABLATION_ARTIFACT = (
    REPO_ROOT
    / "aaai_code"
    / "experiments"
    / "grouping_ablation"
    / "main"
    / "grouping_ablation_fixed_gt.json"
)
CAPABILITY_AUDIT = (
    REPO_ROOT
    / "aaai_code"
    / "benchmark"
    / "annotations"
    / "audits"
    / "PLANTDOC_CAPABILITY_GATE_AUDIT.json"
)
SFT_TRAIN = (
    REPO_ROOT / "aaai_code" / "benchmark" / "annotations" / "sft" / "sft_train_with_source.json"
)
SPLIT_MANIFEST = (
    REPO_ROOT / "aaai_code" / "benchmark" / "annotations" / "splits" / "split_manifest.json"
)
SPLIT_TRAIN = SPLIT_MANIFEST.parent / "train.jsonl"
EVAL_DIR = REPO_ROOT / "aaai_code" / "benchmark" / "annotations" / "eval"
RAW_DIR = REPO_ROOT / "aaai_code" / "benchmark" / "annotations" / "raw"
CONFIG_DIR = REPO_ROOT / "aaai_code" / "configs"
TRI_HEAD_SOURCE = (
    REPO_ROOT / "aaai_code" / "training" / "qwenvl" / "model" / "qwen3vl_dual_head_3reason.py"
)
TRAINER_SOURCE = REPO_ROOT / "aaai_code" / "training" / "qwenvl" / "train" / "train_qwen.py"
RUN_TRAIN = REPO_ROOT / "aaai_code" / "training" / "sft_final" / "run_train.sh"


def _rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["dataset"], row["method"]): row for row in csv.DictReader(handle)}


def _copy_text() -> str:
    """Main paper and technical supplement, concatenated.

    The claims checked here are split across the two submission files, so they are
    joined before matching.
    """
    text = COPY_TEX.read_text(encoding="utf-8") + "\n" + SUPPLEMENT_TEX.read_text(encoding="utf-8")
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _table_block(text: str, label: str) -> str:
    """The whole float carrying ``label``.

    The label may sit either before the tabular body or after the caption, so the
    block is taken from the enclosing ``\\begin{table}`` to its ``\\end{table}``.
    """
    anchor = text.index(f"\\label{{{label}}}")
    start = text.rindex("\\begin{table", 0, anchor)
    end = text.index("\\end{table", anchor)
    return text[start:end]


def _row_values(block: str, label: str) -> tuple[float, ...]:
    match = re.search(rf"(?m)^{re.escape(label)}\s*&(.+?)\\\\", block, flags=re.DOTALL)
    assert match is not None, f"missing table row: {label}"
    return tuple(float(value) for value in re.findall(r"-?(?:\d+)?\.\d+", match.group(1)))


def _comparison(probe: str) -> dict:
    return json.loads(
        (GPT_DIR / probe / "analysis" / "comparison.json").read_text(encoding="utf-8")
    )


def _round_display(value: float, places: int = 3) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def test_copy_matches_canonical_fixed_points() -> None:
    text = _copy_text()
    rows = _rows(ANALYSIS_DIR / "native_comparison.csv")

    expected = {
        "comprehensive": {
            "fca": (0.211, 0.094, 0.132),
            "retention": (0.833, 0.851, 0.976),
            "strict3": (0.660, 0.819, 0.856),
            "e2e": (0.656, 0.705, 0.701),
        },
        "plantwild": {
            "fca": (0.360, 0.155, 0.269),
            "retention": (0.792, 0.764, 0.854),
            "strict3": (0.375, 0.647, 0.694),
        },
        "refcocog": {
            "fca": (0.044, 0.044, 0.083),
            "retention": (0.983, 0.901, 1.000),
            "strict3": (0.762, 0.834, 0.820),
            "e2e": (0.525, 0.645, 0.645),
        },
    }
    methods = ("Base_native", "SFT_native_observed", "ReGate")
    fields = {
        "fca": "unsupported_hallucination_rate",
        "retention": "supported_query_retention",
        "strict3": "strict_3_accuracy",
        "e2e": "a_ground_mean_iou_over_supported",
    }
    for dataset, metrics in expected.items():
        for metric, displayed in metrics.items():
            actual = tuple(
                round(float(rows[(dataset, method)][fields[metric]]), 3) for method in methods
            )
            assert actual == displayed

    assert "from $0.525$ to $0.645$" in text
    assert "improves E2E-mIoU from $0.656$ to $0.705$ in-domain" in text
    assert "T7 inclusion/ladder $0.750/0.781$" in text
    assert _row_values(_table_block(text, "tab:gpt56_full"), "L1")[7] == 0.649
    assert "from $0.525$ to $0.649$" not in text


def test_copy_matches_comprehensive_gpt_comparison() -> None:
    text = _copy_text()
    comparison = json.loads(
        (GPT_DIR / "test_full" / "analysis" / "comparison.json").read_text(encoding="utf-8")
    )
    gpt = comparison["gpt56_sol"]["metrics"]
    regate = comparison["regate_reference"]["metrics"]
    values = (
        gpt["headline"]["FCA"],
        gpt["safeguard"]["A_FU"],
        gpt["safeguard"]["MCC"],
        gpt["semantic"]["Strict_3"],
        gpt["grounding_reason"]["E2E_mIoU"],
        regate["headline"]["FCA"],
        regate["safeguard"]["A_FU"],
        regate["safeguard"]["MCC"],
        regate["semantic"]["Strict_3"],
        regate["grounding_reason"]["E2E_mIoU"],
    )
    rounded = tuple(round(value, 3) for value in values)
    assert rounded == (0.302, 0.076, 0.639, 0.664, 0.625, 0.132, 0.024, 0.849, 0.856, 0.701)
    assert _row_values(_table_block(text, "tab:gpt56_full"), "Comp.")[1::2] == rounded[:5]
    assert "target-unavailable accuracy ($0.951$ versus $0.028$)" in text
    assert "visual-contradiction accuracies are $0.317$ and $0.542$" in text


def test_copy_fixed_point_table_matches_canonical_analysis() -> None:
    text = _copy_text()
    block = _table_block(text, "tab:main_results")
    rows = _rows(ANALYSIS_DIR / "native_comparison.csv")
    datasets = {
        "Comprehensive": "comprehensive",
        "PlantWild": "plantwild",
        "RefCOCOg": "refcocog",
    }
    methods = {
        "Base": "Base_native",
        "SFT native": "SFT_native_observed",
        r"\Method": "ReGate",
    }
    for label, dataset in datasets.items():
        dataset_block = block[block.index(f"{label} & Base") :].split(r"\midrule")[0]
        for interface, method in methods.items():
            row = rows[(dataset, method)]
            expected = [
                _round_display(float(row["unsupported_hallucination_rate"])),
                _round_display(float(row["supported_query_retention"])),
                _round_display(float(row["strict_3_accuracy"])),
            ]
            if row["grounding_scope"] == "bbox":
                expected.append(_round_display(float(row["a_ground_mean_iou_over_supported"])))
            row_label = f"{label} & {interface}" if interface == "Base" else f" & {interface}"
            assert _row_values(dataset_block, row_label) == tuple(expected)


def test_copy_gpt_tables_match_all_comparisons() -> None:
    text = _copy_text()
    full_block = _table_block(text, "tab:gpt56_full")
    probes = {
        "Comp.": "test_full",
        "L1": "L1_discrim",
        "L2": "L2_evidence",
        "L3": "L3_dose",
        "Ref.": "ood_refcocog",
        "PW": "plantwild_real_closed12_v1",
    }
    fields = (
        ("headline", "FCA"),
        ("safeguard", "A_FU"),
        ("safeguard", "MCC"),
        ("semantic", "Strict_3"),
    )
    for label, probe in probes.items():
        comparison = _comparison(probe)
        regate = comparison["regate_reference"]["metrics"]
        gpt = comparison["gpt56_sol"]["metrics"]
        expected: list[float] = []
        for section, key in fields:
            expected.extend((round(regate[section][key], 3), round(gpt[section][key], 3)))
        if label != "PW":
            expected.extend(
                (
                    round(regate["grounding_reason"]["E2E_mIoU"], 3),
                    round(gpt["grounding_reason"]["E2E_mIoU"], 3),
                )
            )
        assert _row_values(full_block, label) == tuple(expected)

    paired = _comparison("test_full")["paired_gpt_minus_regate"]
    for metric, displayed in {
        "FCA": (0.170, 0.147, 0.196),
        "A_FU": (0.051, 0.036, 0.067),
        "Strict_3": (-0.192, -0.213, -0.170),
        "E2E_mIoU": (-0.077, -0.094, -0.058),
    }.items():
        result = paired[metric]
        actual = tuple(
            round(result[key], 3) for key in ("difference_gpt_minus_regate", "ci_low", "ci_high")
        )
        assert actual == displayed
    assert "$0.170$ (95\\% CI $[0.147,0.196]$)" in text
    assert "$0.051$ ($[0.036,0.067]$)" in text
    assert "$-0.192$ ($[-0.213,-0.170]$)" in text
    assert "$-0.077$ ($[-0.094,-0.058]$)" in text


def test_copy_prompt_interface_table_matches_raw_metrics() -> None:
    text = _copy_text()
    block = _table_block(text, "tab:routing_interfaces")
    artifact = json.loads(PROMPT_INTERFACE_ARTIFACT.read_text(encoding="utf-8"))
    rebuilt = build_prompt_interface_artifact(prompt_root=PROMPT_ROOT, regate_root=REGATE_ROOT)
    assert rebuilt == artifact
    datasets = ("comprehensive", "refcocog", "plantwild_closed12")
    rows = {
        r"\Method": "regate",
        "Native LM": "native_lm",
        "Direct": "direct",
        "Reward": "reward",
        "R+E": "reward_evidence",
        "S-CoT": "structured_cot",
    }
    for label, method in rows.items():
        expected: list[float] = []
        for dataset in datasets:
            metrics = artifact["datasets"][dataset]["methods"][method]["metrics"]
            expected.extend(
                (
                    round(metrics["safeguard"]["MCC"], 3),
                    round(metrics["safeguard"]["A_FU"], 3),
                    round(metrics["semantic"]["Strict_3"], 3),
                )
            )
        assert _row_values(block, label) == tuple(expected)
    assert r"\Method & \textbf{.849/.024/.856}" in block
    assert r"\textbf{.920/.000}/.820" in block
    assert r".560/\textbf{.146}/.694" in block
    assert r".858/.099/\textbf{.834}" in block
    assert r"\textbf{.602}/.236/.647" in block
    assert r".581/.250/\textbf{.755}" in block


def test_copy_grouping_ablation_matches_canonical_artifact() -> None:
    text = _copy_text()
    artifact = json.loads(GROUPING_ABLATION_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["primary_mts"] == "MTS_macro"
    assert all(artifact["data_control"]["checks"].values())
    rows = {(row["interface"], row["dataset"]): row for row in artifact["rows"]}

    appendix_block = _table_block(text, "tab:grouping_ablation")
    appendix_rows = [
        ("Comp.-Native", "native", "comprehensive"),
        ("Ref.-Native", "native", "refcocog"),
        ("PW-Native", "native", "plantwild"),
        ("Comp.-Gate", "regate", "comprehensive"),
        ("Ref.-Gate", "regate", "refcocog"),
        ("PW-Gate", "regate", "plantwild"),
        ("L1-Gate", "regate", "L1_discrim"),
        ("L2-Gate", "regate", "L2_evidence"),
        ("L3-Gate", "regate", "L3_dose"),
    ]
    for label, interface, dataset in appendix_rows:
        row = rows[(interface, dataset)]
        expected = tuple(
            _round_display(condition[metric])
            for metric in ("MTS_macro", "MCC", "FCA", "A_FU")
            for condition in (row["grouped"], row["independent_negative"])
        )
        assert _row_values(appendix_block, label) == expected

    controls = artifact["data_control"]["counts"]
    assert controls["positive_rows"] == 2703
    assert controls["replaced_negative_rows"] == 4369
    assert controls["independent_negative_sources"] == 1776
    assert "All $2{,}703$ supported training instances remain unchanged" in text
    assert "For the $4{,}369$ unsupported instances" in text
    assert "contains $1{,}776$ image sources" in text
    selections = artifact["training"]["checkpoint_selection"]
    assert selections["grouped"]["best_step"] == 444
    assert selections["independent_negative"]["best_step"] == 222
    assert selections["grouped"]["policy"] == selections["independent_negative"]["policy"]
    assert selections["grouped"]["candidate_steps"] == [111, 222, 333, 444, 555, 666]
    assert selections["independent_negative"]["candidate_steps"] == [
        111,
        222,
        333,
        444,
        555,
        666,
    ]
    # 当前稿以 dev-loss 选点描述协议；历史 retrospective 叙述只由原产物追溯。
    assert "minimum teacher-forced language-model loss" in text
    assert "the six epoch checkpoints" in text
    assert "hyperparameters, and seed 42" in text
    # RefCOCOg 的 MCC 对比已在上方九行表格逐项检查。
    assert r"reduces PlantWild A\_FU from $0.396$ to $0.146$" in text
    assert "lacks a counterfactual-removal or independently sampled-negative ablation" not in text


def test_copy_representation_and_evidence_metrics_match_artifacts() -> None:
    text = _copy_text()
    probe_results = json.loads(
        (EVIDENCE_DIR / "repr" / "probe_results.json").read_text(encoding="utf-8")
    )["results"]
    representations = {
        "Pre-response prompt token": "last_token",
        "Visual-token mean": "visual_mean",
        "Visual-token max": "visual_max",
        "Visual query-attention": "visual_query_attn",
    }
    for label, key in representations.items():
        aggregate = probe_results[key]["aggregate"]
        expected = (
            round(aggregate["test_mcc_mean"], 3),
            round(aggregate["test_mcc_std"], 3),
        )
        displayed = rf"${expected[0]:.3f}\pm{expected[1]:.3f}$"
        assert (
            f"MCC {displayed}" if key == "last_token" else f"{displayed} for {label.lower()}"
        ) in text

    l2 = json.loads(
        (EVIDENCE_DIR / "evidence" / "L2" / "test" / "analysis.json").read_text(encoding="utf-8")
    )["L2"]
    assert l2["n_pairs"] == 148
    assert l2["n_total_groups"] == 397
    assert l2["incomplete_pairs"] == 249
    assert l2["deduplicated_records"] == 245
    assert round(l2["delta_p_answer_mean"], 3) == 0.786
    assert tuple(round(l2["bootstrap_95ci"][key], 3) for key in ("low", "high")) == (
        0.720,
        0.840,
    )
    assert round(100 * l2["correct_direction_ratio"], 1) == 96.6
    assert "Target removal lowers $p_A$ by $0.786$ on average" in text
    assert "retains 148 groups with both target-retaining and crop-out endpoints" in text
    assert "249 source groups are incomplete after probe balancing" in text
    assert "95\\% CI $[0.720,0.840]$" in text
    assert "expected direction in $96.6\\%$" in text

    l3 = json.loads(
        (EVIDENCE_DIR / "evidence" / "L3" / "test" / "analysis.json").read_text(encoding="utf-8")
    )["L3"]
    assert l3["n_groups"] == 183
    assert round(l3["mean_spearman_rho"], 3) == 0.750
    assert tuple(round(l3["spearman_rho_bootstrap_95ci"][key], 3) for key in ("low", "high")) == (
        0.694,
        0.794,
    )
    assert round(l3["mean_kendall_tau"], 3) == 0.682
    assert tuple(round(l3["kendall_tau_bootstrap_95ci"][key], 3) for key in ("low", "high")) == (
        0.625,
        0.729,
    )
    assert round(100 * l3["mean_adjacent_violation_rate"], 1) == 21.4
    assert tuple(
        round(100 * l3["violation_rate_bootstrap_95ci"][key], 1) for key in ("low", "high")
    ) == (18.3, 24.9)
    assert "Across 183 L3 groups" in text
    assert "Spearman $\\rho=0.750$ (95\\% CI $[0.694,0.794]$)" in text
    assert "Kendall $\\tau=0.682$ (95\\% CI $[0.625,0.729]$)" in text
    assert "$21.4\\%$ (95\\% CI $[18.3\\%,24.9\\%]$)" in text


def test_copy_relation_layer_and_matched_retention_metrics_match_artifacts() -> None:
    text = _copy_text()
    comparison = _comparison("test_full")
    mts = comparison["regate_reference"]["mts"]
    relation_groups = {
        "T1 paraphrase": ("T1",),
        "T2 photometric/crop-out": ("T2", "T2_occluded"),
        "T3 geometry/crop-out": ("T3", "T3_occluded"),
        "T4 control": ("T4_control",),
        "T5 absent query": ("T5",),
        "T6 confusable query": ("T6",),
        "T7 inclusion/ladder": ("T7_include", "T7_ladder"),
        "base crop-out": ("base_occluded",),
    }
    assert {key for keys in relation_groups.values() for key in keys} == set(mts["per_relation"])
    for label, keys in relation_groups.items():
        match = re.search(rf"{re.escape(label)} \$([\d./]+)\$", text)
        assert match is not None
        assert tuple(map(float, match.group(1).split("/"))) == tuple(
            _round_display(mts["per_relation"][key]) for key in keys
        )
    assert f"macro MTS ${_round_display(mts['MTS_macro']):.3f}$" in text

    layer = json.loads(LAYER_RESULT.read_text(encoding="utf-8"))
    assert layer["npz_index"] == 26
    assert layer["transformer_layer"] == 27
    assert layer["best_layer_entry"]["n_train"] == 1599
    assert layer["best_layer_entry"]["n_dev"] == 137
    assert round(layer["best_layer_entry"]["min_score"], 4) == 0.9574
    assert "$1{,}599$ probe-fitting and 137 probe-development records" in text
    assert "$m_{27}=0.9574$" in text

    with (ANALYSIS_DIR / "matched_retention.csv").open(encoding="utf-8", newline="") as handle:
        matched = {(row["dataset"], row["baseline_method"]): row for row in csv.DictReader(handle)}
    for dataset, expected in {
        "comprehensive": (0.036, 0.176, 0.866),
        "plantwild": (0.197, 0.358, 0.713),
        "refcocog": (0.050, 0.273, 0.829),
    }.items():
        row = matched[(dataset, "SFT_native_observed")]
        actual = tuple(
            round(float(row[key]), 3)
            for key in (
                "empirical_unsupported_hallucination_rate",
                "empirical_selective_risk",
                "empirical_strict_3_accuracy",
            )
        )
        assert actual == expected
    assert "$0.197/0.155$ for ReGate/SFT native on PlantWild and $0.050/0.044$ on RefCOCOg" in text
    assert "comprehensive FCA is $0.036$ versus $0.094$" in text
    assert "these curve-matched values characterize routing behavior" in text
    assert "selective risk changes" not in text


def test_copy_margin_kl_table_matches_raw_metrics() -> None:
    text = _copy_text()
    block = _table_block(text, "tab:margin_kl_plantwild")
    expected_rows: dict[str, tuple[float, ...]] = {}
    native = json.loads(
        (PROMPT_ROOT / "native_lm" / "plantwild_real_closed12_v1" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    # PlantWild is label-only, so MTS_micro is its Strict-3 equivalent.
    expected_rows["SFT native"] = (
        round(native["metrics"]["headline"]["FCA"], 3),
        round(native["metrics"]["safeguard"]["A_FU"], 3),
        round(native["metrics"]["safeguard"]["MCC"], 3),
        round(native["mts"]["MTS_micro"], 3),
        round(native["mts"]["per_relation"]["OOD_answer"], 3),
        round(native["mts"]["per_relation"]["OOD_confusion"], 3),
        round(native["mts"]["per_relation"]["OOD_absent"], 3),
    )
    regate = _comparison("plantwild_real_closed12_v1")["regate_reference"]
    expected_rows[r"\Method argmax"] = (
        round(regate["metrics"]["headline"]["FCA"], 3),
        round(regate["metrics"]["safeguard"]["A_FU"], 3),
        round(regate["metrics"]["safeguard"]["MCC"], 3),
        round(regate["metrics"]["semantic"]["Strict_3"], 3),
        round(regate["mts"]["per_relation"]["OOD_answer"], 3),
        round(regate["mts"]["per_relation"]["OOD_confusion"], 3),
        round(regate["mts"]["per_relation"]["OOD_absent"], 3),
    )
    for version, checkpoint in (("v1", 50), ("v2", 60)):
        result = json.loads(
            (
                MARGIN_ROOT
                / version
                / f"checkpoint-{checkpoint}"
                / "eval_paired"
                / "plantwild_real_closed12_v1"
                / "metrics.json"
            ).read_text(encoding="utf-8")
        )
        expected_rows[f"Margin+KL {version}"] = (
            round(result["metrics"]["headline"]["FCA"], 3),
            round(result["metrics"]["safeguard"]["A_FU"], 3),
            round(result["metrics"]["safeguard"]["MCC"], 3),
            round(result["mts"]["MTS_micro"], 3),
            round(result["mts"]["per_relation"]["OOD_answer"], 3),
            round(result["mts"]["per_relation"]["OOD_confusion"], 3),
            round(result["mts"]["per_relation"]["OOD_absent"], 3),
        )
    for label, expected in expected_rows.items():
        assert _row_values(block, label) == expected


def test_copy_dataset_counts_and_capability_table_match_package() -> None:
    text = _copy_text()
    sft_records = json.loads(SFT_TRAIN.read_text(encoding="utf-8"))
    state_counts = Counter(record["head_label"] for record in sft_records)
    assert len(sft_records) == 7072
    assert state_counts == Counter({0: 2703, 1: 1500, 2: 2869})
    distinct = len(
        {json.dumps(record, sort_keys=True, separators=(",", ":")) for record in sft_records}
    )
    assert distinct == 6138
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    assembled = len(sft_records) - (state_counts[1] - split["train"]["t6_count"])
    assert assembled == 6545
    assert "contains $6{,}545$ instances and $6{,}138$ distinct records" in text
    assert "yielding the $7{,}072$-instance composition" in text
    assert "Total & 7,072 & 38:62 & 2,703/1,500/2,869" in text

    with SPLIT_TRAIN.open(encoding="utf-8") as handle:
        train_records = [json.loads(line) for line in handle]
    t6_records = [r for r in train_records if r["metadata"]["relation"] == "T6"]
    assert len(t6_records) == 973
    t6_group_ids = {r["group_id"] for r in t6_records}
    assert len(t6_group_ids) == 372
    t6_anchors = [
        r
        for r in train_records
        if r["metadata"]["relation"] == "base" and r["group_id"] in t6_group_ids
    ]
    assert len(t6_anchors) == 372
    evidence_by_source: dict[str, dict[str, list[dict]]] = {}
    for record in train_records:
        if record["dataset"].startswith("coco"):
            continue
        relation = record["metadata"]["relation"]
        source = evidence_by_source.setdefault(record["source_image_id"], {"vis": [], "crop0": []})
        if relation == "base" and record["decision"] == "answer" and record.get("box"):
            source["vis"].append(record)
        elif (
            relation == "T7_ladder" and record["metadata"].get("achieved_visible_ratio", 1) <= 0.001
        ):
            source["crop0"].append(record)
    paired_sources = [d for d in evidence_by_source.values() if d["vis"] and d["crop0"]]
    assert len(paired_sources) == 281
    evidence_visible = [d["vis"][0] for d in paired_sources]
    evidence_excluded = [d["crop0"][0] for d in paired_sources]
    grouped_instances = t6_anchors + t6_records + evidence_visible + evidence_excluded
    assert len(grouped_instances) == 1907
    assert len({r["sample_id"] for r in grouped_instances}) == 1671
    assert (
        len({r["sample_id"] for r in t6_anchors} & {r["sample_id"] for r in evidence_visible})
        == 236
    )
    assert "contains $1{,}907$ instances" in text
    assert "$372$ supported anchors, $973$ contradiction variants" in text
    assert "$281$ target-retaining plus $281$ target-excluding variants" in text
    # 1,671 distinct records / 236 shared anchors remain artifact-level provenance.

    probe_expectations = {
        "test_L1_discrim.jsonl": Counter({"answer": 205, "abstain": 205}),
        "test_L2_evidence.jsonl": Counter({"answer": 395, "abstain": 395}),
        "test_L3_dose.jsonl": Counter({"answer": 183, "abstain": 148, "ambiguous": 419}),
        "plantwild_real_closed12_paired_408.jsonl": Counter({"answer": 144, "abstain": 264}),
        "ood_refcocog.jsonl": Counter({"answer": 181, "abstain": 181}),
    }
    for filename, expected in probe_expectations.items():
        counts: Counter[str] = Counter()
        with (EVAL_DIR / filename).open(encoding="utf-8") as handle:
            for line in handle:
                metadata = json.loads(line)["metadata"]
                counts[metadata.get("gt_decision", metadata.get("decision"))] += 1
        assert counts == expected
    probes = _table_block(text, "tab:probes")
    for label, counts in {
        "L1": "205/205",
        "L2": "395/395",
        "L3": "183/148",
        "Comp.": "1,323/1,323",
        "PW": "144/264",
        "Ref.": "181/181",
    }.items():
        assert re.search(rf"{re.escape(label)}\s*&\s*{re.escape(counts)}", probes)

    audit = json.loads(CAPABILITY_AUDIT.read_text(encoding="utf-8"))
    assert audit["num_instances"] == 768
    assert audit["observed_label_count"] == 24
    assert audit["capability_retained_label_count"] == 20
    block = _table_block(text, "tab:plantdoc_capability")
    excluded = set(audit["capability_excluded_labels"])
    for label, result in audit["per_label"].items():
        escaped = label.replace("_", r"\_")
        status = "X" if label in excluded else "R"
        expected = (
            rf"\texttt{{{escaped}}} & {result['n']} & {result['correct_rate']:.3f} & {status}"
        )
        assert expected in block


def test_copy_construction_appendix_matches_build_artifacts() -> None:
    text = _copy_text()
    half_visibility_counts: dict[str, Counter[str]] = {}
    for dataset, filename in {
        "PlantDoc": "plantdoc_full.raw.jsonl",
        "COCO": "coco_full.raw.jsonl",
    }.items():
        counts: Counter[str] = Counter()
        with (RAW_DIR / filename).open(encoding="utf-8") as handle:
            for line in handle:
                metadata = json.loads(line).get("metadata", {})
                if (
                    metadata.get("relation") == "T7_ladder"
                    and metadata.get("target_visible_ratio") == 0.5
                ):
                    route = "hard" if metadata["achieved_visible_ratio"] >= 0.5 else "dose"
                    counts[route] += 1
        half_visibility_counts[dataset] = counts
    assert half_visibility_counts == {
        "PlantDoc": Counter({"hard": 483, "dose": 140}),
        "COCO": Counter({"hard": 425, "dose": 142}),
    }
    assert "hard-label splits (PlantDoc: $483$;" in text
    assert "COCO: $425$), while the remainder enter the dose pool (PlantDoc: $140$; COCO:" in text
    assert "$142$)" in text
    assert "samples 300 supported visibility-controlled records" in text

    for filename, expected in {
        "dev_L3_dose.jsonl": (168, 48),
        "test_L3_dose.jsonl": (183, 55),
    }.items():
        levels_by_group: dict[str, set[str]] = {}
        with (EVAL_DIR / filename).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                group = record["images"][0].rsplit("-region-ladder-", 1)[0]
                levels_by_group.setdefault(group, set()).add(
                    record["metadata"]["visible_ratio_bin"]
                )
        assert (
            len(levels_by_group),
            sum(len(levels) == 5 for levels in levels_by_group.values()),
        ) == expected
    assert "48 of 168 development groups and 55 of 183" in text

    plantdoc_config = (CONFIG_DIR / "plantdoc_full.yaml").read_text(encoding="utf-8")
    coco_config = (CONFIG_DIR / "coco_full.yaml").read_text(encoding="utf-8")
    for config in (plantdoc_config, coco_config):
        assert "region_ladder_levels: [1.0, 0.75, 0.5, 0.25, 0.0]" in config
        assert "region_ladder_answer_threshold: 0.5" in config
        assert "collinearity_break: true" in config
    assert "blur_radii_visible: [2, 4]" in plantdoc_config
    assert "blur_radii_occluded: [6, 8]" in plantdoc_config
    assert "pixelate_blocks_visible: [4, 6]" in plantdoc_config
    assert "pixelate_blocks_occluded: [10, 12]" in plantdoc_config
    assert "absent_per_positive: 4" in plantdoc_config
    assert "absent_per_positive: 2" in coco_config

    relations = set(
        json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))["train"]["relation_distribution"]
    )
    assert relations == {
        "T1",
        "T2",
        "T2_occluded",
        "T3",
        "T3_occluded",
        "T4_control",
        "T5",
        "T6",
        "T7_include",
        "T7_ladder",
        "base",
        "base_occluded",
    }


def test_copy_formulas_match_metric_and_training_implementations() -> None:
    text = _copy_text()

    assert (
        "answer-selection rates on $\\mathcal D_{\\mathrm{unsup}}$ and $\\mathcal D_{\\mathrm{sup}}$"
        in text
    )
    assert r"\mathrm{FCA} &= 1-\mathrm{U\_Abs}" not in text
    assert "a final invalid output is a non-answer selection" in text
    assert r"FCA and U\_Abs need not sum to one" in text
    assert "Selective MCC is the standard Matthews coefficient" in text
    assert "the gate defines all binary selection metrics" in text

    assert r"\frac{1}{|\mathcal D_{p,v}|}" in text
    assert r"\sum_{i\in\mathcal D_{p,v}}m_v(o_i)" in text
    assert r"\mathcal V_p" in text
    assert r"\mathfrak{G}_v" not in text
    mts = _comparison("test_full")["regate_reference"]["mts"]
    recomputed_macro = sum(mts["per_relation"].values()) / len(mts["per_relation"])
    assert recomputed_macro == pytest.approx(mts["MTS_macro"])

    gate_parameters = (4096 * 512 + 512) + (512 * 256 + 256) + (256 * 3 + 3)
    assert gate_parameters == 2_229_763
    assert "$2{,}229{,}763$ trainable parameters" in text
    assert "GELU activations and dropout $0.1$" in text
    tri_head_source = TRI_HEAD_SOURCE.read_text(encoding="utf-8")
    assert "nn.Linear(hidden_size, 512)" in tri_head_source
    assert "nn.Linear(512, 256)" in tri_head_source
    assert "nn.Linear(256, num_classes)" in tri_head_source
    assert tri_head_source.count("nn.Dropout(0.1)") == 2

    trainer_source = TRAINER_SOURCE.read_text(encoding="utf-8")
    run_train = RUN_TRAIN.read_text(encoding="utf-8")
    assert "head_loss = (per_sample_loss * weights_per_sample).mean()" in trainer_source
    assert (
        "self.head_lambda * (self.state.global_step / max(self.head_warmup_steps, 1))"
        in trainer_source
    )
    assert "WO_HEAD_WEIGHT=${WO_HEAD_WEIGHT:-1.5}" in run_train
    assert "HEAD_LAMBDA=${HEAD_LAMBDA:-0.4}" in run_train
    assert "HEAD_WARMUP=${HEAD_WARMUP:-10}" in run_train
    assert r"\lambda_t=0.4\min(1,t/10)" in text
