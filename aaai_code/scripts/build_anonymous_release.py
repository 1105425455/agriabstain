#!/usr/bin/env python
"""
build_anonymous_release.py — Build a minimal anonymous supplementary-material
package from the full research repository.

The package is assembled under ``--output-root`` (which must be *outside*
``aaai_code/``). The script operates in **dry-run** mode by default; pass
``--execute`` to actually copy files.

White-listed content (only the following is included):
  1. Code, configs, docs
     - ``aaai_code/configs/``
     - ``aaai_code/reason_abstain/``
     - ``aaai_code/scripts/``
     - ``aaai_code/training/``
     - ``aaai_code/evaluation/``
     - ``aaai_code/experiments/evidence_judge/``  (code only)
     - ``aaai_code/experiments/base_native_regate/`` (code only)
     - ``aaai_code/experiments/selective_grounding/`` (code only)
     - ``aaai_code/experiments/gpt56_sol_l3/`` (code only)
     - ``aaai_code/experiments/layer_selection/`` (code only)
     - ``aaai_code/experiments/grouping_ablation/`` (code + compact ``main/`` artifact)
     - ``aaai_code/README.md``
     - ``aaai_code/pyproject.toml``

  2. Benchmark data (annotations / manifests only — no images, no parquet)
     - ``aaai_code/benchmark/annotations/``
     - ``aaai_code/benchmark/MANIFEST.json``
     - ``aaai_code/benchmark/IMAGE_MANIFEST.json``

   3. Curated experiment artifacts (always included)
     - ``layer_selection``: all JSON + manifest (5 small files)
     - ``evidence_judge``: ``probe_results.json``, L2/L3 test analysis JSON/CSV/README
       (from external output dirs — see ``EVIDENCE_JUDGE_ARTIFACTS``)
     - ``base_native_regate``: canonical analysis CSV/JSON/README
        (from external output dirs — see ``BASE_NATIVE_REGATE_ARTIFACTS``)
     - ``matched_retention_dev``: development-selected threshold audit JSON/CSV/README
     - ``selective_grounding``: canonical analysis CSV/JSON/README
       (from external output dirs — see ``SELECTIVE_GROUNDING_ARTIFACTS``)
      - ``grouping_ablation``: path-free fixed-GT comparison under ``main/``
      - ``margin_kl``: selected v1/v2 trainer states, PlantWild metrics, and summaries
      - ``environment``: original seed-42 runtime environment record
      - ``logs``: selected SFT, grouping, margin+KL, prompt-interface, and layer-scan logs
      - ``gpt56_sol``: six path-sanitised API run configurations

   4. Predictions (only when ``--include-predictions``)
      - 9 Base/SFT/ReGate prediction JSONL files
      - 15 routing-interface prediction JSONL files
      - 6 GPT parsed prediction JSONL files + comparison JSON
      - 2 Margin+KL PlantWild prediction JSONL files
     - Raw API outputs / response IDs are excluded

Sanitisation:
  - All text content is scanned for ``/user/huangyongbin/newrepo/agriabstain``
    and ``/user/huangyongbin`` and replaced with placeholders.
  - JSON structure is preserved (the replacement is string-level).

Final scan:
  - RELEASE_MANIFEST.json (relative path, size, sha256, source category).
  - RELEASE_NOTES.md.
  - Scan output for forbidden patterns: ``huangyongbin``, ``/user/``, secret regex.
  - Single-file size cap (default 100 MB).
  - Fail if any violation is found.

Usage:
    # Dry-run (default):
    python scripts/build_anonymous_release.py \
        --repo-root /path/to/agriabstain \
        --output-root /path/to/output

    # Execute:
    python scripts/build_anonymous_release.py \
        --repo-root /path/to/agriabstain \
        --output-root /path/to/output \
        --execute

    # Include predictions:
    python scripts/build_anonymous_release.py \
        --repo-root /path/to/agriabstain \
        --output-root /path/to/output \
        --execute --include-predictions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ═════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB

# Directories / files to copy from aaai_code (glob patterns)
AAAI_CODE_DIRS = [
    "configs",
    "reason_abstain",
    "scripts",
    "training",
    "evaluation",
]

# Experiment subdirs for which we copy ALL code files (no pycache, no runs, etc.)
EXPERIMENT_CODE_DIRS = [
    "evidence_judge",
    "base_native_regate",
    "selective_grounding",
    "gpt56_sol_l3",
    "layer_selection",
    "grouping_ablation",
    "margin_kl",
]

# Top-level aaai_code directories copied verbatim in addition to AAAI_CODE_DIRS.
# ``environment`` holds the frozen runtime/version record of the reported runs.
EXTRA_TOP_LEVEL_DIRS = [
    "environment",
]

# Benchmark: only annotations/ and manifests (no images, no parquet, no data/)
BENCHMARK_INCLUDE = [
    "annotations",
    "MANIFEST.json",
    "IMAGE_MANIFEST.json",
]

# Forbidden patterns in final scan
FORBIDDEN_PATTERNS = [
    re.compile(r"(?<![:\w])/(?:user|home|Users)/[^<\s\"']+"),
    re.compile(r"(?i)(?:task[_ -]?id)[\"'`\s:=：]+[\"'`]*\d{4,}"),
    re.compile(
        r"(?i)(?:sk-[a-z0-9]{20,}|ghp_[a-zA-Z0-9]{36,}|api_key\s*=\s*['\"][a-zA-Z0-9_\-]{16,})"
    ),
    re.compile(r"(?i)llm-center\.modelbest\.cn"),
    re.compile(r"(?i)modelbest|cybertron|paratera"),
]

# Text file extensions that get sanitisation
TEXT_EXTENSIONS = {
    ".py",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".md",
    ".toml",
    ".cfg",
    ".ini",
    ".txt",
    ".sh",
    ".csv",
    ".tex",
    ".log",
}

# ═════════════════════════════════════════════════════════════════════════
# External curated-artifact paths
# ═════════════════════════════════════════════════════════════════════════

# These are the *source* paths outside aaai_code that we want to pull in.
# They are configurable via CLI overrides but have sensible defaults rooted
# at the repo root.

# Evidence judge probe_results.json
EVIDENCE_JUDGE_PROBE_RESULTS = (
    "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/repr/probe_results.json"
)

# Base / SFT / ReGate canonical analysis CSV/JSON/README
BASE_NATIVE_REGATE_ANALYSIS_DIR = "new_reason/outputs/base_native_regate_20260720/analysis"

# Exact and perceptual duplicate audit behind the PlantWild source-separation claim,
# plus the builder that performs those checks. The builder lives outside aaai_code/
# because it predates the curated layout, but the appendix lists duplicate-check code
# as a released asset, so it ships here.
PLANTWILD_DEDUP_AUDIT = "new_reason/outputs/ood_candidates/plantwild_benchmark/dedup_audit.json"
PLANTWILD_DEDUP_CODE = "new_reason/scripts/build_ood_plantwild.py"

# Development-selected ReGate threshold compact audit artifacts.
MATCHED_RETENTION_ROOT = "new_reason/outputs/matched_retention_dev_20260725"
MATCHED_RETENTION_ARTIFACTS = [
    "README.md",
    "inputs/manifest.json",
    "analysis/result.json",
    "analysis/dev_threshold_sweep.csv",
    "analysis/test_comparison.csv",
]
MATCHED_RETENTION_INPUTS = [
    (
        f"{MATCHED_RETENTION_ROOT}/dev/native_sft/predictions.jsonl",
        "dev_native_predictions.jsonl",
        "native",
    ),
    (
        f"{MATCHED_RETENTION_ROOT}/dev/regate_head/head_probabilities.jsonl",
        "dev_regate_probabilities.jsonl",
        "probability",
    ),
    (
        "new_reason/outputs/base_native_regate_20260720/native/sft/comprehensive/predictions.jsonl",
        "test_native_predictions.jsonl",
        "native",
    ),
    (
        "new_reason/outputs/base_native_regate_20260720/regate/comprehensive/predictions.jsonl",
        "test_regate_predictions.jsonl",
        "regate",
    ),
]

# Selective grounding canonical analysis CSV/JSON/README
LAYER_SELECTION_ARTIFACT_DIR = "new_reason/outputs/head_boundary_refit/layer_scan"
LAYER_SELECTION_MANIFEST = "new_reason/outputs/head_boundary_refit/manifest_extract_alllayers.json"
LAYER_SELECTION_RECORDS = "new_reason/outputs/rl_grpo_trihead/rl_data/rl_samples.jsonl"
LAYER_SELECTION_SPLIT = "new_reason/outputs/rl_grpo_trihead/rl_data/split_assignment.json"

EVIDENCE_ANALYSIS_FILES = [
    "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/evidence/L2/test/analysis.json",
    "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/evidence/L2/test/head_probabilities.jsonl",
    "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/evidence/L3/test/analysis.json",
    "new_reason/outputs/evidence_judge_full_4090_b1_v2_20260719/evidence/L3/test/head_probabilities.jsonl",
]

# GPT-5.6 L3 analysis
GPT56_L3_ANALYSIS_DIR = "aaai_code/experiments/gpt56_sol_l3/runs/l3_full_v1/analysis"

# Predictions (only when --include-predictions)
PREDICTION_FILES = [
    f"new_reason/outputs/base_native_regate_20260720/native/{method}/{dataset}/predictions.jsonl"
    for method in ("base", "sft")
    for dataset in ("comprehensive", "refcocog", "plantwild")
] + [
    f"new_reason/outputs/base_native_regate_20260720/regate/{dataset}/predictions.jsonl"
    for dataset in ("comprehensive", "refcocog", "plantwild")
]

PROMPT_INTERFACE_METHODS = (
    "native_lm",
    "prompt_basic",
    "prompt_reward",
    "prompt_reward_norms",
    "cot_structured",
)
PROMPT_INTERFACE_DATASETS = (
    "test_full",
    "ood_refcocog",
    "plantwild_real_closed12_v1",
)
PROMPT_INTERFACE_PREDICTION_FILES = [
    f"new_reason/baselines/prompt_cot/outputs/{method}/{dataset}/predictions.jsonl"
    for method in PROMPT_INTERFACE_METHODS
    for dataset in PROMPT_INTERFACE_DATASETS
]

GPT_PROBES = (
    "L1_discrim",
    "L2_evidence",
    "L3_dose",
    "test_full",
    "ood_refcocog",
    "plantwild_real_closed12_v1",
)

# Margin+KL selected results now live inside aaai_code/experiments/margin_kl/main/ and are
# copied by the experiment-code stage; nothing is pulled from external output roots.

SELECTED_RUN_LOGS = (
    (
        "new_reason/outputs/logs/fusion_3reason_v3_hl04/train.log",
        "sft_final/train.log",
    ),
    ("new_reason/outputs/logs/paired_retrain.log", "margin_kl/v1_train.log"),
    ("new_reason/outputs/logs/paired_retrain_v2.log", "margin_kl/v2_train.log"),
    (
        "new_reason/outputs/logs/eval_paired_closed12_v1_checkpoint-50.log",
        "margin_kl/v1_plantwild_eval.log",
    ),
    (
        "new_reason/outputs/logs/eval_paired_closed12_v2_checkpoint-60.log",
        "margin_kl/v2_plantwild_eval.log",
    ),
    ("new_reason/baselines/prompt_cot/logs/native_lm.log", "prompt_interfaces/native_lm.log"),
    (
        "new_reason/baselines/prompt_cot/logs/prompt_basic.log",
        "prompt_interfaces/direct.log",
    ),
    (
        "new_reason/baselines/prompt_cot/logs/prompt_reward.log",
        "prompt_interfaces/reward.log",
    ),
    (
        "new_reason/baselines/prompt_cot/logs/full_prompt_reward_norms_002639.log",
        "prompt_interfaces/reward_evidence.log",
    ),
    (
        "new_reason/baselines/prompt_cot/logs/full_cot_structured_005019.log",
        "prompt_interfaces/structured_cot.log",
    ),
    ("new_reason/outputs/logs/extract_alllayers.log", "layer_selection/extract_alllayers.log"),
    ("new_reason/outputs/logs/layer_scan.log", "layer_selection/layer_scan.log"),
)

GROUPING_LOG_DIR = "aaai_code/experiments/grouping_ablation/artifacts/logs"
TRAINING_ENVIRONMENT_LOG = (
    "aaai_code/experiments/grouping_ablation/artifacts/logs/environment.log"
)


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def collect_files(root: Path, *, recursive: bool = True) -> list[Path]:
    """Collect all files under *root* (non-recursive if not *recursive*)."""
    if recursive:
        return sorted(p for p in root.rglob("*") if p.is_file())
    return sorted(p for p in root.iterdir() if p.is_file())


def safe_copy(src: Path, dst: Path, *, allow_symlink: bool = False) -> None:
    """Copy a file from *src* to *dst*, creating parent dirs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if allow_symlink and src.is_symlink():
        link_target = os.readlink(str(src))
        if not dst.exists():
            os.symlink(link_target, str(dst))
        return
    shutil.copy2(src, dst)


def sanitise_text(text: str, repo_root: Path | None = None) -> str:
    """Replace local repository/home paths without embedding an author identity."""
    text = text.replace(
        "https://llm-center.modelbest.cn/llm/v1/chat/completions",
        "<GPT56_ENDPOINT>",
    )
    # The provider's plain-language name identifies the authors' institution just as the
    # endpoint hostname does, and it is baked into already-generated result artifacts.
    text = text.replace(
        "ModelBest LLM Center",
        "OpenAI-compatible commercial API gateway",
    )
    if repo_root is not None:
        text = text.replace(str(repo_root), "<REPO_ROOT>")
        text = text.replace(str(repo_root.parent), "<WORKSPACE_ROOT>")
        parts = repo_root.parts
        for anchor in ("user", "home", "Users"):
            if anchor in parts:
                index = parts.index(anchor)
                if index + 1 < len(parts):
                    text = text.replace(parts[index + 1], "<USER>")
    text = re.sub(r"(?<![:\w])/(?:user|home|Users)/[^/\s\"']+", "<USER_HOME>", text)
    text = re.sub(
        r"(?im)Cybertron task ID[：:]\s*(?:\n[ \t]*)?`?\d+`?[。.]?",
        "",
        text,
    )
    # Scheduler pod names reach runtime logs through the torchrun failure report. Only the
    # project segment identifies the authors, so it alone is replaced and the rest of the
    # host name is left intact.
    text = re.sub(
        r"(pytorchjob-)[A-Za-z0-9]+(-\d+-(?:master|worker)-\d+)",
        r"\1anon\2",
        text,
    )
    # The system prompts of every record name the project. Dropping the self-reference leaves
    # the wording the technical supplement quotes; the remaining mentions become the paper name.
    text = re.sub(r"You are AgriAbstain, (a vision-language)", r"You are \1", text)
    text = text.replace("AgriAbstain", "ReGate")
    text = text.replace("，任务日志见 `logs/pipeline.log`。", "。")
    return text


def write_path_free_matched_input(src: Path, dst: Path, *, kind: str) -> None:
    """Write only fields consumed by the development-threshold analysis."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as source, dst.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if kind == "probability":
                curated = {
                    key: row.get(key)
                    for key in (
                        "sample_id",
                        "original_gt_decision",
                        "variant_id",
                        "p_answer",
                        "p_wo",
                        "p_nt",
                    )
                }
            elif kind == "native":
                curated = {
                    "sample_id": row.get("sample_id"),
                    "relation": row.get("relation"),
                    "gt": row.get("gt"),
                    "pred": row.get("pred"),
                }
            elif kind == "regate":
                curated = {
                    "sample_id": row.get("sample_id"),
                    "gt": row.get("gt"),
                    "p_answer": row.get("p_answer"),
                    "p_wo": row.get("p_wo"),
                    "p_nt": row.get("p_nt"),
                    "answer_candidate": row.get("answer_candidate"),
                }
            else:
                raise ValueError(f"Unknown matched-retention input kind: {kind}")
            target.write(json.dumps(curated, ensure_ascii=False) + "\n")


def sanitise_file(src: Path, dst: Path, repo_root: Path | None = None) -> None:
    """Copy *src* to *dst*, sanitising text content if it is a text file."""
    if is_text_file(src):
        try:
            content = src.read_text(encoding="utf-8")
        except Exception:
            # Fall back to binary copy if text read fails
            safe_copy(src, dst)
            return
        sanitised = sanitise_text(content, repo_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(sanitised, encoding="utf-8")
    else:
        safe_copy(src, dst)


# Component notes written for the authors: Chinese narrative, task-log framing, and
# rationale that duplicates the paper. REVIEWER_README.md is the single reviewer-facing
# guide, so these are left out of the supplement.
AUTHOR_SIDE_DOCS = {
    "README.md",
    "experiments/base_native_regate/README.md",
    "experiments/base_native_regate/analysis/README.md",
    "experiments/evidence_judge/README.md",
    "experiments/grouping_ablation/README.md",
    "experiments/selective_grounding/README.md",
}


def is_author_side_doc(relative: Path | str) -> bool:
    """Return True for author-side notes that the supplement should not carry."""
    return str(relative).replace("\\", "/") in AUTHOR_SIDE_DOCS


def is_excluded_dir(name: str) -> bool:
    """Return True if a directory name should be excluded from code copies."""
    return name in {
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        ".git",
        "runs",
        "smoke",
        "outputs",
        "checkpoints",
        "weights",
        "caches",
        "artifacts",
    }


def is_excluded_file(name: str) -> bool:
    """Return True if a file name / extension should be excluded."""
    if name in {
        "test_build_anonymous_release.py",
        "test_paper_metric_consistency.py",
        "build_anonymous_release.py",
    }:
        return True
    ext = Path(name).suffix.lower()
    if ext == ".tex":
        return True
    if ext in {".npz", ".pt", ".pth", ".safetensors", ".bin", ".pickle", ".pkl", ".parquet"}:
        return True
    if name.endswith(".run_config.json") or name.endswith("api_responses.json"):
        return True
    return False


def copy_code_dir(
    src: Path,
    dst: Path,
    output_root: Path,
    *,
    dry_run: bool,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Copy a code directory tree, skipping excluded dirs/files."""
    entries: list[dict[str, Any]] = []
    if not src.exists():
        return entries
    for item in sorted(src.iterdir()):
        if item.is_dir():
            if is_excluded_dir(item.name):
                continue
            entries.extend(
                copy_code_dir(
                    item,
                    dst / item.name,
                    output_root,
                    dry_run=dry_run,
                    repo_root=repo_root,
                )
            )
        elif item.is_file():
            if is_excluded_file(item.name):
                continue
            relative = (dst / item.name).relative_to(output_root)
            if is_author_side_doc(relative):
                continue
            if not dry_run:
                sanitise_file(item, dst / item.name, repo_root)
            entries.append(
                {
                    "source": str(item.relative_to(src.parent.parent)),
                    "source_path": str(item),
                    "relative": str(relative),
                    "size": file_size(item),
                }
            )
    return entries


def copy_experiment_code(
    src: Path,
    dst: Path,
    output_root: Path,
    *,
    dry_run: bool,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Copy experiment code files only (no subdirs like runs, outputs, etc.)."""
    entries: list[dict[str, Any]] = []
    if not src.exists():
        return entries
    for item in sorted(src.iterdir()):
        if item.is_dir():
            if is_excluded_dir(item.name):
                continue
            entries.extend(
                copy_code_dir(
                    item,
                    dst / item.name,
                    output_root,
                    dry_run=dry_run,
                    repo_root=repo_root,
                )
            )
        elif item.is_file():
            if is_excluded_file(item.name):
                continue
            if is_author_side_doc((dst / item.name).relative_to(output_root)):
                continue
            if not dry_run:
                sanitise_file(item, dst / item.name, repo_root)
            entries.append(
                {
                    "source": str(item.relative_to(src.parent.parent)),
                    "source_path": str(item),
                    "relative": str((dst / item.name).relative_to(output_root)),
                    "size": file_size(item),
                }
            )
    return entries


# ═════════════════════════════════════════════════════════════════════════
# Main build logic
# ═════════════════════════════════════════════════════════════════════════


def build_anonymous_release(
    repo_root: Path,
    output_root: Path,
    *,
    dry_run: bool = True,
    include_predictions: bool = False,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    force: bool = False,
) -> dict[str, Any]:
    """Build the anonymous release package.

    Returns a report dictionary.
    """
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    aaai_code = repo_root / "aaai_code"
    gpt_root = aaai_code / "experiments" / "gpt56_sol_full" / "outputs" / "decision_prompt_v1"

    # ── Validation ────────────────────────────────────────────────────
    if not aaai_code.exists():
        raise FileNotFoundError(f"aaai_code/ not found under {repo_root}")

    if output_root == repo_root or repo_root in output_root.parents:
        raise ValueError("Output root must be outside the research repository.")

    if not dry_run and output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(
                f"output directory is not empty: {output_root}; pass --force to replace it"
            )
        shutil.rmtree(output_root)
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "dry_run": dry_run,
        "include_predictions": include_predictions,
        "steps": {},
        "errors": [],
        "warnings": [],
        "manifest_entries": [],
    }

    copied_count = 0
    total_bytes = 0

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Copy aaai_code top-level docs
    # ══════════════════════════════════════════════════════════════════
    print("Step 1: Copying top-level files...")
    for fname in ["REVIEWER_README.md", "requirements.txt", "pyproject.toml"]:
        src = aaai_code / fname
        if not src.exists():
            report["warnings"].append(f"Top-level file not found: {src}")
            continue
        dst = output_root / fname
        if not dry_run:
            sanitise_file(src, dst, repo_root)
        copied_count += 1
        total_bytes += file_size(src)
        report["manifest_entries"].append(
            {
                "relative": fname,
                "size": file_size(src),
                "sha256": sha256_file(src) if not dry_run else "dry-run",
                "category": "docs",
            }
        )

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Copy aaai_code/ configs, reason_abstain, scripts, training, evaluation
    # ══════════════════════════════════════════════════════════════════
    print("Step 2: Copying code/config directories...")
    for dirname in AAAI_CODE_DIRS + EXTRA_TOP_LEVEL_DIRS:
        src = aaai_code / dirname
        if not src.exists():
            report["warnings"].append(f"Code directory not found: {src}")
            continue
        dst = output_root / dirname
        entries = copy_code_dir(src, dst, output_root, dry_run=dry_run, repo_root=repo_root)
        copied_count += len(entries)
        category = "environment" if dirname in EXTRA_TOP_LEVEL_DIRS else f"code/{dirname}"
        for e in entries:
            src_path = Path(e["source_path"])
            sz = file_size(src_path) if src_path.exists() else e["size"]
            total_bytes += sz
            report["manifest_entries"].append(
                {
                    "relative": e["relative"],
                    "size": sz,
                    "sha256": "dry-run" if dry_run else sha256_file(src_path),
                    "category": category,
                }
            )
        report["steps"][f"code_{dirname}"] = len(entries)

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Copy experiment code (whitelisted dirs only)
    # ══════════════════════════════════════════════════════════════════
    print("Step 3: Copying experiment code...")
    for dirname in EXPERIMENT_CODE_DIRS:
        src = aaai_code / "experiments" / dirname
        if not src.exists():
            report["warnings"].append(f"Experiment directory not found: {src}")
            continue
        dst = output_root / "experiments" / dirname
        entries = copy_experiment_code(src, dst, output_root, dry_run=dry_run, repo_root=repo_root)
        copied_count += len(entries)
        for e in entries:
            src_path = Path(e["source_path"])
            sz = file_size(src_path) if src_path.exists() else e["size"]
            total_bytes += sz
            report["manifest_entries"].append(
                {
                    "relative": e["relative"],
                    "size": sz,
                    "sha256": "dry-run" if dry_run else sha256_file(src_path),
                    "category": f"experiments/{dirname}",
                }
            )
        report["steps"][f"experiments_{dirname}"] = len(entries)

    # ══════════════════════════════════════════════════════════════════
    # Step 4: Copy benchmark annotations / manifests
    # ══════════════════════════════════════════════════════════════════
    print("Step 4: Copying benchmark annotations and manifests...")
    benchmark_src = aaai_code / "benchmark"
    benchmark_dst = output_root / "benchmark"
    benchmark_count = 0

    for item in BENCHMARK_INCLUDE:
        src = benchmark_src / item
        if not src.exists():
            report["warnings"].append(f"Benchmark item not found: {src}")
            continue
        dst = benchmark_dst / item
        if src.is_file():
            if not dry_run:
                sanitise_file(src, dst, repo_root)
            copied_count += 1
            benchmark_count += 1
            # Sanitisation rewrites the system prompts, so the digest has to describe the
            # copy that ships rather than the source file.
            recorded = src if dry_run else dst
            total_bytes += file_size(recorded)
            report["manifest_entries"].append(
                {
                    "relative": f"benchmark/{item}",
                    "size": file_size(recorded),
                    "sha256": "dry-run" if dry_run else sha256_file(dst),
                    "category": "benchmark",
                }
            )
        elif src.is_dir():
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(benchmark_src)
                    dst_file = benchmark_dst / rel
                    if not dry_run:
                        sanitise_file(f, dst_file, repo_root)
                    copied_count += 1
                    benchmark_count += 1
                    recorded = f if dry_run else dst_file
                    total_bytes += file_size(recorded)
                    report["manifest_entries"].append(
                        {
                            "relative": f"benchmark/{rel}",
                            "size": file_size(recorded),
                            "sha256": "dry-run" if dry_run else sha256_file(dst_file),
                            "category": "benchmark",
                        }
                    )
    report["steps"]["benchmark_annotations"] = benchmark_count

    # MANIFEST.json is written by package_benchmark.py to describe the image-bearing,
    # code-free benchmark package. Both of those properties are false for this
    # supplement, so restate the scope here and refresh the recorded digest.
    manifest_relative = "benchmark/MANIFEST.json"
    manifest_dst = benchmark_dst / "MANIFEST.json"
    if not dry_run and manifest_dst.is_file():
        payload = json.loads(manifest_dst.read_text(encoding="utf-8"))
        payload["description"] = (
            "Annotation-only subset accompanying the anonymous supplement: source-ID, "
            "state, relation and split manifests, SFT records, evaluation probes, and "
            "audits. Images are NOT included; IMAGE_MANIFEST.json lists their file names "
            "and SHA-256 so that copies obtained from the original public sources can be "
            "verified. Construction, training, evaluation and analysis code ARE included "
            "(configs/, scripts/, reason_abstain/, training/, evaluation/, experiments/). "
            "See REVIEWER_README.md for the reproduction path."
        )
        payload["supplement_scope"] = {
            "images_included": False,
            "code_included": True,
            "image_acquisition": "REVIEWER_README.md, section 'Obtaining the images'",
        }
        # Sanitisation rewrote the system prompt carried by every record, so the digests this
        # manifest records for the annotation files are recomputed from the shipped copies.
        for entry in payload.get("annotations", []):
            annotation_dst = benchmark_dst / entry["path"]
            if annotation_dst.is_file():
                entry["sha256"] = sha256_file(annotation_dst)
                entry["bytes"] = file_size(annotation_dst)
        write_json(manifest_dst, payload)
        for entry in report["manifest_entries"]:
            if entry["relative"] == manifest_relative:
                entry["size"] = file_size(manifest_dst)
                entry["sha256"] = sha256_file(manifest_dst)
                break

    # ══════════════════════════════════════════════════════════════════
    # Step 5: Copy curated experiment artifacts
    # ══════════════════════════════════════════════════════════════════
    print("Step 5: Copying curated experiment artifacts...")

    # 5a: Evidence judge probe_results
    probe_results_src = repo_root / EVIDENCE_JUDGE_PROBE_RESULTS
    if probe_results_src.exists():
        dst = output_root / "experiments" / "evidence_judge" / "probe_results.json"
        if not dry_run:
            sanitise_file(probe_results_src, dst, repo_root)
        copied_count += 1
        total_bytes += file_size(probe_results_src)
        report["manifest_entries"].append(
            {
                "relative": "experiments/evidence_judge/probe_results.json",
                "size": file_size(probe_results_src),
                "sha256": "dry-run" if dry_run else sha256_file(probe_results_src),
                "category": "artifacts/evidence_judge",
            }
        )
        report["steps"]["evidence_judge_probe_results"] = 1
    else:
        report["warnings"].append(f"Evidence judge probe_results not found: {probe_results_src}")

    # 5a-bis: PlantWild exact/perceptual duplicate audit
    # Both of these carry absolute source paths that sanitisation rewrites, so the
    # digest is taken from the shipped copy rather than the source.
    for source_relative, relative, category, step in (
        (
            PLANTWILD_DEDUP_AUDIT,
            "repro_artifacts/plantwild/dedup_audit.json",
            "artifacts/plantwild",
            "plantwild_dedup_audit",
        ),
        (
            PLANTWILD_DEDUP_CODE,
            "scripts/plantwild_rl/build_ood_plantwild.py",
            "code/scripts",
            "plantwild_dedup_code",
        ),
    ):
        src = repo_root / source_relative
        if not src.exists():
            report["warnings"].append(f"PlantWild asset not found: {src}")
            continue
        dst = output_root / relative
        if not dry_run:
            sanitise_file(src, dst, repo_root)
        size = file_size(dst) if not dry_run else file_size(src)
        copied_count += 1
        total_bytes += size
        report["manifest_entries"].append(
            {
                "relative": relative,
                "size": size,
                "sha256": "dry-run" if dry_run else sha256_file(dst),
                "category": category,
            }
        )
        report["steps"][step] = 1

    # 5b: Base native regate analysis
    base_native_src = repo_root / BASE_NATIVE_REGATE_ANALYSIS_DIR
    if base_native_src.exists():
        dst_dir = output_root / "experiments" / "base_native_regate" / "analysis"
        for f in sorted(base_native_src.rglob("*")):
            if f.is_file():
                rel = f.relative_to(base_native_src)
                dst_file = dst_dir / rel
                if not dry_run:
                    sanitise_file(f, dst_file, repo_root)
                copied_count += 1
                total_bytes += file_size(f)
                report["manifest_entries"].append(
                    {
                        "relative": f"experiments/base_native_regate/analysis/{rel}",
                        "size": file_size(f),
                        "sha256": "dry-run" if dry_run else sha256_file(f),
                        "category": "artifacts/base_native_regate",
                    }
                )
        report["steps"]["base_native_regate_analysis"] = 1
    else:
        report["warnings"].append(f"Base native regate analysis not found: {base_native_src}")

    # 5c: Layer-selection inputs and compact outputs
    layer_files = [
        repo_root / LAYER_SELECTION_MANIFEST,
        repo_root / LAYER_SELECTION_RECORDS,
        repo_root / LAYER_SELECTION_SPLIT,
    ] + [
        repo_root / LAYER_SELECTION_ARTIFACT_DIR / name
        for name in ("best_layer.json", "dev_scan.json", "min_score_curve.json", "ood_scan.json")
    ]
    layer_count = 0
    for src in layer_files:
        if not src.is_file():
            report["warnings"].append(f"Missing layer-selection artifact: {src.name}")
            continue
        dst = output_root / "repro_artifacts" / "layer_selection" / src.name
        if not dry_run:
            sanitise_file(src, dst, repo_root)
        copied_count += 1
        layer_count += 1
        total_bytes += file_size(src)
        report["manifest_entries"].append(
            {
                "relative": str(dst.relative_to(output_root)),
                "size": file_size(src),
                "sha256": "dry-run" if dry_run else sha256_file(src),
                "category": "artifacts/layer_selection",
            }
        )
    report["steps"]["layer_selection_artifacts"] = layer_count

    # 5d: Evidence sensitivity analysis inputs/outputs
    evidence_count = 0
    for relative in EVIDENCE_ANALYSIS_FILES:
        src = repo_root / relative
        if not src.is_file():
            report["warnings"].append(f"Missing evidence artifact: {src.name}")
            continue
        probe = "L2" if "/L2/" in relative else "L3"
        dst = output_root / "repro_artifacts" / "evidence_judge" / probe / src.name
        if not dry_run:
            sanitise_file(src, dst, repo_root)
        copied_count += 1
        evidence_count += 1
        total_bytes += file_size(src)
        report["manifest_entries"].append(
            {
                "relative": str(dst.relative_to(output_root)),
                "size": file_size(src),
                "sha256": "dry-run" if dry_run else sha256_file(src),
                "category": "artifacts/evidence_judge",
            }
        )
    report["steps"]["evidence_analysis_artifacts"] = evidence_count

    # 5e: GPT fixed-denominator comparison summaries (raw API responses excluded)
    gpt_count = 0
    for probe in GPT_PROBES:
        src = gpt_root / probe / "analysis" / "comparison.json"
        if not src.is_file():
            report["warnings"].append(f"Missing GPT comparison: {probe}")
            continue
        dst = output_root / "repro_artifacts" / "gpt56_sol" / probe / "comparison.json"
        if not dry_run:
            sanitise_file(src, dst, repo_root)
        copied_count += 1
        gpt_count += 1
        total_bytes += file_size(src)
        report["manifest_entries"].append(
            {
                "relative": str(dst.relative_to(output_root)),
                "size": file_size(src),
                "sha256": "dry-run" if dry_run else sha256_file(src),
                "category": "artifacts/gpt56_sol",
            }
        )
    report["steps"]["gpt_comparisons"] = gpt_count

    # 5f: Development-selected threshold compact audit artifacts
    matched_count = 0
    matched_root = repo_root / MATCHED_RETENTION_ROOT
    for relative in MATCHED_RETENTION_ARTIFACTS:
        src = matched_root / relative
        if not src.is_file():
            raise FileNotFoundError(f"Required matched-retention artifact is missing: {src}")
        dst = output_root / "repro_artifacts" / "matched_retention_dev" / relative
        if not dry_run:
            sanitise_file(src, dst, repo_root)
        copied_count += 1
        matched_count += 1
        total_bytes += file_size(src)
        report["manifest_entries"].append(
            {
                "relative": str(dst.relative_to(output_root)),
                "size": file_size(src),
                "sha256": "dry-run" if dry_run else sha256_file(src),
                "category": "artifacts/matched_retention_dev",
            }
        )
    report["steps"]["matched_retention_artifacts"] = matched_count

    # 5g: Minimal path-free per-record inputs required to recompute the 52% claim
    matched_input_count = 0
    for relative, output_name, kind in MATCHED_RETENTION_INPUTS:
        src = repo_root / relative
        if not src.is_file():
            raise FileNotFoundError(f"Required matched-retention input is missing: {src}")
        dst = (
            output_root
            / "repro_artifacts"
            / "matched_retention_dev"
            / "inputs"
            / output_name
        )
        if not dry_run:
            write_path_free_matched_input(src, dst, kind=kind)
        copied_count += 1
        matched_input_count += 1
        total_bytes += file_size(src)
        report["manifest_entries"].append(
            {
                "relative": str(dst.relative_to(output_root)),
                "size": file_size(src),
                "sha256": "dry-run" if dry_run else sha256_file(dst),
                "category": "inputs/matched_retention_dev",
            }
        )
    report["steps"]["matched_retention_inputs"] = matched_input_count

    # 5i: GPT API run configurations document the exact request protocol.
    gpt_config_count = 0
    for probe in GPT_PROBES:
        src = gpt_root / probe / "api_outputs.run_config.json"
        if not src.is_file():
            report["warnings"].append(f"Missing GPT run configuration: {probe}")
            continue
        dst = output_root / "repro_artifacts" / "gpt56_sol" / probe / "run_config.json"
        if not dry_run:
            sanitise_file(src, dst, repo_root)
        copied_count += 1
        gpt_config_count += 1
        total_bytes += file_size(src)
        report["manifest_entries"].append(
            {
                "relative": str(dst.relative_to(output_root)),
                "size": file_size(src),
                "sha256": "dry-run" if dry_run else sha256_file(src),
                "category": "protocol/gpt56_sol",
            }
        )
    report["steps"]["gpt_run_configurations"] = gpt_config_count

    # 5j: Sanitised runtime logs and the original seed-42 environment record.
    log_count = 0
    for relative, output_name in SELECTED_RUN_LOGS:
        src = repo_root / relative
        if not src.is_file():
            report["warnings"].append(f"Missing selected run log: {src}")
            continue
        dst = output_root / "repro_artifacts" / "logs" / output_name
        if not dry_run:
            sanitise_file(src, dst, repo_root)
        copied_count += 1
        log_count += 1
        total_bytes += file_size(src)
        report["manifest_entries"].append(
            {
                "relative": str(dst.relative_to(output_root)),
                "size": file_size(src),
                "sha256": "dry-run" if dry_run else sha256_file(src),
                "category": "logs",
            }
        )

    grouping_log_root = repo_root / GROUPING_LOG_DIR
    if grouping_log_root.is_dir():
        for src in sorted(grouping_log_root.iterdir()):
            if not src.is_file():
                continue
            dst = output_root / "repro_artifacts" / "logs" / "grouping" / src.name
            if not dry_run:
                sanitise_file(src, dst, repo_root)
            copied_count += 1
            log_count += 1
            total_bytes += file_size(src)
            report["manifest_entries"].append(
                {
                    "relative": str(dst.relative_to(output_root)),
                    "size": file_size(src),
                    "sha256": "dry-run" if dry_run else sha256_file(src),
                    "category": "logs/grouping_ablation",
                }
            )
    else:
        report["warnings"].append(f"Grouping log directory not found: {grouping_log_root}")

    environment_src = repo_root / TRAINING_ENVIRONMENT_LOG
    if environment_src.is_file():
        dst = output_root / "repro_artifacts" / "environment" / "seed42_training.log"
        if not dry_run:
            sanitise_file(environment_src, dst, repo_root)
        copied_count += 1
        total_bytes += file_size(environment_src)
        report["manifest_entries"].append(
            {
                "relative": str(dst.relative_to(output_root)),
                "size": file_size(environment_src),
                "sha256": "dry-run" if dry_run else sha256_file(environment_src),
                "category": "environment",
            }
        )
    else:
        report["warnings"].append(f"Training environment log not found: {environment_src}")
    report["steps"]["run_logs"] = log_count

    # ══════════════════════════════════════════════════════════════════
    # Step 6: Predictions (optional)
    # ══════════════════════════════════════════════════════════════════
    if include_predictions:
        print("Step 6: Copying predictions (--include-predictions)...")
        pred_count = 0
        prediction_root = repo_root / "new_reason" / "outputs" / "base_native_regate_20260720"
        for relative in PREDICTION_FILES:
            pred_src = repo_root / relative
            if not pred_src.is_file():
                report["warnings"].append(f"Prediction file not found: {pred_src.name}")
                continue
            rel = pred_src.relative_to(prediction_root)
            dst = output_root / "repro_artifacts" / "base_native_regate" / "predictions" / rel
            if not dry_run:
                sanitise_file(pred_src, dst, repo_root)
            pred_count += 1
            total_bytes += file_size(pred_src)
            report["manifest_entries"].append(
                {
                    "relative": str(dst.relative_to(output_root)),
                    "size": file_size(pred_src),
                    "sha256": "dry-run" if dry_run else sha256_file(pred_src),
                    "category": "predictions/base_native_regate",
                }
            )

        prompt_root = repo_root / "new_reason" / "baselines" / "prompt_cot" / "outputs"
        for relative in PROMPT_INTERFACE_PREDICTION_FILES:
            pred_src = repo_root / relative
            if not pred_src.is_file():
                report["warnings"].append(
                    f"Prompt-interface prediction file not found: {pred_src.parent.parent.name}/"
                    f"{pred_src.parent.name}"
                )
                continue
            rel = pred_src.relative_to(prompt_root)
            dst = output_root / "repro_artifacts" / "prompt_interfaces" / "predictions" / rel
            if not dry_run:
                sanitise_file(pred_src, dst, repo_root)
            pred_count += 1
            total_bytes += file_size(pred_src)
            report["manifest_entries"].append(
                {
                    "relative": str(dst.relative_to(output_root)),
                    "size": file_size(pred_src),
                    "sha256": "dry-run" if dry_run else sha256_file(pred_src),
                    "category": "predictions/prompt_interfaces",
                }
            )

        for probe in GPT_PROBES:
            gpt_pred_src = gpt_root / probe / "analysis" / "predictions.jsonl"
            if not gpt_pred_src.is_file():
                report["warnings"].append(f"GPT parsed predictions not found: {probe}")
                continue
            dst = output_root / "repro_artifacts" / "gpt56_sol" / probe / "predictions.jsonl"
            if not dry_run:
                sanitise_file(gpt_pred_src, dst, repo_root)
            pred_count += 1
            total_bytes += file_size(gpt_pred_src)
            report["manifest_entries"].append(
                {
                    "relative": str(dst.relative_to(output_root)),
                    "size": file_size(gpt_pred_src),
                    "sha256": "dry-run" if dry_run else sha256_file(gpt_pred_src),
                    "category": "predictions/gpt56_sol",
                }
            )

        report["steps"]["predictions"] = pred_count
    else:
        report["steps"]["predictions"] = 0
        print("Step 6: Skipping predictions (use --include-predictions to include)")

    # ══════════════════════════════════════════════════════════════════
    # Step 7: Generate RELEASE_MANIFEST.json
    # ══════════════════════════════════════════════════════════════════
    print("Step 7: Generating RELEASE_MANIFEST.json...")

    # Recompute sha256 for all files if not dry-run
    if not dry_run:
        for entry in report["manifest_entries"]:
            fp = output_root / entry["relative"]
            if fp.exists():
                entry["sha256"] = sha256_file(fp)
                entry["size"] = file_size(fp)
        total_bytes = sum(entry["size"] for entry in report["manifest_entries"])
        copied_count = len(report["manifest_entries"])

    manifest = {
        "release_name": "ReGate Anonymous Code and Data Supplement",
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "include_predictions": include_predictions,
        "payload_files": len(report["manifest_entries"]),
        "payload_bytes": total_bytes,
        "files": report["manifest_entries"],
    }

    if not dry_run:
        write_json(output_root / "RELEASE_MANIFEST.json", manifest)
    report["manifest"] = manifest

    # ══════════════════════════════════════════════════════════════════
    # Step 8: Build-side inventory (kept in the build report, not shipped)
    # ══════════════════════════════════════════════════════════════════
    print("Step 8: Recording build inventory...")

    release_notes = f"""# Build inventory

**Build timestamp**: {datetime.now(timezone.utc).isoformat()}
**Dry run**: {dry_run}
**Include predictions**: {include_predictions}

## Contents

### Code, Configs, Docs
- `configs/` — Benchmark construction YAML configs
- `reason_abstain/` — Python package: benchmark construction + evaluation
- `scripts/` — Build, audit, and packaging scripts
- `training/` — SFT, margin+KL, and Qwen3-VL training code
- `evaluation/` — Model inference and evaluation entry point
- `experiments/` — Mechanism experiments (evidence judge, base→ReGate, selective grounding, GPT-5.6 L3, layer selection, grouping ablation)
- `REVIEWER_README.md` — The reviewer guide
- `pyproject.toml` — Lint config

### Benchmark Data
- `benchmark/annotations/` — Curated annotation files (splits, SFT, eval, raw, audits)
- `benchmark/MANIFEST.json` — Benchmark data manifest
- `benchmark/IMAGE_MANIFEST.json` — Image manifest (paths only, no images included)

### Curated Artifacts
- `experiments/evidence_judge/probe_results.json` — Representation probe results
- `experiments/base_native_regate/analysis/` — Canonical Base/SFT/ReGate analysis
- `repro_artifacts/layer_selection/` — Layer-scan inputs and compact outputs
- `repro_artifacts/evidence_judge/` — L2/L3 probabilities and analyses
- `repro_artifacts/matched_retention_dev/` — Development-selected threshold audit and frozen-test comparison
- `repro_artifacts/matched_retention_dev/inputs/` — Four merged, path-free inputs needed to recompute the 52% result
- `repro_artifacts/gpt56_sol/` — Six fixed-denominator comparison summaries
- `experiments/margin_kl/main/` — Selected v1/v2 trainer states, PlantWild metrics, summaries, and predictions
- `experiments/evidence_judge/main/` — Frozen L2/L3 test analyses and manifests
- `experiments/layer_selection/main/post_adaptation/` — Frozen 36-layer post-adaptation scan
- `environment/` — Interpreter, accelerator, and dependency versions of the reported runs
- `repro_artifacts/logs/` — Sanitised SFT, grouping, margin+KL, prompt-interface, and layer-selection logs
- `repro_artifacts/environment/` — Seed-42 training runtime environment record
- `experiments/grouping_ablation/main/` — Path-free grouping-ablation metrics and audits
- `repro_artifacts/plantwild/dedup_audit.json` — Exact and perceptual duplicate audit for the PlantWild probe, with `scripts/plantwild_rl/build_ood_plantwild.py` implementing the checks
"""

    if include_predictions:
        release_notes += """
### Predictions
- `repro_artifacts/base_native_regate/predictions/` — Nine Base/SFT/ReGate JSONL files
- `repro_artifacts/prompt_interfaces/predictions/` — Fifteen routing-interface JSONL files
- `repro_artifacts/gpt56_sol/` — Six GPT parsed prediction files
- Raw API outputs and response IDs are excluded.
"""
    else:
        release_notes += """
### Predictions
- Not included (use `--include-predictions` to include).
"""

    release_notes += f"""
## File Count
- **Payload files**: {len(report["manifest_entries"])}
- **Payload size**: {total_bytes} bytes ({total_bytes / 1024 / 1024:.1f} MB)
"""

    report["release_notes"] = release_notes

    # ══════════════════════════════════════════════════════════════════
    # Step 9: Final security scan
    # ══════════════════════════════════════════════════════════════════
    print("Step 9: Running final security scan...")

    scan_errors: list[str] = []
    if not dry_run:
        dynamic_patterns = list(FORBIDDEN_PATTERNS)
        parts = repo_root.parts
        for anchor in ("user", "home", "Users"):
            if anchor in parts:
                index = parts.index(anchor)
                if index + 1 < len(parts):
                    dynamic_patterns.append(re.compile(re.escape(parts[index + 1]), re.IGNORECASE))

        for fp in sorted(path for path in output_root.rglob("*") if path.is_file()):
            relative = str(fp.relative_to(output_root))

            # Check file size cap
            sz = file_size(fp)
            if sz > max_file_bytes:
                scan_errors.append(f"File exceeds size limit ({sz} > {max_file_bytes}): {relative}")

            # Check content for forbidden patterns (text files only)
            if is_text_file(fp):
                try:
                    content = fp.read_text(encoding="utf-8")
                except Exception:
                    continue
                for pat in dynamic_patterns:
                    if pat.search(content):
                        # Find context
                        for m in pat.finditer(content):
                            start = max(0, m.start() - 20)
                            end = min(len(content), m.end() + 20)
                            context = content[start:end].replace("\n", "\\n")
                            scan_errors.append(
                                f"Forbidden pattern '{pat.pattern}' in {relative}: ...{context}..."
                            )
    else:
        print("  (dry-run: scan skipped)")

    if scan_errors:
        report["scan_errors"] = scan_errors
        report["errors"].extend(scan_errors)
        print(f"  Found {len(scan_errors)} scan violation(s)!")
        for err in scan_errors:
            print(f"    - {err}")
    else:
        report["scan_errors"] = []
        print("  No violations found.")

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    report["summary"] = {
        "total_files_copied": copied_count,
        "total_bytes": total_bytes,
        "scan_violations": len(scan_errors),
    }

    print(f"\n{'=' * 60}")
    print(f"Build {'dry-run' if dry_run else 'complete'}:")
    print(f"  Files to copy: {copied_count}")
    print(f"  Total size:    {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Scan errors:   {len(scan_errors)}")
    if scan_errors:
        print(f"  {'!' * 40}")
        print("  BUILD FAILED due to scan violations")
        print(f"  {'!' * 40}")
    print(f"{'=' * 60}")

    return report


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build anonymous release package from research repo"
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        required=True,
        help="Path to the root of the research repository",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Path to output directory (must be outside aaai_code/)",
    )
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Include prediction files (9 JSONL + GPT parsed + comparison JSON)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually copy files (default is dry-run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a non-empty output directory",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help=f"Maximum single-file size in bytes (default: {DEFAULT_MAX_FILE_BYTES})",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()

    report = build_anonymous_release(
        repo_root,
        output_root,
        dry_run=not args.execute,
        include_predictions=args.include_predictions,
        max_file_bytes=args.max_file_bytes,
        force=args.force,
    )

    if report.get("scan_errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
