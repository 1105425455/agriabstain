#!/usr/bin/env python
"""CPU tests for scripts.build_anonymous_release.

Verifies:
  - CLI argument parsing (--repo-root, --output-root, --include-predictions, --execute)
  - Dry-run mode produces a report without copying files
  - Execute mode copies files and produces REVIEWER_README.md + RELEASE_MANIFEST.json
  - Sanitisation replaces path patterns (/user/huangyongbin/...)
  - Forbidden pattern detection (huangyongbin, /user/, secret regex)
  - File size cap enforcement
  - Output outside aaai_code/ validation
  - Whitelist-only: experiment dirs not in the whitelist are excluded
  - Benchmark: only annotations/manifests, no images/parquet
  - External experiment artifacts are handled gracefully (missing = warning, not error)
  - Predictions included only when --include-predictions
  - JSON structure preserved during sanitisation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_anonymous_release import (  # noqa: E402
    FORBIDDEN_PATTERNS,
    build_anonymous_release,
    is_excluded_dir,
    is_excluded_file,
    is_text_file,
    sanitise_text,
    sha256_file,
    write_json,
    write_text,
)


# ═════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_repo(tmp_path: Path) -> Path:
    """Create a minimal mock repository structure."""
    repo = tmp_path / "repo"
    aaai = repo / "aaai_code"

    # Helper to create file with parent dirs
    def make_file(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    # Top-level files (the reviewer guide ships; author-side README.md stays behind)
    make_file(
        aaai / "REVIEWER_README.md",
        "# AgriAbstain\n\nPath: /user/huangyongbin/newrepo/agriabstain/aaai_code\n",
    )
    make_file(aaai / "pyproject.toml", "[tool.ruff]\nline-length = 100\n")

    # Configs
    make_file(aaai / "configs" / "coco_full.yaml", "dataset: coco\n")
    make_file(aaai / "configs" / "plantdoc_full.yaml", "dataset: plantdoc\n")

    # reason_abstain
    make_file(aaai / "reason_abstain" / "__init__.py", "# package\n")
    make_file(aaai / "reason_abstain" / "bench" / "__init__.py", "# bench\n")
    make_file(aaai / "reason_abstain" / "bench" / "io.py", "def read_yaml(): pass\n")
    make_file(aaai / "reason_abstain" / "eval" / "__init__.py", "# eval\n")
    make_file(aaai / "reason_abstain" / "eval" / "metrics.py", "def score(): pass\n")

    # Scripts
    make_file(aaai / "scripts" / "build_benchmark.py", "#!/usr/bin/env python\n# build script\n")
    make_file(
        aaai / "scripts" / "package_benchmark.py", "#!/usr/bin/env python\n# package script\n"
    )

    # Training
    make_file(aaai / "training" / "sft_final" / "run_train.sh", "#!/bin/bash\necho train\n")
    make_file(
        aaai / "training" / "margin_kl" / "train_paired_qwen.py", "#!/usr/bin/env python\n# train\n"
    )

    # Evaluation
    make_file(aaai / "evaluation" / "eval_qwen3vl_3reason.py", "#!/usr/bin/env python\n# eval\n")

    # Experiments — whitelisted
    for exp in [
        "evidence_judge",
        "base_native_regate",
        "selective_grounding",
        "gpt56_sol_l3",
        "layer_selection",
        "grouping_ablation",
    ]:
        make_file(aaai / "experiments" / exp / "__init__.py", f"# {exp}\n")
        make_file(aaai / "experiments" / exp / "README.md", f"# {exp} readme\n")
        make_file(aaai / "experiments" / exp / "analyze.py", "#!/usr/bin/env python\n# analyze\n")
    make_file(
        aaai / "experiments" / "grouping_ablation" / "main" / "grouping_ablation_fixed_gt.json",
        json.dumps({"protocol": "grouping_ablation", "MTS_macro": 0.8195}),
    )
    make_file(
        aaai
        / "experiments"
        / "layer_selection"
        / "main"
        / "pre_adaptation_candidate_selection.json",
        json.dumps({"candidate_layers": [9, 18, 27], "selected_api_index": 27}),
    )
    make_file(
        aaai / "experiments" / "grouping_ablation" / "artifacts" / "logs" / "train.log",
        "author-side runtime must not appear\n",
    )
    make_file(
        aaai / "experiments" / "base_native_regate" / "test_paper_metric_consistency.py",
        "PAPER_PATH = '../../paper/main.tex'\n",
    )

    # Experiments — excluded (not in whitelist)
    make_file(
        aaai / "experiments" / "gpt56_sol_full" / "outputs" / "dummy.txt", "should not appear\n"
    )
    make_file(aaai / "experiments" / "groundingme_external" / "README.md", "should not appear\n")
    make_file(aaai / "experiments" / "openref_external" / "__init__.py", "should not appear\n")

    # Benchmark — annotations only
    make_file(aaai / "benchmark" / "MANIFEST.json", json.dumps({"version": "1.0"}))
    make_file(aaai / "benchmark" / "IMAGE_MANIFEST.json", json.dumps([{"path": "dummy.jpg"}]))
    make_file(
        aaai / "benchmark" / "annotations" / "splits" / "train.jsonl", '{"id": 1}\n{"id": 2}\n'
    )
    make_file(aaai / "benchmark" / "annotations" / "eval" / "test_L1_discrim.jsonl", '{"id": 3}\n')

    # Benchmark — should NOT be included
    make_file(
        aaai / "benchmark" / "data" / "source" / "plantdoc" / "images" / "leaf.jpg",
        "fake_image_data",
    )
    make_file(aaai / "benchmark" / "data" / "ood" / "refcocog" / "data" / "val.parquet", "parquet")

    # External artifacts (outside aaai_code)
    # Evidence judge probe_results
    make_file(
        repo
        / "new_reason"
        / "outputs"
        / "evidence_judge_full_4090_b1_v2_20260719"
        / "repr"
        / "probe_results.json",
        json.dumps({"accuracy": 0.85}),
    )

    # Base native regate analysis
    make_file(
        repo
        / "new_reason"
        / "outputs"
        / "base_native_regate_20260720"
        / "analysis"
        / "canonical_analysis.csv",
        "method,score\nbase,0.5\nsft,0.7\n",
    )
    make_file(
        repo
        / "new_reason"
        / "outputs"
        / "base_native_regate_20260720"
        / "analysis"
        / "canonical_analysis.json",
        json.dumps({"base": 0.5, "sft": 0.7}),
    )
    make_file(
        repo / "new_reason" / "outputs" / "base_native_regate_20260720" / "analysis" / "README.md",
        "# Analysis\n",
    )

    # Development-selected matched-retention audit (required compact artifacts)
    matched_root = repo / "new_reason" / "outputs" / "matched_retention_dev_20260725"
    make_file(matched_root / "README.md", "# Development-selected threshold\n")
    make_file(matched_root / "inputs" / "manifest.json", json.dumps({"n_dev": 2408}))
    make_file(
        matched_root / "analysis" / "result.json",
        json.dumps({"selected_threshold": 0.9888, "relative_fca_reduction": 0.52}),
    )
    make_file(
        matched_root / "analysis" / "dev_threshold_sweep.csv",
        "threshold,retention,fca\n0.9888,0.8605,0.0357\n",
    )
    make_file(
        matched_root / "analysis" / "test_comparison.csv",
        "method,retention,fca\nSFT,0.8511,0.0945\nReGate,0.8896,0.0454\n",
    )
    make_file(
        matched_root / "dev" / "native_sft" / "predictions.jsonl",
        '{"sample_id":"dev-1","gt":{"decision":"answer"},"pred":{"valid":true,"decision":"answer"}}\n',
    )
    make_file(
        matched_root / "dev" / "regate_head" / "head_probabilities.jsonl",
        '{"sample_id":"dev-1","original_gt_decision":"answer","variant_id":"original","p_answer":0.9,"p_wo":0.05,"p_nt":0.05}\n',
    )

    # GPT-5.6 L3 analysis (within aaai_code)
    make_file(
        aaai
        / "experiments"
        / "gpt56_sol_l3"
        / "runs"
        / "l3_full_v1"
        / "analysis"
        / "comparison.json",
        json.dumps({"gpt_score": 0.75}),
    )
    make_file(
        aaai / "experiments" / "gpt56_sol_l3" / "runs" / "l3_full_v1" / "analysis" / "audit.json",
        json.dumps({"passed": True}),
    )
    make_file(
        aaai
        / "experiments"
        / "gpt56_sol_l3"
        / "runs"
        / "l3_full_v1"
        / "analysis"
        / "answer_rate_by_visibility.csv",
        "visibility,rate\n0.0,0.5\n1.0,0.9\n",
    )
    make_file(
        aaai
        / "experiments"
        / "gpt56_sol_l3"
        / "runs"
        / "l3_full_v1"
        / "analysis"
        / "predictions.jsonl",
        '{"id": 1}\n',
    )

    # Predictions (for --include-predictions)
    prediction_root = repo / "new_reason" / "outputs" / "base_native_regate_20260720"
    for method in ("base", "sft"):
        for dataset in ("comprehensive", "refcocog", "plantwild"):
            make_file(
                prediction_root / "native" / method / dataset / "predictions.jsonl",
                json.dumps({"id": f"{method}-{dataset}"}) + "\n",
            )
    for dataset in ("comprehensive", "refcocog", "plantwild"):
        make_file(
            prediction_root / "regate" / dataset / "predictions.jsonl",
            json.dumps({"id": f"regate-{dataset}"}) + "\n",
        )

    prompt_root = repo / "new_reason" / "baselines" / "prompt_cot" / "outputs"
    for method in (
        "native_lm",
        "prompt_basic",
        "prompt_reward",
        "prompt_reward_norms",
        "cot_structured",
    ):
        for dataset in ("test_full", "ood_refcocog", "plantwild_real_closed12_v1"):
            make_file(
                prompt_root / method / dataset / "predictions.jsonl",
                json.dumps({"id": f"{method}-{dataset}"}) + "\n",
            )

    # GPT parsed predictions and comparisons
    gpt_root = aaai / "experiments" / "gpt56_sol_full" / "outputs" / "decision_prompt_v1"
    for probe in (
        "L1_discrim",
        "L2_evidence",
        "L3_dose",
        "test_full",
        "ood_refcocog",
        "plantwild_real_closed12_v1",
    ):
        make_file(
            gpt_root / probe / "analysis" / "predictions.jsonl",
            json.dumps({"id": probe}) + "\n",
        )
        make_file(
            gpt_root / probe / "analysis" / "comparison.json",
            json.dumps({"probe": probe}),
        )
        make_file(
            gpt_root / probe / "api_outputs.run_config.json",
            json.dumps({"probe": probe, "model_requested": "gpt-5.6-sol"}),
        )

    # Frozen supplement artifacts now living inside aaai_code/
    for version, checkpoint in (("v1", "checkpoint-50"), ("v2", "checkpoint-60")):
        margin_main = aaai / "experiments" / "margin_kl" / "main" / f"{version}_{checkpoint}"
        make_file(margin_main / "trainer_state.json", json.dumps({"global_step": 50}))
        probe = margin_main / "plantwild_real_closed12_v1"
        make_file(probe / "metrics.json", json.dumps({"FCA": 0.155}))
        make_file(probe / "summary.txt", "metrics summary\n")
        make_file(probe / "predictions.jsonl", '{"id": "margin"}\n')

    evidence_main = aaai / "experiments" / "evidence_judge" / "main"
    for probe_name in ("L2_test", "L3_test"):
        make_file(
            evidence_main / probe_name / "analysis.json",
            json.dumps({"probe": probe_name}),
        )
        make_file(
            evidence_main / probe_name / "manifest.json",
            json.dumps({"probe": probe_name}),
        )

    post_adaptation = aaai / "experiments" / "layer_selection" / "main" / "post_adaptation"
    make_file(post_adaptation / "dev_scan.json", json.dumps({"n_layers": 36}))
    make_file(post_adaptation / "best_layer.json", json.dumps({"transformer_layer": 27}))

    make_file(
        aaai / "environment" / "training_environment.json",
        json.dumps({"packages": {"torch": "2.8.0", "peft": "0.17.1"}}),
    )

    return repo


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


# ═════════════════════════════════════════════════════════════════════════
# Helper function tests
# ═════════════════════════════════════════════════════════════════════════


class TestIsExcludedDir:
    def test_excluded_dirs(self):
        for name in [
            "__pycache__",
            ".ruff_cache",
            ".pytest_cache",
            "runs",
            "smoke",
            "outputs",
            "checkpoints",
            "weights",
            "caches",
            "artifacts",
        ]:
            assert is_excluded_dir(name), f"{name} should be excluded"

    def test_included_dirs(self):
        for name in ["bench", "eval", "configs", "scripts", "sft_final", "data", "raw"]:
            assert not is_excluded_dir(name), f"{name} should not be excluded"


class TestIsExcludedFile:
    def test_excluded_extensions(self):
        for name in [
            "model.npz",
            "weights.pt",
            "model.safetensors",
            "data.pkl",
            "model.pth",
            "data.parquet",
            "data.bin",
        ]:
            assert is_excluded_file(name), f"{name} should be excluded"

    def test_included_files(self):
        for name in [
            "train.py",
            "config.yaml",
            "data.json",
            "results.jsonl",
            "README.md",
            "run.sh",
            "metrics.csv",
        ]:
            assert not is_excluded_file(name), f"{name} should not be excluded"

    def test_run_config_excluded(self):
        assert is_excluded_file("api_responses.run_config.json")

    @pytest.mark.parametrize(
        "name",
        ["test_build_anonymous_release.py", "test_paper_metric_consistency.py"],
    )
    def test_author_side_tests_excluded(self, name: str):
        assert is_excluded_file(name)


class TestIsTextFile:
    def test_text_extensions(self):
        for ext in [
            ".py",
            ".json",
            ".jsonl",
            ".yaml",
            ".yml",
            ".md",
            ".toml",
            ".csv",
            ".sh",
            ".txt",
        ]:
            assert is_text_file(Path(f"test{ext}"))

    def test_binary_extensions(self):
        for ext in [".jpg", ".png", ".npz", ".pt", ".safetensors", ".parquet", ".bin"]:
            assert not is_text_file(Path(f"test{ext}"))


class TestSanitiseText:
    def test_long_path_replaced(self):
        root = Path("/user/privateuser/newrepo/agriabstain")
        result = sanitise_text(f"Path: {root}/aaai_code/configs", root)
        assert "<REPO_ROOT>" in result
        assert "/user/huangyongbin/newrepo/agriabstain" not in result

    def test_short_path_replaced(self):
        result = sanitise_text("Home: /user/privateuser/somewhere")
        assert "<USER_HOME>" in result
        assert "/user/privateuser" not in result

    def test_long_path_preferred(self):
        """Longer pattern should match first."""
        root = Path("/user/privateuser/newrepo/agriabstain")
        result = sanitise_text(f"{root}/foo", root)
        assert "<REPO_ROOT>" in result
        # The shorter pattern is *also* contained in the longer one, but since
        # we replace the longer one first, the shorter one won't match again.
        assert "/user/huangyongbin" not in result

    def test_no_path_unchanged(self):
        result = sanitise_text("Hello world")
        assert result == "Hello world"

    def test_empty_string(self):
        assert sanitise_text("") == ""

    def test_internal_task_id_line_removed(self):
        result = sanitise_text("Result verified.\nCybertron task ID: 619527\nDone.\n")
        assert "619527" not in result
        assert "Cybertron" not in result
        assert result == "Result verified.\n\nDone.\n"

    def test_wrapped_internal_task_id_removed(self):
        result = sanitise_text("Log follows. Cybertron task ID：\n`619527`。\n")
        assert "619527" not in result
        assert "Cybertron" not in result

    def test_url_path_is_not_sanitised(self):
        url = "https://example.com/user/profile/page"
        assert sanitise_text(url) == url

    def test_json_structure_preserved(self):
        """Sanitisation should not break JSON structure."""
        root = Path("/user/privateuser/newrepo/agriabstain")
        text = json.dumps({"path": f"{root}/data"})
        sanitised = sanitise_text(text, root)
        parsed = json.loads(sanitised)
        assert "<REPO_ROOT>" in parsed["path"]


class TestForbiddenPatterns:
    def test_private_home_path_detected(self):
        for pat in FORBIDDEN_PATTERNS:
            if pat.search("/user/privateuser/file"):
                break
        else:
            pytest.fail("private home path not detected by any forbidden pattern")

    def test_clean_text_not_detected(self):
        for pat in FORBIDDEN_PATTERNS:
            if pat.search("clean text with no secrets"):
                pytest.fail(f"Clean text matched {pat.pattern}")

    def test_internal_endpoint_is_sanitised(self):
        result = sanitise_text("endpoint=https://llm-center.modelbest.cn/llm/v1/chat/completions")
        assert "llm-center.modelbest.cn" not in result
        assert "<GPT56_ENDPOINT>" in result

    def test_internal_endpoint_is_forbidden(self):
        assert any(pat.search("https://llm-center.modelbest.cn/llm/v1/chat/completions") for pat in FORBIDDEN_PATTERNS)

    def test_api_key_detected(self):
        for pat in FORBIDDEN_PATTERNS:
            if pat.search("sk-abcdefghijklmnopqrstuvwxyz123456"):
                break
        else:
            pytest.fail("API key not detected")

    def test_structured_task_id_detected(self):
        for pat in FORBIDDEN_PATTERNS:
            if pat.search('"task_id": 617455'):
                break
        else:
            pytest.fail("structured task ID not detected")


# ═════════════════════════════════════════════════════════════════════════
# Build logic tests
# ═════════════════════════════════════════════════════════════════════════


class TestBuildDryRun:
    def test_dry_run_returns_report(self, mock_repo: Path, output_dir: Path):
        """Dry-run should produce a report without copying files."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        assert report["dry_run"] is True
        assert "manifest_entries" in report
        assert "steps" in report
        # No files should exist in output
        assert not output_dir.exists() or len(list(output_dir.iterdir())) == 0

    def test_dry_run_contains_expected_keys(self, mock_repo: Path, output_dir: Path):
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        assert "source_root" not in report
        assert "output_root" not in report
        assert "manifest" in report
        assert "release_notes" in report
        assert "summary" in report

    def test_dry_run_manifest_entries(self, mock_repo: Path, output_dir: Path):
        """Dry-run should list all files that would be copied."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        entries = report["manifest_entries"]
        # Should include REVIEWER_README.md, pyproject.toml, configs, reason_abstain, scripts, etc.
        rel_paths = {e["relative"] for e in entries}
        assert "REVIEWER_README.md" in rel_paths
        assert "README.md" not in rel_paths
        assert "pyproject.toml" in rel_paths
        assert any("configs/coco_full.yaml" in r for r in rel_paths)
        assert any("reason_abstain/__init__.py" in r for r in rel_paths)
        assert any("scripts/build_benchmark.py" in r for r in rel_paths)
        assert any("training/sft_final/run_train.sh" in r for r in rel_paths)
        assert any("evaluation/eval_qwen3vl_3reason.py" in r for r in rel_paths)
        assert not any("test_paper_metric_consistency.py" in r for r in rel_paths)

    def test_dry_run_excludes_non_whitelist_experiments(self, mock_repo: Path, output_dir: Path):
        """Experiments not in the whitelist should be excluded."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        rel_paths = {e["relative"] for e in report["manifest_entries"]}
        # gpt56_sol_full should not be included
        assert not any("gpt56_sol_full" in r for r in rel_paths)
        # groundingme_external should not be included
        assert not any("groundingme_external" in r for r in rel_paths)
        # openref_external should not be included
        assert not any("openref_external" in r for r in rel_paths)

    def test_dry_run_excludes_benchmark_images(self, mock_repo: Path, output_dir: Path):
        """Benchmark images and parquet should not be included."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        rel_paths = {e["relative"] for e in report["manifest_entries"]}
        assert not any("data/source" in r for r in rel_paths)
        assert not any("data/ood" in r for r in rel_paths)
        assert not any(".parquet" in r for r in rel_paths)
        # But annotations should be included
        assert any("benchmark/annotations/splits/train.jsonl" in r for r in rel_paths)
        assert any("benchmark/MANIFEST.json" in r for r in rel_paths)

    def test_dry_run_excludes_pycache(self, mock_repo: Path, output_dir: Path):
        """__pycache__ directories should be excluded."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        rel_paths = {e["relative"] for e in report["manifest_entries"]}
        assert not any("__pycache__" in r for r in rel_paths)

    def test_dry_run_includes_experiment_artifacts(self, mock_repo: Path, output_dir: Path):
        """Curated artifacts should be included in dry-run mode."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        rel_paths = {e["relative"] for e in report["manifest_entries"]}
        # Evidence judge code should be included
        assert any("experiments/evidence_judge/__init__.py" in r for r in rel_paths)
        # Base native regate code should be included
        assert any("experiments/base_native_regate/__init__.py" in r for r in rel_paths)

    def test_dry_run_without_predictions(self, mock_repo: Path, output_dir: Path):
        """Without --include-predictions, predictions should not be included."""
        report = build_anonymous_release(
            mock_repo, output_dir, dry_run=True, include_predictions=False
        )
        rel_paths = {e["relative"] for e in report["manifest_entries"]}
        # No top-level predictions/ directory
        pred_paths = [r for r in rel_paths if r.startswith("predictions/")]
        assert len(pred_paths) == 0, f"Found unexpected prediction paths: {pred_paths}"
        assert report["include_predictions"] is False


class TestBuildExecute:
    def test_execute_creates_files(self, mock_repo: Path, output_dir: Path):
        """Execute mode should copy files to output."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=False)
        assert report["dry_run"] is False
        assert output_dir.exists()
        assert (output_dir / "REVIEWER_README.md").exists()
        assert not (output_dir / "README.md").exists()
        assert (output_dir / "pyproject.toml").exists()
        assert (output_dir / "configs" / "coco_full.yaml").exists()
        assert (output_dir / "reason_abstain" / "__init__.py").exists()
        assert (output_dir / "scripts" / "build_benchmark.py").exists()
        assert (output_dir / "benchmark" / "MANIFEST.json").exists()
        assert (output_dir / "benchmark" / "annotations" / "splits" / "train.jsonl").exists()

    def test_execute_creates_manifest(self, mock_repo: Path, output_dir: Path):
        """Execute mode should produce RELEASE_MANIFEST.json."""
        build_anonymous_release(mock_repo, output_dir, dry_run=False)
        manifest_path = output_dir / "RELEASE_MANIFEST.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "release_name" in manifest
        assert "files" in manifest
        assert len(manifest["files"]) > 0

    def test_execute_keeps_release_notes_in_report(self, mock_repo: Path, output_dir: Path):
        """The build inventory lives in the report; no RELEASE_NOTES.md file ships."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=False)
        assert "release_notes" in report
        assert "Payload files" in report["release_notes"]
        assert not (output_dir / "RELEASE_NOTES.md").exists()
        assert (output_dir / "RELEASE_MANIFEST.json").exists()

    def test_execute_sanitises_paths(self, mock_repo: Path, output_dir: Path):
        """Text files should have paths sanitised."""
        build_anonymous_release(mock_repo, output_dir, dry_run=False)
        reviewer_readme = (output_dir / "REVIEWER_README.md").read_text()
        assert "/user/huangyongbin/newrepo/agriabstain" not in reviewer_readme
        assert "<REPO_ROOT>" in reviewer_readme or "<USER_HOME>" in reviewer_readme

    def test_execute_with_predictions(self, mock_repo: Path, output_dir: Path):
        """With --include-predictions, predictions should be included."""
        report = build_anonymous_release(
            mock_repo, output_dir, dry_run=False, include_predictions=True
        )
        # Prediction files should exist
        pred_files = list((output_dir / "repro_artifacts").rglob("predictions.jsonl"))
        assert len(pred_files) == 30
        matched_inputs = output_dir / "repro_artifacts" / "matched_retention_dev" / "inputs"
        assert (matched_inputs / "dev_native_predictions.jsonl").is_file()
        assert (matched_inputs / "dev_regate_probabilities.jsonl").is_file()
        assert any("base_native_regate/predictions/native/base" in str(f) for f in pred_files)
        assert any(
            "prompt_interfaces/predictions/prompt_basic/test_full" in str(f) for f in pred_files
        )
        assert any("gpt56_sol/test_full" in str(f) for f in pred_files)
        assert "Fifteen routing-interface JSONL files" in report["release_notes"]

    def test_execute_includes_frozen_supplement_artifacts(
        self, mock_repo: Path, output_dir: Path
    ):
        """Frozen margin+KL / evidence / layer artifacts and the environment record ship."""
        build_anonymous_release(mock_repo, output_dir, dry_run=False, include_predictions=True)
        exp = output_dir / "experiments"
        margin = exp / "margin_kl" / "main"
        assert (margin / "v1_checkpoint-50" / "trainer_state.json").is_file()
        for version, checkpoint in (("v1", "checkpoint-50"), ("v2", "checkpoint-60")):
            probe = margin / f"{version}_{checkpoint}" / "plantwild_real_closed12_v1"
            assert (probe / "metrics.json").is_file()
            assert (probe / "summary.txt").is_file()
            assert (probe / "predictions.jsonl").is_file()
        assert (exp / "evidence_judge" / "main" / "L2_test" / "analysis.json").is_file()
        assert (exp / "evidence_judge" / "main" / "L3_test" / "analysis.json").is_file()
        post = exp / "layer_selection" / "main" / "post_adaptation"
        assert (post / "dev_scan.json").is_file()
        assert (post / "best_layer.json").is_file()
        assert (output_dir / "environment" / "training_environment.json").is_file()

    def test_execute_excludes_benchmark_images(self, mock_repo: Path, output_dir: Path):
        """Benchmark data/ should not be copied."""
        build_anonymous_release(mock_repo, output_dir, dry_run=False)
        assert not (output_dir / "benchmark" / "data").exists()

    def test_execute_excludes_pycache(self, mock_repo: Path, output_dir: Path):
        """__pycache__ should not be in output."""
        build_anonymous_release(mock_repo, output_dir, dry_run=False)
        pycache_dirs = list(output_dir.rglob("__pycache__"))
        assert len(pycache_dirs) == 0

    def test_execute_includes_compact_grouping_artifact_only(
        self, mock_repo: Path, output_dir: Path
    ):
        build_anonymous_release(mock_repo, output_dir, dry_run=False)
        grouping = output_dir / "experiments" / "grouping_ablation"
        assert (grouping / "main" / "grouping_ablation_fixed_gt.json").is_file()
        assert not (grouping / "artifacts").exists()

    def test_execute_includes_compact_pre_adaptation_layer_artifact(
        self, mock_repo: Path, output_dir: Path
    ):
        build_anonymous_release(mock_repo, output_dir, dry_run=False)
        layer_selection = output_dir / "experiments" / "layer_selection"
        compact = layer_selection / "main" / "pre_adaptation_candidate_selection.json"
        assert compact.is_file()
        artifact = json.loads(compact.read_text(encoding="utf-8"))
        assert artifact["candidate_layers"] == [9, 18, 27]
        assert artifact["selected_api_index"] == 27
        assert not (output_dir / "repro_artifacts" / "layer_selection" / "pre_adaptation").exists()


class TestValidation:
    def test_output_inside_aaai_code_raises(self, mock_repo: Path):
        """Output inside aaai_code/ should raise ValueError."""
        bad_output = mock_repo / "aaai_code" / "release"
        with pytest.raises(ValueError, match="outside"):
            build_anonymous_release(mock_repo, bad_output, dry_run=True)

    def test_missing_repo_raises(self, tmp_path: Path, output_dir: Path):
        """Missing aaai_code/ should raise FileNotFoundError."""
        bad_repo = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="aaai_code"):
            build_anonymous_release(bad_repo, output_dir, dry_run=True)

    def test_missing_external_artifact_warns(self, mock_repo: Path, output_dir: Path):
        """Missing external artifact should produce warning, not error."""
        # Remove the probe_results
        probe_path = (
            mock_repo
            / "new_reason"
            / "outputs"
            / "evidence_judge_full_4090_b1_v2_20260719"
            / "repr"
            / "probe_results.json"
        )
        probe_path.unlink()
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        warnings = [w for w in report["warnings"] if "probe_results" in w.lower()]
        assert len(warnings) > 0


class TestSecurityScan:
    def test_clean_scan_passes(self, mock_repo: Path, output_dir: Path):
        """Clean build should have no scan errors."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=False)
        assert len(report.get("scan_errors", [])) == 0

    def test_dirty_file_fails_scan(self, mock_repo: Path, output_dir: Path):
        """A file containing forbidden pattern should trigger scan error."""
        # Write a file with forbidden content
        (mock_repo / "aaai_code" / "configs" / "secret_config.yaml").write_text(
            "api_key = 'sk-abcdefghijklmnopqrstuvwxyz1234567890abcdefg'"
        )
        report = build_anonymous_release(mock_repo, output_dir, dry_run=False)
        assert len(report.get("scan_errors", [])) > 0
        # The error should mention the file
        error_text = " ".join(report["scan_errors"])
        assert "secret_config.yaml" in error_text or "api_key" in error_text

    def test_private_identity_is_sanitised(self, mock_repo: Path, output_dir: Path):
        """A local account identifier is removed from copied text."""
        (mock_repo / "aaai_code" / "configs" / "leaky_config.yaml").write_text(
            "path: /user/privateuser/project/file.json\n"
        )
        report = build_anonymous_release(mock_repo, output_dir, dry_run=False)
        leaky_file = output_dir / "configs" / "leaky_config.yaml"
        assert leaky_file.exists()
        assert "/user/privateuser" not in leaky_file.read_text()
        assert report.get("scan_errors", []) == []


class TestCategoryAssignment:
    def test_docs_category(self, mock_repo: Path, output_dir: Path):
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        docs_entries = [e for e in report["manifest_entries"] if e["category"] == "docs"]
        assert len(docs_entries) >= 2  # REVIEWER_README.md + pyproject.toml

    def test_code_category(self, mock_repo: Path, output_dir: Path):
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        code_entries = [e for e in report["manifest_entries"] if e["category"].startswith("code/")]
        assert len(code_entries) > 0

    def test_benchmark_category(self, mock_repo: Path, output_dir: Path):
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        bench_entries = [e for e in report["manifest_entries"] if e["category"] == "benchmark"]
        assert len(bench_entries) > 0
        assert report["steps"]["benchmark_annotations"] == len(bench_entries)

    def test_artifacts_category(self, mock_repo: Path, output_dir: Path):
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        artifact_entries = [
            e for e in report["manifest_entries"] if e["category"].startswith("artifacts/")
        ]
        assert len(artifact_entries) > 0


class TestFileSizeCap:
    def test_large_file_flagged(self, mock_repo: Path, output_dir: Path):
        """A file exceeding the size cap should be flagged in scan."""
        # Write a large file
        large_file = mock_repo / "aaai_code" / "configs" / "large_config.yaml"
        large_file.write_text("x" * 200)
        report = build_anonymous_release(mock_repo, output_dir, dry_run=False, max_file_bytes=100)
        size_errors = [e for e in report.get("scan_errors", []) if "exceeds size limit" in e]
        assert len(size_errors) > 0


class TestSha256:
    def test_sha256_deterministic(self, tmp_path: Path):
        p = tmp_path / "test.txt"
        p.write_text("hello")
        assert sha256_file(p) == sha256_file(p)

    def test_sha256_length(self, tmp_path: Path):
        p = tmp_path / "test.txt"
        p.write_text("hello")
        assert len(sha256_file(p)) == 64


class TestWriteHelpers:
    def test_write_json(self, tmp_path: Path):
        p = tmp_path / "test.json"
        write_json(p, {"key": "value"})
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["key"] == "value"

    def test_write_text(self, tmp_path: Path):
        p = tmp_path / "test.txt"
        write_text(p, "hello world")
        assert p.read_text() == "hello world"

    def test_write_json_creates_parents(self, tmp_path: Path):
        p = tmp_path / "a" / "b" / "test.json"
        write_json(p, {"key": "value"})
        assert p.exists()


class TestManifestStructure:
    def test_manifest_has_all_fields(self, mock_repo: Path, output_dir: Path):
        """Each manifest entry should have required fields."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        for entry in report["manifest_entries"]:
            assert "relative" in entry
            assert "size" in entry
            assert "sha256" in entry
            assert "category" in entry

    def test_manifest_sha256_dry_run(self, mock_repo: Path, output_dir: Path):
        """In dry-run mode, sha256 should be 'dry-run'."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        for entry in report["manifest_entries"]:
            assert entry["sha256"] == "dry-run"

    def test_manifest_sha256_execute(self, mock_repo: Path, output_dir: Path):
        """In execute mode, sha256 should be a real hash."""
        report = build_anonymous_release(mock_repo, output_dir, dry_run=False)
        for entry in report["manifest_entries"]:
            assert entry["sha256"] != "dry-run"
            assert len(entry["sha256"]) == 64


class TestSummary:
    def test_summary_counts(self, mock_repo: Path, output_dir: Path):
        report = build_anonymous_release(mock_repo, output_dir, dry_run=True)
        summary = report["summary"]
        assert "total_files_copied" in summary
        assert "total_bytes" in summary
        assert "scan_violations" in summary
        assert summary["total_files_copied"] > 0
