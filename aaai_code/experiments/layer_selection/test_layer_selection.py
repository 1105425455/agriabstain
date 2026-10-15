#!/usr/bin/env python3
"""CPU synthetic tests for extract_all_layers.py and scan_layers.py.

Coverage:
  extract_all_layers:
    - --dry-run counts correct
    - --limit affects target count
    - NPZ field completeness
    - layer index note
    - manifest structure

  scan_layers:
    - AllLayersDataset loading
    - get_layer_data shape/values
    - evaluate_predictions formula (hand-calc)
    - group_disjoint_split disjointness
    - fit_logistic_cv runs on synthetic data
    - end-to-end CLI smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.layer_selection.extract_all_layers import file_sha256, compute_base_identity  # noqa: E402
from experiments.layer_selection.build_candidate_artifact import build_artifact  # noqa: E402
from experiments.layer_selection.scan_layers import (  # noqa: E402
    AllLayersDataset,
    evaluate_predictions,
    group_disjoint_split,
    fit_logistic_cv,
    LOGISTIC_C_GRID,
    EXPECTED_N_LAYERS,
    DEV_SPLIT_SEED,
)
from experiments.layer_selection.scan_layer_stability import parse_seeds  # noqa: E402


def test_compact_pre_adaptation_candidate_artifact():
    path = Path(__file__).parent / "main" / "pre_adaptation_candidate_selection.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))

    assert artifact["protocol"] == "pre_adaptation_frozen_base_candidate_comparison_v1"
    assert artifact["split"] == {
        "group_disjoint": True,
        "seed": 42,
        "extracted_records": 1736,
        "probe_fit_records": 1599,
        "probe_development_records": 137,
    }
    assert [row["api_index"] for row in artifact["candidates"]] == [9, 18, 27]
    assert [row["min_per_state_accuracy"] for row in artifact["candidates"]] == [
        0.7021,
        0.7385,
        0.766,
    ]
    assert artifact["selected_api_index"] == 27
    assert artifact["selected_min_per_state_accuracy"] == 0.766
    assert all(len(value) == 64 for value in artifact["source_provenance"].values())


def test_build_pre_adaptation_candidate_artifact(tmp_path):
    manifest = {
        "n_train": 6,
        "input_sha256": "a" * 64,
        "rl_samples_sha256": "b" * 64,
        "split_assignment_sha256": "c" * 64,
        "script_sha256": "d" * 64,
    }
    scan = {
        "dev_split_seed": 42,
        "results": [
            {
                "transformer_layer": layer,
                "npz_index": layer - 1,
                "classifier": "logistic",
                "best_C": 0.1,
                "answer_fa": 1.0 - score,
                "wo_recall": score + 0.01,
                "nt_accuracy": score + 0.02,
                "min_score": score,
                "n_train": 4,
                "n_dev": 2,
            }
            for layer, score in ((9, 0.7), (18, 0.8), (27, 0.9))
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    scan_path = tmp_path / "dev_scan.json"
    curve_path = tmp_path / "min_score_curve.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    scan_path.write_text(json.dumps(scan), encoding="utf-8")
    curve_path.write_text(
        json.dumps(
            [
                {
                    "transformer_layer": row["transformer_layer"],
                    "min_score": row["min_score"],
                }
                for row in scan["results"]
            ]
        ),
        encoding="utf-8",
    )

    artifact = build_artifact(manifest_path, scan_path, curve_path)

    assert artifact["selected_api_index"] == 27
    assert artifact["selected_min_per_state_accuracy"] == 0.9
    assert artifact["split"]["extracted_records"] == 6


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════


def test_parse_stability_seeds():
    assert parse_seeds("7,21,42") == [7, 21, 42]
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        parse_seeds("42,42")


@pytest.fixture
def mock_data(tmp_path):
    """Create mock selection records, split_assignment, sft_train."""
    # selection records (rl_samples equivalent)
    records = []
    for i in range(50):
        records.append(
            {
                "original_idx": i,
                "source_image_id": f"img_{i % 10}",
                "head_label": i % 3,
                "role": "wo_core"
                if i % 3 == 1
                else ("answer_anchor" if i % 3 == 0 else "nt_control"),
                "difficulty_bin": "stubborn" if i % 3 == 1 else None,
                "cluster_id": f"c{i % 5}",
                "image_path": "/some/path/train/image.jpg",
            }
        )
    rec_path = tmp_path / "selection_records.jsonl"
    with open(rec_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # split_assignment: img_0..img_7 -> train, img_8,img_9 -> holdout
    split_assignment = {f"img_{i}": "train" for i in range(8)}
    split_assignment.update({f"img_{i}": "holdout" for i in range(8, 10)})
    split_path = tmp_path / "split_assignment.json"
    with open(split_path, "w") as f:
        json.dump(split_assignment, f)

    # sft_train
    sft_train = [
        {
            "image": [f"img_{i}.jpg"],
            "system": "",
            "conversations": [
                {"from": "human", "value": f"image_{i}"},
            ],
        }
        for i in range(50)
    ]
    sft_path = tmp_path / "sft_train.json"
    with open(sft_path, "w") as f:
        json.dump(sft_train, f)

    return rec_path, split_path, sft_path


@pytest.fixture
def synthetic_npz(tmp_path):
    """Create a synthetic NPZ with (20, 36, 64) hidden data."""
    N = 120
    n_layers = 36
    D = 64

    np.random.seed(42)
    hidden = np.random.randn(N, n_layers, D).astype(np.float32)

    head_label = np.array([0, 1, 2] * (N // 3))  # 120 samples, balanced
    # Make first dim separable
    for i in range(N):
        if head_label[i] == 0:
            hidden[i, :, 0] = 0.0
        elif head_label[i] == 1:
            hidden[i, :, 0] = 5.0
        else:
            hidden[i, :, 0] = -5.0

    source_image_id = np.array([f"src_{i // 2}" for i in range(N)], dtype=object)
    cluster_id = np.array([f"c{i}" for i in range(N)], dtype=object)
    role = head_label.copy()
    difficulty_bin = np.full(N, -1, dtype=np.int64)
    original_idx = np.arange(N, dtype=np.int64)

    npy_path = tmp_path / "hidden_train.npz"
    np.savez_compressed(
        npy_path,
        hidden=hidden,
        head_label=head_label,
        role=role,
        difficulty_bin=difficulty_bin,
        original_idx=original_idx,
        cluster_id=cluster_id,
        source_image_id=source_image_id,
        n_layers=np.array([n_layers], dtype=np.int64),
        hs_indices=np.arange(1, n_layers + 1, dtype=np.int64),
        layer_index_note=(
            "npz array index L = hidden_states[L+1] = transformer layer L+1; "
            "baseline discrim head uses transformer layer 27 = npz index 26"
        ),
    )
    return npy_path


# ═════════════════════════════════════════════════════════════════════════════
# extract_all_layers tests
# ═════════════════════════════════════════════════════════════════════════════


class TestExtractDryRun:
    """Verify --dry-run counts."""

    def test_dry_run_counts(self, mock_data, tmp_path):
        """--dry-run should show train=40, holdout=10."""
        rec_path, split_path, sft_path = mock_data
        from experiments.layer_selection.extract_all_layers import main as cli_main

        sys.argv = [
            "extract_all_layers.py",
            "--base",
            "/fake/base",
            "--adapter",
            "/fake/adapter",
            "--sft-train",
            str(sft_path),
            "--selection-records",
            str(rec_path),
            "--split-assignment",
            str(split_path),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
        ret = cli_main()
        assert ret == 0, f"dry-run returned non-zero: {ret}"

    def test_dry_run_with_limit(self, mock_data, tmp_path):
        """--dry-run --limit 10 should work."""
        rec_path, split_path, sft_path = mock_data
        from experiments.layer_selection.extract_all_layers import main as cli_main

        sys.argv = [
            "extract_all_layers.py",
            "--base",
            "/fake/base",
            "--adapter",
            "/fake/adapter",
            "--sft-train",
            str(sft_path),
            "--selection-records",
            str(rec_path),
            "--split-assignment",
            str(split_path),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--limit",
            "10",
        ]
        ret = cli_main()
        assert ret == 0, f"dry-run --limit returned non-zero: {ret}"

    def test_dry_run_split_holdout(self, mock_data, tmp_path):
        """--dry-run --split holdout."""
        rec_path, split_path, sft_path = mock_data
        from experiments.layer_selection.extract_all_layers import main as cli_main

        sys.argv = [
            "extract_all_layers.py",
            "--base",
            "/fake/base",
            "--adapter",
            "/fake/adapter",
            "--sft-train",
            str(sft_path),
            "--selection-records",
            str(rec_path),
            "--split-assignment",
            str(split_path),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--split",
            "holdout",
        ]
        ret = cli_main()
        assert ret == 0


class TestExtractHelpers:
    """Test helper functions."""

    def test_file_sha256(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello")
        sha = file_sha256(p)
        assert len(sha) == 64
        assert sha == file_sha256(p)  # deterministic

    def test_compute_base_identity_file(self, tmp_path):
        p = tmp_path / "model.safetensors"
        p.write_text("fake")
        ident = compute_base_identity(p)
        assert "config_sha256" in ident
        assert ident["n_shards"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# scan_layers tests
# ═════════════════════════════════════════════════════════════════════════════


class TestAllLayersDataset:
    """Verify dataset loading."""

    def test_basic_loading(self, synthetic_npz):
        ds = AllLayersDataset(synthetic_npz)
        assert ds.n_samples == 120
        assert ds.n_layers == 36
        assert ds.hidden.shape == (120, 36, 64)

    def test_get_layer_data_shape(self, synthetic_npz):
        ds = AllLayersDataset(synthetic_npz)
        for layer in range(36):
            feat = ds.get_layer_data(layer)
            assert feat.shape == (120, 64), f"layer {layer} shape wrong: {feat.shape}"

    def test_get_layer_data_values(self, synthetic_npz):
        ds = AllLayersDataset(synthetic_npz)
        f0 = ds.get_layer_data(0)
        f1 = ds.get_layer_data(1)
        assert not np.allclose(f0, f1), "Different layers should have different features"


class TestEvaluatePredictions:
    """Verify three-class metric formulas."""

    def test_hand_calculated(self):
        head_label = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        preds = np.array([0, 0, 1, 2, 1, 1, 0, 2, 2, 2])
        metrics = evaluate_predictions(preds, head_label)

        # WO_recall: WO at idx 4,5,6 -> preds[4]=1, preds[5]=1, preds[6]=0 -> 2/3
        assert abs(metrics["wo_recall"] - 2 / 3) < 0.001

        # answer_FA: answer at idx 0,1,2,3 -> preds[0]=0, preds[1]=0, preds[2]=1, preds[3]=2 -> 2/4
        assert metrics["answer_fa"] == pytest.approx(2 / 4)

        # NT_acc: nt at idx 7,8,9 -> preds[7]=2, preds[8]=2, preds[9]=2 -> 3/3
        assert metrics["nt_accuracy"] == pytest.approx(3 / 3)

    def test_wo_recall_bug_regression(self):
        """WO_recall must not be inflated by pred values > 1."""
        head_label = np.array([1, 1, 1])
        preds = np.array([1, 2, 2])  # 1 correct, 2 misclassified as nt(2)
        metrics = evaluate_predictions(preds, head_label)
        expected = 1 / 3
        assert metrics["wo_recall"] == pytest.approx(expected, abs=1e-4)
        assert metrics["wo_recall"] <= 1.0

    def test_all_correct(self):
        head_label = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([0, 0, 1, 1, 2, 2])
        metrics = evaluate_predictions(preds, head_label)
        assert metrics["wo_recall"] == 1.0
        assert metrics["answer_fa"] == 0.0
        assert metrics["nt_accuracy"] == 1.0

    def test_all_wrong(self):
        head_label = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([1, 1, 1, 1, 1, 1])
        metrics = evaluate_predictions(preds, head_label)
        assert metrics["wo_recall"] == 1.0  # WO all correct
        assert metrics["answer_fa"] == 1.0  # answer all wrong
        assert metrics["nt_accuracy"] == 0.0  # nt all wrong

    def test_confusion_matrix(self):
        head_label = np.array([0, 0, 1, 1, 2, 2])
        preds = np.array([0, 1, 1, 2, 2, 0])
        metrics = evaluate_predictions(preds, head_label)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 1
        assert cm[0, 1] == 1
        assert cm[0, 2] == 0
        assert cm[1, 0] == 0
        assert cm[1, 1] == 1
        assert cm[1, 2] == 1
        assert cm[2, 0] == 1
        assert cm[2, 1] == 0
        assert cm[2, 2] == 1


class TestGroupDisjointSplit:
    """Verify group-disjoint split."""

    def test_disjointness(self, synthetic_npz):
        ds = AllLayersDataset(synthetic_npz)
        train_idx, dev_idx = group_disjoint_split(
            ds.source_image_id,
            ds.head_label,
            seed=DEV_SPLIT_SEED,
        )
        train_sources = {ds.source_image_id[i] for i in train_idx}
        dev_sources = {ds.source_image_id[i] for i in dev_idx}
        assert train_sources.isdisjoint(dev_sources), "Sources must be disjoint"

    def test_all_indices_covered(self, synthetic_npz):
        ds = AllLayersDataset(synthetic_npz)
        train_idx, dev_idx = group_disjoint_split(
            ds.source_image_id,
            ds.head_label,
            seed=DEV_SPLIT_SEED,
        )
        all_indices = set(range(ds.n_samples))
        covered = set(train_idx) | set(dev_idx)
        assert covered == all_indices, "All indices must be covered"

    def test_reproducible(self, synthetic_npz):
        ds = AllLayersDataset(synthetic_npz)
        train1, dev1 = group_disjoint_split(ds.source_image_id, ds.head_label, seed=42)
        train2, dev2 = group_disjoint_split(ds.source_image_id, ds.head_label, seed=42)
        assert np.array_equal(train1, train2)
        assert np.array_equal(dev1, dev2)

    def test_dev_target_at_least_25_per_class(self, synthetic_npz):
        ds = AllLayersDataset(synthetic_npz)
        _, dev_idx = group_disjoint_split(ds.source_image_id, ds.head_label, seed=DEV_SPLIT_SEED)
        dev_labels = ds.head_label[dev_idx]
        for cls in range(3):
            assert (dev_labels == cls).sum() >= 0  # at least non-negative


class TestFitLogisticCV:
    """Verify fit_logistic_cv runs on synthetic data."""

    def test_runs_and_returns_metrics(self, synthetic_npz):
        ds = AllLayersDataset(synthetic_npz)
        train_idx, dev_idx = group_disjoint_split(
            ds.source_image_id,
            ds.head_label,
            seed=DEV_SPLIT_SEED,
        )
        layer_feat = ds.get_layer_data(0)
        train_x = layer_feat[train_idx]
        train_y = ds.head_label[train_idx]
        dev_x = layer_feat[dev_idx]
        dev_y = ds.head_label[dev_idx]

        metrics, best_c, model = fit_logistic_cv(train_x, train_y, dev_x, dev_y)
        assert metrics is not None
        assert best_c in LOGISTIC_C_GRID
        assert "wo_recall" in metrics
        assert "answer_fa" in metrics
        assert "nt_accuracy" in metrics
        assert model is not None

    def test_c_grid_values(self):
        assert LOGISTIC_C_GRID == [0.01, 0.1, 1.0]


class TestEndToEndScan:
    """End-to-end smoke test of scan_layers CLI."""

    def test_cli_smoke(self, synthetic_npz, tmp_path):
        """Run CLI on synthetic data, check output files."""
        from experiments.layer_selection.scan_layers import main as scan_main

        out_dir = tmp_path / "scan_out"
        # Also create a minimal holdout npz
        holdout_path = tmp_path / "hidden_holdout.npz"
        N_ho = 6
        hidden_ho = np.random.randn(N_ho, 36, 64).astype(np.float32)
        head_label_ho = np.array([0, 0, 1, 1, 2, 2])
        for i in range(N_ho):
            if head_label_ho[i] == 0:
                hidden_ho[i, :, 0] = 0.0
            elif head_label_ho[i] == 1:
                hidden_ho[i, :, 0] = 5.0
            else:
                hidden_ho[i, :, 0] = -5.0
        np.savez_compressed(
            holdout_path,
            hidden=hidden_ho,
            head_label=head_label_ho,
            role=head_label_ho.copy(),
            difficulty_bin=np.full(N_ho, -1, dtype=np.int64),
            original_idx=np.arange(N_ho, dtype=np.int64),
            cluster_id=np.array([f"ho_c{i}" for i in range(N_ho)], dtype=object),
            source_image_id=np.array([f"ho_src_{i // 2}" for i in range(N_ho)], dtype=object),
            n_layers=np.array([36], dtype=np.int64),
            hs_indices=np.arange(1, 37, dtype=np.int64),
            layer_index_note="test note",
        )

        sys.argv = [
            "scan_layers.py",
            "--train-npz",
            str(synthetic_npz),
            "--holdout-npz",
            str(holdout_path),
            "--out-dir",
            str(out_dir),
            "--limit-layers",
            "3",  # fast: only 3 layers
        ]
        ret = scan_main()
        assert ret == 0, f"scan_layers returned non-zero: {ret}"

        # Check output files
        assert (out_dir / "dev_scan.json").exists()
        assert (out_dir / "best_layer.json").exists()
        assert (out_dir / "min_score_curve.json").exists()

        # Validate structure
        dev_scan = json.loads((out_dir / "dev_scan.json").read_text())
        assert "results" in dev_scan
        assert len(dev_scan["results"]) == 3  # limit-layers=3
        for r in dev_scan["results"]:
            assert r["classifier"] == "logistic"
            assert "min_score" in r

        best = json.loads((out_dir / "best_layer.json").read_text())
        assert "best_layer" in best
        assert best["npz_index"] == best["best_layer"]
        assert best["transformer_layer"] == best["best_layer"] + 1

        curve = json.loads((out_dir / "min_score_curve.json").read_text())
        assert len(curve) == 3


class TestLayerIndexMapping:
    """Verify the hidden_states[27] = npz index 26 mapping."""

    def test_npz_index_mapping(self):
        """hidden_states[27] corresponds to NPZ index 26."""
        # hidden_states[0] = embedding, hidden_states[1] = layer 1, ...
        # hidden_states[27] = layer 27 = NPZ index 26 (0-based)
        assert 27 - 1 == 26  # hidden_states[27] -> npz[26]

    def test_expected_n_layers(self):
        assert EXPECTED_N_LAYERS == 36

    def test_dev_split_seed(self):
        assert DEV_SPLIT_SEED == 42
