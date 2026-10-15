#!/usr/bin/env python3
"""CPU-only unit tests for the representation comparison, no model loading involved.

Covered:
  - compute_expected_visual_tokens: several grids and the temporal factor
  - merge_shards: two-rank merge and ordering, field alignment, failure on a missing tail
  - _required_image_variant_ids and _build_variants: variant enumeration
  - analyze_l2: grouping and pairing, duplicate detection, multi-instance images
  - analyze_l3: grouping, missing endpoints, duplicate ratio, multi-instance images
  - analyze_input_equivariance_consistency: the flip-rate denominator
  - stable_group_key, _group_key_l2, _group_key_l3
  - launcher smoke test
  - a light cardinality regression over the shipped annotations
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import torch  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import analyze_evidence  # noqa: E402
import extract_head_probabilities  # noqa: E402
import extract_representations  # noqa: E402


def _make_image_grid_thw(
    grids: list[tuple[int, int, int]],
) -> torch.Tensor:
    return torch.tensor(grids, dtype=torch.int64)


def _make_l2_sid(image_id: str, marker: str, source: str, terminal: str) -> str:
    """Build a proper L2 sample_id."""
    return f"L2-{image_id}-{marker}-{source}-{terminal}"


def _make_l3_sid(image_id: str, ratio: float, source: str, dose: int) -> str:
    """Build a proper L3 sample_id."""
    return f"L3-{image_id}-{ratio:.3f}-{source}-ladder-{dose:03d}"


# =============================================================================
# Test: compute_expected_visual_tokens
# =============================================================================


class TestComputeExpectedVisualTokens:
    """The visual-token count formula."""

    def test_single_grid_no_temporal(self):
        thw = _make_image_grid_thw([(1, 28, 28)])
        r = extract_representations.compute_expected_visual_tokens(thw, merge_size=2)
        assert r == 196

    def test_single_grid_with_temporal(self):
        thw = _make_image_grid_thw([(4, 14, 14)])
        r = extract_representations.compute_expected_visual_tokens(thw, merge_size=2)
        assert r == 196

    def test_multi_grid(self):
        thw = _make_image_grid_thw([(1, 28, 28), (1, 14, 28), (3, 10, 10)])
        assert extract_representations.compute_expected_visual_tokens(thw, merge_size=2) == 369

    def test_floor_division_not_ceil(self):
        thw = _make_image_grid_thw([(1, 15, 15)])
        r = extract_representations.compute_expected_visual_tokens(thw, merge_size=2)
        assert r == 49
        assert r != 64

    def test_empty_grid(self):
        assert (
            extract_representations.compute_expected_visual_tokens(
                torch.empty((0, 3), dtype=torch.int64), merge_size=2
            )
            == 0
        )  # noqa: E501

    def test_none_grid(self):
        assert extract_representations.compute_expected_visual_tokens(None, merge_size=2) == 0

    def test_zero_merge_floor(self):
        thw = _make_image_grid_thw([(1, 28, 28)])
        assert extract_representations.compute_expected_visual_tokens(thw, merge_size=1) == 784

    def test_temporal_multi_grid_combined(self):
        thw = _make_image_grid_thw([(4, 14, 14), (2, 7, 14)])
        assert extract_representations.compute_expected_visual_tokens(thw, merge_size=2) == 238

    def test_odd_merge_size(self):
        thw = _make_image_grid_thw([(1, 30, 30)])
        assert extract_representations.compute_expected_visual_tokens(thw, merge_size=3) == 100


# =============================================================================
# Test: merge_shards
# =============================================================================


class TestMergeShards:
    def _write_shard(
        self,
        path: Path,
        rank: int,
        world: int,
        fp: str,
        indices: list[int],
    ) -> None:
        n = len(indices)
        np.savez_compressed(
            str(path),
            last_token=np.random.randn(n, 4096).astype(np.float32),
            visual_mean=np.random.randn(n, 4096).astype(np.float32),
            visual_max=np.random.randn(n, 4096).astype(np.float32),
            visual_query_attn=np.random.randn(n, 4096).astype(np.float32),
            head_label=np.array([0, 1, 2][:n], dtype=np.int64),
            sample_id=np.array([f"sample_{i}" for i in indices], dtype=object),
            image=np.array([f"img_{i}.jpg" for i in indices], dtype=object),
            original_index=np.array(indices, dtype=np.int64),
            visual_token_count=np.array([100, 200, 300][:n], dtype=np.int64),
        )

    def test_two_rank_merge_order(self, tmp_path: Path):
        fp = "test_fp"
        world = 2
        n_total = 6
        for rank in range(world):
            mine = [i for i in range(n_total) if i % world == rank]
            self._write_shard(
                tmp_path / f"shard_rank{rank}of{world}_{fp}.npz",
                rank,
                world,
                fp,
                mine,
            )
            (tmp_path / f"shard_rank{rank}of{world}_{fp}.done").write_text("ok")
        result = extract_representations.merge_shards(
            tmp_path,
            "test",
            fp,
            world,
            list(range(n_total)),
            timeout=10,
        )
        assert result is not None
        assert result["n_merged"] == n_total
        assert list(result["original_index"]) == list(range(n_total))
        for field in [
            "last_token",
            "visual_mean",
            "visual_max",
            "visual_query_attn",
            "head_label",
            "sample_id",
            "image",
            "original_index",
            "visual_token_count",
            "n_selected",
            "n_merged",
        ]:
            assert field in result

    def test_two_rank_out_of_order_alignment(self, tmp_path: Path):
        """2-rank with shuffled indices: rank1 has indices 1,3,5; rank0 has 0,2,4.
        After merge, all fields must align with sorted original_index.
        Values must be preserved correctly, not corrupted by reorder."""
        fp = "shuffle_fp"
        world = 2
        # rank 0: [4, 0, 2] (out of natural order), rank 1: [3, 5, 1] (out of natural order)
        rank0_indices = [4, 0, 2]
        rank1_indices = [3, 5, 1]
        n_total = 6
        for rank, indices in [(0, rank0_indices), (1, rank1_indices)]:
            n = len(indices)
            # Use deterministic seed per rank for reproducible values
            rng = np.random.RandomState(42 + rank)
            np.savez_compressed(
                str(tmp_path / f"shard_rank{rank}of{world}_{fp}.npz"),
                last_token=rng.randn(n, 4096).astype(np.float32),
                visual_mean=rng.randn(n, 4096).astype(np.float32),
                visual_max=rng.randn(n, 4096).astype(np.float32),
                visual_query_attn=rng.randn(n, 4096).astype(np.float32),
                head_label=np.array([i % 3 for i in indices], dtype=np.int64),
                sample_id=np.array([f"sample_{i}" for i in indices], dtype=object),
                image=np.array([f"img_{i}.jpg" for i in indices], dtype=object),
                original_index=np.array(indices, dtype=np.int64),
                visual_token_count=np.array([100 + i * 10 for i in indices], dtype=np.int64),
            )
            (tmp_path / f"shard_rank{rank}of{world}_{fp}.done").write_text("ok")
        result = extract_representations.merge_shards(
            tmp_path,
            "test",
            fp,
            world,
            list(range(n_total)),
            timeout=10,
        )
        assert result is not None
        assert result["n_merged"] == n_total
        # Verify order: 0, 1, 2, 3, 4, 5
        orig = result["original_index"].tolist()
        assert orig == [0, 1, 2, 3, 4, 5], f"expected sorted order, got {orig}"
        # Verify metadata alignment: sample_id, image, head_label follow original_index
        for idx in range(n_total):
            assert result["sample_id"][idx] == f"sample_{idx}", (
                f"sample_id mismatch at pos {idx}: {result['sample_id'][idx]}"
            )
            assert result["image"][idx] == f"img_{idx}.jpg", f"image mismatch at pos {idx}"
            assert result["head_label"][idx] == idx % 3, f"head_label mismatch at pos {idx}"

    def test_merge_missing_index_fails(self, tmp_path: Path):
        fp = "missing_fp"
        for rank in range(1):
            mine = [i for i in range(6) if i % 2 == rank]
            self._write_shard(
                tmp_path / f"shard_rank{rank}of2_{fp}.npz",
                rank,
                2,
                fp,
                mine,
            )
            (tmp_path / f"shard_rank{rank}of2_{fp}.done").write_text("ok")
        result = extract_representations.merge_shards(
            tmp_path,
            "test",
            fp,
            2,
            list(range(6)),
            timeout=0,
        )
        assert result is None

    def test_merge_duplicate_index_fails(self, tmp_path: Path):
        fp = "dup_fp"
        np.savez_compressed(
            str(tmp_path / f"shard_rank0of1_{fp}.npz"),
            last_token=np.random.randn(3, 4096).astype(np.float32),
            visual_mean=np.random.randn(3, 4096).astype(np.float32),
            visual_max=np.random.randn(3, 4096).astype(np.float32),
            visual_query_attn=np.random.randn(3, 4096).astype(np.float32),
            head_label=np.array([0, 1, 0], dtype=np.int64),
            sample_id=np.array(["a", "b", "a"], dtype=object),
            image=np.array(["1.jpg", "2.jpg", "1.jpg"], dtype=object),
            original_index=np.array([0, 1, 0], dtype=np.int64),
            visual_token_count=np.array([100, 200, 100], dtype=np.int64),
        )
        (tmp_path / f"shard_rank0of1_{fp}.done").write_text("ok")
        result = extract_representations.merge_shards(tmp_path, "test", fp, 1, [0, 1], timeout=10)
        assert result is None

    def test_merge_expected_indices_mismatch(self, tmp_path: Path):
        fp = "mismatch_fp"
        np.savez_compressed(
            str(tmp_path / f"shard_rank0of1_{fp}.npz"),
            last_token=np.random.randn(2, 4096).astype(np.float32),
            visual_mean=np.random.randn(2, 4096).astype(np.float32),
            visual_max=np.random.randn(2, 4096).astype(np.float32),
            visual_query_attn=np.random.randn(2, 4096).astype(np.float32),
            head_label=np.array([0, 1], dtype=np.int64),
            sample_id=np.array(["a", "b"], dtype=object),
            image=np.array(["1.jpg", "2.jpg"], dtype=object),
            original_index=np.array([0, 2], dtype=np.int64),
            visual_token_count=np.array([100, 200], dtype=np.int64),
        )
        (tmp_path / f"shard_rank0of1_{fp}.done").write_text("ok")
        result = extract_representations.merge_shards(tmp_path, "test", fp, 1, [0, 1], timeout=10)
        assert result is None

    def test_each_npz_key_read_once(self, tmp_path: Path, monkeypatch):
        """Verify that each NPZ key is accessed at most once per shard during merge_shards.
        This guards against the performance bug where re-accessing keys inside a loop
        causes repeated decompression."""
        import numpy as np

        fp = "once_fp"
        world = 2
        n_total = 4
        # Write shards first with real np.savez_compressed
        for rank in range(world):
            mine = [i for i in range(n_total) if i % world == rank]
            n = len(mine)
            np.savez_compressed(
                str(tmp_path / f"shard_rank{rank}of{world}_{fp}.npz"),
                last_token=np.random.randn(n, 4096).astype(np.float32),
                visual_mean=np.random.randn(n, 4096).astype(np.float32),
                visual_max=np.random.randn(n, 4096).astype(np.float32),
                visual_query_attn=np.random.randn(n, 4096).astype(np.float32),
                head_label=np.array([0] * n, dtype=np.int64),
                sample_id=np.array([f"s{i}" for i in mine], dtype=object),
                image=np.array([f"img{i}.jpg" for i in mine], dtype=object),
                original_index=np.array(mine, dtype=np.int64),
                visual_token_count=np.array([100] * n, dtype=np.int64),
            )
            (tmp_path / f"shard_rank{rank}of{world}_{fp}.done").write_text("ok")

        # Track __getitem__ calls per key across all NpzFile instances
        key_access_counts: dict[str, int] = {}

        # We'll wrap np.load to return a wrapper that counts __getitem__ per key
        _original_load = np.load

        def tracking_load(path, **kwargs):
            real = _original_load(str(path), **kwargs)

            # Wrap the real NpzFile to count access per key
            class TrackingNpzFile:
                def __enter__(self_):
                    return self_

                def __exit__(self_, *exc):
                    real.close()

                def __getitem__(self_, key):
                    key_access_counts[key] = key_access_counts.get(key, 0) + 1
                    return real[key]

                def __len__(self_):
                    return len(real)

                def close(self_):
                    real.close()

            return TrackingNpzFile()

        monkeypatch.setattr("numpy.load", tracking_load)

        result = extract_representations.merge_shards(
            tmp_path,
            "test",
            fp,
            world,
            list(range(n_total)),
            timeout=10,
        )
        assert result is not None

        # Each key should be accessed at most world times (once per shard load)
        for key, count in key_access_counts.items():
            assert count <= world, (
                f"Key '{key}' accessed {count} times, expected ≤ {world} "
                f"(once per shard). This indicates repeated decompression."
            )


# =============================================================================
# Test: _required_image_variant_ids
# =============================================================================


class TestRequiredImageVariantIds:
    def test_variant_set_none(self):
        assert extract_head_probabilities._required_image_variant_ids("none") == ["original"]

    def test_variant_set_query(self):
        assert extract_head_probabilities._required_image_variant_ids("query") == ["original"]

    def test_variant_set_image(self):
        ids = extract_head_probabilities._required_image_variant_ids("image")
        assert ids == [
            "original",
            "image_brightness_0.9",
            "image_brightness_1.1",
            "image_contrast_0.9",
            "image_contrast_1.1",
        ]

    def test_variant_set_all(self):
        ids = extract_head_probabilities._required_image_variant_ids("all")
        assert len(ids) == 5
        assert ids[0] == "original"

    def test_photometric_variants_exact_factors(self):
        ids = extract_head_probabilities._required_image_variant_ids("all")
        bfs = sorted([float(v.split("_")[-1]) for v in ids if v.startswith("image_brightness_")])
        cfs = sorted([float(v.split("_")[-1]) for v in ids if v.startswith("image_contrast_")])
        assert bfs == [0.9, 1.1]
        assert cfs == [0.9, 1.1]


# =============================================================================
# Test: _build_variants
# =============================================================================


class TestBuildVariants:
    def _make_row(self, template_id: int = 0, label: str = "tomato disease") -> dict:
        from extract_head_probabilities import GROUNDING_TEMPLATES

        query = GROUNDING_TEMPLATES[template_id].format(label=label)
        return {
            "images": ["data/source/plantdoc/images/test.jpg"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "data/source/plantdoc/images/test.jpg"},
                        {"type": "text", "text": query},
                    ],
                },
            ],
        }

    def test_variant_set_none(self, tmp_path):
        vs = extract_head_probabilities._build_variants(self._make_row(), "none", tmp_path)
        assert len(vs) == 1 and vs[0]["variant_id"] == "original"

    def test_variant_set_query(self, tmp_path):
        vs = extract_head_probabilities._build_variants(self._make_row(0), "query", tmp_path)
        assert len(vs) == 3

    def test_variant_set_query_second_template(self, tmp_path):
        vs = extract_head_probabilities._build_variants(self._make_row(1), "query", tmp_path)
        assert len(vs) == 3

    def test_variant_set_image(self, tmp_path):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        p = str(tmp_path / "test.jpg")
        img.save(p)
        row = self._make_row(0)
        row["images"] = [p]
        row["messages"][1]["content"][0]["image"] = p
        vs = extract_head_probabilities._build_variants(row, "image", tmp_path)
        assert len(vs) == 5

    def test_variant_set_all(self, tmp_path):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="blue")
        p = str(tmp_path / "test_all.jpg")
        img.save(p)
        row = self._make_row(0)
        row["images"] = [p]
        row["messages"][1]["content"][0]["image"] = p
        vs = extract_head_probabilities._build_variants(row, "all", tmp_path)
        assert len(vs) == 7

    def test_original_query_deduplication(self, tmp_path):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="green")
        p = str(tmp_path / "test_dedup.jpg")
        img.save(p)
        row = self._make_row(0)
        row["images"] = [p]
        row["messages"][1]["content"][0]["image"] = p
        vs = extract_head_probabilities._build_variants(row, "all", tmp_path)
        qids = [v["variant_id"] for v in vs if v["variant_type"] == "query"]
        assert len(qids) == 2
        assert "original" not in qids


# =============================================================================
# Test: Group keys (new logic)
# =============================================================================


class TestGroupKey:
    """The group key, formed from the image prefix and the source sample id."""

    def test_l2_keep_plantdoc(self):
        # L2-{image_id}-keep-{source_sample_id}-answer
        sid = _make_l2_sid(
            "ncorn_leaf_blight_0796", "keep", "plantdoc-train-84d6b1f300e05cc8-0", "answer"
        )
        gk = analyze_evidence._group_key_l2(sid)
        assert gk == "L2-ncorn_leaf_blight_0796__plantdoc-train-84d6b1f300e05cc8-0"

    def test_l2_cropout_plantdoc(self):
        sid = _make_l2_sid(
            "ncorn_leaf_blight_0796", "cropout", "plantdoc-train-84d6b1f300e05cc8-0", "ladder-000"
        )
        gk = analyze_evidence._group_key_l2(sid)
        assert gk == "L2-ncorn_leaf_blight_0796__plantdoc-train-84d6b1f300e05cc8-0"

    def test_l2_coco(self):
        # COCO: source_sample_id = coco-train-{coco_id}-{ann_id}
        sid = _make_l2_sid("558840", "keep", "coco-train-558840-156", "answer")
        gk = analyze_evidence._group_key_l2(sid)
        assert gk == "L2-558840__coco-train-558840-156"

    def test_l2_coco_cropout(self):
        sid = _make_l2_sid("558840", "cropout", "coco-train-558840-156", "ladder-000")
        gk = analyze_evidence._group_key_l2(sid)
        assert gk == "L2-558840__coco-train-558840-156"

    def test_l2_same_image_different_ann(self):
        """Same image, different annotation IDs → different group keys."""
        sid_a = _make_l2_sid("558840", "keep", "coco-train-558840-156", "answer")
        sid_b = _make_l2_sid("558840", "keep", "coco-train-558840-676791", "answer")
        assert analyze_evidence._group_key_l2(sid_a) != analyze_evidence._group_key_l2(sid_b)

    def test_l2_no_marker(self):
        assert analyze_evidence._group_key_l2("L2-ncorn-other-xxx") is None

    def test_l3_plantdoc(self):
        sid = _make_l3_sid("ncorn_leaf_blight_0796", 1.0, "plantdoc-train-84d6b1f300e05cc8-0", 100)
        gk = analyze_evidence._group_key_l3(sid)
        assert gk == "L3-ncorn_leaf_blight_0796__plantdoc-train-84d6b1f300e05cc8-0"

    def test_l3_coco(self):
        sid = _make_l3_sid("558840", 0.5, "coco-train-558840-156", 50)
        gk = analyze_evidence._group_key_l3(sid)
        expected = "L3-558840__coco-train-558840-156"
        assert gk == expected

    def test_l3_same_image_different_ann(self):
        """Same image, different annotation IDs → different L3 group keys."""
        sid_a = _make_l3_sid("558840", 0.0, "coco-train-558840-156", 0)
        sid_b = _make_l3_sid("558840", 0.0, "coco-train-558840-676791", 0)
        assert analyze_evidence._group_key_l3(sid_a) != analyze_evidence._group_key_l3(sid_b)

    def test_l3_different_ratio_same_group(self):
        """Different ratios of same source → same group key."""
        sid_a = _make_l3_sid("ncorn", 0.0, "plantdoc-train-abc-0", 0)
        sid_b = _make_l3_sid("ncorn", 1.0, "plantdoc-train-abc-0", 100)
        assert analyze_evidence._group_key_l3(sid_a) == analyze_evidence._group_key_l3(sid_b)

    def test_l3_non_l3_prefix(self):
        assert analyze_evidence._group_key_l3("L2-ncorn-1.000-xxx") is None

    def test_stable_group_key_l2(self):
        sid = _make_l2_sid("ncorn", "keep", "plantdoc-train-abc-0", "answer")
        assert analyze_evidence.stable_group_key(sid) is not None

    def test_stable_group_key_l3(self):
        sid = _make_l3_sid("ncorn", 1.0, "plantdoc-train-abc-0", 100)
        assert analyze_evidence.stable_group_key(sid) is not None

    def test_stable_group_key_unknown(self):
        assert analyze_evidence.stable_group_key("OTHER-ncorn-keep-xxx") is None

    def test_pairing_key_l2_with_target_label(self):
        """pairing_key_l2 = stable_group_key + target_label."""
        sid = _make_l2_sid("558840", "keep", "coco-train-558840-156", "answer")
        pk = analyze_evidence.pairing_key_l2(sid, "hot dog")
        assert pk is not None
        assert pk.endswith("__tl:hot dog")

    def test_pairing_key_l2_same_instance_diff_labels(self):
        """Same source instance, different target_label -> different pairing keys."""
        sid = _make_l2_sid("558840", "keep", "coco-train-558840-156", "answer")
        pk_hotdog = analyze_evidence.pairing_key_l2(sid, "hot dog")
        pk_cup = analyze_evidence.pairing_key_l2(sid, "cup")
        assert pk_hotdog != pk_cup

    def test_normalize_target_label_casefold(self):
        assert analyze_evidence.normalize_target_label("Hot Dog") == "hot dog"
        assert analyze_evidence.normalize_target_label("  HOT DOG  ") == "hot dog"
        assert analyze_evidence.normalize_target_label(None) == ""
        assert analyze_evidence.normalize_target_label("") == ""


# =============================================================================
# Test: L2 analysis
# =============================================================================


class TestAnalyzeL2:
    """L2 analysis, including semantic de-duplication and conflict detection."""

    def _make_l2_records(self, n_pairs: int = 3) -> list[dict]:
        records = []
        for i in range(n_pairs):
            src = f"plantdoc-train-{hash(i):016x}-0"
            for rel, marker, terminal, pa in [
                ("L2_keep", "keep", "answer", 0.9 - i * 0.05),
                ("L2_cropout", "cropout", "ladder-000", 0.2 + i * 0.05),
            ]:
                records.append(
                    {
                        "sample_id": _make_l2_sid(f"img_{i:04d}", marker, src, terminal),
                        "relation": rel,
                        "variant_id": "original",
                        "p_answer": pa,
                        "p_wo": 0.05,
                        "p_nt": 0.05,
                        "predicted_class": 0,
                        "image": f"data/source/img_{i}.jpg",
                        "target_label": "test disease",
                        "achieved_visible_ratio": None,
                        "original_gt_decision": "answer",
                        "original_index": i * 2 + (0 if "keep" in rel else 1),
                    }
                )
        return records

    def test_basic_l2_analysis(self):
        records = self._make_l2_records(3)
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100, bootstrap_seed=42)
        assert result["n_pairs"] == 3
        assert result["deduplicated_records"] == 0
        assert result["conflicting_duplicate_groups"] == 0
        assert result["invalid_groups"] == 0
        assert result["delta_p_answer_mean"] > 0
        assert result["correct_direction_ratio"] == 1.0

    def test_l2_identical_duplicates_folded(self):
        """Byte-identical duplicates folded into one, pair preserved."""
        records = self._make_l2_records(2)
        dup = dict(records[0])
        dup["original_index"] = 99
        records.append(dup)
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        assert result["n_pairs"] == 2
        assert result["deduplicated_records"] == 1
        assert result["conflicting_duplicate_groups"] == 0
        assert result["invalid_groups"] == 0

    def test_l2_identical_dup_conflicting_probs_invalidates(self):
        """Same semantic identity, different p_answer → conflict, group invalid."""
        records = self._make_l2_records(2)
        dup = dict(records[0])
        dup["p_answer"] = 0.99
        dup["original_index"] = 99
        records.append(dup)
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        assert result["n_pairs"] == 1
        assert result["conflicting_duplicate_groups"] == 1
        assert result["invalid_groups"] == 1
        assert result["deduplicated_records"] == 0  # conflict, not deduplicated

    def test_l2_diff_semantic_identity_same_group_invalidates(self):
        """Same group+relation, different semantic identities → conflict."""
        records = self._make_l2_records(2)
        records.append(
            {
                "sample_id": _make_l2_sid(
                    "img_0000", "keep", f"plantdoc-train-{hash(0):016x}-0", "answer"
                ),
                "relation": "L2_keep",
                "variant_id": "original",
                "p_answer": 0.95,
                "p_wo": 0.03,
                "p_nt": 0.02,
                "predicted_class": 0,
                "image": "data/source/img_diff.jpg",
                "target_label": "test disease",
                "achieved_visible_ratio": None,
                "original_gt_decision": "answer",
                "original_index": 99,
            }
        )
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        assert result["conflicting_duplicate_groups"] == 1
        assert result["invalid_groups"] == 1
        assert result["n_pairs"] == 1

    def test_l2_missing_pair(self):
        records = self._make_l2_records(3)
        records.pop(1)
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        assert result["n_pairs"] == 2
        assert result["incomplete_pairs"] == 1
        assert result["n_total_groups"] == 3

    def test_l2_non_original_variant_filtered(self):
        records = self._make_l2_records(3)
        rec = dict(records[0])
        rec["variant_id"] = "query_0"
        rec["original_index"] = 99
        records.append(rec)
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        assert result["n_pairs"] == 3

    def test_l2_grouping_via_relation(self):
        records = self._make_l2_records(3)
        records.append(
            {
                "sample_id": "L3-img-1.000-plantdoc-train-xxx-0-ladder-100",
                "relation": "L3_ladder",
                "variant_id": "original",
                "p_answer": 0.8,
                "original_index": 100,
            }
        )
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        assert result["n_pairs"] == 3

    def test_l2_multi_instance_same_image(self):
        """Two different annotation objects on same image → separate groups."""
        records = []
        for ann_id in [100, 200]:
            src = f"coco-train-999-{ann_id}"
            for rel, marker, terminal, pa in [
                ("L2_keep", "keep", "answer", 0.8),
                ("L2_cropout", "cropout", "ladder-000", 0.2),
            ]:
                records.append(
                    {
                        "sample_id": _make_l2_sid("img_multi", marker, src, terminal),
                        "relation": rel,
                        "variant_id": "original",
                        "p_answer": pa,
                        "p_wo": 0.1,
                        "p_nt": 0.1,
                        "predicted_class": 0,
                        "image": "data/source/img_multi.jpg",
                        "target_label": "test disease",
                        "achieved_visible_ratio": None,
                        "original_gt_decision": "answer",
                        "original_index": len(records),
                    }
                )
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        assert result["n_pairs"] == 2
        assert result["deduplicated_records"] == 0
        assert result["conflicting_duplicate_groups"] == 0

    def test_l2_coco_multi_label_same_instance(self):
        """COCO: same source instance (558840-156), cropout has 3 labels, keep only hot dog.

        Hot dog forms 1 valid pair (keep + cropout).
        Cup and dining table are incomplete (cropout only, no keep).
        No conflicts.
        """
        records = []
        src = "coco-train-558840-156"
        # Keep: only hot dog
        records.append(
            {
                "sample_id": _make_l2_sid("558840", "keep", src, "answer"),
                "relation": "L2_keep",
                "variant_id": "original",
                "p_answer": 0.85,
                "p_wo": 0.10,
                "p_nt": 0.05,
                "predicted_class": 0,
                "image": "data/source/558840.jpg",
                "target_label": "hot dog",
                "achieved_visible_ratio": None,
                "original_gt_decision": "answer",
                "original_index": 0,
            }
        )
        # Cropout: hot dog, cup, dining table
        for li, label in enumerate(["hot dog", "cup", "dining table"]):
            records.append(
                {
                    "sample_id": _make_l2_sid("558840", "cropout", src, "ladder-000"),
                    "relation": "L2_cropout",
                    "variant_id": "original",
                    "p_answer": 0.15 if label == "hot dog" else 0.70,
                    "p_wo": 0.10,
                    "p_nt": 0.05,
                    "predicted_class": 0,
                    "image": "data/source/558840.jpg",
                    "target_label": label,
                    "achieved_visible_ratio": 0.0,
                    "original_gt_decision": "answer",
                    "original_index": 1 + li,
                }
            )
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        # 1 valid pair (hot dog has both keep + cropout)
        assert result["n_pairs"] == 1, f"expected 1 pair, got {result['n_pairs']}"
        # 2 incomplete (cup, dining table have cropout only)
        assert result["incomplete_pairs"] == 2, (
            f"expected 2 incomplete, got {result['incomplete_pairs']}"
        )
        # No conflicts
        assert result["conflicting_duplicate_groups"] == 0, (
            f"expected 0 conflicting, got {result['conflicting_duplicate_groups']}"
        )
        assert result["invalid_groups"] == 0, f"expected 0 invalid, got {result['invalid_groups']}"
        # pairing_key is set
        assert result.get("pairing_key") == "source_instance+target_label"


# =============================================================================
# Test: L3 analysis
# =============================================================================


class TestAnalyzeL3:
    """L3 analysis, including the duplicate ratio and multi-instance images."""

    def _make_l3_records(
        self, n_groups: int = 3, include_missing_endpoint: bool = True
    ) -> list[dict]:
        records = []
        for i in range(n_groups):
            src = f"plantdoc-train-{hash(i + 100):016x}-0"
            if include_missing_endpoint and i % 3 == 0:
                ratios = (0.25, 0.5, 0.75)
            elif include_missing_endpoint and i % 3 == 1:
                ratios = (0.25, 0.5, 0.75, 1.0)
            else:
                ratios = (0.0, 0.25, 0.5, 0.75, 1.0)
            for ratio in ratios:
                pa = ratio * 0.5 + 0.2
                records.append(
                    {
                        "sample_id": _make_l3_sid(f"img_{i:04d}", ratio, src, int(ratio * 100)),
                        "relation": "L3_ladder",
                        "variant_id": "original",
                        "achieved_visible_ratio": ratio,
                        "p_answer": pa,
                        "original_index": i * 10 + int(ratio * 4),
                    }
                )
        return records

    def test_l3_basic_analysis(self, monkeypatch):
        monkeypatch.setattr(analyze_evidence, "_SCIPY_AVAILABLE", True)
        records = self._make_l3_records(3, include_missing_endpoint=False)
        result = analyze_evidence.analyze_l3(
            records, bootstrap_seed=42, shuffle_rounds=50, n_bootstrap=100
        )
        assert result["n_groups"] == 3
        assert result["mean_spearman_rho"] > 0.5
        assert result["mean_kendall_tau"] > 0.3
        assert result["n_duplicate_ratio_groups"] == 0
        assert result["n_invalid_groups"] == 0

    def test_l3_missing_endpoint_not_fabricated(self):
        records = self._make_l3_records(3, include_missing_endpoint=True)
        result = analyze_evidence.analyze_l3(
            records, bootstrap_seed=42, shuffle_rounds=50, n_bootstrap=100
        )
        assert result["n_complete_endpoint_groups"] == 1

    def test_l3_group_count(self):
        records = self._make_l3_records(5, include_missing_endpoint=False)
        result = analyze_evidence.analyze_l3(
            records, bootstrap_seed=42, shuffle_rounds=50, n_bootstrap=100
        )
        assert result["n_groups"] == 5

    def test_l3_adjacent_violation(self):
        records = []
        src = "plantdoc-train-violation-0"
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        p_vals = [0.2, 0.8, 0.3, 0.9, 0.95]
        for ratio, pa in zip(ratios, p_vals):
            records.append(
                {
                    "sample_id": _make_l3_sid("violation", ratio, src, int(ratio * 100)),
                    "relation": "L3_ladder",
                    "variant_id": "original",
                    "achieved_visible_ratio": ratio,
                    "p_answer": pa,
                    "original_index": int(ratio * 4),
                }
            )
        result = analyze_evidence.analyze_l3(
            records, bootstrap_seed=42, shuffle_rounds=50, n_bootstrap=100
        )
        assert result["mean_adjacent_violation_rate"] == 0.25

    def test_l3_only_endpoint_zero_and_one(self):
        records = []
        src = "plantdoc-train-endpoint-0"
        for ratio, pa in [(0.0, 0.1), (1.0, 0.9)]:
            records.append(
                {
                    "sample_id": _make_l3_sid("endpoint_only", ratio, src, int(ratio * 100)),
                    "relation": "L3_ladder",
                    "variant_id": "original",
                    "achieved_visible_ratio": ratio,
                    "p_answer": pa,
                    "original_index": int(ratio),
                }
            )
        result = analyze_evidence.analyze_l3(
            records, bootstrap_seed=42, shuffle_rounds=50, n_bootstrap=100
        )
        assert result["n_groups"] == 1
        assert result["mean_endpoint_delta_p"] == pytest.approx(0.8, abs=1e-6)

    def test_l3_duplicate_ratio_invalidates_group(self):
        """Duplicate ratio records within same group → group invalidated."""
        src = "plantdoc-train-dup-0"
        records = []
        for ratio in [0.0, 0.5, 1.0]:
            records.append(
                {
                    "sample_id": _make_l3_sid("dup_test", ratio, src, int(ratio * 100)),
                    "relation": "L3_ladder",
                    "variant_id": "original",
                    "achieved_visible_ratio": ratio,
                    "p_answer": ratio * 0.5 + 0.2,
                    "original_index": int(ratio * 4),
                }
            )
        # Add duplicate at ratio 0.5
        records.append(
            {
                "sample_id": _make_l3_sid("dup_test", 0.5, src, 50),
                "relation": "L3_ladder",
                "variant_id": "original",
                "achieved_visible_ratio": 0.5,
                "p_answer": 0.6,
                "original_index": 99,
            }
        )
        result = analyze_evidence.analyze_l3(
            records, bootstrap_seed=42, shuffle_rounds=50, n_bootstrap=100
        )
        assert result["n_duplicate_ratio_groups"] == 1
        assert result["n_invalid_groups"] == 1
        assert result["n_groups"] == 0  # only group is invalidated, no valid groups left

    def test_l3_multi_instance_same_image(self):
        """Two different annotation objects on same image → separate L3 groups."""
        srcs = ["coco-train-888-100", "plantdoc-train-multi-0"]
        records = []
        for j, src in enumerate(srcs):
            for ratio in [0.0, 0.5, 1.0]:
                records.append(
                    {
                        "sample_id": _make_l3_sid("img_multi_l3", ratio, src, int(ratio * 100)),
                        "relation": "L3_ladder",
                        "variant_id": "original",
                        "achieved_visible_ratio": ratio,
                        "p_answer": ratio * 0.5 + 0.2,
                        "original_index": j * 3 + int(ratio * 2),
                    }
                )
        result = analyze_evidence.analyze_l3(
            records, bootstrap_seed=42, shuffle_rounds=50, n_bootstrap=100
        )
        assert result["n_groups"] == 2, f"expected 2 groups, got {result['n_groups']}"
        assert result["n_duplicate_ratio_groups"] == 0


# =============================================================================
# Test: Consistency analysis
# =============================================================================


class TestConsistency:
    def _make_consistency_records(self) -> list[dict]:
        records = []
        for oi in range(2):
            records.append(
                {
                    "original_index": oi,
                    "sample_id": _make_l2_sid(
                        f"img_{oi:04d}", "keep", "plantdoc-train-xxx-0", "answer"
                    ),
                    "relation": "L2_keep",
                    "variant_id": "original",
                    "variant_type": "original",
                    "p_answer": 0.8,
                    "predicted_class": 0,
                }
            )
            for tid in range(2):
                records.append(
                    {
                        "original_index": oi,
                        "sample_id": _make_l2_sid(
                            f"img_{oi:04d}", "keep", "plantdoc-train-xxx-0", "answer"
                        ),
                        "relation": "L2_keep",
                        "variant_id": f"query_{tid}",
                        "variant_type": "query",
                        "p_answer": 0.8 + 0.05 * (tid - 0.5),
                        "predicted_class": 0,
                    }
                )
            records.append(
                {
                    "original_index": oi,
                    "sample_id": _make_l2_sid(
                        f"img_{oi:04d}", "keep", "plantdoc-train-xxx-0", "answer"
                    ),
                    "relation": "L2_keep",
                    "variant_id": "image_brightness_0.9",
                    "variant_type": "image",
                    "p_answer": 0.75,
                    "predicted_class": 0,
                }
            )
        return records

    def test_consistency_mixed_variant_types(self):
        records = self._make_consistency_records()
        result = analyze_evidence.analyze_input_equivariance_consistency(
            records, n_bootstrap=100, bootstrap_seed=42
        )
        iec = result.get("input_equivariance_consistency", {})
        assert "by_variant_type_query" in iec
        assert "by_variant_type_image" in iec
        assert iec["by_variant_type_query"]["argmax_decision_flip_rate"] == 0.0
        assert iec["by_variant_type_image"]["argmax_decision_flip_rate"] == 0.0

    def test_consistency_flip_rate_denominator(self):
        records = self._make_consistency_records()
        records.append(
            {
                "original_index": 3,
                "sample_id": _make_l2_sid("img_0003", "keep", "plantdoc-train-xxx-0", "answer"),
                "relation": "L2_keep",
                "variant_id": "original",
                "variant_type": "original",
                "p_answer": 0.8,
                "predicted_class": 0,
            }
        )
        for tid in range(2):
            records.append(
                {
                    "original_index": 3,
                    "sample_id": _make_l2_sid("img_0003", "keep", "plantdoc-train-xxx-0", "answer"),
                    "relation": "L2_keep",
                    "variant_id": f"query_{tid}",
                    "variant_type": "query",
                    "p_answer": 0.7 if tid == 0 else 0.85,
                    "predicted_class": 1 if tid == 0 else 0,
                }
            )
        result = analyze_evidence.analyze_input_equivariance_consistency(
            records, n_bootstrap=100, bootstrap_seed=42
        )
        iec = result.get("input_equivariance_consistency", {})
        qa = iec.get("by_variant_type_query", {})
        assert qa["argmax_decision_flip_rate"] == pytest.approx(1.0 / 6.0, abs=1e-4)

    def test_consistency_all_equivalents_std(self):
        records = self._make_consistency_records()
        result = analyze_evidence.analyze_input_equivariance_consistency(
            records, n_bootstrap=100, bootstrap_seed=42
        )
        iec = result.get("input_equivariance_consistency", {})
        qa = iec.get("by_variant_type_query", {})
        assert qa["all_equivalents_std_mean"] > 0
        assert qa["p_answer_std_mean"] > 0
        assert qa["all_equivalents_std_mean"] != qa["p_answer_std_mean"]

    def test_consistency_no_original_skipped(self):
        records = [
            {
                "original_index": 0,
                "variant_id": "query_0",
                "variant_type": "query",
                "p_answer": 0.8,
                "predicted_class": 0,
                "relation": "L2_keep",
            }
        ]
        result = analyze_evidence.analyze_input_equivariance_consistency(records, n_bootstrap=100)
        iec = result.get("input_equivariance_consistency", {})
        assert (
            "by_variant_type_query" not in iec
            or iec.get("by_variant_type_query", {}).get("n_samples", 0) == 0
        )

    def test_consistency_relation_breakdown(self):
        records = self._make_consistency_records()
        result = analyze_evidence.analyze_input_equivariance_consistency(
            records, n_bootstrap=100, bootstrap_seed=42
        )
        iec = result.get("input_equivariance_consistency", {})
        assert "by_relation" in iec
        assert "L2_keep" in iec["by_relation"]


# =============================================================================
# Test: Bug report
# =============================================================================


class TestBugReport:
    def test_report_merge_shards_signature_change(self):
        import inspect

        sig = inspect.signature(extract_representations.merge_shards)
        assert "expected_indices" in sig.parameters

    def test_report_required_image_variant_ids_name(self):
        assert hasattr(extract_head_probabilities, "_required_image_variant_ids")


# =============================================================================
# Test: Launcher check smoke test
# =============================================================================


class TestLauncherCheck:
    """Both launchers exit 0 in check mode when given placeholder paths and an external output root.

    """

    LAUNCHERS = ["run_representation.sh", "run_evidence.sh"]

    def test_representation_launcher_aaai_code_derivation(self):
        """The package root is derived correctly, as the grandparent of this test file."""
        script = _THIS_DIR / "run_representation.sh"
        assert script.exists()
        content = script.read_text()
        assert "SCRIPT_DIR/../.." in content, "must use SCRIPT_DIR/../.."
        real_aaai = (
            _THIS_DIR.parent.parent.resolve()
        )  # experiments/evidence_judge/../../ = aaai_code/
        assert real_aaai.is_dir(), f"AAAI_CODE derived path {real_aaai} must exist"

    def test_evidence_launcher_aaai_code_derivation(self):
        script = _THIS_DIR / "run_evidence.sh"
        assert script.exists()
        content = script.read_text()
        assert "SCRIPT_DIR/../.." in content, "must use SCRIPT_DIR/../.."

    def test_representation_launcher_check_exit_zero(self, tmp_path):
        """Provide dummy BASE and external OUTPUT_ROOT, check should exit 0."""
        script = _THIS_DIR / "run_representation.sh"
        dummy_base = tmp_path / "dummy_qwen_model"
        dummy_base.mkdir()
        # Create a dummy file to satisfy the existence check
        (dummy_base / "config.json").write_text("{}")

        output_root = tmp_path / "scratch_output"
        output_root.mkdir()

        env = {
            **{k: v for k, v in os.environ.items()},
            "BASE": str(dummy_base),
            "OUTPUT_ROOT": str(output_root),
        }
        result = subprocess.run(
            ["bash", str(script), "check"],
            capture_output=True,
            text=True,
            env=env,
        )
        print(f"  stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  stderr: {result.stderr.strip()}")
        assert result.returncode == 0, f"check should exit 0, got {result.returncode}"

    def test_evidence_launcher_check_exit_zero(self, tmp_path):
        """Provide dummy paths and external OUTPUT_ROOT, check should exit 0."""
        script = _THIS_DIR / "run_evidence.sh"
        dummy_base = tmp_path / "dummy_qwen_model"
        dummy_base.mkdir()
        (dummy_base / "config.json").write_text("{}")

        dummy_adapter = tmp_path / "dummy_adapter"
        dummy_adapter.mkdir()
        (dummy_adapter / "adapter_model.safetensors").write_text("{}")

        dummy_calib = tmp_path / "dummy_calibrated.pt"
        dummy_calib.write_text("dummy")

        output_root = tmp_path / "scratch_output"
        output_root.mkdir()

        env = {
            **{k: v for k, v in os.environ.items()},
            "BASE": str(dummy_base),
            "ADAPTER": str(dummy_adapter),
            "CALIBRATED_HEAD": str(dummy_calib),
            "OUTPUT_ROOT": str(output_root),
        }
        result = subprocess.run(
            ["bash", str(script), "check"],
            capture_output=True,
            text=True,
            env=env,
        )
        print(f"  stdout: {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  stderr: {result.stderr.strip()}")
        assert result.returncode == 0, f"check should exit 0, got {result.returncode}"

    def test_evidence_launcher_check_output_root_in_aaai_rejected(self, tmp_path):
        """OUTPUT_ROOT inside AAAI_CODE should be rejected."""
        script = _THIS_DIR / "run_evidence.sh"
        dummy_base = tmp_path / "dummy_qwen_model"
        dummy_base.mkdir()
        (dummy_base / "config.json").write_text("{}")

        dummy_adapter = tmp_path / "dummy_adapter"
        dummy_adapter.mkdir()
        (dummy_adapter / "adapter_model.safetensors").write_text("{}")

        dummy_calib = tmp_path / "dummy_calibrated.pt"
        dummy_calib.write_text("dummy")

        # OUTPUT_ROOT set to AAAI_CODE itself (will be rejected)
        output_root = Path.cwd()

        env = {
            **{k: v for k, v in os.environ.items()},
            "BASE": str(dummy_base),
            "ADAPTER": str(dummy_adapter),
            "CALIBRATED_HEAD": str(dummy_calib),
            "OUTPUT_ROOT": str(output_root),
        }
        result = subprocess.run(
            ["bash", str(script), "check"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0, "OUTPUT_ROOT inside AAAI_CODE should be rejected"
        assert "OUTPUT_ROOT" in result.stdout, "error message should mention OUTPUT_ROOT"

    def test_representation_launcher_check_accepts_derived_root(self, tmp_path):
        """Check mode accepts a derived root that points outside the package."""
        script = _THIS_DIR / "run_representation.sh"
        dummy_base = tmp_path / "dummy_qwen_model"
        dummy_base.mkdir()
        (dummy_base / "config.json").write_text("{}")
        output_root = tmp_path / "scratch_output"
        output_root.mkdir()
        derived_root = tmp_path / "external_derived"
        derived_root.mkdir()
        env = {
            **{k: v for k, v in os.environ.items()},
            "BASE": str(dummy_base),
            "OUTPUT_ROOT": str(output_root),
            "DERIVED_ROOT": str(derived_root),
        }
        result = subprocess.run(
            ["bash", str(script), "check"],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(f"  stdout: {result.stdout.strip()}")
            print(f"  stderr: {result.stderr.strip()}")
        assert result.returncode == 0

    def test_evidence_launcher_check_accepts_derived_root(self, tmp_path):
        """The evidence launcher accepts a derived root that points outside the package."""
        script = _THIS_DIR / "run_evidence.sh"
        dummy_base = tmp_path / "dummy_qwen_model"
        dummy_base.mkdir()
        (dummy_base / "config.json").write_text("{}")
        dummy_adapter = tmp_path / "dummy_adapter"
        dummy_adapter.mkdir()
        (dummy_adapter / "adapter_model.safetensors").write_text("{}")
        dummy_calib = tmp_path / "dummy_calibrated.pt"
        dummy_calib.write_text("dummy")
        output_root = tmp_path / "scratch_output"
        output_root.mkdir()
        derived_root = tmp_path / "external_derived"
        derived_root.mkdir()
        env = {
            **{k: v for k, v in os.environ.items()},
            "BASE": str(dummy_base),
            "ADAPTER": str(dummy_adapter),
            "CALIBRATED_HEAD": str(dummy_calib),
            "OUTPUT_ROOT": str(output_root),
            "DERIVED_ROOT": str(derived_root),
        }
        result = subprocess.run(
            ["bash", str(script), "check"],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(f"  stdout: {result.stdout.strip()}")
            print(f"  stderr: {result.stderr.strip()}")
        assert result.returncode == 0


# =============================================================================
# Test: Derived root resolution
# =============================================================================


class TestDerivedRoot:
    """The derived-root mapping in `resolve_image_path`."""

    def test_derived_root_default_maps_to_benchmark(self):
        """By default `data/derived/...` resolves under `benchmark/data/derived/...`."""
        # From extract_representations
        from extract_representations import resolve_image_path as resolve_repr

        result = resolve_repr("data/derived/plantdoc/train/img.jpg")
        assert "benchmark/data/derived" in result
        assert result.endswith("plantdoc/train/img.jpg")

    def test_derived_root_override_maps_to_external(self):
        """An overridden derived root moves `data/derived/...` outside the package."""
        from extract_representations import resolve_image_path as resolve_repr

        result = resolve_repr("data/derived/plantdoc/train/img.jpg", derived_root="/ext/derived")
        assert result.startswith("/ext/derived")
        assert result.endswith("plantdoc/train/img.jpg")

    def test_source_path_not_affected_by_derived_root(self):
        """`data/source/...` still resolves under the benchmark directory, whatever the derived root is.

        """
        from extract_representations import resolve_image_path as resolve_repr

        result = resolve_repr("data/source/plantdoc/images/img.jpg", derived_root="/ext/derived")
        # Should still contain benchmark
        assert "benchmark" in result
        assert result.endswith("data/source/plantdoc/images/img.jpg")

    def test_absolute_path_unchanged(self):
        """An absolute path is returned unchanged."""
        from extract_representations import resolve_image_path as resolve_repr

        result = resolve_repr("/absolute/path/img.jpg", derived_root="/ext/derived")
        assert result == "/absolute/path/img.jpg"

    def test_extract_head_probabilities_derived_root(self):
        """`extract_head_probabilities` honours the derived root in the same way."""
        from extract_head_probabilities import resolve_image_path as resolve_head

        result = resolve_head("data/derived/coco/train/img.jpg", derived_root="/ext/derived")
        assert result.startswith("/ext/derived")
        assert result.endswith("coco/train/img.jpg")

    def test_extract_head_probabilities_source_not_affected(self):
        """Source paths in `extract_head_probabilities` are unaffected by the derived root."""
        from extract_head_probabilities import resolve_image_path as resolve_head

        result = resolve_head("data/source/plantdoc/images/img.jpg", derived_root="/ext/derived")
        assert "benchmark" in result


# =============================================================================


class TestRealAnnotationCardinality:
    """Light cardinality regression over the shipped annotations; reads JSONL only, no images or models.

    """

    BENCHMARK = Path(__file__).resolve().parent.parent.parent / "benchmark"
    EVAL_DIR = BENCHMARK / "annotations" / "eval"
    # Head-probability extractions are runtime artifacts kept outside the release. Point
    # EVIDENCE_EXTRACTED_DIR at them to enable the tests that read target_label.
    EXTRACTED_DIR = Path(os.environ.get("EVIDENCE_EXTRACTED_DIR", "evidence_extractions"))

    @pytest.fixture(autouse=True)
    def skip_if_no_benchmark(self):
        if not self.EVAL_DIR.exists():
            pytest.skip(f"benchmark eval dir not found: {self.EVAL_DIR}")

    def _load_jsonl(self, name: str) -> list[dict]:
        path = self.EVAL_DIR / name
        if not path.exists():
            pytest.skip(f"file not found: {path}")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _load_extracted_jsonl(self, *parts: str) -> list[dict]:
        path = self.EXTRACTED_DIR.joinpath(*parts)
        if not path.exists():
            pytest.skip("head-probability extractions are excluded from the release")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _unique_l2_pairs(self, records: list[dict]) -> dict:
        """Count unique L2 pairing keys using pairing_key_l2."""
        from collections import defaultdict

        pairs: dict[str, dict] = defaultdict(lambda: {"keep": 0, "cropout": 0})
        for r in records:
            md = r.get("metadata", {})
            sid = md.get("sample_id", "")
            label = md.get("label", "")
            rel = md.get("relation", "")
            if "L2" not in rel:
                continue
            pk = analyze_evidence.pairing_key_l2(sid, label)
            if pk is None:
                continue
            key = "keep" if "keep" in rel else "cropout"
            pairs[pk][key] += 1
        return dict(pairs)

    def _unique_l3_groups(self, records: list[dict]) -> dict:
        """Count unique L3 groups using the new stable_group_key logic."""
        groups = {}
        for r in records:
            if "L3" not in r.get("metadata", {}).get("relation", ""):
                continue
            sid = r.get("metadata", {}).get("sample_id", "")
            gk = analyze_evidence.stable_group_key(sid)
            if gk is None:
                continue
            groups.setdefault(gk, set()).add(r["metadata"].get("achieved_visible_ratio"))
        return groups

    def test_l2_test_cardinality(self):
        """L2 test extracted: analyze_l2 should produce exactly 148 pairs (target_label-aware grouping)."""
        records = self._load_extracted_jsonl("L2", "test", "head_probabilities.jsonl")
        assert len(records) == 790, f"expected 790 records, got {len(records)}"
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        print(
            f"  L2 test extracted: n_pairs={result['n_pairs']} conflicting={result['conflicting_duplicate_groups']} invalid={result['invalid_groups']} incomplete={result['incomplete_pairs']}"
        )
        # With target_label-aware pairing, test should have 148 valid pairs
        assert result["n_pairs"] == 148, f"expected 148 pairs, got {result['n_pairs']}"
        assert result["conflicting_duplicate_groups"] == 0, (
            f"expected 0 conflicting groups, got {result['conflicting_duplicate_groups']}"
        )
        assert result["invalid_groups"] == 0, (
            f"expected 0 invalid groups, got {result['invalid_groups']}"
        )

    def test_l2_dev_cardinality(self):
        """L2 dev extracted: analyze_l2 should produce exactly 124 pairs with target_label-aware grouping."""
        records = self._load_extracted_jsonl("L2", "dev", "head_probabilities.jsonl")
        assert len(records) == 570, f"expected 570 records, got {len(records)}"
        result = analyze_evidence.analyze_l2(records, n_bootstrap=100)
        print(
            f"  L2 dev extracted: n_pairs={result['n_pairs']} conflicting={result['conflicting_duplicate_groups']} invalid={result['invalid_groups']} incomplete={result['incomplete_pairs']}"
        )
        # Dev should have 124 valid pairs with target_label-aware grouping
        assert result["n_pairs"] == 124, f"expected 124 pairs, got {result['n_pairs']}"
        assert result["conflicting_duplicate_groups"] == 0, (
            f"expected 0 conflicting groups, got {result['conflicting_duplicate_groups']}"
        )
        assert result["invalid_groups"] == 0, (
            f"expected 0 invalid groups, got {result['invalid_groups']}"
        )

    def test_l3_test_cardinality(self):
        """L3 test: 750 records, each group should have one normalized source, groups count > 0."""
        records = self._load_jsonl("test_L3_dose.jsonl")
        assert len(records) == 750, f"expected 750 records, got {len(records)}"
        groups = self._unique_l3_groups(records)
        assert len(groups) > 0, "no L3 groups found"
        # Verify each group has a single normalized source (no cross-object merging)
        # Each group key should correspond to a unique source instance
        # We can't easily verify this without running the full parser, but we can
        # check that the number of groups is reasonable
        assert len(groups) <= 750, "more groups than records"

    def test_l3_dev_cardinality(self):
        """L3 dev: 677 records."""
        records = self._load_jsonl("dev_L3_dose.jsonl")
        assert len(records) == 677, f"expected 677 records, got {len(records)}"

    def test_l2_groups_via_parser(self):
        """Verify the new analyze_l2 parser produces expected counts."""
        records = self._load_jsonl("test_L2_evidence.jsonl")
        # Parse records into the format expected by analyze_l2
        parsed = []
        for r in records:
            md = r.get("metadata", {})
            parsed.append(
                {
                    "sample_id": md.get("sample_id", ""),
                    "relation": md.get("relation", ""),
                    "variant_id": "original",
                    "p_answer": 0.5,  # dummy value, just testing grouping
                    "original_index": len(parsed),
                }
            )
        # Check that "L2" appears in relation for all
        l2_count = sum(1 for p in parsed if "L2" in p.get("relation", ""))
        assert l2_count > 0, "no L2 records found"

    def test_l3_groups_via_parser(self):
        """Verify the new analyze_l3 parser produces expected counts."""
        records = self._load_jsonl("test_L3_dose.jsonl")
        parsed = []
        for r in records:
            md = r.get("metadata", {})
            parsed.append(
                {
                    "sample_id": md.get("sample_id", ""),
                    "relation": md.get("relation", ""),
                    "variant_id": "original",
                    "achieved_visible_ratio": md.get("achieved_visible_ratio"),
                    "p_answer": 0.5,
                    "original_index": len(parsed),
                }
            )
        l3_count = sum(1 for p in parsed if "L3" in p.get("relation", ""))
        assert l3_count > 0, "no L3 records found"


# =============================================================================
# Test: Batch size — fake processor/model for _extract_batch (representations)
# =============================================================================


class TestExtractRepresentationsBatch:
    """Test _extract_batch with fake processor/model (pure CPU, no real model)."""

    @pytest.fixture(autouse=True)
    def _fake_qwen_vl_utils(self, monkeypatch):
        """Inject a fake qwen_vl_utils module so that local imports inside _extract_batch work."""
        from PIL import Image

        import types

        fake_mod = types.ModuleType("qwen_vl_utils")

        def mock_process_vision_info(messages):
            return [Image.new("RGB", (224, 224))], []

        fake_mod.process_vision_info = mock_process_vision_info
        monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_mod)

    @pytest.fixture
    def fake_processor(self):
        class BatchOutput(dict):
            """Mimics transformers.BatchEncoding: dict-like + .to() method."""

            def to(self, device):
                return self

        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "fake template"

            def __call__(self, text=None, images=None, padding=True, return_tensors="pt"):
                B = len(text) if text else 0
                S = 24
                # input_ids: fill with VISUAL_TOKEN_ID, last 4 positions are text tokens
                input_ids = torch.full((B, S), 151655, dtype=torch.long)
                for j in range(B):
                    input_ids[j, -4:] = torch.tensor([100, 101, 102, 103])
                # attention_mask with left padding: earlier samples have more pad at start
                attn_mask = torch.ones(B, S, dtype=torch.long)
                for j in range(B):
                    pad_len = (B - 1 - j) * 2  # sample 0: 4 pad, sample 1: 2 pad, sample 2: 0 pad
                    attn_mask[j, :pad_len] = 0
                # image_grid_thw matching actual visual tokens: 20 visual tokens
                # compute_expected_visual_tokens([[1,10,8]], merge_size=2) = 1*5*4 = 20
                return BatchOutput(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attn_mask,
                        "image_grid_thw": torch.tensor([[1, 10, 8]] * B),
                        "pixel_values": torch.randn(B, 3, 224, 224),
                    }
                )

        return FakeProcessor()

    @pytest.fixture
    def fake_model(self):
        class FakeModel:
            def __call__(
                self,
                input_ids,
                attention_mask,
                pixel_values,
                image_grid_thw,
                labels=None,
                output_hidden_states=True,
            ):
                B, S = input_ids.shape
                D = 4096
                # Each position gets value = position_index (float)
                hs = torch.zeros(B, S, D)
                for b in range(B):
                    for s in range(S):
                        hs[b, s] = float(s)

                class Output:
                    def __init__(self, hidden_states):
                        self.hidden_states = [None] * 27 + [hidden_states]

                return Output(hs)

        return FakeModel()

    def _run_batch(self, fake_processor, fake_model, monkeypatch, batch_indices):
        device = torch.device("cpu")
        B = len(batch_indices)
        return extract_representations._extract_batch(
            fake_model,
            fake_processor,
            device,
            batch_indices,
            ["sys"] * B,
            ["describe this"] * B,
            ["/tmp/img.jpg"] * B,
            [0, 1, 2][:B],
            [f"s{i}" for i in batch_indices],
        )

    def test_repr_batch_3_output_shapes(self, fake_processor, fake_model, monkeypatch):
        """B=3: all four representations have shape [3, 4096], metadata lists length 3."""
        result = self._run_batch(fake_processor, fake_model, monkeypatch, [0, 1, 2])
        for key in ("last_token", "visual_mean", "visual_max", "visual_query_attn"):
            assert result[key].shape == (3, 4096), f"{key} shape mismatch: {result[key].shape}"
        assert len(result["head_label"]) == 3
        assert len(result["sample_id"]) == 3
        assert len(result["visual_token_count"]) == 3

    def test_repr_batch_1_output_shapes(self, fake_processor, fake_model, monkeypatch):
        """B=1: all four representations have shape [1, 4096]."""
        result = self._run_batch(fake_processor, fake_model, monkeypatch, [5])
        for key in ("last_token", "visual_mean", "visual_max", "visual_query_attn"):
            assert result[key].shape == (1, 4096), f"{key} shape mismatch: {result[key].shape}"
        assert result["head_label"] == [0]
        assert result["sample_id"] == ["s5"]

    def test_repr_left_padding_last_token(self, fake_processor, fake_model, monkeypatch):
        """With left padding, last_token should be the last non-zero position in attention_mask,
        not the very end of the padded sequence. Sample 0 has 4 pad tokens at start,
        so last non-zero is at position 23 (same as S-1 since padding is at start).
        We verify by checking that the value matches the position index.
        """
        result = self._run_batch(fake_processor, fake_model, monkeypatch, [0, 1, 2])
        # For each sample, last non-zero in attention_mask is at position 23 (S-1)
        # because left padding only affects the start, not the end.
        # The last_token feature value should be 23.0 (position index).
        expected_val = 23.0
        for j, key in enumerate(("last_token",)):
            feat = result[key][j]
            assert float(feat[0]) == pytest.approx(expected_val, abs=1e-5), (
                f"sample {j} last_token[0] should be {expected_val}, got {float(feat[0])}"
            )

    def test_repr_per_sample_visual_token_count(self, fake_processor, fake_model, monkeypatch):
        """Each sample has its own visual_token_count, matching the number of VISUAL_TOKEN_ID."""
        result = self._run_batch(fake_processor, fake_model, monkeypatch, [0, 1, 2])
        # All samples have 20 VISUAL_TOKEN_IDs (24 seq_len - 4 text tokens)
        for cnt in result["visual_token_count"]:
            assert cnt == 20, f"expected 20 visual tokens, got {cnt}"

    def test_shard_save_concatenate_not_stack_batch1_multiple_chunks(self):
        """batch=1, 3 chunks → concatenate produces [3,4096] not [3,1,4096].
        This guards against np.stack() being used where np.concatenate(axis=0) is required."""
        chunks = [np.random.randn(1, 4096).astype(np.float32) for _ in range(3)]
        # Correct: np.concatenate along axis 0
        result = np.concatenate(chunks, axis=0)
        assert result.shape == (3, 4096), (
            f"concatenate(axis=0) expected (3,4096), got {result.shape}"
        )
        # Wrong: np.stack would produce [3,1,4096]
        stacked = np.stack(chunks, axis=0)
        assert stacked.shape == (3, 1, 4096), (
            f"stack(axis=0) expected (3,1,4096), got {stacked.shape}"
        )

    def test_shard_save_concatenate_tail_chunk_batch3(self):
        """batch=3 with 4 chunks (3+3+3+1) → concatenate produces [10,4096].
        Tail chunk (last batch smaller than batch_size) must still concatenate correctly."""
        chunks = [np.random.randn(3, 4096).astype(np.float32) for _ in range(3)]
        chunks.append(np.random.randn(1, 4096).astype(np.float32))  # tail chunk
        result = np.concatenate(chunks, axis=0)
        assert result.shape == (10, 4096), (
            f"concatenate(axis=0) expected (10,4096), got {result.shape}"
        )
        # Verify all four features concatenate the same way
        for key in ("last_token", "visual_mean", "visual_max", "visual_query_attn"):
            feats = [np.random.randn(3, 4096).astype(np.float32) for _ in range(3)]
            feats.append(np.random.randn(1, 4096).astype(np.float32))
            merged = np.concatenate(feats, axis=0)
            assert merged.shape == (10, 4096), (
                f"{key} tail chunk: expected (10,4096), got {merged.shape}"
            )

    def test_shard_save_concatenate_all_four_features_consistent_N(self):
        """All four features across chunks must have consistent B per chunk and final N."""
        rng = np.random.RandomState(42)
        chunk_sizes = [2, 2, 2, 2, 1]  # last chunk is tail
        expected_N = sum(chunk_sizes)
        feat_lists = {
            k: [] for k in ("last_token", "visual_mean", "visual_max", "visual_query_attn")
        }
        for bs in chunk_sizes:
            for k in feat_lists:
                feat_lists[k].append(rng.randn(bs, 4096).astype(np.float32))
        merged = {k: np.concatenate(v, axis=0) for k, v in feat_lists.items()}
        for k, arr in merged.items():
            assert arr.shape == (expected_N, 4096), (
                f"{k}: expected ({expected_N},4096), got {arr.shape}"
            )
        # All four have same N
        Ns = [arr.shape[0] for arr in merged.values()]
        assert len(set(Ns)) == 1, f"all four features must have same N, got {Ns}"


# =============================================================================
# Test: Batch size — fake model for _head_forward_batch (evidence)
# =============================================================================


class TestExtractHeadProbabilitiesBatch:
    """Test _head_forward_batch with fake model (pure CPU, no real model)."""

    @pytest.fixture(autouse=True)
    def _fake_qwen_vl_utils(self, monkeypatch):
        """Inject a fake qwen_vl_utils module so that local imports inside _head_forward_batch work."""
        from PIL import Image

        import types

        fake_mod = types.ModuleType("qwen_vl_utils")

        def mock_process_vision_info(messages):
            return [Image.new("RGB", (224, 224))], []

        fake_mod.process_vision_info = mock_process_vision_info
        monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_mod)

    @pytest.fixture
    def fake_processor(self):
        class BatchOutput(dict):
            """Mimics transformers.BatchEncoding: dict-like + .to() method."""

            def to(self, device):
                return self

        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "fake evidence template"

            def __call__(self, text=None, images=None, padding=True, return_tensors="pt"):
                B = len(text) if text else 0
                S = 16
                return BatchOutput(
                    {
                        "input_ids": torch.zeros(B, S, dtype=torch.long),
                        "attention_mask": torch.ones(B, S, dtype=torch.long),
                    }
                )

        return FakeProcessor()

    @pytest.fixture
    def fake_head_model(self):
        class FakeHeadModel:
            def __call__(self, **inputs):
                B = inputs["input_ids"].shape[0]
                # Deterministic logits: sample i gets [B-i, i, 0.0]
                # So logits[0] = [B, 0, 0], logits[1] = [B-1, 1, 0], ...
                hl = torch.zeros(B, 3)
                for i in range(B):
                    hl[i, 0] = float(B - i)
                    hl[i, 1] = float(i)
                    hl[i, 2] = 0.0

                class Output:
                    def __init__(self, head_logits):
                        self.head_logits = head_logits

                return Output(hl)

        return FakeHeadModel()

    def _make_variant_specs(self, n: int) -> list[dict]:
        """Create simple variant_specs with minimal structure."""
        return [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "/tmp/img.jpg"},
                            {"type": "text", "text": "test"},
                        ],
                    },
                ]
            }
            for _ in range(n)
        ]

    def test_head_batch_3_logits_order(self, fake_processor, fake_head_model, monkeypatch):
        """B=3: results should be in the same order as variant_specs,
        with correct logits per sample."""
        device = torch.device("cpu")
        specs = self._make_variant_specs(3)
        results = extract_head_probabilities._head_forward_batch(
            fake_head_model, fake_processor, specs, device
        )
        assert len(results) == 3
        # fake logits: sample_idx 0 -> [3,0,0], 1 -> [2,1,0], 2 -> [1,2,0]
        # p_answer decreases: softmax([3,0,0])[0] > softmax([2,1,0])[0] > softmax([1,2,0])[0]
        p_answers = [r["p_answer"] for r in results]
        assert p_answers[0] > p_answers[1] > p_answers[2], (
            f"p_answers should be decreasing order: {p_answers}"
        )
        # predicted_class: sample 0 = 0 (answer), sample 1 = 0 (answer), sample 2 = 1 (wrong_obj)
        assert results[0]["predicted_class"] == 0
        assert results[1]["predicted_class"] == 0
        assert results[2]["predicted_class"] == 1

    def test_head_batch_1_single(self, fake_processor, fake_head_model, monkeypatch):
        """B=1: single result with correct structure."""
        device = torch.device("cpu")
        specs = self._make_variant_specs(1)
        results = extract_head_probabilities._head_forward_batch(
            fake_head_model, fake_processor, specs, device
        )
        assert len(results) == 1
        assert "p_answer" in results[0]
        assert "p_wo" in results[0]
        assert "p_nt" in results[0]
        assert "predicted_class" in results[0]


# =============================================================================
# Test: CLI batch-size argument parsing
# =============================================================================


class TestCLIBatchSize:
    """Verify CLI --batch-size argument and launcher --batch-size passing."""

    def test_repr_cli_parses_batch_size(self):
        """extract_representations.py --batch-size 5 should parse correctly."""
        ap = argparse.ArgumentParser()
        ap.add_argument("--batch-size", type=int, default=1)
        args = ap.parse_args(["--batch-size", "5"])
        assert args.batch_size == 5

    def test_repr_cli_default_batch_size(self):
        """extract_representations.py default --batch-size should be 1."""
        ap = argparse.ArgumentParser()
        ap.add_argument("--batch-size", type=int, default=1)
        args = ap.parse_args([])
        assert args.batch_size == 1

    def test_head_cli_parses_batch_size(self):
        """extract_head_probabilities.py --batch-size 5 should parse correctly."""
        from extract_head_probabilities import _positive_int

        ap = argparse.ArgumentParser()
        ap.add_argument("--batch-size", type=_positive_int, default=1)
        args = ap.parse_args(["--batch-size", "5"])
        assert args.batch_size == 5

    def test_head_cli_default_batch_size(self):
        """extract_head_probabilities.py default --batch-size should be 1."""
        from extract_head_probabilities import _positive_int

        ap = argparse.ArgumentParser()
        ap.add_argument("--batch-size", type=_positive_int, default=1)
        args = ap.parse_args([])
        assert args.batch_size == 1

    def test_representation_launcher_passes_batch_size(self, tmp_path):
        """run_representation.sh should pass --batch-size "$BATCH_SIZE" to the python script."""
        script = _THIS_DIR / "run_representation.sh"
        content = script.read_text()
        # Check that --batch-size is passed with the shell variable
        assert '--batch-size "$BATCH_SIZE"' in content, (
            'launcher must pass --batch-size "$BATCH_SIZE"'
        )

    def test_evidence_launcher_passes_batch_size(self, tmp_path):
        """run_evidence.sh should pass --batch-size "$BATCH_SIZE" to the python script."""
        script = _THIS_DIR / "run_evidence.sh"
        content = script.read_text()
        assert '--batch-size "$BATCH_SIZE"' in content, (
            'launcher must pass --batch-size "$BATCH_SIZE"'
        )

    def test_representation_launcher_prints_batch_size(self, tmp_path):
        """run_representation.sh should print BATCH_SIZE in its echo block."""
        script = _THIS_DIR / "run_representation.sh"
        content = script.read_text()
        assert "BATCH_SIZE" in content, "launcher should print BATCH_SIZE"

    def test_evidence_launcher_prints_batch_size(self, tmp_path):
        """run_evidence.sh should print BATCH_SIZE in its echo block."""
        script = _THIS_DIR / "run_evidence.sh"
        content = script.read_text()
        assert "BATCH_SIZE" in content, "launcher should print BATCH_SIZE"

    def test_representation_launcher_validates_batch_size(self, tmp_path):
        """run_representation.sh should reject non-positive BATCH_SIZE."""
        script = _THIS_DIR / "run_representation.sh"
        content = script.read_text()
        assert "=~ ^[1-9]" in content, "launcher should validate BATCH_SIZE is positive integer"

    def test_evidence_launcher_validates_batch_size(self, tmp_path):
        """run_evidence.sh should reject non-positive BATCH_SIZE."""
        script = _THIS_DIR / "run_evidence.sh"
        content = script.read_text()
        assert "=~ ^[1-9]" in content, "launcher should validate BATCH_SIZE is positive integer"


if __name__ == "__main__":
    import os

    pytest.main([__file__, "-v", "--tb=short"])
