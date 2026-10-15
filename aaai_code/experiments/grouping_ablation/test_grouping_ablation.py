from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from experiments.grouping_ablation.build_independent_negatives import (  # noqa: E402
    CandidateImage,
    IndependentNegativeSampler,
    audit_output,
    build_ablation,
    join_sft_to_raw,
)
from experiments.grouping_ablation.build_release_artifact import (  # noqa: E402
    build_artifact,
)


GROUPING_ROOT = Path(__file__).resolve().parent
RUNTIME_ARTIFACT_ROOT = GROUPING_ROOT / "artifacts"
CANONICAL_ARTIFACT = GROUPING_ROOT / "main" / "grouping_ablation_fixed_gt.json"


def _sft_row(state: int, image: str, query: str, source_id: str) -> dict:
    if state == 0:
        target = {"decision": "answer", "label": "tomato early blight leaf", "box": [1, 2, 3, 4]}
        decision_label, reason_label = 1, -1
    else:
        reason = "wrong_object" if state == 1 else "no_target"
        target = {"decision": "abstain", "reason": reason}
        decision_label, reason_label = 0, state - 1
    return {
        "image": [image],
        "system": "system",
        "decision_label": decision_label,
        "reason_label": reason_label,
        "head_label": state,
        "source_image_id": source_id,
        "conversations": [
            {"from": "human", "value": f"<image>{query}"},
            {"from": "gpt", "value": json.dumps(target, separators=(",", ":"))},
        ],
    }


def _raw_row(
    state: int,
    image: str,
    query: str,
    source_id: str,
    queried: str,
    visible: str,
) -> dict:
    reason = None if state == 0 else ("wrong_object" if state == 1 else "no_target")
    metadata = {"relation": "base"}
    if state == 1:
        metadata = {"relation": "T6", "mismatch_label": queried}
    elif state == 2:
        metadata = {"relation": "T5", "absent_label": queried}
    return {
        "sample_id": f"raw-{state}",
        "dataset": "plantdoc_vg",
        "source_image_id": source_id,
        "image_path": image,
        "query": query,
        "decision": "answer" if state == 0 else "abstain",
        "reason": reason,
        "label": visible if state == 0 else None,
        "source_target": {"label": visible},
        "metadata": metadata,
    }


def test_build_replaces_only_negatives_and_preserves_targets(tmp_path: Path) -> None:
    positive_path = str(tmp_path / "base/positive.jpg")
    wo_path = str(tmp_path / "base/wo.jpg")
    nt_path = str(tmp_path / "base/nt.jpg")
    independent_wo_path = str(tmp_path / "independent/wo.jpg")
    independent_nt_path = str(tmp_path / "independent/nt.jpg")
    sft_rows = [
        _sft_row(0, positive_path, "Locate disease A.", "positive"),
        _sft_row(1, wo_path, "Locate disease B.", "wo-source"),
        _sft_row(2, nt_path, "Locate disease C.", "nt-source"),
    ]
    raw_rows = [
        _raw_row(
            0,
            positive_path,
            "Locate disease A.",
            "positive",
            "tomato early blight leaf",
            "tomato early blight leaf",
        ),
        _raw_row(
            1,
            wo_path,
            "Locate disease B.",
            "wo-source",
            "tomato late blight leaf",
            "tomato early blight leaf",
        ),
        _raw_row(
            2,
            nt_path,
            "Locate disease C.",
            "nt-source",
            "corn rust leaf",
            "tomato early blight leaf",
        ),
    ]
    joined = join_sft_to_raw(sft_rows, raw_rows)
    candidates = [
        CandidateImage(
            dataset="plantdoc_vg",
            source_id="independent-wo",
            image_path=independent_wo_path,
            labels=frozenset({"tomato early blight leaf"}),
        ),
        CandidateImage(
            dataset="plantdoc_vg",
            source_id="independent-nt",
            image_path=independent_nt_path,
            labels=frozenset({"apple scab leaf"}),
        ),
    ]

    output, assignments = build_ablation(
        sft_rows,
        joined,
        IndependentNegativeSampler(candidates, seed=42),
    )

    assert output[0] == sft_rows[0]
    assert output[1]["image"] == [independent_wo_path]
    assert output[2]["image"][0].startswith(str(tmp_path / "independent"))
    assert output[1]["conversations"] == sft_rows[1]["conversations"]
    assert output[2]["conversations"] == sft_rows[2]["conversations"]
    assert len(assignments) == 2

    for row in output:
        image_path = Path(row["image"][0])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()

    audit = audit_output(
        sft_rows,
        output,
        joined,
        assignments,
        {"plantdoc_vg": {"positive", "wo-source", "nt-source"}},
    )
    assert all(audit["checks"].values())
    assert audit["counts"]["states"] == {
        "answer": 1,
        "wrong_object": 1,
        "no_target": 1,
    }


def test_wrong_object_requires_visible_confusable_label() -> None:
    sft_rows = [_sft_row(1, "/base/wo.jpg", "Locate disease B.", "wo-source")]
    raw_rows = [
        _raw_row(
            1,
            "/base/wo.jpg",
            "Locate disease B.",
            "wo-source",
            "tomato late blight leaf",
            "tomato early blight leaf",
        )
    ]
    candidates = [
        CandidateImage(
            dataset="plantdoc_vg",
            source_id="wrong-visible",
            image_path="/independent/wrong.jpg",
            labels=frozenset({"apple scab leaf"}),
        )
    ]

    with pytest.raises(ValueError, match="No independent candidate"):
        build_ablation(
            sft_rows,
            join_sft_to_raw(sft_rows, raw_rows),
            IndependentNegativeSampler(candidates, seed=42),
        )


def test_join_rejects_unmatched_training_row() -> None:
    with pytest.raises(ValueError, match="Failed to join"):
        join_sft_to_raw(
            [_sft_row(0, "/missing.jpg", "Locate disease.", "source")],
            [],
        )


def test_canonical_artifact_is_path_free_and_uses_macro_mts() -> None:
    text = CANONICAL_ARTIFACT.read_text(encoding="utf-8")
    artifact = json.loads(text)
    assert not re.search(r"(?<![\w.])/\w+/", text), "artifact must stay free of absolute paths"
    assert artifact["primary_mts"] == "MTS_macro"
    assert all(artifact["data_control"]["checks"].values())
    selections = artifact["training"]["checkpoint_selection"]
    assert selections["grouped"]["best_step"] == 444
    assert selections["grouped"]["best_val_loss"] == pytest.approx(0.278026)
    assert selections["grouped"]["candidate_steps"] == [111, 222, 333, 444, 555, 666]
    assert selections["independent_negative"]["best_step"] == 222
    assert selections["independent_negative"]["best_val_loss"] == pytest.approx(0.289195)
    assert selections["independent_negative"]["candidate_steps"] == [
        111,
        222,
        333,
        444,
        555,
        666,
    ]
    assert selections["grouped"]["policy"] == selections["independent_negative"]["policy"]
    assert selections["grouped"]["dev_subset_sha256"] == selections["independent_negative"][
        "dev_subset_sha256"
    ]
    assert len(selections["grouped"]["dev_loss_log_sha256"]) == 6
    assert len(selections["independent_negative"]["dev_loss_log_sha256"]) == 6
    assert selections["grouped"]["recovered_candidates"] == {
        "steps": [111],
    }
    assert selections["independent_negative"]["recovered_candidates"] is None
    rows = {(row["interface"], row["dataset"]): row for row in artifact["rows"]}
    comprehensive = rows[("regate", "comprehensive")]
    assert comprehensive["grouped"]["MTS_macro"] == pytest.approx(0.8195042768467142)
    assert comprehensive["independent_negative"]["MTS_macro"] == pytest.approx(0.5267407545130093)
    assert comprehensive["grouped"]["MCC"] == pytest.approx(0.8485085166883631)
    assert comprehensive["independent_negative"]["FCA"] == pytest.approx(0.4860166288737717)
    assert rows[("regate", "L1_discrim")]["grouped"]["E2E_mIoU"] == pytest.approx(
        0.810333283165365
    )
    plantwild = rows[("regate", "plantwild")]
    assert plantwild["metric_applicability"]["E2E_mIoU"] is False
    assert plantwild["grouped"]["E2E_mIoU"] is None


def test_canonical_artifact_rebuilds_from_author_runtime() -> None:
    if not RUNTIME_ARTIFACT_ROOT.is_dir():
        pytest.skip("author-side runtime artifacts are excluded from the release")
    expected = json.loads(CANONICAL_ARTIFACT.read_text(encoding="utf-8"))
    runtime_selection = json.loads(
        (
            RUNTIME_ARTIFACT_ROOT
            / "grouped_dev_selection"
            / "selection"
            / "checkpoint_selection.json"
        ).read_text(encoding="utf-8")
    )
    assert set(runtime_selection["extra_checkpoint_task_ids"]) == {"111"}
    assert all(
        isinstance(value, int) and value > 0
        for value in runtime_selection["extra_checkpoint_task_ids"].values()
    )
    assert build_artifact(RUNTIME_ARTIFACT_ROOT) == expected
