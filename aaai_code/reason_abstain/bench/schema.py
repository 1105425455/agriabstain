from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


class Reason(str, Enum):
    """The two abstention reasons the model can emit.

    They partition abstention by its cause and both references are discrete and threshold free:
      - WRONG_OBJECT: the image contains a visually similar category, but not the one the query
        refers to, for instance a related disease on the same crop. The reference comes from the
        near-miss pairing and is therefore discrete.
      - NO_TARGET: the queried target cannot be located, either because it is absent from the
        frame after a cross-category substitution or because it has been cropped out entirely.
        The reference comes from category mismatch or from crop geometry and is discrete.

    Degradations such as blur, pixelation and mean replacement carry **no reason reference**.
    They only enter the dose-response curve, which measures decision behaviour, because there is
    no objective threshold at which an image becomes too degraded to read. Such samples are built
    with `reason=None` and are marked `metadata.dose_only=True`.
    """

    WRONG_OBJECT = "wrong_object"
    NO_TARGET = "no_target"

    TARGET_ABSENT = "target_absent"
    VISUAL_EVIDENCE_INSUFFICIENT = "visual_evidence_insufficient"
    QUERY_IMAGE_MISMATCH = "query_image_mismatch"


class DatasetName(str, Enum):
    COCO = "coco_ground_abstain"
    PLANTDOC = "plantdoc_vg"


class SampleType(str, Enum):
    ANSWERABLE_POSITIVE = "answerable_positive"
    PARAPHRASE_ANSWERABLE = "paraphrase_answerable"
    PHOTOMETRIC_PRESERVE = "photometric_preserve"
    GEOMETRIC_EQUIVARIANCE = "geometric_equivariance"
    HARD_POSITIVE = "hard_positive"
    TARGET_ABSENT_NEGATIVE = "target_absent_negative"
    EVIDENCE_REMOVED_NEGATIVE = "evidence_removed_negative"
    EVIDENCE_INSUFFICIENT_NEGATIVE = "evidence_insufficient_negative"
    NON_TARGET_REMOVAL_CONTROL = "non_target_removal_control"
    REGION_INCLUDE = "region_include"
    REGION_EXCLUDE = "region_exclude"
    REGION_LADDER_VISIBLE = "region_ladder_visible"
    REGION_LADDER_OCCLUDED = "region_ladder_occluded"
    QUERY_IMAGE_MISMATCH_NEGATIVE = "query_image_mismatch_negative"
    PHOTOMETRIC_OCCLUDED = "photometric_occluded"
    GEOMETRIC_OCCLUDED = "geometric_occluded"


@dataclass(frozen=True)
class Box:
    """Box coordinates in Qwen-style [0, 1000] relative coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class BenchmarkTarget:
    label: str
    box: Box | None = None
    source_box_xywh: list[float] | None = None
    category_id: int | None = None
    instance_id: int | str | None = None


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    group_id: str
    dataset: DatasetName
    split: str
    image_path: str
    query: str
    decision: Decision
    sample_type: SampleType
    label: str | None = None
    box: Box | None = None
    reason: Reason | None = None
    source_image_id: int | str | None = None
    source_target: BenchmarkTarget | None = None
    transform: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        if self.box is not None:
            record["box"] = self.box.as_list()
        if self.source_target is not None and self.source_target.box is not None:
            record["source_target"]["box"] = self.source_target.box.as_list()
        return record

    def expected_output(self) -> dict[str, Any]:
        if self.decision == Decision.ANSWER:
            if self.label is None or self.box is None:
                raise ValueError(f"Answer sample {self.sample_id} is missing label or box.")
            return {"decision": "answer", "label": self.label, "box": self.box.as_list()}

        if self.reason is None:
            raise ValueError(f"Abstain sample {self.sample_id} is missing reason.")
        return {"decision": "abstain", "reason": self.reason.value}


COCO_REASONS = {Reason.NO_TARGET}
PLANTDOC_REASONS = {
    Reason.WRONG_OBJECT,
    Reason.NO_TARGET,
}
