from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .boxes import xywh_to_qwen_box
from .schema import Box
from .io import read_json
from .schema import (
    BenchmarkSample,
    BenchmarkTarget,
    DatasetName,
    Decision,
    Reason,
    SampleType,
)
from .templates import GROUNDING_TEMPLATES, build_query, build_query_varied
from .transforms import (
    save_brightness_contrast,
    save_gaussian_blur_region,
    save_horizontal_flip,
    save_inclusion_crop,
    save_mean_replace_region,
    save_visibility_ladder_crop,
)


def _image_path(images_dir: Path, file_name: str) -> str:
    return str((images_dir / file_name).resolve())


def _display(label: str) -> str:
    """Convert canonical underscore label to the space form Qwen3-VL prefers."""
    return label.replace("_", " ")


def build_coco_ground_abstain(config: dict[str, Any]) -> list[BenchmarkSample]:
    annotation_path = Path(config["annotation_path"])
    images_dir = Path(config["images_dir"])
    split = str(config.get("split", "train"))
    max_samples = config.get("max_samples")
    absent_per_positive = config.get("absent_per_positive", "all")
    transform_images_dir = Path(
        config.get(
            "transform_images_dir",
            f"outputs/benchmarks/transformed_images/coco/{split}",
        )
    )
    # Minimum target box size in Qwen [0,1000] coords; tiny annotations (e.g. a 1x3 px
    # cell phone) are un-groundable noise and are skipped.
    min_box_side = float(config.get("min_box_side", 8))
    min_box_area = float(config.get("min_box_area", 64))
    metamorphic = config.get("metamorphic", {})
    enabled_relations = set(
        metamorphic.get("enabled_relations", ["T1", "T2", "T3", "T4", "T5", "T7"])
    )
    removal_operators = metamorphic.get("removal_operators", ["gaussian_blur", "mean_replace"])
    region_ladder_enabled = bool(metamorphic.get("region_ladder", True))
    region_ladder_levels = [
        float(v) for v in metamorphic.get("region_ladder_levels", [0.5, 0.25, 0.0])
    ]
    region_ladder_answer_threshold = float(metamorphic.get("region_ladder_answer_threshold", 0.5))
    # Collinearity break (P0): emit an abstain twin for base by occluding the target on the
    # SAME image, so relation type no longer 100% predicts the decision.
    collinearity_break = bool(metamorphic.get("collinearity_break", False))

    coco = read_json(annotation_path)
    categories = {cat["id"]: cat["name"].replace(" ", "_") for cat in coco["categories"]}
    all_category_ids = set(categories)
    images = {image["id"]: image for image in coco["images"]}

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        if annotation.get("bbox") is None:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    samples: list[BenchmarkSample] = []
    for image_id, image_annotations in tqdm(
        annotations_by_image.items(), desc="building COCO-Ground-Abstain"
    ):
        image = images[image_id]
        width = int(image["width"])
        height = int(image["height"])
        present_category_ids = {int(annotation["category_id"]) for annotation in image_annotations}
        counts_by_category: dict[int, int] = defaultdict(int)
        for annotation in image_annotations:
            counts_by_category[int(annotation["category_id"])] += 1

        # First version: use only unique-instance classes to avoid ambiguous "locate the dog" cases.
        for annotation in image_annotations:
            category_id = int(annotation["category_id"])
            if counts_by_category[category_id] != 1:
                continue

            label = categories[category_id]
            disp = _display(label)
            box = xywh_to_qwen_box(annotation["bbox"], width=width, height=height)
            box_w, box_h = box.x2 - box.x1, box.y2 - box.y1
            if box_w < min_box_side or box_h < min_box_side or box_w * box_h < min_box_area:
                continue
            group_id = f"coco-{split}-{image_id}-{annotation['id']}"
            image_path = _image_path(images_dir, image["file_name"])
            source_target = BenchmarkTarget(
                label=disp,
                box=box,
                source_box_xywh=[float(v) for v in annotation["bbox"]],
                category_id=category_id,
                instance_id=annotation["id"],
            )

            group_samples: list[BenchmarkSample] = []
            group_samples.append(
                BenchmarkSample(
                    sample_id=f"{group_id}-answer",
                    group_id=group_id,
                    dataset=DatasetName.COCO,
                    split=split,
                    image_path=image_path,
                    query=build_query(label),
                    decision=Decision.ANSWER,
                    sample_type=SampleType.ANSWERABLE_POSITIVE,
                    label=disp,
                    box=box,
                    source_image_id=image_id,
                    source_target=source_target,
                    metadata={"relation": "base", "template_id": 0},
                )
            )

            if collinearity_break:
                # Same base image, target occluded -> abstain twin. Breaks "base always answers" shortcut.
                occ_path = transform_images_dir / f"{group_id}-base-occluded.jpg"
                _, occ_achieved, occ_wrote = save_visibility_ladder_crop(
                    image_path, box, occ_path, visible_ratio=0.0
                )
                if occ_wrote:
                    group_samples.append(
                        BenchmarkSample(
                            sample_id=f"{group_id}-base-occluded",
                            group_id=group_id,
                            dataset=DatasetName.COCO,
                            split=split,
                            image_path=str(occ_path.resolve()),
                            query=build_query_varied(
                                label, f"{group_id}-base_occluded-{label}", is_answer=False
                            ),
                            decision=Decision.ABSTAIN,
                            sample_type=SampleType.TARGET_ABSENT_NEGATIVE,
                            reason=Reason.NO_TARGET,
                            source_image_id=image_id,
                            source_target=source_target,
                            transform="base+ladder_occlude",
                            metadata={
                                "relation": "base_occluded",
                                "achieved_visible_ratio": occ_achieved,
                                "collinearity_break": True,
                            },
                        )
                    )

            if "T1" in enabled_relations:
                for template_id in range(1, len(GROUNDING_TEMPLATES)):
                    group_samples.append(
                        BenchmarkSample(
                            sample_id=f"{group_id}-paraphrase-{template_id}",
                            group_id=group_id,
                            dataset=DatasetName.COCO,
                            split=split,
                            image_path=image_path,
                            query=build_query(label, template_id=template_id),
                            decision=Decision.ANSWER,
                            sample_type=SampleType.PARAPHRASE_ANSWERABLE,
                            label=disp,
                            box=box,
                            source_image_id=image_id,
                            source_target=source_target,
                            transform="query_paraphrase",
                            metadata={"relation": "T1", "template_id": template_id},
                        )
                    )

            if "T2" in enabled_relations:
                photometric_path = transform_images_dir / f"{group_id}-photometric.jpg"
                save_brightness_contrast(image_path, photometric_path, brightness=1.1, contrast=1.1)
                group_samples.append(
                    BenchmarkSample(
                        sample_id=f"{group_id}-photometric",
                        group_id=group_id,
                        dataset=DatasetName.COCO,
                        split=split,
                        image_path=str(photometric_path.resolve()),
                        query=build_query_varied(label, f"{group_id}-T2-{label}", is_answer=True),
                        decision=Decision.ANSWER,
                        sample_type=SampleType.PHOTOMETRIC_PRESERVE,
                        label=disp,
                        box=box,
                        source_image_id=image_id,
                        source_target=source_target,
                        transform="brightness_contrast",
                        metadata={"relation": "T2", "brightness": 1.1, "contrast": 1.1},
                    )
                )

            if "T3" in enabled_relations:
                flip_path = transform_images_dir / f"{group_id}-hflip.jpg"
                flip_box = save_horizontal_flip(image_path, box, flip_path)
                group_samples.append(
                    BenchmarkSample(
                        sample_id=f"{group_id}-hflip",
                        group_id=group_id,
                        dataset=DatasetName.COCO,
                        split=split,
                        image_path=str(flip_path.resolve()),
                        query=build_query_varied(label, f"{group_id}-T3-{label}", is_answer=True),
                        decision=Decision.ANSWER,
                        sample_type=SampleType.GEOMETRIC_EQUIVARIANCE,
                        label=disp,
                        box=flip_box,
                        source_image_id=image_id,
                        source_target=source_target,
                        transform="horizontal_flip",
                        metadata={"relation": "T3", "operator": "horizontal_flip"},
                    )
                )

            absent_ids = _select_absent_ids(
                all_category_ids - present_category_ids,
                absent_per_positive,
                seed_key=group_id,
            )
            for absent_category_id in absent_ids:
                absent_label = _display(categories[absent_category_id])
                group_samples.append(
                    BenchmarkSample(
                        sample_id=f"{group_id}-absent-{absent_category_id}",
                        group_id=group_id,
                        dataset=DatasetName.COCO,
                        split=split,
                        image_path=image_path,
                        query=build_query_varied(
                            absent_label, f"{group_id}-T5-{absent_label}", is_answer=False
                        ),
                        decision=Decision.ABSTAIN,
                        sample_type=SampleType.TARGET_ABSENT_NEGATIVE,
                        reason=Reason.NO_TARGET,
                        source_image_id=image_id,
                        source_target=source_target,
                        transform="target_absent_swap",
                        metadata={"relation": "T5", "absent_label": absent_label},
                    )
                )

            if "T4" in enabled_relations:
                for operator in removal_operators:
                    removed_path = transform_images_dir / f"{group_id}-target-{operator}.jpg"
                    _save_removal(operator, image_path, box, removed_path)
                    group_samples.append(
                        BenchmarkSample(
                            sample_id=f"{group_id}-target-{operator}",
                            group_id=group_id,
                            dataset=DatasetName.COCO,
                            split=split,
                            image_path=str(removed_path.resolve()),
                            query=build_query_varied(
                                label, f"{group_id}-T4-{operator}-{label}", is_answer=False
                            ),
                            decision=Decision.ABSTAIN,
                            sample_type=SampleType.EVIDENCE_REMOVED_NEGATIVE,
                            source_image_id=image_id,
                            source_target=source_target,
                            transform=f"target_{operator}",
                            metadata={"relation": "T4", "operator": operator, "dose_only": True},
                        )
                    )

                    occupied_boxes = [
                        xywh_to_qwen_box(ann["bbox"], width=width, height=height)
                        for ann in image_annotations
                    ]
                    control_box = _select_background_control_box(occupied_boxes, box)
                    if control_box is not None:
                        control_path = transform_images_dir / f"{group_id}-nontarget-{operator}.jpg"
                        _save_removal(operator, image_path, control_box, control_path)
                        group_samples.append(
                            BenchmarkSample(
                                sample_id=f"{group_id}-nontarget-{operator}",
                                group_id=group_id,
                                dataset=DatasetName.COCO,
                                split=split,
                                image_path=str(control_path.resolve()),
                                query=build_query_varied(
                                    label,
                                    f"{group_id}-T4_control-{operator}-{label}",
                                    is_answer=True,
                                ),
                                decision=Decision.ANSWER,
                                sample_type=SampleType.NON_TARGET_REMOVAL_CONTROL,
                                label=disp,
                                box=box,
                                source_image_id=image_id,
                                source_target=source_target,
                                transform=f"non_target_{operator}",
                                metadata={
                                    "relation": "T4_control",
                                    "operator": operator,
                                    "control_source": "background_non_target",
                                    "control_category_id": None,
                                    "control_box": control_box.as_list(),
                                },
                            )
                        )

            if "T7" in enabled_relations:
                include_path = transform_images_dir / f"{group_id}-region-include.jpg"
                include_box = save_inclusion_crop(image_path, box, include_path)
                group_samples.append(
                    BenchmarkSample(
                        sample_id=f"{group_id}-region-include",
                        group_id=group_id,
                        dataset=DatasetName.COCO,
                        split=split,
                        image_path=str(include_path.resolve()),
                        query=build_query_varied(
                            label, f"{group_id}-T7_include-{label}", is_answer=True
                        ),
                        decision=Decision.ANSWER,
                        sample_type=SampleType.REGION_INCLUDE,
                        label=disp,
                        box=include_box,
                        source_image_id=image_id,
                        source_target=source_target,
                        transform="region_include_crop",
                        metadata={"relation": "T7_include"},
                    )
                )

                # T7 severity ladder (visible-ratio), see build_plantdoc for rationale.
                if region_ladder_enabled:
                    for level in region_ladder_levels:
                        ladder_path = (
                            transform_images_dir
                            / f"{group_id}-region-ladder-{int(round(level * 100)):03d}.jpg"
                        )
                        new_box, achieved, wrote_file = save_visibility_ladder_crop(
                            image_path, box, ladder_path, visible_ratio=level
                        )
                        if not wrote_file:
                            continue
                        is_answer = (
                            achieved >= region_ladder_answer_threshold and new_box is not None
                        )
                        ladder_meta = {
                            "relation": "T7_ladder",
                            "target_visible_ratio": level,
                            "achieved_visible_ratio": achieved,
                            "answer_threshold": region_ladder_answer_threshold,
                        }
                        if is_answer:
                            group_samples.append(
                                BenchmarkSample(
                                    sample_id=f"{group_id}-ladder-{int(round(level * 100)):03d}",
                                    group_id=group_id,
                                    dataset=DatasetName.COCO,
                                    split=split,
                                    image_path=str(ladder_path.resolve()),
                                    query=build_query_varied(
                                        label,
                                        f"{group_id}-T7_ladder-{level}-{label}",
                                        is_answer=True,
                                    ),
                                    decision=Decision.ANSWER,
                                    sample_type=SampleType.REGION_LADDER_VISIBLE,
                                    label=disp,
                                    box=new_box,
                                    source_image_id=image_id,
                                    source_target=source_target,
                                    transform="region_ladder_crop",
                                    metadata=ladder_meta,
                                )
                            )
                        else:
                            if level <= 0.001:
                                ladder_reason = Reason.NO_TARGET
                                ladder_meta["dose_only"] = False
                            else:
                                ladder_reason = None
                                ladder_meta["dose_only"] = True
                            group_samples.append(
                                BenchmarkSample(
                                    sample_id=f"{group_id}-ladder-{int(round(level * 100)):03d}",
                                    group_id=group_id,
                                    dataset=DatasetName.COCO,
                                    split=split,
                                    image_path=str(ladder_path.resolve()),
                                    query=build_query_varied(
                                        label,
                                        f"{group_id}-T7_ladder-{level}-{label}",
                                        is_answer=False,
                                    ),
                                    decision=Decision.ABSTAIN,
                                    sample_type=SampleType.REGION_LADDER_OCCLUDED,
                                    reason=ladder_reason,
                                    source_image_id=image_id,
                                    source_target=source_target,
                                    transform="region_ladder_crop",
                                    metadata=ladder_meta,
                                )
                            )

            if max_samples is not None and len(samples) + len(group_samples) > int(max_samples):
                return samples
            samples.extend(group_samples)

    return samples


def _select_absent_ids(
    absent_ids: set[int],
    absent_per_positive: int | str | None,
    seed_key: str = "",
) -> list[int]:
    """Pick which absent classes to query for a positive sample.

    `all` keeps every absent class (causes T5 to dominate). A small integer deterministically
    samples that many absent classes (unbiased, per-group seeded) so T5 stays balanced against
    the evidence-grounded relations.
    """
    sorted_absent_ids = sorted(absent_ids)
    if absent_per_positive in (None, "all"):
        return sorted_absent_ids
    n = int(absent_per_positive)
    if n >= len(sorted_absent_ids):
        return sorted_absent_ids
    import random

    rng = random.Random(f"absent:{seed_key}")
    return sorted(rng.sample(sorted_absent_ids, n))


def _save_removal(operator: str, image_path: str, box: Any, output_path: Path) -> None:
    if operator == "gaussian_blur":
        save_gaussian_blur_region(image_path, box, output_path)
    elif operator == "mean_replace":
        save_mean_replace_region(image_path, box, output_path)
    else:
        raise ValueError(f"Unknown removal operator: {operator}")


def _select_background_control_box(
    occupied_boxes: list[Box],
    target_box: Box,
) -> Box | None:
    """Grid-search the whole image, in 1000x1000 coordinates, for an empty background region that
    avoids every occupied box.

    Candidate sizes are the target box scaled by 1.0, 0.75 and 0.5, and the candidate furthest
    from the target centre is returned. The degraded region therefore never overlaps any target.
    """
    target_width = target_box.x2 - target_box.x1
    target_height = target_box.y2 - target_box.y1

    for scale in (1.0, 0.75, 0.5):
        width = max(50, round(target_width * scale))
        height = max(50, round(target_height * scale))
        if width >= 1000 or height >= 1000:
            continue
        candidates: list[Any] = []
        for x1 in _grid_positions(1000 - width):
            for y1 in _grid_positions(1000 - height):
                candidate = Box(x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)
                if any(
                    _boxes_overlap_with_margin(candidate, occupied) for occupied in occupied_boxes
                ):
                    continue
                candidates.append(candidate)
        if candidates:
            return max(candidates, key=lambda candidate: _center_distance(candidate, target_box))
    return None


def _grid_positions(max_start: int, steps: int = 8) -> list[int]:
    """Enumerate the start positions of the grid search."""
    if max_start < 0:
        return []
    if max_start == 0:
        return [0]
    return sorted({round(max_start * idx / steps) for idx in range(steps + 1)})


def _boxes_overlap_with_margin(a: Any, b: Any, margin: int = 20) -> bool:
    """Whether two boxes overlap, allowing for a margin."""
    return not (
        a.x2 <= b.x1 - margin
        or a.x1 >= b.x2 + margin
        or a.y2 <= b.y1 - margin
        or a.y1 >= b.y2 + margin
    )


def _box_area(box: Any) -> int:
    return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)


def _box_iou(a: Any, b: Any) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union else 0.0


def _center_distance(a: Any, b: Any) -> float:
    ax = (a.x1 + a.x2) / 2
    ay = (a.y1 + a.y2) / 2
    bx = (b.x1 + b.x2) / 2
    by = (b.y1 + b.y2) / 2
    return (ax - bx) ** 2 + (ay - by) ** 2
