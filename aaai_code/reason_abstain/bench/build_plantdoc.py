from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .schema import (
    BenchmarkSample,
    BenchmarkTarget,
    Box,
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
    save_pixelate_region,
    save_visibility_ladder_crop,
)


def _display(label: str) -> str:
    """Convert canonical underscore label to the space form Qwen3-VL prefers.

    The emitted `label` field and human-facing metadata use this form so the model's
    natural output (e.g. "tomato septoria leaf spot") matches the gold label after the
    eval-side normalization in `reason_abstain.eval.parser`.
    """
    return label.replace("_", " ")


def _read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
        class_id = int(parts[0])
        cx, cy, w, h = [float(v) for v in parts[1:]]
        rows.append((class_id, cx, cy, w, h))
    return rows


def _yolo_to_qwen_box(cx: float, cy: float, w: float, h: float) -> Box:
    x1 = round((cx - w / 2) * 1000)
    y1 = round((cy - h / 2) * 1000)
    x2 = round((cx + w / 2) * 1000)
    y2 = round((cy + h / 2) * 1000)
    return Box(
        x1=max(0, min(1000, x1)),
        y1=max(0, min(1000, y1)),
        x2=max(0, min(1000, x2)),
        y2=max(0, min(1000, y2)),
    )


def build_plantdoc_vg(config: dict[str, Any]) -> list[BenchmarkSample]:
    images_dir = Path(config["images_dir"])
    labels_dir = Path(config.get("labels_dir", images_dir))
    split = str(config.get("split", "train"))
    class_names = list(config["class_names"])
    excluded_class_ids = {int(class_id) for class_id in config.get("excluded_class_ids", [])}
    require_unique_class_per_image = bool(config.get("require_unique_class_per_image", True))
    image_extensions = tuple(config.get("image_extensions", [".jpg", ".jpeg", ".png"]))
    absent_per_positive = config.get("absent_per_positive", "all")
    mismatch_pairs = config.get("mismatch_pairs", {})
    max_samples = config.get("max_samples")
    # Minimum target box size in Qwen [0,1000] coords; tiny annotations are un-groundable
    # noise and are skipped (mirrors build_coco).
    min_box_side = float(config.get("min_box_side", 8))
    min_box_area = float(config.get("min_box_area", 64))
    transform_images_dir = Path(
        config.get(
            "transform_images_dir",
            f"outputs/benchmarks/transformed_images/plantdoc/{split}",
        )
    )
    metamorphic = config.get("metamorphic", {})
    enabled_relations = set(
        metamorphic.get("enabled_relations", ["T1", "T2", "T3", "T4", "T5", "T6", "T7"])
    )
    removal_operators = metamorphic.get("removal_operators", ["gaussian_blur", "mean_replace"])
    target_removal_source = str(metamorphic.get("target_removal_source", "region_include_crop"))
    region_exclude_skip_labels = set(metamorphic.get("region_exclude_skip_labels", []))
    # T7 severity ladder (visible-ratio): geometric objective anchor for Risk-Coverage.
    # Levels are achieved visible fractions of the target; >=0.5 -> answer, <0.5 -> abstain
    # (final decision is later confirmed by the VLM, see docs §5.1/§6).
    region_ladder_enabled = bool(metamorphic.get("region_ladder", True))
    region_ladder_levels = [
        float(v) for v in metamorphic.get("region_ladder_levels", [0.5, 0.25, 0.0])
    ]
    region_ladder_answer_threshold = float(metamorphic.get("region_ladder_answer_threshold", 0.5))
    # Collinearity break (P0): emit an abstain twin for T2/T3 by occluding the target on the
    # SAME transformed image, so relation type no longer 100% predicts the decision. The twin
    # shares the group_id with its answer sibling -> GRPO sees both decisions within one group.
    collinearity_break = bool(metamorphic.get("collinearity_break", False))
    class_ids_by_name = {name: idx for idx, name in enumerate(class_names)}

    samples: list[BenchmarkSample] = []
    image_paths = sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in image_extensions
    )
    for image_path in tqdm(image_paths, desc="building PlantDoc-VG"):
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        labels = [row for row in _read_yolo_labels(label_path) if row[0] not in excluded_class_ids]
        if not labels:
            continue

        present_class_ids = {row[0] for row in labels}
        counts_by_class: dict[int, int] = {}
        for row in labels:
            counts_by_class[row[0]] = counts_by_class.get(row[0], 0) + 1
        with Image.open(image_path) as image:
            width, height = image.size

        for instance_idx, (class_id, cx, cy, w, h) in enumerate(labels):
            if require_unique_class_per_image and counts_by_class[class_id] != 1:
                continue
            if class_id >= len(class_names):
                raise ValueError(f"class_id {class_id} is outside class_names in {label_path}")
            label = class_names[class_id]
            disp = _display(label)
            box = _yolo_to_qwen_box(cx, cy, w, h)
            box_w, box_h = box.x2 - box.x1, box.y2 - box.y1
            if box_w < min_box_side or box_h < min_box_side or box_w * box_h < min_box_area:
                continue
            group_id = f"plantdoc-{split}-{_short_id(image_path)}-{instance_idx}"
            source_target = BenchmarkTarget(
                label=disp,
                box=box,
                source_box_xywh=[cx * width, cy * height, w * width, h * height],
                category_id=class_id,
                instance_id=instance_idx,
            )

            group_samples: list[BenchmarkSample] = []
            group_samples.append(
                BenchmarkSample(
                    sample_id=f"{group_id}-answer",
                    group_id=group_id,
                    dataset=DatasetName.PLANTDOC,
                    split=split,
                    image_path=str(image_path.resolve()),
                    query=build_query(label),
                    decision=Decision.ANSWER,
                    sample_type=SampleType.ANSWERABLE_POSITIVE,
                    label=disp,
                    box=box,
                    source_image_id=image_path.stem,
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
                            dataset=DatasetName.PLANTDOC,
                            split=split,
                            image_path=str(occ_path.resolve()),
                            query=build_query_varied(
                                label, f"{group_id}-base_occluded-{label}", is_answer=False
                            ),
                            decision=Decision.ABSTAIN,
                            sample_type=SampleType.TARGET_ABSENT_NEGATIVE,
                            reason=Reason.NO_TARGET,
                            source_image_id=image_path.stem,
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
                            dataset=DatasetName.PLANTDOC,
                            split=split,
                            image_path=str(image_path.resolve()),
                            query=build_query(label, template_id=template_id),
                            decision=Decision.ANSWER,
                            sample_type=SampleType.PARAPHRASE_ANSWERABLE,
                            label=disp,
                            box=box,
                            source_image_id=image_path.stem,
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
                        dataset=DatasetName.PLANTDOC,
                        split=split,
                        image_path=str(photometric_path.resolve()),
                        query=build_query_varied(label, f"{group_id}-T2-{label}", is_answer=True),
                        decision=Decision.ANSWER,
                        sample_type=SampleType.PHOTOMETRIC_PRESERVE,
                        label=disp,
                        box=box,
                        source_image_id=image_path.stem,
                        source_target=source_target,
                        transform="brightness_contrast",
                        metadata={"relation": "T2", "brightness": 1.1, "contrast": 1.1},
                    )
                )

                if collinearity_break:
                    # Same photometric image, target occluded -> abstain twin (box unchanged
                    # under brightness/contrast). Breaks "T2 always answers" shortcut.
                    occ_path = transform_images_dir / f"{group_id}-photometric-occluded.jpg"
                    _, occ_achieved, occ_wrote = save_visibility_ladder_crop(
                        photometric_path, box, occ_path, visible_ratio=0.0
                    )
                    if occ_wrote:
                        group_samples.append(
                            BenchmarkSample(
                                sample_id=f"{group_id}-photometric-occluded",
                                group_id=group_id,
                                dataset=DatasetName.PLANTDOC,
                                split=split,
                                image_path=str(occ_path.resolve()),
                                query=build_query_varied(
                                    label, f"{group_id}-T2_occluded-{label}", is_answer=False
                                ),
                                decision=Decision.ABSTAIN,
                                sample_type=SampleType.PHOTOMETRIC_OCCLUDED,
                                reason=Reason.NO_TARGET,
                                source_image_id=image_path.stem,
                                source_target=source_target,
                                transform="brightness_contrast+ladder_occlude",
                                metadata={
                                    "relation": "T2_occluded",
                                    "brightness": 1.1,
                                    "contrast": 1.1,
                                    "achieved_visible_ratio": occ_achieved,
                                    "collinearity_break": True,
                                },
                            )
                        )

            if "T3" in enabled_relations:
                flip_path = transform_images_dir / f"{group_id}-hflip.jpg"
                flip_box = save_horizontal_flip(image_path, box, flip_path)
                group_samples.append(
                    BenchmarkSample(
                        sample_id=f"{group_id}-hflip",
                        group_id=group_id,
                        dataset=DatasetName.PLANTDOC,
                        split=split,
                        image_path=str(flip_path.resolve()),
                        query=build_query_varied(label, f"{group_id}-T3-{label}", is_answer=True),
                        decision=Decision.ANSWER,
                        sample_type=SampleType.GEOMETRIC_EQUIVARIANCE,
                        label=disp,
                        box=flip_box,
                        source_image_id=image_path.stem,
                        source_target=source_target,
                        transform="horizontal_flip",
                        metadata={"relation": "T3", "operator": "horizontal_flip"},
                    )
                )

                if collinearity_break:
                    # Same flipped image, target occluded -> abstain twin. Must use flip_path +
                    # flip_box (geometry changed by hflip), else the crop misses the target.
                    occ_path = transform_images_dir / f"{group_id}-hflip-occluded.jpg"
                    _, occ_achieved, occ_wrote = save_visibility_ladder_crop(
                        flip_path, flip_box, occ_path, visible_ratio=0.0
                    )
                    if occ_wrote:
                        group_samples.append(
                            BenchmarkSample(
                                sample_id=f"{group_id}-hflip-occluded",
                                group_id=group_id,
                                dataset=DatasetName.PLANTDOC,
                                split=split,
                                image_path=str(occ_path.resolve()),
                                query=build_query_varied(
                                    label, f"{group_id}-T3_occluded-{label}", is_answer=False
                                ),
                                decision=Decision.ABSTAIN,
                                sample_type=SampleType.GEOMETRIC_OCCLUDED,
                                reason=Reason.NO_TARGET,
                                source_image_id=image_path.stem,
                                source_target=source_target,
                                transform="horizontal_flip+ladder_occlude",
                                metadata={
                                    "relation": "T3_occluded",
                                    "operator": "horizontal_flip",
                                    "achieved_visible_ratio": occ_achieved,
                                    "collinearity_break": True,
                                },
                            )
                        )

            target_crop = label.split("_")[0]
            absent_ids = [
                idx
                for idx in range(len(class_names))
                if idx not in present_class_ids
                and idx not in excluded_class_ids
                and class_names[idx].split("_")[0] != target_crop
            ]
            for absent_class_id in _select_absent_ids(
                absent_ids, absent_per_positive, seed_key=group_id
            ):
                absent_label = _display(class_names[absent_class_id])
                group_samples.append(
                    BenchmarkSample(
                        sample_id=f"{group_id}-absent-{absent_class_id}",
                        group_id=group_id,
                        dataset=DatasetName.PLANTDOC,
                        split=split,
                        image_path=str(image_path.resolve()),
                        query=build_query_varied(
                            absent_label, f"{group_id}-T5-{absent_label}", is_answer=False
                        ),
                        decision=Decision.ABSTAIN,
                        sample_type=SampleType.TARGET_ABSENT_NEGATIVE,
                        reason=Reason.NO_TARGET,
                        source_image_id=image_path.stem,
                        source_target=source_target,
                        transform="target_absent_swap",
                        metadata={"relation": "T5", "absent_label": absent_label},
                    )
                )

            if "T4" in enabled_relations:
                blur_radii_visible = metamorphic.get("blur_radii_visible", [2, 4])
                blur_radii_occluded = metamorphic.get("blur_radii_occluded", [6, 8])
                pixelate_blocks_visible = metamorphic.get("pixelate_blocks_visible", [4, 6])
                pixelate_blocks_occluded = metamorphic.get("pixelate_blocks_occluded", [10, 12])

                for operator in removal_operators:
                    if operator == "gaussian_blur":
                        for radius in blur_radii_visible:
                            removed_path = (
                                transform_images_dir / f"{group_id}-target-{operator}-r{radius}.jpg"
                            )
                            removal_metadata: dict[str, Any] = {
                                "relation": "T4",
                                "operator": operator,
                                "radius": radius,
                                "strength": "visible",
                                "dose_only": True,
                            }
                            if target_removal_source == "region_include_crop":
                                crop_box = _save_removal_on_inclusion_crop(
                                    operator, image_path, box, removed_path, radius=radius
                                )
                                removal_metadata["removal_source"] = "region_include_crop"
                                removal_metadata["crop_target_box"] = crop_box.as_list()
                            elif target_removal_source == "full_image":
                                _save_removal(
                                    operator, image_path, box, removed_path, radius=radius
                                )
                                removal_metadata["removal_source"] = "full_image"
                            else:
                                raise ValueError(
                                    f"Unknown PlantDoc target_removal_source: {target_removal_source}"
                                )
                            group_samples.append(
                                BenchmarkSample(
                                    sample_id=f"{group_id}-target-{operator}-r{radius}",
                                    group_id=group_id,
                                    dataset=DatasetName.PLANTDOC,
                                    split=split,
                                    image_path=str(removed_path.resolve()),
                                    query=build_query_varied(
                                        label,
                                        f"{group_id}-T4-{operator}-r{radius}-{label}",
                                        is_answer=True,
                                    ),
                                    decision=Decision.ANSWER,
                                    sample_type=SampleType.ANSWERABLE_POSITIVE,
                                    label=disp,
                                    box=box,
                                    source_image_id=image_path.stem,
                                    source_target=source_target,
                                    transform=f"target_{operator}_r{radius}",
                                    metadata=removal_metadata,
                                )
                            )

                        for radius in blur_radii_occluded:
                            removed_path = (
                                transform_images_dir / f"{group_id}-target-{operator}-r{radius}.jpg"
                            )
                            removal_metadata = {
                                "relation": "T4",
                                "operator": operator,
                                "radius": radius,
                                "strength": "occluded",
                                "dose_only": True,
                            }
                            if target_removal_source == "region_include_crop":
                                crop_box = _save_removal_on_inclusion_crop(
                                    operator, image_path, box, removed_path, radius=radius
                                )
                                removal_metadata["removal_source"] = "region_include_crop"
                                removal_metadata["crop_target_box"] = crop_box.as_list()
                            elif target_removal_source == "full_image":
                                _save_removal(
                                    operator, image_path, box, removed_path, radius=radius
                                )
                                removal_metadata["removal_source"] = "full_image"
                            else:
                                raise ValueError(
                                    f"Unknown PlantDoc target_removal_source: {target_removal_source}"
                                )
                            group_samples.append(
                                BenchmarkSample(
                                    sample_id=f"{group_id}-target-{operator}-r{radius}",
                                    group_id=group_id,
                                    dataset=DatasetName.PLANTDOC,
                                    split=split,
                                    image_path=str(removed_path.resolve()),
                                    query=build_query_varied(
                                        label,
                                        f"{group_id}-T4-{operator}-r{radius}-{label}",
                                        is_answer=False,
                                    ),
                                    decision=Decision.ABSTAIN,
                                    sample_type=SampleType.EVIDENCE_INSUFFICIENT_NEGATIVE,
                                    source_image_id=image_path.stem,
                                    source_target=source_target,
                                    transform=f"target_{operator}_r{radius}",
                                    metadata=removal_metadata,
                                )
                            )

                    elif operator == "pixelate":
                        for block_size in pixelate_blocks_visible:
                            removed_path = (
                                transform_images_dir
                                / f"{group_id}-target-{operator}-b{block_size}.jpg"
                            )
                            removal_metadata = {
                                "relation": "T4",
                                "operator": operator,
                                "block_size": block_size,
                                "strength": "visible",
                                "dose_only": True,
                            }
                            if target_removal_source == "region_include_crop":
                                crop_box = _save_removal_on_inclusion_crop(
                                    operator, image_path, box, removed_path, block_size=block_size
                                )
                                removal_metadata["removal_source"] = "region_include_crop"
                                removal_metadata["crop_target_box"] = crop_box.as_list()
                            elif target_removal_source == "full_image":
                                _save_removal(
                                    operator, image_path, box, removed_path, block_size=block_size
                                )
                                removal_metadata["removal_source"] = "full_image"
                            else:
                                raise ValueError(
                                    f"Unknown PlantDoc target_removal_source: {target_removal_source}"
                                )
                            group_samples.append(
                                BenchmarkSample(
                                    sample_id=f"{group_id}-target-{operator}-b{block_size}",
                                    group_id=group_id,
                                    dataset=DatasetName.PLANTDOC,
                                    split=split,
                                    image_path=str(removed_path.resolve()),
                                    query=build_query_varied(
                                        label,
                                        f"{group_id}-T4-{operator}-b{block_size}-{label}",
                                        is_answer=True,
                                    ),
                                    decision=Decision.ANSWER,
                                    sample_type=SampleType.ANSWERABLE_POSITIVE,
                                    label=disp,
                                    box=box,
                                    source_image_id=image_path.stem,
                                    source_target=source_target,
                                    transform=f"target_{operator}_b{block_size}",
                                    metadata=removal_metadata,
                                )
                            )

                        for block_size in pixelate_blocks_occluded:
                            removed_path = (
                                transform_images_dir
                                / f"{group_id}-target-{operator}-b{block_size}.jpg"
                            )
                            removal_metadata = {
                                "relation": "T4",
                                "operator": operator,
                                "block_size": block_size,
                                "strength": "occluded",
                                "dose_only": True,
                            }
                            if target_removal_source == "region_include_crop":
                                crop_box = _save_removal_on_inclusion_crop(
                                    operator, image_path, box, removed_path, block_size=block_size
                                )
                                removal_metadata["removal_source"] = "region_include_crop"
                                removal_metadata["crop_target_box"] = crop_box.as_list()
                            elif target_removal_source == "full_image":
                                _save_removal(
                                    operator, image_path, box, removed_path, block_size=block_size
                                )
                                removal_metadata["removal_source"] = "full_image"
                            else:
                                raise ValueError(
                                    f"Unknown PlantDoc target_removal_source: {target_removal_source}"
                                )
                            group_samples.append(
                                BenchmarkSample(
                                    sample_id=f"{group_id}-target-{operator}-b{block_size}",
                                    group_id=group_id,
                                    dataset=DatasetName.PLANTDOC,
                                    split=split,
                                    image_path=str(removed_path.resolve()),
                                    query=build_query_varied(
                                        label,
                                        f"{group_id}-T4-{operator}-b{block_size}-{label}",
                                        is_answer=False,
                                    ),
                                    decision=Decision.ABSTAIN,
                                    sample_type=SampleType.EVIDENCE_INSUFFICIENT_NEGATIVE,
                                    source_image_id=image_path.stem,
                                    source_target=source_target,
                                    transform=f"target_{operator}_b{block_size}",
                                    metadata=removal_metadata,
                                )
                            )

                    elif operator == "mean_replace":
                        # mean_replace: no reason GT, dose_only
                        removed_path = transform_images_dir / f"{group_id}-target-{operator}.jpg"
                        removal_metadata = {
                            "relation": "T4",
                            "operator": operator,
                            "dose_only": True,
                        }
                        if target_removal_source == "region_include_crop":
                            crop_box = _save_removal_on_inclusion_crop(
                                operator, image_path, box, removed_path
                            )
                            removal_metadata["removal_source"] = "region_include_crop"
                            removal_metadata["crop_target_box"] = crop_box.as_list()
                        elif target_removal_source == "full_image":
                            _save_removal(operator, image_path, box, removed_path)
                            removal_metadata["removal_source"] = "full_image"
                        else:
                            raise ValueError(
                                f"Unknown PlantDoc target_removal_source: {target_removal_source}"
                            )
                        group_samples.append(
                            BenchmarkSample(
                                sample_id=f"{group_id}-target-{operator}",
                                group_id=group_id,
                                dataset=DatasetName.PLANTDOC,
                                split=split,
                                image_path=str(removed_path.resolve()),
                                query=build_query_varied(
                                    label, f"{group_id}-T4-{operator}-{label}", is_answer=False
                                ),
                                decision=Decision.ABSTAIN,
                                sample_type=SampleType.EVIDENCE_REMOVED_NEGATIVE,
                                source_image_id=image_path.stem,
                                source_target=source_target,
                                transform=f"target_{operator}",
                                metadata=removal_metadata,
                            )
                        )

                    control_box, control_source = _select_non_target_control_box(
                        labels,
                        target_class_id=class_id,
                        target_box=box,
                    )
                    if control_box is not None:
                        control_path = transform_images_dir / f"{group_id}-nontarget-{operator}.jpg"
                        _save_removal(operator, image_path, control_box, control_path)
                        group_samples.append(
                            BenchmarkSample(
                                sample_id=f"{group_id}-nontarget-{operator}",
                                group_id=group_id,
                                dataset=DatasetName.PLANTDOC,
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
                                source_image_id=image_path.stem,
                                source_target=source_target,
                                transform=f"non_target_{operator}",
                                metadata={
                                    "relation": "T4_control",
                                    "operator": operator,
                                    "control_source": control_source,
                                    "control_box": control_box.as_list(),
                                },
                            )
                        )

            if "T6" in enabled_relations:
                for mismatch_label in mismatch_pairs.get(label, []):
                    mismatch_class_id = class_ids_by_name.get(mismatch_label)
                    if (
                        mismatch_class_id is None
                        or mismatch_class_id in present_class_ids
                        or mismatch_class_id in excluded_class_ids
                    ):
                        continue
                    group_samples.append(
                        BenchmarkSample(
                            sample_id=f"{group_id}-mismatch-{mismatch_label}",
                            group_id=group_id,
                            dataset=DatasetName.PLANTDOC,
                            split=split,
                            image_path=str(image_path.resolve()),
                            query=build_query_varied(
                                mismatch_label, f"{group_id}-T6-{mismatch_label}", is_answer=False
                            ),
                            decision=Decision.ABSTAIN,
                            sample_type=SampleType.QUERY_IMAGE_MISMATCH_NEGATIVE,
                            reason=Reason.WRONG_OBJECT,
                            source_image_id=image_path.stem,
                            source_target=source_target,
                            transform="query_image_mismatch",
                            metadata={"relation": "T6", "mismatch_label": _display(mismatch_label)},
                        )
                    )

            if "T7" in enabled_relations:
                include_path = transform_images_dir / f"{group_id}-region-include.jpg"
                include_box = save_inclusion_crop(image_path, box, include_path)
                group_samples.append(
                    BenchmarkSample(
                        sample_id=f"{group_id}-region-include",
                        group_id=group_id,
                        dataset=DatasetName.PLANTDOC,
                        split=split,
                        image_path=str(include_path.resolve()),
                        query=build_query_varied(
                            label, f"{group_id}-T7_include-{label}", is_answer=True
                        ),
                        decision=Decision.ANSWER,
                        sample_type=SampleType.REGION_INCLUDE,
                        label=disp,
                        box=include_box,
                        source_image_id=image_path.stem,
                        source_target=source_target,
                        transform="region_include_crop",
                        metadata={"relation": "T7_include"},
                    )
                )

                # T7 severity ladder: slide a crop edge across the target so a controlled
                # fraction stays visible (geometric objective anchor). This replaces the old
                # binary region_exclude (which silently dropped centered/large targets), and
                # yields the Risk-Coverage ladder. L=0.0 == full evidence removal (abstain).
                if region_ladder_enabled and label not in region_exclude_skip_labels:
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
                                    dataset=DatasetName.PLANTDOC,
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
                                    source_image_id=image_path.stem,
                                    source_target=source_target,
                                    transform="region_ladder_crop",
                                    metadata=ladder_meta,
                                )
                            )
                        else:
                            if level <= 0.001:
                                ladder_sample_type = SampleType.TARGET_ABSENT_NEGATIVE
                                ladder_meta["dose_only"] = False
                                group_samples.append(
                                    BenchmarkSample(
                                        sample_id=f"{group_id}-ladder-{int(round(level * 100)):03d}",
                                        group_id=group_id,
                                        dataset=DatasetName.PLANTDOC,
                                        split=split,
                                        image_path=str(ladder_path.resolve()),
                                        query=build_query_varied(
                                            label,
                                            f"{group_id}-T7_ladder-{level}-{label}",
                                            is_answer=False,
                                        ),
                                        decision=Decision.ABSTAIN,
                                        sample_type=ladder_sample_type,
                                        reason=Reason.NO_TARGET,
                                        source_image_id=image_path.stem,
                                        source_target=source_target,
                                        transform="region_ladder_crop",
                                        metadata=ladder_meta,
                                    )
                                )
                            else:
                                ladder_sample_type = SampleType.REGION_LADDER_OCCLUDED
                                ladder_meta["dose_only"] = True
                                group_samples.append(
                                    BenchmarkSample(
                                        sample_id=f"{group_id}-ladder-{int(round(level * 100)):03d}",
                                        group_id=group_id,
                                        dataset=DatasetName.PLANTDOC,
                                        split=split,
                                        image_path=str(ladder_path.resolve()),
                                        query=build_query_varied(
                                            label,
                                            f"{group_id}-T7_ladder-{level}-{label}",
                                            is_answer=False,
                                        ),
                                        decision=Decision.ABSTAIN,
                                        sample_type=ladder_sample_type,
                                        source_image_id=image_path.stem,
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
    absent_ids: list[int],
    absent_per_positive: int | str | None,
    seed_key: str = "",
) -> list[int]:
    """Pick which absent classes to query for a positive sample.

    `all` keeps every absent class (legacy, causes T5 to dominate the pool). A small integer
    is the recommended setting: it deterministically samples that many absent classes so T5
    stays balanced against the evidence-grounded relations (T4/T7). Deterministic per group
    via `seed_key`, and unbiased (not just the lowest class ids).
    """
    if absent_per_positive in (None, "all"):
        return absent_ids
    n = int(absent_per_positive)
    if n >= len(absent_ids):
        return absent_ids
    import random

    rng = random.Random(f"absent:{seed_key}")
    return sorted(rng.sample(absent_ids, n))


def _save_removal(
    operator: str,
    image_path: Path,
    box: Box,
    output_path: Path,
    radius: float | None = None,
    block_size: int | None = None,
) -> None:
    if operator == "gaussian_blur":
        r = radius if radius is not None else 8.0
        save_gaussian_blur_region(image_path, box, output_path, radius=r)
    elif operator == "mean_replace":
        save_mean_replace_region(image_path, box, output_path)
    elif operator == "pixelate":
        bs = block_size if block_size is not None else 12
        save_pixelate_region(image_path, box, output_path, block_size=bs)
    else:
        raise ValueError(f"Unknown removal operator: {operator}")


def _save_removal_on_inclusion_crop(
    operator: str,
    image_path: Path,
    box: Box,
    output_path: Path,
    radius: float | None = None,
    block_size: int | None = None,
) -> Box:
    temp_path = output_path.with_name(f"{output_path.stem}.source_crop.tmp{output_path.suffix}")
    crop_box = save_inclusion_crop(image_path, box, temp_path)
    try:
        _save_removal(
            operator, temp_path, crop_box, output_path, radius=radius, block_size=block_size
        )
    finally:
        temp_path.unlink(missing_ok=True)
    return crop_box


def _select_non_target_control_box(
    labels: list[tuple[int, float, float, float, float]],
    target_class_id: int,
    target_box: Box,
) -> tuple[Box | None, str | None]:
    """Degraded region for the T4 control: grid-search an empty background region that avoids every
    annotated box, as in the COCO builder.

    The search is driven by avoidance of the annotated boxes rather than by picking another
    annotated object, because an IoU test does not detect a large box containing a small one and
    would let the control cover a target. Coverage of any target is therefore zero by
    construction.
    """
    annotated_boxes = [_yolo_to_qwen_box(cx, cy, w, h) for _, cx, cy, w, h in labels]
    background_box = _select_background_control_box(annotated_boxes, target_box)
    if background_box is None:
        return None, None
    return background_box, "background_non_target"


def _select_background_control_box(occupied_boxes: list[Box], target_box: Box) -> Box | None:
    target_width = target_box.x2 - target_box.x1
    target_height = target_box.y2 - target_box.y1
    for scale in (1.0, 0.75, 0.5):
        width = max(50, round(target_width * scale))
        height = max(50, round(target_height * scale))
        if width >= 1000 or height >= 1000:
            continue
        candidates: list[Box] = []
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
    if max_start < 0:
        return []
    if max_start == 0:
        return [0]
    return sorted({round(max_start * idx / steps) for idx in range(steps + 1)})


def _box_area(box: Box) -> int:
    return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)


def _box_iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union else 0.0


def _boxes_overlap_with_margin(a: Box, b: Box, margin: int = 20) -> bool:
    return not (
        a.x2 <= b.x1 - margin
        or a.x1 >= b.x2 + margin
        or a.y2 <= b.y1 - margin
        or a.y1 >= b.y2 + margin
    )


def _center_distance(a: Box, b: Box) -> float:
    ax = (a.x1 + a.x2) / 2
    ay = (a.y1 + a.y2) / 2
    bx = (b.x1 + b.x2) / 2
    by = (b.y1 + b.y2) / 2
    return (ax - bx) ** 2 + (ay - by) ** 2


def _short_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
