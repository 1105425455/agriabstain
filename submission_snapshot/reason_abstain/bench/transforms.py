from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .boxes import qwen_box_to_xyxy
from .schema import Box


Resampling = Image.Resampling

# Qwen3-VL's image processor rejects images whose absolute aspect ratio exceeds 200. Crops that
# slice a thin strip off a near-full-frame target can produce such degenerate shapes, so we guard
# generation with a safe ceiling well below that limit.
MAX_ASPECT_RATIO = 100.0


def _aspect_ok(w: int, h: int) -> bool:
    if w <= 0 or h <= 0:
        return False
    return (max(w, h) / min(w, h)) <= MAX_ASPECT_RATIO


def save_gaussian_blur_region(
    image_path: str | Path,
    box: Box,
    output_path: str | Path,
    radius: float = 8.0,
) -> None:
    """Apply Gaussian blur to the target region. Supports multiple radii via caller."""
    image = Image.open(image_path).convert("RGB")
    region_xyxy = _clip_xyxy(qwen_box_to_xyxy(box, *image.size), *image.size)
    region = image.crop(region_xyxy).filter(ImageFilter.GaussianBlur(radius=radius))
    image.paste(region, region_xyxy)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_mean_replace_region(
    image_path: str | Path,
    box: Box,
    output_path: str | Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image).copy()
    x1, y1, x2, y2 = _clip_xyxy(qwen_box_to_xyxy(box, *image.size), *image.size)
    mean_color = array.mean(axis=(0, 1)).astype(array.dtype)
    array[y1:y2, x1:x2, :] = mean_color
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path)


def save_pixelate_region(
    image_path: str | Path,
    box: Box,
    output_path: str | Path,
    block_size: int = 12,
) -> None:
    """Apply pixelation to the target region. Supports multiple block sizes via caller."""
    image = Image.open(image_path).convert("RGB")
    region_xyxy = _clip_xyxy(qwen_box_to_xyxy(box, *image.size), *image.size)
    region = image.crop(region_xyxy)
    downsampled_size = (
        max(1, region.width // block_size),
        max(1, region.height // block_size),
    )
    region = region.resize(downsampled_size, resample=Resampling.BILINEAR).resize(
        region.size,
        resample=Resampling.NEAREST,
    )
    image.paste(region, region_xyxy)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_brightness_contrast(
    image_path: str | Path,
    output_path: str | Path,
    brightness: float = 1.1,
    contrast: float = 1.1,
) -> None:
    image = Image.open(image_path).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_horizontal_flip(
    image_path: str | Path,
    box: Box,
    output_path: str | Path,
) -> Box:
    image = Image.open(image_path).convert("RGB")
    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    flipped.save(output_path)
    return Box(
        x1=1000 - box.x2,
        y1=box.y1,
        x2=1000 - box.x1,
        y2=box.y2,
    )


def save_inclusion_crop(
    image_path: str | Path,
    box: Box,
    output_path: str | Path,
    context_ratio: float = 0.10,
) -> Box:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = _clip_xyxy(qwen_box_to_xyxy(box, width, height), width, height)
    pad_x = round((x2 - x1) * context_ratio)
    pad_y = round((y2 - y1) * context_ratio)
    crop_xyxy = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    crop = image.crop(crop_xyxy)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    cx1, cy1, _, _ = crop_xyxy
    crop_width, crop_height = crop.size
    return Box(
        x1=round(1000 * (x1 - cx1) / crop_width),
        y1=round(1000 * (y1 - cy1) / crop_height),
        x2=round(1000 * (x2 - cx1) / crop_width),
        y2=round(1000 * (y2 - cy1) / crop_height),
    )


def save_exclusion_crop(
    image_path: str | Path,
    box: Box,
    output_path: str | Path,
    min_area_ratio: float = 0.60,
) -> bool:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = _clip_xyxy(qwen_box_to_xyxy(box, width, height), width, height)
    candidates = [
        (0, 0, x1, height),
        (x2, 0, width, height),
        (0, 0, width, y1),
        (0, y2, width, height),
    ]
    min_area = width * height * min_area_ratio
    candidates = [
        candidate
        for candidate in candidates
        if (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]) >= min_area
    ]
    if not candidates:
        return False
    crop_xyxy = max(candidates, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    crop = image.crop(crop_xyxy)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return True


def save_visibility_ladder_crop(
    image_path: str | Path,
    box: Box,
    output_path: str | Path,
    visible_ratio: float,
    context_ratio: float = 0.10,
) -> tuple[Box | None, float, bool]:
    """Crop so that exactly ``visible_ratio`` of the target's area remains in frame.

    This is the geometric objective anchor (W1) for the T7 severity ladder: instead of the
    binary "exclude the whole box", we slide a crop edge across the target so a controlled
    fraction of it stays visible (1.0 -> 0.5 -> 0.25 -> 0.0). The crop direction picks the
    side with the largest surrounding context so the retained image is meaningful.

    Returns ``(updated_box_or_None, achieved_visible_ratio, wrote_file)``:
    - ``visible_ratio >= 0.999``: inclusion crop (full target), box returned, file written.
    - ``0 < visible_ratio < 1``: box clipped to the visible part, file written.
    - ``visible_ratio <= 0``: target fully removed, box is None, file written — abstain sample.
    ``wrote_file`` is False when the requested crop degenerates (too small to be meaningful);
    callers MUST skip emitting a sample in that case (otherwise it points to a missing image).
    The achieved ratio is recomputed from integer pixels and reported for auditing.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = _clip_xyxy(qwen_box_to_xyxy(box, width, height), width, height)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None, 0.0, False

    visible_ratio = max(0.0, min(1.0, float(visible_ratio)))

    if visible_ratio >= 0.999:
        new_box = save_inclusion_crop(image_path, box, output_path, context_ratio=context_ratio)
        return new_box, 1.0, True

    # Choose the crop edge: cut from the side that has the LEAST context, sliding inward across
    # the target, so the kept image still shows surrounding scene rather than empty padding.
    context = {
        "cut_left": x1,  # space to the left of the box
        "cut_right": width - x2,  # space to the right
        "cut_top": y1,
        "cut_bottom": height - y2,
    }
    direction = min(context, key=context.get)

    if visible_ratio <= 0.0:
        # Remove the entire target: keep the largest complementary region along the chosen axis.
        if direction in ("cut_left", "cut_right"):
            crop_xyxy = (x2, 0, width, height) if direction == "cut_left" else (0, 0, x1, height)
        else:
            crop_xyxy = (0, y2, width, height) if direction == "cut_top" else (0, 0, width, y1)
        cw0, ch0 = crop_xyxy[2] - crop_xyxy[0], crop_xyxy[3] - crop_xyxy[1]
        if cw0 < 8 or ch0 < 8 or not _aspect_ok(cw0, ch0):
            return None, 0.0, False
        crop = image.crop(crop_xyxy)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)
        return None, 0.0, True

    # Partial visibility: cut a fraction (1 - visible_ratio) off the target along one axis.
    cut = 1.0 - visible_ratio
    if direction in ("cut_left", "cut_right"):
        cut_px = round(bw * cut)
        if direction == "cut_left":
            new_x1, new_x2 = x1 + cut_px, x2
            crop_xyxy = (new_x1, 0, width, height)
        else:
            new_x1, new_x2 = x1, x2 - cut_px
            crop_xyxy = (0, 0, new_x2, height)
        kept_box_px = (new_x1, y1, new_x2, y2)
    else:
        cut_px = round(bh * cut)
        if direction == "cut_top":
            new_y1, new_y2 = y1 + cut_px, y2
            crop_xyxy = (0, new_y1, width, height)
        else:
            new_y1, new_y2 = y1, y2 - cut_px
            crop_xyxy = (0, 0, width, new_y2)
        kept_box_px = (x1, new_y1, x2, new_y2)

    cw, ch = crop_xyxy[2] - crop_xyxy[0], crop_xyxy[3] - crop_xyxy[1]
    kw = kept_box_px[2] - kept_box_px[0]
    kh = kept_box_px[3] - kept_box_px[1]
    if cw < 8 or ch < 8 or kw < 4 or kh < 4 or not _aspect_ok(cw, ch):
        return None, 0.0, False

    crop = image.crop(crop_xyxy)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)

    achieved = (kw * kh) / float(bw * bh)
    new_box = Box(
        x1=round(1000 * (kept_box_px[0] - crop_xyxy[0]) / cw),
        y1=round(1000 * (kept_box_px[1] - crop_xyxy[1]) / ch),
        x2=round(1000 * (kept_box_px[2] - crop_xyxy[0]) / cw),
        y2=round(1000 * (kept_box_px[3] - crop_xyxy[1]) / ch),
    )
    return new_box, round(achieved, 4), True


def _clip_xyxy(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return (
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(1, min(width, round(x2))),
        max(1, min(height, round(y2))),
    )
