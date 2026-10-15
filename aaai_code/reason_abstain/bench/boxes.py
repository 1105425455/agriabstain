from __future__ import annotations

from .schema import Box


def xywh_to_qwen_box(xywh: list[float], width: int, height: int) -> Box:
    x, y, w, h = xywh
    x1 = round(1000 * x / width)
    y1 = round(1000 * y / height)
    x2 = round(1000 * (x + w) / width)
    y2 = round(1000 * (y + h) / height)
    return Box(
        x1=max(0, min(1000, x1)),
        y1=max(0, min(1000, y1)),
        x2=max(0, min(1000, x2)),
        y2=max(0, min(1000, y2)),
    )


def qwen_box_to_xyxy(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        box.x1 * width / 1000,
        box.y1 * height / 1000,
        box.x2 * width / 1000,
        box.y2 * height / 1000,
    )


def box_area(box: Box) -> int:
    return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)
