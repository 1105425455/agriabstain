from __future__ import annotations

import json
import re
from typing import Any

VALID_REASONS = {
    "wrong_object",
    "no_target",
    "target_absent",
    "visual_evidence_insufficient",
    "query_image_mismatch",
}


def normalize_label(label: object) -> str:
    """Normalize a class label for comparison.

    Qwen3-VL outputs labels like "tomato septoria leaf spot" while gold labels may use
    underscores or different casing. Comparison therefore happens on a normalized form:
    lowercase, underscores to spaces, punctuation stripped, the generic modifiers `leaf`,
    `plant` and `disease` removed, and whitespace collapsed.

    The same normalization is applied symmetrically to the prediction and the reference, so it
    cannot favour either side. It is specified in the technical supplement, in the section on
    the output format.
    """
    if not isinstance(label, str):
        return ""
    text = label.lower().strip().replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\b(leaf|plant|disease)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_structured_output(text: str) -> dict[str, Any]:
    payload = _extract_json_object(text)
    if payload is None:
        return {"valid": False, "error": "no_json_object"}

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"valid": False, "error": "invalid_json"}

    if not isinstance(parsed, dict):
        return {"valid": False, "error": "not_object"}

    if "present" in parsed:
        present = parsed["present"]
        if present is True:
            label = parsed.get("label")
            box = parsed.get("box")
            if not isinstance(label, str) or not _valid_box(box):
                return {"valid": False, "error": "invalid_answer_fields", "raw": parsed}
            return {"valid": True, "decision": "answer", "label": label, "box": box}
        if present is False:
            reason = parsed.get("reason")
            if reason is None or reason not in VALID_REASONS:
                return {"valid": False, "error": "missing_reason", "raw": parsed}
            return {"valid": True, "decision": "abstain", "reason": reason}
        return {"valid": False, "error": "invalid_present_value", "raw": parsed}

    decision = parsed.get("decision")
    if decision == "answer":
        label = parsed.get("label")
        box = parsed.get("box")
        if not isinstance(label, str) or not _valid_box(box):
            return {"valid": False, "error": "invalid_answer_fields", "raw": parsed}
        return {"valid": True, "decision": "answer", "label": label, "box": box}

    if decision == "abstain":
        reason = parsed.get("reason")
        if reason not in VALID_REASONS:
            return {"valid": False, "error": "invalid_reason", "raw": parsed}
        return {"valid": True, "decision": "abstain", "reason": reason}

    return {"valid": False, "error": "invalid_decision", "raw": parsed}


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL)


def _extract_json_object(text: str) -> str | None:
    text = text.strip()
    tag = _ANSWER_TAG_RE.search(text)
    if tag is not None:
        inner = tag.group(1).strip()
        if inner.startswith("{") and inner.endswith("}"):
            return inner
        inner_match = re.search(r"\{.*\}", inner, flags=re.DOTALL)
        if inner_match:
            return inner_match.group(0)
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _valid_box(box: object) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(v, int | float) for v in box):
        return False
    x1, y1, x2, y2 = [float(v) for v in box]
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
