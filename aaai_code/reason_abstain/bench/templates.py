from __future__ import annotations

import hashlib

GROUNDING_TEMPLATES = [
    "Locate the {label} in this image.",
    "Find the {label} and return its bounding box.",
    "Where is the {label} in the image?",
]

ABSTAIN_TEMPLATE_IDS = [0, 2]
ANSWER_TEMPLATE_IDS = [0, 1, 2]


def build_query(label: str, template_id: int = 0) -> str:
    template = GROUNDING_TEMPLATES[template_id % len(GROUNDING_TEMPLATES)]
    return template.format(label=label.replace("_", " "))


def _stable_choice(vary_key: str, pool: list[int]) -> int:
    """Pick a template id from the pool deterministically, by hashing the contents of `vary_key`."""
    h = int(hashlib.md5(vary_key.encode()).hexdigest(), 16)
    return pool[h % len(pool)]


def build_query_varied(label: str, vary_key: str, is_answer: bool) -> str:
    """Vary the phrasing by scenario: answers draw on all templates, abstentions on the neutral
    ones only, deterministically so the result is reproducible.
    """
    pool = ANSWER_TEMPLATE_IDS if is_answer else ABSTAIN_TEMPLATE_IDS
    tid = _stable_choice(vary_key, pool)
    return build_query(label, template_id=tid)
