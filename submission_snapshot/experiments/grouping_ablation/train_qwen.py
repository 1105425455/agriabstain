#!/usr/bin/env python3
"""Isolated SFT entry point for the grouping ablation.

This wrapper adds one experiment-local dataset entry at runtime and then delegates to
the unchanged SFT_FINAL Qwen trainer. No shared training registry is modified.
"""

from __future__ import annotations

import os
from pathlib import Path

from qwenvl.data import data_dict


def main() -> None:
    annotation_path = Path(os.environ["GROUPING_ABLATION_TRAIN_DATA"]).resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Grouping-ablation training data not found: {annotation_path}")

    data_dict["grouping_ablation_independent_negatives"] = {
        "annotation_path": str(annotation_path),
        # Absolute media paths are emitted by the builder. Path(base) / absolute_path
        # remains the absolute path in the shared data processor.
        "data_path": "/",
    }

    from qwenvl.train.train_qwen import train

    train(attn_implementation="sdpa")


if __name__ == "__main__":
    main()
