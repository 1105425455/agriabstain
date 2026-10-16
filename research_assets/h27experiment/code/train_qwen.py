#!/usr/bin/env python3
"""Register one experiment-local dataset and delegate to the released trainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
AAAI_ROOT = REPO_ROOT / "aaai_code"
TRAINING_ROOT = AAAI_ROOT / "training"
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(AAAI_ROOT))

from qwenvl.data import data_dict  # noqa: E402


def main() -> None:
    annotation_path = Path(os.environ["H27_TRAIN_DATA"]).resolve()
    media_root = Path(os.environ["H27_MEDIA_ROOT"]).resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Training data not found: {annotation_path}")
    if not media_root.is_dir():
        raise FileNotFoundError(f"Media root not found: {media_root}")

    data_dict["h27_multiseed"] = {
        "annotation_path": str(annotation_path),
        "data_path": str(media_root),
    }

    from qwenvl.train.train_qwen import train

    train(attn_implementation="sdpa")


if __name__ == "__main__":
    main()
