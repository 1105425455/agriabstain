#!/usr/bin/env python3
"""Seed the released Stage-2 calibration script without modifying aaai_code."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATE_SCRIPT = EXPERIMENT_ROOT.parent / "aaai_code/training/sft_final/calibrate_head.py"


def pin_visible_device() -> None:
    """Give each torchrun worker exactly one GPU before importing torch."""
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1")))
    if local_rank < 0:
        return
    visible = [
        device.strip()
        for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]
    selected = visible[local_rank] if local_rank < len(visible) else str(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    print(
        f"[h27-calibrate] local_rank={local_rank}; CUDA_VISIBLE_DEVICES={selected}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-seed", type=int, required=True)
    known, remaining = parser.parse_known_args()

    pin_visible_device()
    import numpy as np
    import torch

    seed = known.run_seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(
        f"[h27-calibrate] run_seed={seed}; dev_subset_sampling_seed=42; "
        f"delegate={CALIBRATE_SCRIPT}",
        flush=True,
    )

    if not CALIBRATE_SCRIPT.is_file():
        raise FileNotFoundError(CALIBRATE_SCRIPT)
    sys.argv = [str(CALIBRATE_SCRIPT), *remaining]
    runpy.run_path(str(CALIBRATE_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
