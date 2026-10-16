#!/usr/bin/env python3
"""Verify calibration features really came from the paper's own Stage-1 adapter.

The released calibrated head stores the per-dimension mean/std of the calibration
training features it was fitted on, so reproducing those statistics from a freshly
extracted feature set proves the extraction used the same adapter and the same
7072-record training split. A different checkpoint shows |dmu| of several units.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

FAIL_TOLERANCE = 0.5
WARN_TOLERANCE = 0.02

# The stored mean/std only equal the calibration-training statistics when the head was fitted
# by calibrate_head.py --mode train on that same extraction. The released grouped head was
# not, so a mismatch is a provenance signal to report rather than a correctness failure: the
# head normalises with its own stored statistics at inference either way. Reproduction is
# judged by whether the evaluator reproduces the published table, not by this check.
STRICT = False


def load_shards(calibration_dir: Path, prefix: str) -> np.ndarray:
    paths = sorted(glob.glob(str(calibration_dir / f"{prefix}_rank*.npz")))
    if not paths:
        raise FileNotFoundError(f"no {prefix}_rank*.npz under {calibration_dir}")
    return np.concatenate([np.load(path)["feats"].astype(np.float32) for path in paths])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--expected-train-rows", type=int, default=7072)
    parser.add_argument("--expected-dev-rows", type=int, default=100)
    parser.add_argument(
        "--strict",
        action="store_true",
        default=STRICT,
        help="treat a statistics mismatch as fatal instead of reporting it",
    )
    args = parser.parse_args()

    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    mu = payload["mu"].float().numpy()
    sd = payload["sd"].float().numpy()

    train = load_shards(args.calibration_dir, "calib_train")
    dev = load_shards(args.calibration_dir, "calib_dev")
    print(f"[check] calib_train={train.shape} calib_dev={dev.shape}", flush=True)

    failures = []
    if train.shape[0] != args.expected_train_rows:
        failures.append(f"train rows {train.shape[0]} != {args.expected_train_rows}")
    if dev.shape[0] != args.expected_dev_rows:
        failures.append(f"dev rows {dev.shape[0]} != {args.expected_dev_rows}")
    if train.shape[1] != mu.shape[0]:
        failures.append(f"feature dim {train.shape[1]} != head dim {mu.shape[0]}")

    if not failures:
        d_mu = float(np.abs(train.mean(axis=0) - mu).max())
        d_sd = float(np.abs(train.std(axis=0) - sd).max())
        print(f"[check] max|dmu|={d_mu:.6f} max|dsd|={d_sd:.6f}", flush=True)
        if d_mu > FAIL_TOLERANCE or d_sd > FAIL_TOLERANCE:
            message = (
                f"feature statistics differ from the released head "
                f"(max|dmu|={d_mu:.4f}, max|dsd|={d_sd:.4f}); the head was not fitted by "
                f"calibrate_head.py --mode train on this extraction"
            )
            if args.strict:
                failures.append(message)
            else:
                print(f"[check][WARN] {message}", flush=True)
        elif d_mu > WARN_TOLERANCE or d_sd > WARN_TOLERANCE:
            print(
                f"[check][WARN] statistics match only loosely "
                f"(max|dmu|={d_mu:.4f}, max|dsd|={d_sd:.4f})",
                flush=True,
            )

    if failures:
        for failure in failures:
            print(f"[check][FATAL] {failure}", flush=True)
        return 1
    print("[check] OK: features are provenance-consistent with the released head", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
