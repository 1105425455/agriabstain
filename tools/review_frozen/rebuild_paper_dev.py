#!/usr/bin/env python3
"""Rebuild the paper's 100-record stratified dev subset and test identity-matchability
against the frozen cache's dev split.

The paper's Stage-2 dev subset (see source
`new_reason/scripts/calibrate_head_3reason.py`, replicated identically here) is:
  * group the 2408 dev.jsonl rows by head class (0=answer, 1=wrong_object, 2=no_target);
  * per_class = 100 // 3 = 33 rows from each class after an RNG(seed=42) shuffle;
  * fill the remaining 1 row from the shuffled leftover; then shuffle the 100.

This script writes the 100 sample_ids to --output and then reports whether those ids (or
the derived image identity) exist anywhere in the frozen cache's dev npz, which carries no
semantic sample ids (only sft-json row indices).

Findings are written to --report as JSON so the RESULTS.md caveat is machine checkable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def build_paper_100_dev(dev_path: Path) -> tuple[list[dict], Counter]:
    """Exactly reproduce the stratified sampling in calibrate_head_3reason.py."""
    dev_rows = [json.loads(line) for line in dev_path.open(encoding="utf-8") if line.strip()]
    rng = random.Random(42)
    dev_by_class = {0: [], 1: [], 2: []}
    for r in dev_rows:
        decision = r.get("decision")
        reason = r.get("reason")
        if decision == "answer":
            cls = 0
        elif decision == "abstain" and reason == "wrong_object":
            cls = 1
        elif decision == "abstain" and reason == "no_target":
            cls = 2
        else:
            continue
        dev_by_class[cls].append(r)
    dev_max = 100
    per_class = dev_max // 3
    picked = []
    for cls in [0, 1, 2]:
        rng.shuffle(dev_by_class[cls])
        picked.extend(dev_by_class[cls][:per_class])
    if len(picked) < dev_max:
        remaining = []
        for cls in [0, 1, 2]:
            remaining.extend(dev_by_class[cls][per_class:])
        rng.shuffle(remaining)
        picked.extend(remaining[: dev_max - len(picked)])
    rng.shuffle(picked)
    counts = Counter(
        "answer"
        if r.get("decision") == "answer"
        else ("wrong_object" if r.get("reason") == "wrong_object" else "no_target")
        for r in picked
    )
    return picked, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-jsonl", required=True, type=Path)
    ap.add_argument("--repr-dev-npz", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()

    picked, counts = build_paper_100_dev(args.dev_jsonl)
    sids = [r["sample_id"] for r in picked]
    labels = [
        0 if r["decision"] == "answer" else (1 if r["reason"] == "wrong_object" else 2)
        for r in picked
    ]

    pay = np.load(args.repr_dev_npz, allow_pickle=True)
    # The frozen dev npz carries only sft-json row indices as sample_id.
    npz_sids = set(str(s) for s in pay["sample_id"])
    # no real sample ids: check whether any npz sample_id string coincides (should be trivial)
    overlap = sorted(set(sids) & npz_sids)

    # Try image-base matching: paper dev jsonl image_path basenames vs npz images
    import os
    paper_img_bases = {os.path.basename(r["image_path"]) for r in picked}
    npz_imgs = {os.path.basename(p) for p in pay["image"]}
    img_overlap = sorted(paper_img_bases & npz_imgs)

    report = {
        "n_picked": len(picked),
        "class_counts": dict(counts),
        "stratification_spec": "rng=random.Random(42); per_class=33 from classes [0,1,2]; "
                               "+1 leftover; final shuffle",
        "paper_dev_source": str(args.dev_jsonl.resolve()),
        "frozen_dev_manifests_data": "sft_dev.json (see npz manifest_json); npz sample_id = "
                                     "sft-json row index only (no semantic ids)",
        "npz_sampleid_vs_paper_sid_overlap": len(overlap),
        "image_basename_overlap": len(img_overlap),
        "matchable_in_frozen_cache": False,
        "reason": "frozen dev npz carries only row-index sample_ids and its derived-image "
                  "identity does not map to dev.jsonl (different derived-image build, e.g. "
                  "'region-ladder-100' vs 'ladder-100'); the paper 100-record subset cannot "
                  "be reconstructed inside the frozen cache.",
    }
    (args.report.parent).mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as handle:
        for sid, lab in zip(sids, labels):
            handle.write(f"{sid}\t{lab}\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()