#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reason_abstain.bench.build_coco import build_coco_ground_abstain
from reason_abstain.bench.build_plantdoc import build_plantdoc_vg
from reason_abstain.bench.io import read_yaml, write_json, write_jsonl, write_yaml
from reason_abstain.bench.quality import summarize_records


def _default_sidecar(output_path: str, suffix: str) -> str:
    path = Path(output_path)
    return str(path.with_suffix("").with_name(path.with_suffix("").name + suffix))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coco", "plantdoc"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--config-snapshot-output")
    args = parser.parse_args()

    config = read_yaml(args.config)
    if args.dataset == "coco":
        samples = build_coco_ground_abstain(config)
    else:
        samples = build_plantdoc_vg(config)

    records = [sample.to_record() for sample in samples]
    write_jsonl(args.output, records)
    summary_output = args.summary_output or _default_sidecar(args.output, ".summary.json")
    config_snapshot_output = args.config_snapshot_output or _default_sidecar(
        args.output,
        ".config.snapshot.yaml",
    )
    write_json(summary_output, summarize_records(records))
    write_yaml(config_snapshot_output, config)
    print(f"Wrote {len(samples)} samples to {args.output}", flush=True)
    print(f"Wrote summary to {summary_output}", flush=True)
    print(f"Wrote config snapshot to {config_snapshot_output}", flush=True)


if __name__ == "__main__":
    main()
