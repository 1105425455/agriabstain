"""导出已保存 gate 日志的评估点，不插值未记录 epoch。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "Grouped": "new_reason/outputs/logs/fusion_3reason_v3_hl04/calib_train_checkpoint-444.log",
    "Independent": "aaai_code/experiments/grouping_ablation/artifacts/logs/calibrate_train.log",
}


def main() -> None:
    rows = []
    provenance = {}
    pattern = (
        r"epoch (\d+)/1000, train_loss=([\d.]+), dev_loss=([\d.]+), dev_MCC=([\d.]+)"
    )
    for arm, relative in SOURCES.items():
        text = (ROOT / relative).read_text()
        provenance[arm] = {
            "source": relative,
            "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
            "recorded_setup_and_support": [
                line
                for line in text.splitlines()
                if any(
                    token in line
                    for token in (
                        "train:",
                        "dev:",
                        "MLP(",
                        "answer(0)",
                        "wrong_object(1)",
                        "no_target(2)",
                    )
                )
            ],
        }
        for match in re.finditer(pattern, (ROOT / relative).read_text()):
            epoch, train, dev, mcc = match.groups()
            rows.append(
                {
                    "arm": arm,
                    "epoch": int(epoch),
                    "train_loss": float(train),
                    "dev_loss": float(dev),
                    "dev_multiclass_MCC": float(mcc),
                    "source": relative,
                }
            )
    output = ROOT / "docs/review_evidence/gate_loss_curves.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} logged evaluation points -> {output.relative_to(ROOT)}")
    code_path = ROOT / "aaai_code/training/sft_final/calibrate_head.py"
    provenance["current_code_recipe"] = {
        "source": str(code_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        "excerpts": [
            f"{i}: {line}"
            for i, line in enumerate(code_path.read_text().splitlines(), 1)
            if any(
                token in line
                for token in (
                    "random.Random(42)",
                    "X_train.mean",
                    "X_train.std",
                    "torch.optim.Adam",
                    "CrossEntropyLoss",
                    "patience_limit =",
                    "eval_every =",
                    "manual_seed",
                )
            )
        ],
    }
    (output.parent / "gate_protocol_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
