#!/usr/bin/env python3
"""Select the validation-best checkpoint and persist training diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEV_LOSS_PATTERN = re.compile(r"^DEV_LOSS=([0-9.eE+-]+)$", re.MULTILINE)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint directory name: {path}") from exc


def read_dev_losses(
    log_dir: Path,
    checkpoints: list[Path],
    log_overrides: dict[int, Path] | None = None,
) -> dict[int, float]:
    losses: dict[int, float] = {}
    overrides = log_overrides or {}
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        log_path = overrides.get(step, log_dir / f"dev_loss_checkpoint-{step}.log")
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing validation log: {log_path}")
        matches = DEV_LOSS_PATTERN.findall(log_path.read_text(encoding="utf-8"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one DEV_LOSS in {log_path}, found {len(matches)}")
        loss = float(matches[0])
        if not math.isfinite(loss):
            raise ValueError(f"Non-finite DEV_LOSS in {log_path}: {loss}")
        losses[step] = loss
    return losses


def epoch_train_losses(trainer_state: dict[str, Any]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in trainer_state.get("log_history", []):
        if "loss" not in row or "epoch" not in row:
            continue
        epoch = max(1, math.ceil(float(row["epoch"])))
        grouped[epoch].append(float(row["loss"]))
    return {epoch: mean(values) for epoch, values in sorted(grouped.items()) if values}


def epoch_learning_rates(trainer_state: dict[str, Any]) -> dict[int, float]:
    rates: dict[int, float] = {}
    for row in trainer_state.get("log_history", []):
        if "learning_rate" not in row or "epoch" not in row:
            continue
        epoch = max(1, math.ceil(float(row["epoch"])))
        rates[epoch] = float(row["learning_rate"])
    return rates


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dev-log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-checkpoint", action="append", default=[], type=Path)
    parser.add_argument(
        "--extra-dev-log",
        action="append",
        default=[],
        metavar="STEP=PATH",
    )
    parser.add_argument(
        "--extra-checkpoint-task-id",
        action="append",
        default=[],
        metavar="STEP=TASK_ID",
    )
    args = parser.parse_args()

    checkpoints_by_step = {
        checkpoint_step(path): path
        for path in args.checkpoint_root.glob("checkpoint-*")
        if path.is_dir()
    }
    for path in args.extra_checkpoint:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing extra checkpoint: {path}")
        step = checkpoint_step(path)
        if step in checkpoints_by_step:
            raise ValueError(f"Duplicate checkpoint step: {step}")
        checkpoints_by_step[step] = path
    checkpoints = [checkpoints_by_step[step] for step in sorted(checkpoints_by_step)]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints below {args.checkpoint_root}")

    log_overrides: dict[int, Path] = {}
    for value in args.extra_dev_log:
        step_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Invalid --extra-dev-log value: {value}")
        log_overrides[int(step_text)] = Path(path_text)
    extra_task_ids: dict[str, int] = {}
    for value in args.extra_checkpoint_task_id:
        step_text, separator, task_text = value.partition("=")
        if not separator:
            raise ValueError(f"Invalid --extra-checkpoint-task-id value: {value}")
        extra_task_ids[step_text] = int(task_text)
    dev_losses = read_dev_losses(args.dev_log_dir, checkpoints, log_overrides)
    best_step = min(dev_losses, key=lambda step: (dev_losses[step], step))
    best_checkpoint = checkpoints_by_step[best_step]

    trainer_state_path = args.checkpoint_root / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    train_losses = epoch_train_losses(trainer_state)
    learning_rates = epoch_learning_rates(trainer_state)
    steps_per_epoch = int(trainer_state["max_steps"]) / int(trainer_state["num_train_epochs"])

    rows = []
    best_so_far = math.inf
    timestamp = datetime.now(timezone.utc).isoformat()
    for step, val_loss in sorted(dev_losses.items()):
        epoch = max(1, round(step / steps_per_epoch))
        best_so_far = min(best_so_far, val_loss)
        rows.append(
            {
                "epoch": epoch,
                "step": step,
                "train_loss": train_losses.get(epoch),
                "val_loss": val_loss,
                "lr": learning_rates.get(epoch),
                "timestamp": timestamp,
                "best_val_loss": best_so_far,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = args.output_dir / "metrics.jsonl"
    with metrics_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        epochs,
        [row["train_loss"] for row in rows],
        marker="o",
        label="train loss",
    )
    plt.plot(
        epochs,
        [row["val_loss"] for row in rows],
        marker="o",
        label="validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Grouping ablation training curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    curve_path = args.output_dir / "loss_curves.png"
    plt.savefig(curve_path, dpi=180)
    plt.close()

    val_series = [row["val_loss"] for row in rows]
    if len(val_series) >= 2 and val_series[-1] > min(val_series[:-1]):
        analysis = (
            "Validation loss reached its minimum before the final epoch; extending training "
            "is not supported and would risk overfitting."
        )
    elif len(val_series) >= 2 and val_series[-1] < val_series[-2]:
        analysis = (
            "Validation loss was still decreasing at the final epoch, but the fixed six-epoch "
            "recipe is retained for controlled comparison."
        )
    else:
        analysis = "Validation loss plateaued under the fixed six-epoch recipe."

    final_log = next(
        (row for row in reversed(trainer_state.get("log_history", [])) if "train_runtime" in row),
        {},
    )
    summary = {
        "best_checkpoint": str(best_checkpoint.resolve()),
        "best_step": best_step,
        "best_epoch": max(1, round(best_step / steps_per_epoch)),
        "best_val_loss": dev_losses[best_step],
        "final_train_loss": final_log.get("train_loss"),
        "train_runtime_seconds": final_log.get("train_runtime"),
        "loss_curve": str(curve_path.resolve()),
        "metrics_jsonl": str(metrics_jsonl.resolve()),
        "analysis": analysis,
        "selection_policy": "minimum teacher-forced LM loss on the fixed 100-record dev subset",
        "extra_checkpoint_task_ids": extra_task_ids,
    }
    write_json(args.output_dir / "final_metrics.json", summary)
    write_json(
        args.output_dir / "checkpoint_selection.json",
        {
            "best_checkpoint": str(best_checkpoint.resolve()),
            "best_step": best_step,
            "dev_losses": {str(step): loss for step, loss in sorted(dev_losses.items())},
            "extra_checkpoint_task_ids": extra_task_ids,
        },
    )
    (args.output_dir / "process.md").write_text(
        "\n".join(
            [
                "# Grouping ablation 训练记录",
                "",
                f"- validation-best checkpoint: `{best_checkpoint}`",
                f"- best epoch: {summary['best_epoch']}",
                f"- best validation loss: {summary['best_val_loss']:.6f}",
                f"- final training loss: {summary['final_train_loss']}",
                f"- training time: {summary['train_runtime_seconds']} seconds",
                f"- loss curve: `{curve_path}`",
                f"- 分析：{analysis}",
                "- 决策：保持与主方法完全相同的六 epoch recipe，不通过增加 epoch 优化结果。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"BEST_CHECKPOINT={best_checkpoint.resolve()}", flush=True)
    print(f"BEST_VAL_LOSS={dev_losses[best_step]:.8f}", flush=True)
    print(f"ANALYSIS={analysis}", flush=True)


if __name__ == "__main__":
    main()
