from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = ("comprehensive", "refcocog", "plantwild")
METHODS = (
    "native_lm",
    "prompt_basic",
    "prompt_reward",
    "prompt_reward_norms",
    "cot_structured",
)
DATASET_LABELS = {
    "comprehensive": "Comprehensive",
    "refcocog": "RefCOCOg",
    "plantwild": "PlantWild",
}
METHOD_LABELS = {
    "native_lm": "Native LM",
    "prompt_basic": "Direct Prompt",
    "prompt_reward": "Reward Prompt",
    "prompt_reward_norms": "Reward + Norms",
    "cot_structured": "CoT",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _advantage_matrices(input_dir: Path):
    import numpy as np

    rows = _read_csv(input_dir / "matched_retention.csv")
    indexed = {(row["dataset"], row["baseline_method"]): row for row in rows}
    hallucination = np.zeros((len(DATASETS), len(METHODS)))
    risk = np.zeros_like(hallucination)
    for row_index, dataset in enumerate(DATASETS):
        for column_index, method in enumerate(METHODS):
            row = indexed[(dataset, method)]
            hallucination[row_index, column_index] = 100 * (
                float(row["baseline_unsupported_hallucination_rate"])
                - float(row["regate_empirical_unsupported_hallucination_rate"])
            )
            risk[row_index, column_index] = 100 * (
                float(row["baseline_selective_risk"])
                - float(row["regate_empirical_selective_risk"])
            )
    return hallucination, risk


def _draw_heatmap(axis, values, title: str, limit: float, *, show_method_labels: bool) -> None:
    from matplotlib.colors import TwoSlopeNorm

    image = axis.imshow(
        values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    axis.set_title(title, fontsize=14, pad=14, weight="semibold")
    axis.set_xticks(range(len(METHODS)))
    axis.set_xticklabels([METHOD_LABELS[name] for name in METHODS] if show_method_labels else [])
    axis.set_yticks(range(len(DATASETS)), [DATASET_LABELS[name] for name in DATASETS])
    axis.tick_params(axis="x", rotation=28)
    axis.tick_params(length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "white" if abs(value) > limit * 0.58 else "#151515"
            axis.text(
                column,
                row,
                f"{value:+.1f} pp",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                weight="semibold",
            )
    for boundary in range(values.shape[0] + 1):
        axis.axhline(boundary - 0.5, color="white", linewidth=2)
    for boundary in range(values.shape[1] + 1):
        axis.axvline(boundary - 0.5, color="white", linewidth=2)
    return image


def make_summary_figure(input_dir: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    hallucination, risk = _advantage_matrices(input_dir)
    limit = float(np.ceil(max(abs(hallucination).max(), abs(risk).max()) / 5) * 5)

    fig, axes = plt.subplots(2, 1, figsize=(12.6, 7.8))
    image = _draw_heatmap(
        axes[0],
        hallucination,
        "Unsupported Hallucination Reduction at Matched Retention",
        limit,
        show_method_labels=False,
    )
    _draw_heatmap(
        axes[1],
        risk,
        "Selective Risk Reduction at Matched Retention",
        limit,
        show_method_labels=True,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("ReGate advantage (percentage points)", rotation=90)
    fig.suptitle("Where Does ReGate Win?", fontsize=18, weight="semibold", y=0.985)
    fig.text(
        0.5,
        0.018,
        "Positive (green): ReGate is lower and better   |   Negative (red): baseline is lower",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )
    fig.subplots_adjust(left=0.13, right=0.91, top=0.91, bottom=0.16, hspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_summary_figure(args.input_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
