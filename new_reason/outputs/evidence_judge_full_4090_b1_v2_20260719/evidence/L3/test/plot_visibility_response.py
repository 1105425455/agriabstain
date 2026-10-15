#!/usr/bin/env python3
"""Plot checkpoint-444 answer probability across T7 visibility levels."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "head_probabilities.jsonl"
ANALYSIS = ROOT / "analysis.json"
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 10_000


def group_key(sample_id: str) -> str:
    """Recover the source-instance key used by the canonical L3 analysis."""
    rest = sample_id.removeprefix("L3-")
    match = re.search(r"-(?:0|1)\.\d{3}-", rest)
    if match is None:
        raise ValueError(f"Cannot parse L3 sample id: {sample_id}")
    image_prefix = rest[: match.start()]
    source = re.sub(r"-ladder-\d{3}$", "", rest[match.end() :])
    return f"{image_prefix}__{source}"


def load_rows() -> list[dict]:
    with INPUT.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cluster_bootstrap(
    groups: dict[str, dict[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap source groups, preserving within-group visibility dependence."""
    keys = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((N_BOOTSTRAP, len(LEVELS)), dtype=np.float64)

    for bootstrap_idx in range(N_BOOTSTRAP):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        for level_idx, level in enumerate(LEVELS):
            values = [groups[key][level] for key in sampled if level in groups[key]]
            draws[bootstrap_idx, level_idx] = np.mean(values)

    low = np.percentile(draws, 2.5, axis=0)
    high = np.percentile(draws, 97.5, axis=0)
    return low, high


def main() -> None:
    rows = load_rows()
    by_level: dict[float, list[float]] = defaultdict(list)
    groups: dict[str, dict[float, float]] = defaultdict(dict)

    for row in rows:
        level = float(row["visible_ratio_bin"])
        probability = float(row["p_answer"])
        by_level[level].append(probability)
        groups[group_key(row["sample_id"])][level] = probability

    means = np.asarray([np.mean(by_level[level]) for level in LEVELS])
    medians = np.asarray([np.median(by_level[level]) for level in LEVELS])
    counts = np.asarray([len(by_level[level]) for level in LEVELS])
    ci_low, ci_high = cluster_bootstrap(groups)

    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))["L3"]
    rho = analysis["mean_spearman_rho"]
    tau = analysis["mean_kendall_tau"]
    violation = analysis["mean_adjacent_violation_rate"]

    plt.rcParams.update(
        {
            # Render directly at AAAI single-column size. STIXGeneral is the
            # available Times-compatible face; 9.5 pt stays above the 9 pt
            # minimum without relying on post-hoc scaling.
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    # AAAI single-column width is 3.3 inches. Generate at final physical size
    # so LaTeX can include the PDF at \columnwidth without shrinking its text.
    fig, ax = plt.subplots(figsize=(3.3, 2.8))

    x = np.asarray(LEVELS) * 100
    ax.fill_between(
        x,
        ci_low,
        ci_high,
        color="#4C78A8",
        alpha=0.18,
        linewidth=0,
        label="95% bootstrap CI",
    )
    ax.plot(x, means, color="#4C78A8", linewidth=1.8, label=r"Mean $p_A$")
    ax.plot(
        x,
        medians,
        color="#6B6B6B",
        linewidth=1.2,
        linestyle="--",
        label=r"Median $p_A$",
    )
    ax.scatter(x[1:-1], means[1:-1], s=30, color="#4C78A8", zorder=3)
    ax.scatter(
        x[[0, -1]],
        means[[0, -1]],
        s=40,
        marker="s",
        color="#F28E2B",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="Endpoints",
    )

    ax.set_xlabel("Retained target visibility (%)")
    ax.xaxis.set_label_coords(0.5, 0.025)
    ax.xaxis.label.set_verticalalignment("bottom")
    ax.set_ylabel(r"Answer probability $p_A$")
    ax.set_xticks(
        x,
        [f"{int(level)}\n(n={count})" for level, count in zip(x, counts, strict=True)],
    )
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlim(-3, 103)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(1.0, 0.42),
        frameon=False,
        handlelength=2.2,
        labelspacing=0.25,
        borderaxespad=0.35,
    )
    fig.tight_layout(pad=0.4)

    png_path = ROOT / "checkpoint444_visibility_response.png"
    pdf_path = ROOT / "checkpoint444_visibility_response.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "input": str(INPUT),
        "checkpoint": 444,
        "n_records": len(rows),
        "n_source_instance_groups": len(groups),
        "bootstrap": {
            "unit": "source_instance_group",
            "n_resamples": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
        },
        "levels": [
            {
                "visibility": level,
                "n": int(counts[idx]),
                "mean_p_answer": float(means[idx]),
                "median_p_answer": float(medians[idx]),
                "mean_ci_95_low": float(ci_low[idx]),
                "mean_ci_95_high": float(ci_high[idx]),
            }
            for idx, level in enumerate(LEVELS)
        ],
        "mean_spearman_rho": rho,
        "mean_kendall_tau": tau,
        "mean_adjacent_violation_rate": violation,
    }
    (ROOT / "checkpoint444_visibility_response.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
