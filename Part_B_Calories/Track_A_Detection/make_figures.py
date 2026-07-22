"""
make_figures.py  --  CSC3014 Project Part 2, Track A (G17)

Builds the report figures from test_results.json. No training, no re-evaluation.
Reads the six cached runs and renders:

  fig1_overall.png    baseline vs CLAHE on the four headline metrics,
                      seed spread drawn as error bars
  fig2_per_class.png  per-class mAP50, baseline vs CLAHE, sorted by baseline

The error bars are the point of fig1: they show the reader that the precision
gap clears the seed noise while the mAP and recall gaps do not.

Usage (from C:\\Users\\PF4B3\\Desktop\\g17):
    python make_figures.py
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "test_results.json"

CONDITIONS = {
    "baseline": ["baseline3", "baseline_s1", "baseline_s2"],
    "clahe": ["clahe", "clahe_s1", "clahe_s2"],
}
METRICS = ["mAP50", "mAP50-95", "precision", "recall"]
METRIC_LABELS = ["mAP@50", "mAP@50-95", "Precision", "Recall"]

# muted, print-safe, colourblind-friendly. No gray colormap, no viridis needed here.
C_BASE = "#3B6EA5"
C_CLAHE = "#C1666B"


def load():
    if not CACHE.exists():
        sys.exit(f"missing {CACHE} -- run aggregate_runs.py first")
    return json.loads(CACHE.read_text())


def mean_range(values):
    return statistics.mean(values), (max(values) - min(values)) / 2.0


def figure_overall(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METRICS))
    width = 0.38

    for offset, (cond, runs) in zip((-width / 2, width / 2), CONDITIONS.items()):
        means, halves = [], []
        for metric in METRICS:
            vals = [results[r][metric] for r in runs if r in results]
            m, h = mean_range(vals)
            means.append(m)
            halves.append(h)
        colour = C_BASE if cond == "baseline" else C_CLAHE
        label = "Baseline" if cond == "baseline" else "CLAHE"
        bars = ax.bar(x + offset, means, width, yerr=halves, capsize=5,
                      color=colour, label=label, edgecolor="white", linewidth=0.5)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.012,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("Score (test set, mean of 3 seeds)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline vs CLAHE, error bars show seed spread (half-range)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = HERE / "fig1_overall.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def figure_per_class(results):
    classes = sorted(
        results["baseline3"]["per_class_map50"].keys(),
        key=lambda c: results["baseline3"]["per_class_map50"][c],
        reverse=True,
    )

    def cell(cond, cls):
        vals = [results[r]["per_class_map50"][cls]
                for r in CONDITIONS[cond]
                if r in results and cls in results[r]["per_class_map50"]]
        return mean_range(vals)

    base_m = [cell("baseline", c)[0] for c in classes]
    base_h = [cell("baseline", c)[1] for c in classes]
    clahe_m = [cell("clahe", c)[0] for c in classes]
    clahe_h = [cell("clahe", c)[1] for c in classes]

    y = np.arange(len(classes))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(y + height / 2, base_m, height, xerr=base_h, capsize=3,
            color=C_BASE, label="Baseline", edgecolor="white", linewidth=0.5)
    ax.barh(y - height / 2, clahe_m, height, xerr=clahe_h, capsize=3,
            color=C_CLAHE, label="CLAHE", edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel("mAP@50 (test set, mean of 3 seeds)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-class mAP@50, baseline vs CLAHE")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = HERE / "fig2_per_class.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    results = load()
    f1 = figure_overall(results)
    f2 = figure_per_class(results)
    print("written:")
    print(f"  {f1}")
    print(f"  {f2}")


if __name__ == "__main__":
    main()
