"""
make_figures.py -- Track C (G17)

Builds the report figures from outputs/comparison_table.csv. No models loaded,
no re-evaluation -- run compare.py first.

    fig_c1_calorie_error.png   calorie MAE, classifier vs detector, split by
                               single-item / multi-item. THE headline figure.
    fig_c2_multilabel.png      multi-label precision/recall/F1, same split.

Colours are lifted from Track A's make_figures.py so every figure in the report
belongs to the same set. That palette was checked for colourblind safety --
worst adjacent pair dE 10.8 under protanopia, 21.4 at normal vision, both above
the usable floor.

Design choices worth knowing:
  - grouped bars, not stacked: the two models are compared against each other,
    not summed into a whole
  - one y-axis only, never two: the classic way to make a chart lie
  - direct value labels on every bar and a legend, so identity never depends on
    colour alone
  - recessive grid, no chartjunk, no gradients

Usage:
    python src/make_figures.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display needed; write straight to file
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

SUBSET_LABELS = {"all": "All images", "single": "Single-item", "multi": "Multi-item"}


def load_rows():
    if not cfg.COMPARISON_CSV.exists():
        raise SystemExit(
            f"missing {cfg.COMPARISON_CSV}\n"
            f"  run:  python src/compare.py"
        )
    with open(cfg.COMPARISON_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_simulated(rows):
    """True if compare.py produced these numbers with --simulate-classifier."""
    return any(r.get("simulated") == "1" for r in rows)


def stamp_proxy(ax):
    """Make a proxy figure impossible to mistake for a real one.

    Two independent markers, because a watermark alone can wash out when the
    figure is scaled down into a report or a slide -- and the whole purpose of
    the stamp is that it survives exactly that journey:
      1. a diagonal watermark across the plot area
      2. a literal prefix on the title, which cannot fade
    """
    ax.text(0.5, 0.5, "SIMULATED PROXY", transform=ax.transAxes,
            ha="center", va="center", fontsize=34, fontweight="bold",
            color=cfg.C_CLASSIFY, alpha=0.22, rotation=20, zorder=10)
    # get_title() defaults to the CENTRE title, which is empty here -- both
    # figures set theirs with loc="left". Asking for the wrong one silently
    # returns "" and erases the real title.
    ax.set_title("[ SIMULATED PROXY - NOT FOR THE REPORT ]\n"
                 + ax.get_title(loc="left"),
                 fontsize=11, loc="left", color=cfg.C_CLASSIFY,
                 fontweight="bold")


def pick(rows, metric):
    """-> (subset_labels, classifier_values, detector_values, n_per_subset)."""
    keep = [r for r in rows if r["metric"] == metric]
    order = [s for s in ("all", "single", "multi")
             if any(r["subset"] == s for r in keep)]
    by_subset = {r["subset"]: r for r in keep}
    return (order,
            [float(by_subset[s]["classifier"]) for s in order],
            [float(by_subset[s]["detector"]) for s in order],
            [int(by_subset[s]["n"]) for s in order])


def grouped_bars(ax, subsets, clf_values, det_values, counts, value_fmt):
    x = np.arange(len(subsets))
    width = 0.38

    bars_clf = ax.bar(x - width / 2, clf_values, width,
                      label="Classifier (EfficientNetB0)", color=cfg.C_CLASSIFY,
                      edgecolor="white", linewidth=2)
    bars_det = ax.bar(x + width / 2, det_values, width,
                      label="Detector (YOLOv8n)", color=cfg.C_DETECT,
                      edgecolor="white", linewidth=2)

    # Direct labels: identity and magnitude readable without the legend.
    for bars in (bars_clf, bars_det):
        for bar in bars:
            ax.annotate(value_fmt.format(bar.get_height()),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{SUBSET_LABELS.get(s, s)}\n(n={n})"
                        for s, n in zip(subsets, counts)])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors="#555555")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)


def figure_calorie_error(rows):
    subsets, clf, det, counts = pick(rows, "calorie_mae")
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_bars(ax, subsets, clf, det, counts, "{:.0f}")
    ax.set_ylabel("Calorie MAE (kcal) — lower is better")
    ax.set_title("Downstream calorie error: single-label classifier vs detection\n"
                 "Both scored through Track B's calorie table, same ground truth",
                 fontsize=11, loc="left")
    ax.set_ylim(0, max(clf + det) * 1.18)
    if is_simulated(rows):
        stamp_proxy(ax)
    fig.tight_layout()
    out = cfg.FIGURE_DIR / "fig_c1_calorie_error.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def figure_multilabel(rows):
    if not any(r["metric"] == "multilabel_f1" for r in rows):
        return None
    subsets, clf, det, counts = pick(rows, "multilabel_f1")
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_bars(ax, subsets, clf, det, counts, "{:.3f}")
    ax.set_ylabel("Micro-averaged F1 — higher is better")
    ax.set_title("Image-level multi-label F1: which foods are on the plate\n"
                 "The classifier emits one class by construction, so recall is capped",
                 fontsize=11, loc="left")
    ax.set_ylim(0, 1.0)
    if is_simulated(rows):
        stamp_proxy(ax)
    fig.tight_layout()
    out = cfg.FIGURE_DIR / "fig_c2_multilabel.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    rows = load_rows()
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for out in (figure_calorie_error(rows), figure_multilabel(rows)):
        if out:
            print(f"wrote {out}")
        else:
            print("skipped multi-label figure -- run derive_labels.py + compare.py "
                  "with the dataset present")


if __name__ == "__main__":
    main()
