"""
classifier_diagnostics.py -- Track C (G17)

Per-class diagnostics for the baseline classifier: confusion matrix, Macro-F1,
and per-class precision/recall/F1.

WHY THIS SCRIPT EXISTS -- it closes two commitments our own literature review made
--------------------------------------------------------------------------------
Section 9.3 of the review ("Recommendations for Implementation") states:

    "For classification, we should report Top-1 accuracy, Macro-F1 (for rare
     dishes/Malaysian foods), and a confusion matrix [23]."

Before this script, Track C reported Top-1 only. Macro-F1 and the confusion
matrix were missing, so the implementation was not honouring its own review.

The confusion matrix is the more important of the two. Section 7.3 records:

    "although it is the obvious tool for showing which dishes a model mistakes
     for which, not one of the ten papers reports a full confusion matrix as a
     primary result, so the question of which foods get confused with which goes
     largely unanswered."

and Section 9.1 repeats it as a headline finding. Publishing one is therefore not
housekeeping -- it is a stated contribution of this project, and Track A already
supplies the detection-side matrix. This supplies the classification side.

MACRO vs MICRO -- the distinction the review asks us to make
-----------------------------------------------------------
Section 7.2 sets it out: macro-averaging weights every class equally regardless
of how many samples it has; micro-averaging lets frequent classes dominate.
Section 7.2 also warns, via Niu et al. [23], that "accuracy can be misleading on
imbalanced datasets, because a strong overall figure can conceal weak results on
infrequent classes."

Our test split is exactly that case -- Lo-Mein has 10 images, Sambal and
Boiled-Egg have 1 each. So:

    Micro-F1 == Top-1 accuracy      (for single-label multi-class, precision,
                                     recall and F1 all collapse to accuracy --
                                     every error is simultaneously one FP and
                                     one FN)
    Macro-F1  < Micro-F1            whenever rare classes do worse, which is the
                                     effect the review predicted

Reporting both, and the gap between them, is the honest answer to "does this
model actually handle Malaysian rare dishes, or just the common noodles?"

Usage:
    python src/classifier_diagnostics.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def load_predictions():
    if not cfg.CLASSIFIER_PREDS.exists():
        raise SystemExit(
            f"missing {cfg.CLASSIFIER_PREDS}\n"
            f"  run:  python src/predict_classifier.py"
        )
    with open(cfg.CLASSIFIER_PREDS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def confusion(rows):
    """-> (17x17 int array, index[true][pred]).

    Built over ALL 17 classes, not just those present, so the matrix shape is
    stable across splits and directly comparable with Track A's.
    """
    idx = {name: i for i, name in enumerate(cfg.CLASS_NAMES)}
    m = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=int)
    for r in rows:
        m[idx[r["true_label"]], idx[r["pred_label"]]] += 1
    return m


def per_class(m):
    """Precision / recall / F1 / support for every class, from the matrix."""
    out = []
    for i, name in enumerate(cfg.CLASS_NAMES):
        tp = int(m[i, i])
        fp = int(m[:, i].sum() - tp)
        fn = int(m[i, :].sum() - tp)
        support = int(m[i, :].sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out.append({"class": name, "support": support, "tp": tp, "fp": fp,
                    "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f1, 4)})
    return out


def averages(stats, n_images):
    """Macro / weighted / micro.

    Macro is averaged over classes with support > 0 ONLY. Including absent
    classes would score them 0 and drag the average down for a reason that has
    nothing to do with the model -- Chicken-Rendang and Fried-Egg simply never
    appear as a top-1 ground-truth label in this split.
    """
    present = [s for s in stats if s["support"] > 0]
    macro_f1 = sum(s["f1"] for s in present) / len(present)
    macro_p = sum(s["precision"] for s in present) / len(present)
    macro_r = sum(s["recall"] for s in present) / len(present)
    weighted_f1 = sum(s["f1"] * s["support"] for s in present) / n_images
    tp = sum(s["tp"] for s in stats)
    micro_f1 = tp / n_images          # == accuracy for single-label multi-class
    return {
        "n_images": n_images,
        "classes_present": len(present),
        "classes_total": cfg.NUM_CLASSES,
        "micro_f1_eq_top1_accuracy": round(micro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_micro_gap": round(micro_f1 - macro_f1, 4),
    }


def top_confusions(m, k=6):
    """Largest off-diagonal cells -- 'which dish is mistaken for which'."""
    pairs = []
    for i in range(cfg.NUM_CLASSES):
        for j in range(cfg.NUM_CLASSES):
            if i != j and m[i, j] > 0:
                pairs.append((int(m[i, j]), cfg.CLASS_NAMES[i], cfg.CLASS_NAMES[j]))
    pairs.sort(reverse=True)
    return pairs[:k]


def figure_confusion(m, out_path):
    """Row-normalised confusion matrix, styled to match Track A's figure."""
    support = m.sum(axis=1, keepdims=True)
    keep = support.ravel() > 0                      # drop rows with no test images
    names = [n for n, k in zip(cfg.CLASS_NAMES, keep) if k]
    norm = np.divide(m, np.maximum(support, 1))[keep][:, keep]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True (largest annotated box)", fontsize=10)
    ax.set_title("Track C baseline classifier: confusion matrix (row-normalised)\n"
                 "EfficientNetB0, 76 test images. Rows with no test images omitted.",
                 fontsize=11, loc="left")
    for i in range(len(names)):
        for j in range(len(names)):
            v = norm[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "#333333")
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of that class's images")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    rows = load_predictions()
    m = confusion(rows)
    stats = per_class(m)
    avg = averages(stats, len(rows))
    top3 = sum(r["true_label"] in r["top3"].split("|") for r in rows) / len(rows)

    print("=" * 78)
    print("TRACK C CLASSIFIER DIAGNOSTICS")
    print("=" * 78)
    print(f"  images                    {avg['n_images']}")
    print(f"  classes present in test   {avg['classes_present']} of {avg['classes_total']}")
    print(f"  Top-1 accuracy (= Micro-F1) {avg['micro_f1_eq_top1_accuracy']:.4f}")
    print(f"  Top-3 accuracy            {top3:.4f}")
    print(f"  Macro-F1                  {avg['macro_f1']:.4f}")
    print(f"  Weighted-F1               {avg['weighted_f1']:.4f}")
    print(f"  Macro-Micro gap           {avg['macro_micro_gap']:.4f}")
    print()
    print("  For single-label multi-class, micro-P = micro-R = micro-F1 = accuracy:")
    print("  every mistake is simultaneously one false positive and one false")
    print("  negative, so they cannot diverge. Macro-F1 is the informative one.")
    print()

    print("=" * 78)
    print("PER-CLASS  (sorted by support, rarest last -- the review's concern)")
    print("=" * 78)
    print(f"{'class':<20}{'n':>4}{'prec':>8}{'recall':>8}{'F1':>8}")
    for s in sorted(stats, key=lambda s: -s["support"]):
        if s["support"] == 0:
            continue
        print(f"{s['class']:<20}{s['support']:>4}{s['precision']:>8.3f}"
              f"{s['recall']:>8.3f}{s['f1']:>8.3f}")
    absent = [s["class"] for s in stats if s["support"] == 0]
    if absent:
        print(f"\n  never a ground-truth label in this split: {', '.join(absent)}")

    print()
    print("=" * 78)
    print("MOST CONFUSED PAIRS  (the question our review says nobody answered)")
    print("=" * 78)
    for n, true_c, pred_c in top_confusions(m):
        print(f"  {n:>2}x   {true_c:<18} -> predicted {pred_c}")

    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with open(cfg.OUTPUTS / "classifier_per_class.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader(); w.writerows(stats)
    with open(cfg.OUTPUTS / "classifier_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(avg.keys()))
        w.writeheader(); w.writerow(avg)
    np.savetxt(cfg.OUTPUTS / "classifier_confusion_matrix.csv", m, fmt="%d",
               delimiter=",", header=",".join(cfg.CLASS_NAMES), comments="")

    out_fig = cfg.FIGURE_DIR / "fig_c3_confusion_matrix.png"
    figure_confusion(m, out_fig)

    print()
    print(f"wrote {cfg.OUTPUTS / 'classifier_per_class.csv'}")
    print(f"wrote {cfg.OUTPUTS / 'classifier_summary.csv'}")
    print(f"wrote {cfg.OUTPUTS / 'classifier_confusion_matrix.csv'}")
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
