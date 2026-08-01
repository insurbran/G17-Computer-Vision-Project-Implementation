"""
compare.py -- Track C (G17)

THE deliverable. Scores the single-label classifier against the detection
pipeline on three axes, and writes the comparison table the report is built
around.

WHY THREE METRICS AND NOT ONE
-----------------------------
You cannot compare "classifier accuracy" against "detector mAP". They measure
different things, so printing them side by side is two unrelated numbers
sharing a table, not a comparison. A marker will say so.

So we score both models only on axes they can BOTH occupy:

  M1  Top-1 dish accuracy         -- the classifier's home turf. Split by
                                     single-item vs multi-item images. This is
                                     the fair-fight control: if the classifier
                                     is not competitive on single-item plates,
                                     something is wrong with our training and
                                     the whole comparison is void.

  M2  Image-level multi-label F1  -- treat each image's output as a SET of
                                     classes. The detector emits a set; the
                                     classifier emits a set of size 1, always.
                                     Recall collapses on multi-item plates by
                                     construction. This is the structural point,
                                     measured.

  M3  Downstream calorie MAE      -- run each model's output through Track B's
                                     calorie table. Same units, same ground
                                     truth, directly comparable to Brandon's
                                     110.7 kcal. THIS IS THE HEADLINE: it is the
                                     only place the project produces a positive,
                                     quantified finding rather than another
                                     negative result.

GROUND TRUTH SOURCES -- keep these straight
-------------------------------------------
  M1, M2 -> the dataset's own test label .txt files (via derive_labels.py).
            Human-drawn boxes. Never model output.
  M3     -> Brandon's data/ground_truth.csv `expected_kcal` column.

            Do NOT use that file's `items_visible` column for M1 or M2. It is a
            verbatim copy of the detector's predictions (verified: 76/76 rows
            match exactly at conf 0.4). Only `expected_kcal` was hand-corrected.
            Scoring the detector against its own output would give it a free
            100%.

Usage:
    python src/compare.py
    python src/compare.py --simulate-classifier    (no trained model needed)

--simulate-classifier substitutes the largest-area DETECTION as a stand-in for
what a strong classifier would say. Use it to sanity-check the harness before
you have a trained model. It is an upper bound, not a result -- a real
classifier will do worse, because the proxy has already seen the detector's
localisation. Never put proxy numbers in the report.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


# --- loading ---------------------------------------------------------------

def load_calorie_table():
    cfg.require(cfg.CALORIE_TABLE, "Track B's calorie table",
                "It ships in the repo at Track_B_Calories/data/calorie_table.csv")
    with open(cfg.CALORIE_TABLE, newline="", encoding="utf-8") as f:
        return {r["class_name"]: float(r["kcal_per_portion"]) for r in csv.DictReader(f)}


def load_expected_kcal():
    cfg.require(cfg.GROUND_TRUTH, "Track B's ground truth",
                "It ships in the repo at Track_B_Calories/data/ground_truth.csv")
    out = {}
    with open(cfg.GROUND_TRUTH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["file"]] = float(row["expected_kcal"])
            except (TypeError, ValueError):
                continue
    return out


def load_detections(conf: float):
    """image_id -> [ {class, conf, area}, ... ] above the confidence floor."""
    cfg.require(cfg.PREDICTIONS_JSON, "Track A's predictions.json",
                "It ships in the repo at Track_B_Calories/from_A/predictions.json")
    data = json.loads(cfg.PREDICTIONS_JSON.read_text(encoding="utf-8"))
    out = {}
    for entry in data["predictions"]:
        boxes = []
        for b in entry["boxes"]:
            if b["conf"] < conf:
                continue
            x1, y1, x2, y2 = b["xyxy"]
            boxes.append({"class": b["class"], "conf": b["conf"],
                          "area": max(0.0, x2 - x1) * max(0.0, y2 - y1)})
        out[entry["image_id"]] = boxes
    return out


def load_test_manifest():
    """Ground-truth image label + the full class set + the multi-item flag.

    Returns None if derive_labels.py has not been run -- the calorie metric can
    still be computed without it, so we degrade instead of dying.
    """
    path = cfg.MANIFEST_DIR / "test.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return {r["file"]: {
            "label": r["label"],
            "classes": set(r["all_classes"].split("|")) if r["all_classes"] else set(),
            "is_multi": bool(int(r["is_multi_item"])),
        } for r in csv.DictReader(f)}


def load_classifier_preds(simulate: bool, detections):
    """-> file -> predicted class name."""
    if simulate:
        preds = {}
        for filename, boxes in detections.items():
            if boxes:
                preds[filename] = max(boxes, key=lambda b: b["area"])["class"]
        return preds, True
    if not cfg.CLASSIFIER_PREDS.exists():
        raise SystemExit(
            f"missing {cfg.CLASSIFIER_PREDS}\n"
            f"  run:  python src/predict_classifier.py\n"
            f"  or:   python src/compare.py --simulate-classifier"
        )
    with open(cfg.CLASSIFIER_PREDS, newline="", encoding="utf-8") as f:
        return {r["file"]: r["pred_label"] for r in csv.DictReader(f)}, False


# --- metrics ---------------------------------------------------------------

def detector_top1(boxes):
    """Largest-area detection.

    Matched deliberately to derive_labels.py's largest-area rule so the
    detector and the ground-truth label are produced by the same convention.
    Scoring the detector by highest-CONFIDENCE instead would be comparing it
    against a target defined a different way.
    """
    return max(boxes, key=lambda b: b["area"])["class"] if boxes else None


def accuracy(pairs):
    """pairs = [(predicted, truth), ...] -> accuracy, ignoring None predictions
    only by counting them as wrong (a no-answer is not a free pass)."""
    if not pairs:
        return float("nan"), 0
    correct = sum(1 for pred, truth in pairs if pred is not None and pred == truth)
    return correct / len(pairs), len(pairs)


def micro_set_f1(pairs):
    """pairs = [(predicted_set, truth_set), ...] -> (precision, recall, f1, n).

    Micro-averaged: pool TP/FP/FN across all images, then compute once. Chosen
    over macro because macro would let a single image with one easy class carry
    the same weight as a seven-item nasi lemak plate, which is exactly the case
    we care about.
    """
    tp = fp = fn = 0
    for pred, truth in pairs:
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, len(pairs)


def mae(pairs):
    """pairs = [(estimated, expected), ...] -> (MAE, MAPE, bias, n)."""
    if not pairs:
        return float("nan"), float("nan"), float("nan"), 0
    errors = [est - exp for est, exp in pairs]
    abs_errors = [abs(e) for e in errors]
    mape = sum(abs(e) / max(1.0, exp) for e, (_, exp) in zip(errors, pairs)) / len(pairs)
    return (sum(abs_errors) / len(pairs), 100 * mape,
            sum(errors) / len(pairs), len(pairs))


def fmt(value, nd=3):
    return "n/a" if value != value else f"{value:.{nd}f}"   # NaN check


# --- report ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=cfg.CONF_THRESHOLD,
                    help="detection confidence floor (default matches Track B)")
    ap.add_argument("--simulate-classifier", action="store_true")
    args = ap.parse_args()

    kcal_table = load_calorie_table()
    expected = load_expected_kcal()
    detections = load_detections(args.conf)
    manifest = load_test_manifest()
    clf_preds, simulated = load_classifier_preds(args.simulate_classifier, detections)

    if simulated:
        print("\n*** SIMULATED CLASSIFIER (largest-area detection as proxy) ***")
        print("*** Upper bound only. Do NOT put these numbers in the report. ***")
    if manifest is None:
        print("\nNOTE: outputs/manifests/test.csv not found, so M1 and M2 are skipped.")
        print("      Run `python src/derive_labels.py` once you have the dataset.")
        print("      M3 (calories) does not need it and runs below.\n")

    # Two file lists, on purpose.
    #
    # files_cal -- images with both a detection entry and a ground-truth kcal.
    #   M3 scores over ALL of these, which is what makes its detector column
    #   reproduce Track B's 110.7 exactly. Narrowing it would silently shift the
    #   number and destroy the cross-check against Brandon's pipeline.
    #
    # files_lbl -- the subset that ALSO has a ground-truth label row. An image
    #   whose label file is empty (no annotated boxes) has no true top-1 class
    #   and no true class set, so it cannot be scored on M1 or M2. It is still
    #   perfectly scorable on M3, because a calorie ground truth exists for it.
    files_cal = sorted(set(detections) & set(expected))
    if not files_cal:
        raise SystemExit("no overlap between predictions.json and ground_truth.csv")
    files_lbl = [f for f in files_cal if f in manifest] if manifest else []

    if manifest is not None and len(files_lbl) < len(files_cal):
        dropped = len(files_cal) - len(files_lbl)
        print(f"NOTE: {dropped} image(s) have detections and a kcal ground truth "
              f"but no annotated boxes.\n"
              f"      Scored on M3, excluded from M1/M2. Watch the n columns.\n")

    def subset(name, base):
        """Split a file list into all / single-item / multi-item."""
        if name == "all":
            return base
        if manifest is None:
            return []
        want_multi = (name == "multi")
        return [f for f in base
                if f in manifest and manifest[f]["is_multi"] == want_multi]

    rows = []

    # ---- M1: top-1 dish accuracy ------------------------------------------
    if manifest is not None:
        print("=" * 74)
        print("M1  TOP-1 DISH ACCURACY  (ground truth = largest annotated box)")
        print("=" * 74)
        print(f"{'subset':<14}{'n':>5}{'classifier':>14}{'detector':>12}")
        for name in ("all", "single", "multi"):
            group = subset(name, files_lbl)
            if not group:
                continue
            clf_acc, n = accuracy([(clf_preds.get(f), manifest[f]["label"]) for f in group])
            det_acc, _ = accuracy([(detector_top1(detections[f]), manifest[f]["label"])
                                   for f in group])
            print(f"{name:<14}{n:>5}{fmt(clf_acc):>14}{fmt(det_acc):>12}")
            rows.append({"metric": "top1_accuracy", "subset": name, "n": n,
                         "classifier": round(clf_acc, 4), "detector": round(det_acc, 4)})
        print("\n  Read this as the CONTROL. On single-item plates the classifier")
        print("  should be roughly competitive. If it is not, the training is at")
        print("  fault and nothing below is trustworthy.")
        if simulated:
            print("\n  The two columns are IDENTICAL here and that is not a bug: the")
            print("  proxy classifier IS the largest-area detection, which is exactly")
            print("  what the detector's top-1 is. M1 only becomes informative once a")
            print("  real classifier has been trained.")
        print()

    # ---- M2: multi-label set F1 -------------------------------------------
    if manifest is not None:
        print("=" * 74)
        print("M2  IMAGE-LEVEL MULTI-LABEL F1  (set of classes present)")
        print("=" * 74)
        print(f"{'subset':<10}{'n':>4}{'clf P':>8}{'clf R':>8}{'clf F1':>9}"
              f"{'det P':>8}{'det R':>8}{'det F1':>9}")
        for name in ("all", "single", "multi"):
            group = subset(name, files_lbl)
            if not group:
                continue
            clf_p, clf_r, clf_f1, n = micro_set_f1(
                [({clf_preds[f]} if f in clf_preds else set(), manifest[f]["classes"])
                 for f in group])
            det_p, det_r, det_f1, _ = micro_set_f1(
                [({b["class"] for b in detections[f]}, manifest[f]["classes"])
                 for f in group])
            print(f"{name:<10}{n:>4}{fmt(clf_p):>8}{fmt(clf_r):>8}{fmt(clf_f1):>9}"
                  f"{fmt(det_p):>8}{fmt(det_r):>8}{fmt(det_f1):>9}")
            rows.append({"metric": "multilabel_f1", "subset": name, "n": n,
                         "classifier": round(clf_f1, 4), "detector": round(det_f1, 4)})
        print("\n  The classifier's RECALL is the structural point: its output set")
        print("  has size 1 by construction, so on a 7-item plate it can recover")
        print("  at most 1/7 of what is there no matter how accurate it is.\n")

    # ---- M3: downstream calorie error -------------------------------------
    print("=" * 74)
    print("M3  DOWNSTREAM CALORIE ERROR  (both models -> Track B's calorie table)")
    print("=" * 74)
    print(f"{'subset':<10}{'n':>4}{'clf MAE':>10}{'clf MAPE':>10}"
          f"{'det MAE':>10}{'det MAPE':>10}")
    for name in ("all", "single", "multi"):
        group = subset(name, files_cal)
        if not group:
            continue
        clf_pairs, det_pairs = [], []
        for f in group:
            exp = expected[f]
            pred = clf_preds.get(f)
            clf_pairs.append((kcal_table.get(pred, 0.0), exp))
            det_pairs.append((sum(kcal_table.get(b["class"], 0.0)
                                  for b in detections[f]), exp))
        clf_mae, clf_mape, clf_bias, n = mae(clf_pairs)
        det_mae, det_mape, det_bias, _ = mae(det_pairs)
        print(f"{name:<10}{n:>4}{fmt(clf_mae, 1):>10}{fmt(clf_mape, 1):>10}"
              f"{fmt(det_mae, 1):>10}{fmt(det_mape, 1):>10}")
        rows.append({"metric": "calorie_mae", "subset": name, "n": n,
                     "classifier": round(clf_mae, 1), "detector": round(det_mae, 1)})
        rows.append({"metric": "calorie_mape", "subset": name, "n": n,
                     "classifier": round(clf_mape, 1), "detector": round(det_mape, 1)})

    print("\n  Detector 'all' MAE should land on ~110.7 -- Track B's box-method")
    print("  baseline. If it does, this harness agrees with Brandon's pipeline")
    print("  and the classifier column is trustworthy. If it does not, stop and")
    print("  reconcile before writing anything.\n")

    # ---- write out ---------------------------------------------------------
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Carry the proxy flag into the CSV, not just the console banner and the .md.
    # make_figures.py reads this CSV, so without a flag here a proxy run produces
    # a figure that is visually indistinguishable from a real one -- exactly the
    # way placeholder numbers end up in a submitted report.
    for r in rows:
        r["simulated"] = int(simulated)

    with open(cfg.COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "subset", "n",
                                               "classifier", "detector",
                                               "simulated"])
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Track C comparison table", ""]
    if simulated:
        lines += ["> SIMULATED classifier (proxy). Not for the report.", ""]
    lines += ["| Metric | Subset | n | Classifier (EfficientNetB0) | Detector (YOLOv8n) |",
              "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['metric']} | {r['subset']} | {r['n']} "
                     f"| {r['classifier']} | {r['detector']} |")
    cfg.COMPARISON_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {cfg.COMPARISON_CSV}")
    print(f"wrote {cfg.COMPARISON_MD}")


if __name__ == "__main__":
    main()
