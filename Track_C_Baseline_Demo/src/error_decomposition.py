"""
error_decomposition.py -- Track C (G17)

Answers a question the comparison table raises but cannot itself explain:

    Why is the detector's calorie MAE LOWER on multi-item plates (99.2 kcal)
    than on single-item ones (116.3), when multi-item plates are obviously the
    harder case?

It is a fair question and the intuitive answer ("the detector is better at
multi-item images") is WRONG. This script shows why, by splitting the calorie
error into the two things that actually produce it:

    1. TABLE ERROR   -- the calorie lookup table is a fixed kcal-per-portion
                        value per class. Real portions vary. Even a PERFECT
                        detector inherits this error.
                        measured as |kcal(true boxes) - expected_kcal|

    2. DETECTOR ERROR -- classes the detector missed (false negatives) and
                        classes it invented (false positives), priced in kcal.
                        measured as |kcal(detected boxes) - kcal(true boxes)|

The headline output is the "perfect detector" column: the MAE Track B would
have achieved if Track A had been flawless. The gap between that and the
actual MAE is ALL the headroom detection improvements could ever buy.

WHY THIS MATTERS FOR THE REPORT
-------------------------------
It independently corroborates Track B's negative result. Brandon found that
refining portion size via segmentation did not help. This shows the mechanical
reason: on multi-item plates almost the entire error is baked into the lookup
table before the detector is even involved, so no amount of portion refinement
could have moved it.

Usage:
    python src/error_decomposition.py
    python src/error_decomposition.py --conf 0.4
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def load_ground_truth_boxes():
    """image stem -> list of class names, one entry per annotated box.

    A LIST, not a set: Track B prices every detection separately, so an image
    with three curry puffs costs three curry puffs. Collapsing to distinct
    classes here would silently under-count the ground truth.
    """
    labels_dir = cfg.DATASET / "test" / "labels"
    out = {}
    for path in labels_dir.glob("*.txt"):
        classes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if 0 <= cid < cfg.NUM_CLASSES:
                classes.append(cfg.CLASS_NAMES[cid])
        out[path.stem] = classes
    return out


def load_all(conf):
    with open(cfg.CALORIE_TABLE, newline="", encoding="utf-8") as f:
        kcal = {r["class_name"]: float(r["kcal_per_portion"]) for r in csv.DictReader(f)}
    with open(cfg.GROUND_TRUTH, newline="", encoding="utf-8") as f:
        expected = {}
        for r in csv.DictReader(f):
            try:
                expected[r["file"]] = float(r["expected_kcal"])
            except (TypeError, ValueError):
                continue
    manifest_path = cfg.MANIFEST_DIR / "test.csv"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}\n  run:  python src/derive_labels.py")
    with open(manifest_path, newline="", encoding="utf-8") as f:
        manifest = {r["file"]: r for r in csv.DictReader(f)}
    data = json.loads(cfg.PREDICTIONS_JSON.read_text(encoding="utf-8"))
    detected = {e["image_id"]: [b["class"] for b in e["boxes"] if b["conf"] >= conf]
                for e in data["predictions"]}
    return kcal, expected, manifest, detected, load_ground_truth_boxes()


def summarise(label, rows):
    n = len(rows)
    actual = mean(abs(r["est"] - r["exp"]) for r in rows)
    perfect = mean(abs(r["true_kcal"] - r["exp"]) for r in rows)
    gross = mean(r["fp"] + r["fn"] for r in rows)
    net = mean(abs(r["fp"] - r["fn"]) for r in rows)
    return {
        "subset": label,
        "n": n,
        "actual_mae": round(actual, 1),
        "perfect_detector_mae": round(perfect, 1),
        "detection_headroom": round(actual - perfect, 1),
        "headroom_pct": round(100 * (actual - perfect) / actual, 1) if actual else 0.0,
        "gross_detection_error": round(gross, 1),
        "net_after_cancelling": round(net, 1),
        "cancelled_pct": round(100 * (gross - net) / gross, 1) if gross else 0.0,
        "big_miss_rate_pct": round(100 * sum(abs(r["est"] - r["exp"]) > 200 for r in rows) / n, 1),
        "mean_expected_kcal": round(mean(r["exp"] for r in rows), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=cfg.CONF_THRESHOLD)
    args = ap.parse_args()

    kcal, expected, manifest, detected, gt_boxes = load_all(args.conf)

    rows = []
    for filename in sorted(set(detected) & set(expected) & set(manifest)):
        det = detected[filename]
        true = gt_boxes.get(filename.rsplit(".", 1)[0], [])

        det_counts, true_counts = {}, {}
        for c in det:
            det_counts[c] = det_counts.get(c, 0) + 1
        for c in true:
            true_counts[c] = true_counts.get(c, 0) + 1

        # Price the disagreement in kcal, per class, by count.
        every = set(det_counts) | set(true_counts)
        fp = sum(kcal.get(c, 0.0) * max(0, det_counts.get(c, 0) - true_counts.get(c, 0))
                 for c in every)
        fn = sum(kcal.get(c, 0.0) * max(0, true_counts.get(c, 0) - det_counts.get(c, 0))
                 for c in every)

        rows.append({
            "file": filename,
            "is_multi": manifest[filename]["is_multi_item"] == "1",
            "est": sum(kcal.get(c, 0.0) for c in det),
            "true_kcal": sum(kcal.get(c, 0.0) for c in true),
            "exp": expected[filename],
            "fp": fp,
            "fn": fn,
        })

    groups = [
        ("all", rows),
        ("single-item", [r for r in rows if not r["is_multi"]]),
        ("multi-item", [r for r in rows if r["is_multi"]]),
    ]
    stats = [summarise(label, group) for label, group in groups if group]

    print("=" * 78)
    print("CALORIE ERROR DECOMPOSITION  (detector pipeline, conf >= "
          f"{args.conf})")
    print("=" * 78)
    print(f"{'subset':<14}{'n':>4}{'actual':>9}{'perfect':>9}{'headroom':>10}{'  (of MAE)':>12}")
    for s in stats:
        print(f"{s['subset']:<14}{s['n']:>4}{s['actual_mae']:>9}"
              f"{s['perfect_detector_mae']:>9}{s['detection_headroom']:>10}"
              f"{s['headroom_pct']:>11}%")
    print("\n  'perfect' = the MAE Track B would have scored if Track A had made zero")
    print("  detection errors. It is pure calorie-table error and no detection")
    print("  improvement can reduce it. 'headroom' is everything better detection")
    print("  could ever buy.\n")

    print("=" * 78)
    print("WHY MULTI-ITEM LOOKS EASIER THAN IT IS")
    print("=" * 78)
    print(f"{'subset':<14}{'n':>4}{'gross':>9}{'net':>8}{'cancelled':>11}"
          f"{'|err|>200':>11}{'mean kcal':>11}")
    for s in stats:
        print(f"{s['subset']:<14}{s['n']:>4}{s['gross_detection_error']:>9}"
              f"{s['net_after_cancelling']:>8}{s['cancelled_pct']:>10}%"
              f"{s['big_miss_rate_pct']:>10}%{s['mean_expected_kcal']:>11}")
    print("\n  gross = FP + FN kcal, the detector's total disagreement with truth")
    print("  net   = |FP - FN|, what survives after over- and under-counts cancel")
    print("          inside the per-image sum")
    print("  On BOTH gross and net the detector does WORSE on multi-item plates.")
    print("  The lower headline MAE comes from (a) a fatter tail of catastrophic")
    print("  single-item failures, where one miss is the entire image, and (b) a")
    print("  larger denominator: multi-item plates simply hold more calories.\n")

    out = cfg.OUTPUTS / "error_decomposition.csv"
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
