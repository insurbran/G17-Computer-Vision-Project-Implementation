"""Main Part B pipeline: predictions.json -> per-image calorie estimates.

kcal(detection) = kcal_per_portion(class) * portion_factor
portion_factor  = clip(food_frac / REF_FOOD_FRAC, MIN_F, MAX_F)

food_frac is the OpenCV-segmented food area divided by box area. In
--method box mode the factor is fixed at 1.0 (the naive baseline we
compare against in the report).

Usage:
  python src/calorie_estimator.py --predictions outputs/fake/predictions.json \
      --images outputs/fake/images --out outputs/fake [--method segment|box] \
      [--conf 0.4] [--save-overlays]
"""
import argparse
import csv
import json
from pathlib import Path

import cv2

from segmentation import segment_food, draw_overlay

REF_FOOD_FRAC = 0.65   # typical fraction of a box that is actually food
MIN_F, MAX_F = 0.5, 1.6  # clamp so one bad mask can't nuke an estimate


def load_calorie_table(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["class_name"]: float(r["kcal_per_portion"]) for r in csv.DictReader(f)}


def portion_factor(food_frac: float) -> float:
    return max(MIN_F, min(MAX_F, food_frac / REF_FOOD_FRAC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--images", required=True, help="folder with the test images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["segment", "box"], default="segment")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--table", default=str(Path(__file__).parent.parent / "data" / "calorie_table.csv"))
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--ref-frac", type=float, default=None,
                    help="override REF_FOOD_FRAC (calibrate on the VALID set, not test)")
    args = ap.parse_args()
    global REF_FOOD_FRAC
    if args.ref_frac:
        REF_FOOD_FRAC = args.ref_frac

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / f"overlays_{args.method}"
    if args.save_overlays:
        overlay_dir.mkdir(exist_ok=True)

    kcal_table = load_calorie_table(Path(args.table))
    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    rows, detail = [], []
    for img_entry in preds["predictions"]:
        img_path = Path(args.images) / img_entry["image_id"]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"WARN missing image: {img_path}")
            continue
        total = 0.0
        overlay = image
        for det in img_entry["boxes"]:
            if det["conf"] < args.conf:
                continue
            cls = det["class"]
            if cls not in kcal_table:
                print(f"WARN class not in calorie table: {cls}")
                continue
            if args.method == "segment":
                seg = segment_food(image, det["xyxy"])
                factor = portion_factor(seg.food_frac)
                seg_info = {"food_frac": round(seg.food_frac, 3), "seg_method": seg.method}
            else:
                seg, factor, seg_info = None, 1.0, {"food_frac": 1.0, "seg_method": "box"}
            kcal = kcal_table[cls] * factor
            total += kcal
            detail.append({"file": img_entry["image_id"], "class_name": cls,
                           "confidence": det["conf"], "portion_factor": round(factor, 3),
                           "kcal": round(kcal, 1), **seg_info})
            if args.save_overlays and seg is not None:
                overlay = draw_overlay(overlay, det["xyxy"], seg, f"{cls} {kcal:.0f}kcal")
        rows.append({"file": img_entry["image_id"], "estimated_kcal": round(total, 1),
                     "n_items": sum(1 for d in detail if d["file"] == img_entry["image_id"])})
        if args.save_overlays:
            cv2.imwrite(str(overlay_dir / img_entry["image_id"]), overlay)

    est_csv = out_dir / f"estimates_{args.method}.csv"
    with open(est_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "estimated_kcal", "n_items"])
        wr.writeheader(); wr.writerows(rows)
    (out_dir / f"detail_{args.method}.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    print(f"Wrote {est_csv} ({len(rows)} images, method={args.method})")


if __name__ == "__main__":
    main()
