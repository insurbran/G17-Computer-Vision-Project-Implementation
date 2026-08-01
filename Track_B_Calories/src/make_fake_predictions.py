"""Generate synthetic plate images + a fake predictions.json in the agreed
schema, plus a matching ground_truth.csv — so Part B runs end-to-end before
A's detector is ready. Replace with real data later; nothing else changes.
"""
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["Anchovies", "Boiled-Egg", "Char-Kuey-Teow", "Chicken-Rendang",
           "Curry-Puff", "Fried-Chicken", "Fried-Egg", "Fried-Rice",
           "Hokkien-Mee", "Lo-Mein", "Mee-Rebus", "Mee-Siam", "Peanuts",
           "Rice", "Roti-Canai", "Sambal", "Slices-Cucumber"]

FOOD_COLORS = {  # BGR-ish plausible colours so HSV/GrabCut has something real to segment
    "Rice": (235, 240, 245), "Sambal": (30, 30, 180), "Boiled-Egg": (200, 230, 250),
    "Anchovies": (60, 90, 130), "Peanuts": (70, 120, 170), "Fried-Chicken": (50, 100, 160),
    "Fried-Egg": (120, 200, 250), "Slices-Cucumber": (120, 200, 140),
}
DEFAULT_COLOR = (60, 110, 170)


def make_image(w, h, items, rng):
    img = np.full((h, w, 3), 210, np.uint8)                      # table
    cv2.circle(img, (w // 2, h // 2), int(min(w, h) * 0.46), (245, 245, 250), -1)  # plate
    noise = rng.integers(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    detections = []
    for cls, (cx, cy, rx, ry) in items:
        color = FOOD_COLORS.get(cls, DEFAULT_COLOR)
        cv2.ellipse(img, (cx, cy), (rx, ry), rng.integers(0, 180), 0, 360, color, -1)
        pad_x, pad_y = int(rx * rng.uniform(1.15, 1.5)), int(ry * rng.uniform(1.15, 1.5))
        bbox = [max(0, cx - pad_x), max(0, cy - pad_y), min(w, cx + pad_x), min(h, cy + pad_y)]
        detections.append({"class": cls, "class_id": CLASSES.index(cls),
                           "conf": round(rng.uniform(0.55, 0.95), 2),
                           "xyxy": [float(v) for v in bbox]})
    return img, detections


def main():
    rng = np.random.default_rng(42)
    random.seed(42)
    out = Path(__file__).parent.parent / "outputs" / "fake"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    table = {r["class_name"]: float(r["kcal_per_portion"]) for r in csv.DictReader(
        open(Path(__file__).parent.parent / "data" / "calorie_table.csv", encoding="utf-8"))}

    entries, gt_rows = [], []
    combos = [
        ["Rice", "Sambal", "Boiled-Egg", "Anchovies", "Peanuts"],  # nasi lemak
        ["Char-Kuey-Teow"], ["Fried-Rice", "Fried-Egg"], ["Roti-Canai"],
        ["Mee-Rebus"], ["Fried-Chicken", "Rice", "Slices-Cucumber"],
        ["Hokkien-Mee"], ["Curry-Puff", "Curry-Puff"],
    ]
    w, h = 640, 640
    for i, combo in enumerate(combos, 1):
        spots = []
        for j, cls in enumerate(combo):
            angle = 2 * np.pi * j / max(1, len(combo))
            r = 0 if len(combo) == 1 else int(min(w, h) * 0.22)
            cx, cy = int(w / 2 + r * np.cos(angle)), int(h / 2 + r * np.sin(angle))
            size = int(min(w, h) * (0.3 if len(combo) == 1 else 0.11))
            spots.append((cls, (cx, cy, size + int(rng.integers(-8, 12)), size + int(rng.integers(-8, 12)))))
        img, dets = make_image(w, h, spots, rng)
        fname = f"fake_{i:04d}.jpg"
        cv2.imwrite(str(img_dir / fname), img)
        entries.append({"image_id": fname, "image_size": [w, h], "boxes": dets})
        true_kcal = sum(table[c] for c in combo) * rng.uniform(0.85, 1.15)  # portion noise
        gt_rows.append({"file": fname, "expected_kcal": round(float(true_kcal), 1),
                        "items_visible": "+".join(combo), "portion_notes": "synthetic"})

    (out / "predictions.json").write_text(json.dumps(
        {"schema_version": 1, "model": "fake-detector-v0", "conf_threshold": 0.25,
         "iou_threshold": 0.70, "classes": CLASSES, "predictions": entries}, indent=2),
        encoding="utf-8")
    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "expected_kcal", "items_visible", "portion_notes"])
        wr.writeheader(); wr.writerows(gt_rows)
    print(f"Wrote {len(entries)} fake images + predictions.json + ground_truth.csv -> {out}")


if __name__ == "__main__":
    main()
