"""
emit_predictions.py  --  CSC3014 Project Part 2, Track A (G17)

Runs the trained detector over a folder of images and writes predictions.json,
the handoff contract consumed by Track B (calorie estimation) and Track C
(classifier baseline + demo).

Coordinates are absolute pixel xyxy in the ORIGINAL image resolution -- YOLO's
letterboxing is already undone by Ultralytics before results are returned, so
consumers can index straight into the source image without rescaling.

The detector deliberately does not emit calories. It reports what food is where
and how confident it is; mapping that to kcal belongs to Track B.

Usage (from C:\\Users\\PF4B3\\Desktop\\g17):
    python emit_predictions.py
    python emit_predictions.py --weights C:/Users/PF4B3/runs/detect/clahe/weights/best.pt --out predictions_clahe.json
    python emit_predictions.py --conf 0.10 --out predictions_lowconf.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = "C:/Users/PF4B3/runs/detect/baseline3/weights/best.pt"
DEFAULT_SOURCE = HERE / "dataset" / "test" / "images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def collect(source):
    folder = Path(source)
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")
    found = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not found:
        sys.exit(f"no images in {folder}")
    return found


def box_records(result, names):
    """Flatten one Ultralytics Result into a list of plain dicts."""
    out = []
    if result.boxes is None:
        return out
    for box in result.boxes:
        cid = int(box.cls.item())
        x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0].tolist())
        out.append({
            "class": names[cid],
            "class_id": cid,
            "conf": round(float(box.conf.item()), 4),
            "xyxy": [x1, y1, x2, y2],
        })
    out.sort(key=lambda b: b["conf"], reverse=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="emit predictions.json for Tracks B and C")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(HERE / "predictions.json"))
    parser.add_argument("--conf", type=float, default=0.25, help="confidence floor")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold")
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"weights not found: {weights}")

    model = YOLO(str(weights))
    names = model.names
    images = collect(args.source)

    print(f"weights: {weights}")
    print(f"source:  {args.source}  ({len(images)} images)")
    print(f"conf:    {args.conf}   iou: {args.iou}\n")

    predictions = []
    empty = 0

    for index, path in enumerate(images, start=1):
        result = model.predict(str(path), conf=args.conf, iou=args.iou, verbose=False)[0]
        height, width = result.orig_shape
        boxes = box_records(result, names)
        if not boxes:
            empty += 1
        predictions.append({
            "image_id": path.name,
            "image_size": [width, height],
            "boxes": boxes,
        })
        if index % 25 == 0 or index == len(images):
            print(f"  {index}/{len(images)}")

    payload = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": weights.name,
        "run": weights.parent.parent.name,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "classes": [names[i] for i in range(len(names))],
        "predictions": predictions,
    }

    target = Path(args.out)
    target.write_text(json.dumps(payload, indent=2))

    total = sum(len(p["boxes"]) for p in predictions)
    print("-" * 46)
    print(f"images:      {len(predictions)}")
    print(f"boxes:       {total}")
    print(f"no boxes:    {empty}")
    print(f"avg per img: {total / len(predictions):.2f}")
    print(f"\nwritten: {target}")


if __name__ == "__main__":
    main()
