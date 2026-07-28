"""
derive_labels.py -- Track C (G17)

Turns the detection dataset into a classification dataset.

WHY THIS SCRIPT EXISTS
----------------------
The Roboflow dataset labels BOXES, not images. Every .txt file holds one line
per object:

    class_id  cx  cy  w  h        (all normalised 0-1)

A classifier needs exactly ONE label per image. So we have to invent a rule for
collapsing a list of boxes into a single label, and that rule IS the experiment
design -- pick a bad one and the comparison is rigged.

THE RULE WE USE: largest box area wins.
    label(image) = class of the box with the greatest w*h

Why largest-area and not the alternatives:

  largest area   -> deterministic, scale-invariant (w,h are normalised), and it
                    matches what a human means by "what dish is this?" The main
                    subject of a food photo is the biggest thing on the plate.
  majority count -> REJECTED. Biases toward small repeated garnishes. A nasi
                    lemak with 6 peanuts and 1 rice portion would be labelled
                    "Peanuts". Curry-Puff would swamp the dataset.
  first listed   -> REJECTED. Label file order is arbitrary.
  detector output-> REJECTED, and specifically do NOT use ground_truth.csv's
                    `items_visible` column: it is a verbatim copy of the
                    detector's own predictions (verified, 76/76 rows). Training
                    on it and then comparing against the detector is circular.

WHAT ELSE THIS EMITS
--------------------
The single-item vs multi-item split of the TEST set, derived from ground-truth
boxes (not from any model's output). That split is the spine of the whole Track
C argument: a single-label classifier should be competitive on single-item
plates and should collapse on multi-item ones.

Usage:
    python src/derive_labels.py
    python src/derive_labels.py --splits test          (just re-check one split)
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_boxes(label_path: Path):
    """Parse one YOLO label file -> [(class_id, area_normalised), ...].

    Malformed lines are skipped rather than crashing the run: Roboflow exports
    are usually clean, but a single bad line should not cost you the dataset.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            w, h = float(parts[3]), float(parts[4])
        except ValueError:
            continue
        if not (0 <= cid < cfg.NUM_CLASSES):
            print(f"  WARN class_id {cid} out of range in {label_path.name}")
            continue
        boxes.append((cid, max(0.0, w) * max(0.0, h)))
    return boxes


def find_image(images_dir: Path, stem: str):
    """Label files and image files share a stem but not an extension."""
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_split(split: str):
    """-> (rows, stats). One row per image that has at least one box."""
    images_dir = cfg.DATASET / split / "images"
    labels_dir = cfg.DATASET / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(
            f"missing {images_dir} or {labels_dir}\n"
            f"  Is the dataset unzipped at {cfg.DATASET}? "
            f"Expected <split>/images and <split>/labels inside it."
        )

    rows, skipped_empty, skipped_noimage = [], 0, 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        boxes = read_boxes(label_path)
        if not boxes:
            skipped_empty += 1
            continue
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            skipped_noimage += 1
            continue

        # THE RULE: biggest box wins.
        top_cid, _ = max(boxes, key=lambda b: b[1])
        present = sorted({cid for cid, _ in boxes})

        rows.append({
            "file": image_path.name,
            "label_id": top_cid,
            "label": cfg.CLASS_NAMES[top_cid],
            "n_boxes": len(boxes),
            "n_distinct_classes": len(present),
            "is_multi_item": int(len(present) > 1),
            "all_classes": "|".join(cfg.CLASS_NAMES[c] for c in present),
        })

    stats = {
        "split": split,
        "images": len(rows),
        "skipped_empty_label": skipped_empty,
        "skipped_missing_image": skipped_noimage,
        "single_item": sum(1 for r in rows if not r["is_multi_item"]),
        "multi_item": sum(r["is_multi_item"] for r in rows),
    }
    return rows, stats


def write_manifest(split: str, rows):
    cfg.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.MANIFEST_DIR / f"{split}.csv"
    fields = ["file", "label_id", "label", "n_boxes",
              "n_distinct_classes", "is_multi_item", "all_classes"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def report_balance(split: str, rows):
    """Class imbalance is worth seeing before you train, not after."""
    counts = Counter(r["label"] for r in rows)
    missing = [c for c in cfg.CLASS_NAMES if c not in counts]
    print(f"  label distribution ({len(counts)}/{cfg.NUM_CLASSES} classes present):")
    for name, n in counts.most_common():
        bar = "#" * max(1, round(40 * n / max(counts.values())))
        print(f"    {name:<18}{n:>5}  {bar}")
    if missing:
        print(f"  NOT PRESENT as a top-1 label in {split}: {', '.join(missing)}")
        print("    (these classes only ever appear as the smaller item in an image;")
        print("     the classifier can never learn them -- say so in the report)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = ap.parse_args()

    cfg.require_dataset()

    all_stats = []
    for split in args.splits:
        print(f"\n=== {split} ===")
        rows, stats = build_split(split)
        out = write_manifest(split, rows)
        all_stats.append(stats)
        print(f"  wrote {out}  ({stats['images']} images)")
        if stats["skipped_empty_label"]:
            print(f"  skipped {stats['skipped_empty_label']} images with no boxes")
        if stats["skipped_missing_image"]:
            print(f"  skipped {stats['skipped_missing_image']} labels with no image")
        print(f"  single-item: {stats['single_item']}   "
              f"multi-item: {stats['multi_item']}")
        report_balance(split, rows)

    print("\n=== FOR THE REPORT ===")
    print("Test-set composition (from ground-truth boxes, not model output):")
    for s in all_stats:
        if s["split"] == "test":
            total = s["images"]
            print(f"  {total} images = {s['single_item']} single-item "
                  f"+ {s['multi_item']} multi-item")
            print(f"  multi-item share: {100 * s['multi_item'] / max(1, total):.1f}%")
            print("\n  n for the multi-item subset is small. Report the number "
                  "plainly and do NOT claim statistical significance --\n"
                  "  Track A set that precedent with its seed-spread analysis.")


if __name__ == "__main__":
    main()
