"""
predict_classifier.py -- Track C (G17)

Runs the trained classifier over the test split and writes one row per image.
Kept separate from training so you can re-run the comparison without retraining.

Emits top-1 plus top-3, because "the right answer was in its top 3" is a fair
question a marker will ask, and having the number ready is better than not.

Usage:
    python src/predict_classifier.py
"""

import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from dataset import FoodClassificationDataset, pick_device
from train_classifier import build_model


def load_classifier(device):
    if not cfg.CLASSIFIER_CKPT.exists():
        raise SystemExit(
            f"missing {cfg.CLASSIFIER_CKPT}\n"
            f"  run:  python src/train_classifier.py"
        )
    ckpt = torch.load(cfg.CLASSIFIER_CKPT, map_location=device, weights_only=False)
    # pretrained=False: the checkpoint replaces every weight anyway.
    model = build_model(cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def main():
    device = pick_device()
    model, ckpt = load_classifier(device)
    print(f"loaded classifier (valid acc {ckpt.get('valid_acc', float('nan')):.4f})"
          f" on {device}")

    test_ds = FoodClassificationDataset("test", train=False)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    rows = []
    for images, labels, files in loader:
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        top3 = probs.topk(k=min(3, cfg.NUM_CLASSES), dim=1)
        for i, filename in enumerate(files):
            ids = top3.indices[i].tolist()
            scores = top3.values[i].tolist()
            rows.append({
                "file": filename,
                "true_label": cfg.CLASS_NAMES[int(labels[i])],
                "pred_label": cfg.CLASS_NAMES[ids[0]],
                "pred_conf": round(scores[0], 4),
                "top3": "|".join(cfg.CLASS_NAMES[c] for c in ids),
                "top3_conf": "|".join(f"{s:.4f}" for s in scores),
            })

    cfg.CLASSIFIER_PREDS.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.CLASSIFIER_PREDS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top1 = sum(r["true_label"] == r["pred_label"] for r in rows) / max(1, len(rows))
    top3_acc = sum(r["true_label"] in r["top3"].split("|") for r in rows) / max(1, len(rows))
    print(f"test top-1 {top1:.4f}   top-3 {top3_acc:.4f}   ({len(rows)} images)")
    print(f"wrote {cfg.CLASSIFIER_PREDS}")
    print("\nnext:  python src/compare.py")


if __name__ == "__main__":
    main()
