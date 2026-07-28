"""
train_classifier.py -- Track C (G17)

Fine-tunes EfficientNetB0 to classify a food photo into one of 17 classes.
This is the BASELINE the report compares detection against.

WHY EFFICIENTNETB0 AND NOT RESNET50
-----------------------------------
    YOLOv8n        =  3.01M parameters   (Track A's detector, counted from best.pt)
    EfficientNetB0 =  4.03M parameters   <- what we use, with a 17-class head
    ResNet50       = 23.5M parameters    (17-class head, for comparison)

Counted with a 17-class head so the comparison is like-for-like. The commonly
quoted "EfficientNetB0 = 5.3M" is the stock 1000-class ImageNet model; replacing
that head with 17 classes removes 1.26M parameters.

The point of this baseline is to demonstrate a STRUCTURAL limitation -- a
single-label classifier cannot enumerate the items on a plate, so it cannot
feed Track B -- not to show that YOLO extracts better features. If we picked a
weaker or smaller model, the obvious objection is "you handicapped the
baseline." EfficientNetB0 gives the classifier ~1.34x MORE capacity than the
detector it is being compared against, in the same order of magnitude. When it
still loses, that objection is not available.

ResNet50 is the more conventional baseline, but at 23.5M parameters on ~1600
training images it overfits hard and you would spend your week fighting
regularisation instead of writing the report.

TRAINING RECIPE
---------------
Two phases, which is standard practice for fine-tuning a pretrained net:

  Phase 1 (head warm-up): freeze the backbone, train only the new 17-way
      output layer. The head starts at random; letting its large early
      gradients flow into carefully pretrained features would damage them.
  Phase 2 (full fine-tune): unfreeze everything at a low learning rate so the
      features adapt to food photos without being destroyed.

Cosine learning-rate decay, AdamW, class-weighted cross-entropy. Best
checkpoint is selected on VALIDATION accuracy -- never on test. Touching the
test set for model selection is the single most common way student projects
quietly inflate their numbers.

Usage:
    python src/train_classifier.py
    python src/train_classifier.py --epochs 8 --batch-size 16   (slow machine)
    python src/train_classifier.py --device cpu
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from dataset import (FoodClassificationDataset, class_weights, pick_device)


def set_seed(seed: int):
    """Track A made reproducibility a theme of this project; we match it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNetB0 with an ImageNet backbone and a fresh 17-way head.

    pretrained=False is for the inference path: when a trained checkpoint is
    about to be loaded over the top, downloading ImageNet weights first is a
    pointless download and a confusing warning if it fails.
    """
    if not pretrained:
        model = models.efficientnet_b0(weights=None)
    else:
        try:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:  # older torchvision, or no network for the download
            print("WARN could not fetch pretrained ImageNet weights -- training")
            print("     from scratch. Accuracy will be MUCH lower and the baseline")
            print("     will look unfairly weak. Fix your connection and re-run.")
            model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool):
    for param in model.features.parameters():
        param.requires_grad = not frozen


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / max(1, seen), correct / max(1, seen)


def run_epoch(model, loader, device, criterion, optimizer, scheduler):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / max(1, seen), correct / max(1, seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20, help="phase-2 epochs")
    ap.add_argument("--warmup-epochs", type=int, default=3, help="phase-1 (frozen) epochs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-full", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 is safest on Windows; raise on Linux/Mac for speed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-class-weights", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    train_ds = FoodClassificationDataset("train", train=True)
    valid_ds = FoodClassificationDataset("valid", train=False)
    print(f"train: {len(train_ds)} images    valid: {len(valid_ds)} images")

    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    valid_loader = DataLoader(valid_ds, shuffle=False, **common)

    model = build_model(cfg.NUM_CLASSES).to(device)

    if args.no_class_weights:
        weights = None
        print("class weighting: OFF")
    else:
        weights = class_weights(train_ds).to(device)
        print("class weighting: ON (inverse frequency, mean-normalised)")
    criterion = nn.CrossEntropyLoss(weight=weights)

    history, best_acc, best_epoch = [], -1.0, -1
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ---- Phase 1: head only ------------------------------------------------
    set_backbone_frozen(model, True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_head, weight_decay=args.weight_decay)
    print(f"\n--- phase 1: head warm-up ({args.warmup_epochs} epochs, backbone frozen) ---")
    for epoch in range(1, args.warmup_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer, None)
        va_loss, va_acc = evaluate(model, valid_loader, device, criterion)
        history.append({"phase": 1, "epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "valid_loss": va_loss, "valid_acc": va_acc})
        print(f"  e{epoch:<3} train {tr_acc:.3f} / {tr_loss:.3f}   "
              f"valid {va_acc:.3f} / {va_loss:.3f}")
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save({"model": model.state_dict(), "classes": cfg.CLASS_NAMES,
                        "valid_acc": va_acc, "arch": "efficientnet_b0"},
                       cfg.CLASSIFIER_CKPT)

    # ---- Phase 2: full fine-tune ------------------------------------------
    set_backbone_frozen(model, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_full,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(train_loader)))
    print(f"\n--- phase 2: full fine-tune ({args.epochs} epochs) ---")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion,
                                    optimizer, scheduler)
        va_loss, va_acc = evaluate(model, valid_loader, device, criterion)
        history.append({"phase": 2, "epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "valid_loss": va_loss, "valid_acc": va_acc})
        flag = ""
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, args.warmup_epochs + epoch
            torch.save({"model": model.state_dict(), "classes": cfg.CLASS_NAMES,
                        "valid_acc": va_acc, "arch": "efficientnet_b0"},
                       cfg.CLASSIFIER_CKPT)
            flag = "  <- best"
        print(f"  e{epoch:<3} train {tr_acc:.3f} / {tr_loss:.3f}   "
              f"valid {va_acc:.3f} / {va_loss:.3f}{flag}")

    elapsed = time.time() - started
    summary = {
        "arch": "efficientnet_b0",
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "best_valid_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "epochs_total": args.warmup_epochs + args.epochs,
        "seed": args.seed,
        "class_weighted": not args.no_class_weights,
        "minutes": round(elapsed / 60, 1),
        "history": history,
    }
    (cfg.OUTPUTS / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nbest validation accuracy {best_acc:.4f} at epoch {best_epoch}")
    print(f"checkpoint -> {cfg.CLASSIFIER_CKPT}")
    print(f"summary    -> {cfg.OUTPUTS / 'training_summary.json'}")
    print(f"took {elapsed / 60:.1f} min")
    print("\nnext:  python src/predict_classifier.py")


if __name__ == "__main__":
    main()
