"""
dataset.py -- Track C (G17)

PyTorch Dataset that reads the manifests produced by derive_labels.py.

A note on augmentation, because it is easy to get wrong here:

The Roboflow TRAIN split is ALREADY augmented (757 source photos -> 1593 train
images) by Roboflow's own pipeline. So we add only light extra augmentation on
top -- horizontal flip and mild colour jitter. Piling on heavy augmentation
would be augmenting augmented data, which mostly just blurs the signal.

VALID and TEST get no augmentation at all, only resize + normalise. Augmenting
an evaluation set makes the score meaningless.

Normalisation uses ImageNet statistics because we start from ImageNet-pretrained
weights -- the network expects inputs distributed the way its pretraining data
was.
"""

import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def read_manifest(split: str):
    path = cfg.MANIFEST_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(
            f"missing manifest {path}\n"
            f"  run:  python src/derive_labels.py"
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class FoodClassificationDataset(Dataset):
    """One image -> one label. The whole point of the baseline."""

    def __init__(self, split: str, train: bool = False):
        self.split = split
        self.rows = read_manifest(split)
        self.images_dir = cfg.DATASET / split / "images"
        self.tf = build_transforms(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = self.images_dir / row["file"]
        # convert("RGB") matters: some Roboflow exports carry palette PNGs or
        # greyscale, which would otherwise arrive with the wrong channel count.
        image = Image.open(path).convert("RGB")
        return self.tf(image), int(row["label_id"]), row["file"]

    def label_counts(self):
        counts = torch.zeros(cfg.NUM_CLASSES, dtype=torch.long)
        for row in self.rows:
            counts[int(row["label_id"])] += 1
        return counts


def class_weights(dataset: FoodClassificationDataset) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Why bother: the label distribution is badly skewed (Curry-Puff appears many
    times over, Fried-Egg barely at all). Without weighting, the model can score
    respectably by ignoring the rare classes entirely -- and since our whole
    argument is about what the classifier CANNOT do, we must not hand a critic
    the easy objection that we crippled it through neglect. Weighting is the
    defensible default.

    Classes absent from the split get weight 0 rather than infinity.
    """
    counts = dataset.label_counts().float()
    weights = torch.where(counts > 0, counts.sum() / (counts * (counts > 0).sum()),
                          torch.zeros_like(counts))
    present = weights > 0
    if present.any():
        weights[present] = weights[present] / weights[present].mean()
    return weights


def pick_device(requested: str = "auto") -> torch.device:
    """CUDA if you have it, Apple Silicon MPS if you're on a Mac, else CPU.

    CPU works. EfficientNetB0 on ~1600 images at 224px is roughly 10-15 min per
    epoch on a laptop CPU, so use --epochs 8 if that is all you have.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
