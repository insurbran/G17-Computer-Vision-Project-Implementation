"""
config.py -- Track C (Baseline Classifier + Demo), G17

Every path the Track C scripts need, in one place. If something cannot be
found, change it HERE and nowhere else.

Layout this assumes (matches how the repo is already organised):

    G17-Computer-Vision-Project-Implementation/
        Track_A_Detection/          <- Oscar's scripts
        Part_B_Calories/            <- Brandon's pipeline
        Track_C_Baseline_Demo/      <- this folder
        dataset/                    <- you download this (Roboflow, gitignored)
        best.pt                     <- you get this from Oscar (gitignored)

Neither `dataset/` nor `best.pt` is in the repo. Both are gitignored because
they are large. See README.md for how to obtain them.
"""

from pathlib import Path

# --- roots -----------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Detector weights from Track A. Oscar must send this file -- predictions.json
# is a frozen output for 76 fixed images and CANNOT process a new photo.
WEIGHTS = REPO / "best.pt"

# Roboflow "Malaysian Food Recognition 2" v7, YOLOv8 format.
#
# The Roboflow zip does NOT unpack to a folder called "dataset" -- it unpacks to
# a folder named after the project and version, e.g.
#     "Malaysian Food Recognition 2.v7i.yolov8"
# Every teammate who downloads it hits that, so hardcoding "dataset" means the
# scripts break on every machine but the one that renamed the folder. Resolve it
# by content instead: find the directory that actually holds a data.yaml and a
# test/images. An explicit `dataset/` still wins if someone did rename it.


def _find_dataset(root: Path) -> Path:
    explicit = root / "dataset"
    if (explicit / "data.yaml").exists():
        return explicit
    for candidate in sorted(p for p in root.iterdir() if p.is_dir()):
        if candidate.name.startswith("."):
            continue
        if (candidate / "data.yaml").exists() and (candidate / "test" / "images").is_dir():
            return candidate
    # Nothing found. Return the conventional path so require_dataset() can print
    # a useful "expected at ..." message rather than something arbitrary.
    return explicit


DATASET = _find_dataset(REPO)

# Track B's folder. We import Brandon's segmentation + calorie code from here
# rather than copying it, so his fixes reach us automatically.
PART_B = REPO / "Part_B_Calories"
CALORIE_TABLE = PART_B / "data" / "calorie_table.csv"
GROUND_TRUTH = PART_B / "data" / "ground_truth.csv"
PART_B_SRC = PART_B / "src"

# Track A's frozen predictions on the 76 test images. Used by compare.py so the
# comparison can run before Oscar sends the weights.
PREDICTIONS_JSON = PART_B / "from_A" / "predictions.json"

# --- our outputs -----------------------------------------------------------

OUTPUTS = HERE / "outputs"
MANIFEST_DIR = OUTPUTS / "manifests"
CLASSIFIER_CKPT = OUTPUTS / "classifier_best.pt"
CLASSIFIER_PREDS = OUTPUTS / "classifier_test_preds.csv"
COMPARISON_CSV = OUTPUTS / "comparison_table.csv"
COMPARISON_MD = OUTPUTS / "comparison_table.md"
FIGURE_DIR = OUTPUTS / "figures"

# --- shared constants ------------------------------------------------------

# Class order MUST match Track A's. It is the order in predictions.json
# ("classes") and in the dataset's data.yaml. class_id indexes into this list.
# Do not sort it, do not re-order it -- the label .txt files store integers.
CLASS_NAMES = [
    "Anchovies", "Boiled-Egg", "Char-Kuey-Teow", "Chicken-Rendang",
    "Curry-Puff", "Fried-Chicken", "Fried-Egg", "Fried-Rice",
    "Hokkien-Mee", "Lo-Mein", "Mee-Rebus", "Mee-Siam",
    "Peanuts", "Rice", "Roti-Canai", "Sambal", "Slices-Cucumber",
]
NUM_CLASSES = len(CLASS_NAMES)

# Confidence floor for detections. 0.4 is what Track B used to produce its
# headline numbers (217 detections over 76 images). Keep it identical or our
# calorie comparison is not comparable to Brandon's 110.7 kcal.
CONF_THRESHOLD = 0.4

# Classifier input size. 224 is EfficientNetB0's native training resolution.
IMG_SIZE = 224

# Report figure colours -- taken from Track A's make_figures.py so every figure
# in the report matches. Validated colourblind-safe (worst adjacent pair
# dE 10.8 protan, 21.4 normal vision).
C_DETECT = "#3B6EA5"   # detector / our system
C_CLASSIFY = "#C1666B"  # classifier baseline


def require(path: Path, what: str, how: str) -> Path:
    """Fail loudly and usefully instead of throwing a bare FileNotFoundError."""
    if not path.exists():
        raise SystemExit(
            f"\nMISSING: {what}\n"
            f"  expected at: {path}\n"
            f"  how to get it: {how}\n"
        )
    return path


def require_weights() -> Path:
    return require(
        WEIGHTS, "Track A detector weights (best.pt)",
        "Ask Oscar for runs/detect/baseline3/weights/best.pt (~6 MB). "
        "predictions.json is NOT a substitute -- it only holds answers for "
        "the 76 test images and cannot process a new photo.",
    )


def require_dataset() -> Path:
    return require(
        DATASET, "the Roboflow dataset",
        "Download 'Malaysian Food Recognition 2' v7 in YOLOv8 format from "
        "https://universe.roboflow.com/malaysian-dishes-classification/"
        "malaysian-food-recognition-2-0kwwr/dataset/7 and unzip it to "
        f"{DATASET}",
    )
