# Track C — Baseline Classifier + Demo (G17)

Two deliverables:

1. **A single-label classifier baseline** (EfficientNetB0) and a three-metric
   comparison against Track A's detector — the evidence that the detection
   architecture was the right choice.
2. **A Streamlit demo** — upload a photo, see boxes, see a calorie total, see
   the classifier fail on the same image.

---

## Before anything runs, you need two files that are not in this repo

| File | Where it goes | How to get it |
|---|---|---|
| `best.pt` | repo root | **Ask Oscar** for `runs/detect/baseline3/weights/best.pt` (~6 MB). `predictions.json` is *not* a substitute — it is a frozen output for 76 fixed images and cannot process a new photo. |
| `dataset/` | repo root | Download [Roboflow "Malaysian Food Recognition 2" v7](https://universe.roboflow.com/malaysian-dishes-classification/malaysian-food-recognition-2-0kwwr/dataset/7) in **YOLOv8** format and unzip. |

Both are gitignored deliberately — they are large and regenerable.

Expected layout:

```
G17-Computer-Vision-Project-Implementation/
├── best.pt                  <- from Oscar
├── dataset/                 <- from Roboflow
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   └── test/{images,labels}/
├── Track_A_Detection/
├── Part_B_Calories/
└── Track_C_Baseline_Demo/   <- this folder
```

If a path is wrong, fix it in `config.py` and nowhere else.

---

## Run order

```bash
pip install -r requirements.txt

python src/derive_labels.py         # boxes -> one label per image
python src/train_classifier.py      # fine-tune EfficientNetB0
python src/predict_classifier.py    # classifier output on the test split
python src/compare.py               # the three metrics + comparison table
python src/make_figures.py          # report figures

streamlit run app.py                # the demo
```

**You can run `compare.py` today, before Oscar sends anything:**

```bash
python src/compare.py --simulate-classifier
```

That substitutes the largest-area detection for a classifier and computes the
calorie metric from files already in the repo. Its detector column reproduces
Track B's published **110.7 kcal MAE / 16.7% MAPE** exactly, which is how you
know the harness agrees with Brandon's pipeline. The classifier column is an
upper bound proxy — **never put those numbers in the report.**

---

## What each script does and why

### `derive_labels.py` — boxes to labels

The dataset labels *boxes*. A classifier needs *one label per image*. The rule
for collapsing them **is** the experiment design.

```
label(image) = class of the largest box (normalised w × h)
```

- **largest area** → deterministic, scale-invariant, matches what a human means by "what dish is this?"
- **majority count** → rejected; a nasi lemak with 6 peanuts and 1 rice becomes "Peanuts"
- **`ground_truth.csv → items_visible`** → **never**. Verified: it is a verbatim copy of the detector's own predictions (76/76 rows match at conf 0.4). Only `expected_kcal` was hand-corrected. Training on it and then comparing against the detector is circular.

Also emits the **single-item vs multi-item split of the test set**, derived from
ground-truth boxes, not from any model's output. That split is the spine of the
whole argument.

### `train_classifier.py` — the baseline

**EfficientNetB0, not ResNet50:**

```
YOLOv8n         = 3.01M parameters   (Track A's detector, counted from best.pt)
EfficientNetB0  = 4.03M parameters   <- our baseline, 17-class head
ResNet50        = 23.5M parameters   (17-class head, for comparison)
```

All three counted with a 17-class head so the comparison is like-for-like. Note
that the widely quoted "EfficientNetB0 = 5.3M" is the stock ImageNet model with a
1000-class head; swapping that for 17 classes removes 1.26M parameters.

The point is a *structural* limitation — one output slot cannot enumerate a
plate — not that YOLO extracts better features. Giving the classifier ~1.34×
more capacity than the detector kills the "you handicapped the baseline"
objection before anyone raises it. ResNet50 at 23.5M params on ~1600 images
just overfits and eats your week.

Two-phase fine-tune: freeze the backbone and warm up the new head (so its
random-init gradients don't wreck pretrained features), then unfreeze
everything at a low LR. Class-weighted loss, because ignoring rare classes is
exactly the shortcut that would let a critic say we crippled the baseline.

**Best checkpoint is selected on validation accuracy, never on test.**

### `compare.py` — the three metrics

You cannot compare accuracy against mAP. So both models are scored only on axes
they can both occupy:

| | What it shows |
|---|---|
| **M1** Top-1 dish accuracy | The **control**. On single-item plates the classifier should be roughly competitive. If it isn't, your training is broken and nothing else is trustworthy. |
| **M2** Multi-label set F1 | The structural point, measured. The classifier's output set has size 1 by construction — on a 7-item plate it recovers at most 1/7 no matter how accurate it is. |
| **M3** Downstream calorie MAE | **The headline.** Both models pushed through Track B's calorie table, same ground truth, same units — directly comparable to Brandon's 110.7. |

Ground-truth sources, keep them straight:

```
M1, M2 -> dataset test label .txt files    (human-drawn boxes)
M3     -> Part_B_Calories/data/ground_truth.csv, `expected_kcal` column
```

### `app.py` — the demo

Three panels off one upload: detection boxes → Brandon's segmentation overlay
and calorie table → the classifier's single guess beside the detector's total.

On a multi-item plate the app spells out the gap in kcal. That is the argument,
live, on the marker's own photo.

---

## Writing it up

**Lead with M3.** Tracks A and B both landed on negative results (CLAHE didn't
help; segmentation didn't help). Both are honest and well-analysed, but if
Track C reads as "we confirmed the obvious," the report is three chapters of
nothing worked. M3 is the one place the project produces a **positive,
quantified finding**: detection was measurably the right architecture, and here
is the cost in kilocalories of choosing wrong.

**Reframe the brief.** "Show it does worse" decides the answer before running
the experiment, and it will leak into your design. Write it as *"measure what a
single-label classifier can and cannot do on this task."* Same result, honestly
derived, consistent with how A and B reported theirs.

**Do not oversell the multi-item subset.** It is 25 of 76 images (re-derived from ground-truth boxes; the 33 figure came from detector output and was wrong).
Report n plainly and claim no statistical significance — Track A set that
precedent with its seed-spread analysis, and a marker who noticed it there will
be looking for it here.

**Limitations to state:**

- Training images are 320×320; a phone photo is far larger. The demo will
  underperform the 87.2% headline on real photos. Test this early — it is the
  most likely demo-day question.
- 17 classes only. Anything else is undetectable, not misdetected.
- The demo reports calorie *estimates*. Ground truth exists for the 76 test
  images and nowhere else.
- The image-level label rule (largest box) is a design choice, not a
  ground-truth fact. State it and say why.

---

## Worth raising with the team

- **0.872 vs 0.882** — Track A's README and report draft use the single-run
  number; `fig1_overall.png` uses the 3-seed mean. Both correct. Pick one
  convention for the report or the two contradict each other.
- **`ground_truth.csv` typos** — columns `remakrs` and `quantity if have`.
- **GT description** — if the report calls `items_visible` "human-annotated
  item lists," that is inaccurate. Only `expected_kcal` was corrected.
- **Test-set calibration** — Brandon tuned `ref_frac = 0.574` on the test set
  because valid-split predictions weren't available, and flagged it himself.
  Once you have `best.pt`, run Track A's `emit_predictions.py` on the valid
  split and hand it over. Cheap fix, patches a real hole.
