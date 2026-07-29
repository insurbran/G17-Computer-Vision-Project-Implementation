# Track C — Baseline Classifier + Demo (G17)

Two deliverables:

1. **A single-label classifier baseline** (EfficientNetB0) and a three-metric
   comparison against Track A's detector — the evidence that the detection
   architecture was the right choice.
2. **A Streamlit demo** — upload a photo, see boxes, see a calorie total, see
   the classifier fail on the same image.

> **Results are not in this file.** This README explains how the code works and
> why it was built this way. For final numbers, the framing argument, and the
> limitations to carry into the report, see **`REPORT_NOTES.md`**.

**Status: delivered.** Classifier trained, all three metrics computed, figures
generated, demo verified end-to-end.

---

## Two large files are not in this repo

| File | Where it goes | How to get it |
|---|---|---|
| `best.pt` | repo root | Track A's detector weights, `runs/detect/baseline3/weights/best.pt` (~6 MB), from Oscar. **Already obtained and verified** — see `REPORT_NOTES.md` §9. `predictions.json` is *not* a substitute; it is a frozen output for 76 fixed images and cannot process a new photo. |
| the dataset | anywhere in the repo root | Download [Roboflow "Malaysian Food Recognition 2" v7](https://universe.roboflow.com/malaysian-dishes-classification/malaysian-food-recognition-2-0kwwr/dataset/7) in **YOLOv8** format and unzip. |

Both are gitignored deliberately — they are large and regenerable.

**You do not need to rename the dataset folder.** The Roboflow zip unpacks to
something like `Malaysian Food Recognition 2.v7i.yolov8`, and `config.py` locates
it by searching the repo root for a directory containing `data.yaml` and
`test/images`. An explicitly named `dataset/` still wins if you prefer that.
Hardcoding the name meant the scripts broke on every machine but the one that
renamed the folder.

Expected layout:

```
G17-Computer-Vision-Project-Implementation/
├── best.pt                                  <- from Oscar (gitignored)
├── Malaysian Food Recognition 2.v7i.yolov8/ <- from Roboflow (gitignored)
│   ├── data.yaml
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   └── test/{images,labels}/
├── Track_A_Detection/
├── Part_B_Calories/
└── Track_C_Baseline_Demo/                   <- this folder
```

If something still cannot be found, fix it in `config.py` and nowhere else.

---

## Run order

```bash
pip install -r requirements.txt

python src/derive_labels.py         # boxes -> one label per image
python src/train_classifier.py      # fine-tune EfficientNetB0 (~10 min on MPS)
python src/predict_classifier.py    # classifier output on the test split
python src/compare.py                 # the three metrics + comparison table
python src/classifier_diagnostics.py  # Macro-F1, per-class, confusion matrix
python src/error_decomposition.py     # table error vs detector error
python src/efficiency_benchmark.py    # latency, file size, throughput
python src/make_figures.py            # report figures

streamlit run app.py                # the demo
```

`outputs/` is gitignored and fully regenerable from the above.

### Running the comparison without a trained model

```bash
python src/compare.py --simulate-classifier
```

Substitutes the largest-area detection for a classifier, so the harness can be
exercised before a model exists. Its detector column reproduces Track B's
published **110.7 kcal MAE / 16.7% MAPE** exactly, which is how you know the
harness agrees with Brandon's pipeline.

The classifier column is an upper-bound proxy — **never put those numbers in the
report.** To make that hard to do by accident, a simulated run writes
`simulated=1` into `comparison_table.csv`, and `make_figures.py` then stamps the
figures with a diagonal watermark *and* prefixes the title with
`[ SIMULATED PROXY - NOT FOR THE REPORT ]`. If you see that stamp, regenerate
without the flag.

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

Also emits the **single-item vs multi-item split of the test set** (51 / 25),
derived from ground-truth boxes, not from any model's output. That split is the
spine of the whole argument.

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
| **M1** Top-1 dish accuracy | The **control**. On single-item plates the classifier should be roughly competitive. If it isn't, your training is broken and nothing else is trustworthy. *(It was: 0.941 vs the detector's 0.824 — the baseline wins its home turf.)* |
| **M2** Multi-label set F1 | The structural point, measured. The classifier's output set has size 1 by construction — on a 7-item plate it recovers at most 1/7 no matter how accurate it is. |
| **M3** Downstream calorie MAE | **The headline.** Both models pushed through Track B's calorie table, same ground truth, same units — directly comparable to Brandon's 110.7. |

Ground-truth sources, keep them straight:

```
M1, M2 -> dataset test label .txt files    (human-drawn boxes)
M3     -> Part_B_Calories/data/ground_truth.csv, `expected_kcal` column
```

### `error_decomposition.py` — whose fault is the calorie error?

Splits Track B's calorie error into the part a *perfect* detector would still
incur (the fixed-portion lookup table) and the part detection contributes. The
headline output is the "perfect detector" column: the MAE Track B would have
scored had Track A been flawless. Everything between that and the actual MAE is
the entire headroom better detection could ever buy.

It answers a question `compare.py` raises but cannot explain — why the detector's
calorie MAE looks *better* on multi-item plates than single-item ones. Full
analysis in `REPORT_NOTES.md` §6.

### `classifier_diagnostics.py` — Macro-F1 and the confusion matrix

Section 9.3 of our literature review commits us to reporting "Top-1 accuracy,
Macro-F1 (for rare dishes/Malaysian foods), and a confusion matrix". Top-1 alone
was not enough. This produces the other two.

The confusion matrix matters most: Section 7.3 of the review records that **none
of the ten papers surveyed publishes one**, leaving "which foods get confused with
which largely unanswered." Publishing it is a stated contribution, not
housekeeping. Track A supplies the detection-side matrix; this supplies the
classification side.

Headline: **Macro-F1 0.554 against Top-1 0.724.** The 0.169 gap is exactly the
effect the review predicted — a strong overall accuracy concealing total failure
on rare classes. Every class scoring 1.000 is a standalone noodle or bread dish;
every class scoring 0.000 is a nasi lemak component.

### `efficiency_benchmark.py` — inference cost

Section 8.5 names deployment cost as an unresolved gap and Section 9.3 asks for a
model that runs on a phone. Parameters are a proxy for size, not speed, so this
measures wall-clock latency and file size directly. The detector has **fewer**
parameters yet runs **2.04x slower** — reporting parameters alone would have
implied the opposite conclusion.

### `app.py` — the demo

Three panels off one upload: detection boxes → Brandon's segmentation overlay
and calorie table → the classifier's single guess beside the detector's total.

On a multi-item plate the app spells out the gap in kcal. That is the argument,
live, on the marker's own photo.

---

## Writing it up

See **`REPORT_NOTES.md`** — it has the numbers, the framing paragraph, and the
limitations in the form they should reach the report. The short version:

**Lead with M3.** Tracks A and B both landed on negative results (CLAHE didn't
help; segmentation didn't help). M3 is the one place the project produces a
**positive, quantified finding**: detection was measurably the right
architecture, and here is the cost in kilocalories of choosing wrong.

**Do not write "the classifier performed worse."** It beat the detector on
single-item plates (0.941 vs 0.824 top-1) and reached parity on single-item
calorie error. It collapses only on multi-item plates, and *that* is the finding.
Because the baseline wins its home turf, the collapse cannot be blamed on a weak
model — which is what makes the structural argument hold.

**Do not oversell the multi-item subset.** It is 25 of 76 images, re-derived from
ground-truth boxes (the 33 figure came from detector output and was wrong).
Report n plainly and claim no statistical significance — Track A set that
precedent with its seed-spread analysis.

**Limitations to state:**

- **Demo failure mode is framing, not resolution.** This was measured, not
  assumed. Upscaling a 320×320 image to 3024×3024 returns the *same* detections,
  because YOLO resizes to 640 internally. What breaks it is the plate occupying a
  small share of the frame — below roughly 10% of frame area, detection returns
  nothing at all. Aspect-ratio distortion also hurts (stretching to 4:3 dropped 5
  detections to 3; letterboxing preserved 6). Demo rule: **fill the frame with the
  plate.** Table in `REPORT_NOTES.md` §7.
- 17 classes only. Anything else is undetectable, not misdetected.
- The demo reports calorie *estimates*. Ground truth exists for the 76 test
  images and nowhere else.
- Classifier confidences shown in the demo are **not calibrated** — validation
  loss rose while accuracy improved, meaning the model is overconfident on its
  errors. Harmless for M1/M2/M3 (top-1 only), but don't present them as
  probabilities.
- The image-level label rule (largest box) is a design choice, not a
  ground-truth fact. State it and say why.

---

## Worth raising with the team

- **0.872 vs 0.882** — these are not two roundings of one number. 0.872 is run
  `baseline3`; 0.882 is the 3-seed mean. Track A's README uses the first,
  `fig1_overall.png` uses the second. Pick one convention or they contradict.
- **`baseline3` is the weakest of the three baseline seeds** (0.8718 vs 0.8863
  and 0.8871). `predictions.json` comes from that run, so every detector number in
  Tracks B and C derives from a below-average detector. Conservative, but say so.
- **`ground_truth.csv` typos** — columns `remakrs` and `quantity if have`.
- **GT description** — if the report calls `items_visible` "human-annotated
  item lists," that is inaccurate. Only `expected_kcal` was corrected.
- **Test-set calibration** — Brandon tuned `ref_frac = 0.574` on the test set
  because valid-split predictions weren't available, and flagged it himself.
  `best.pt` is now in the repo, so this is fixable: run Track A's
  `emit_predictions.py` on the valid split and hand it over. **Still outstanding.**
