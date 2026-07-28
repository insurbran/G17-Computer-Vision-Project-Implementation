# Track C (Baseline Classifier + Demo) — everything the report needs

Written for whoever drafts the Part II report. All numbers below are final and
reproducible from this folder. Format follows Part B's `REPORT_NOTES.md`.

## 1. What Track C does

Builds the obvious simpler alternative to the detection pipeline — a single-label
image classifier — and scores it against Track A's detector on three axes both
models can occupy. The purpose is to establish whether the detection architecture
was the right engineering choice, and to quantify the cost of choosing wrong.

Classifier: **EfficientNetB0**, ImageNet-pretrained, fine-tuned to 17 classes.

## 2. Design decisions (these ARE the experiment)

**2.1 Deriving image labels from box labels.** The Roboflow dataset labels boxes,
not images. A classifier needs one label per image, so a collapsing rule was
required and the choice of rule determines the result.

```
label(image) = class of the largest annotated box (normalised w x h)
```

- *largest area* — deterministic, scale-invariant, matches what a human means by
  "what dish is this?"
- *majority count* — rejected. Biases toward small repeated garnishes; a nasi
  lemak with 6 peanuts and 1 rice would be labelled "Peanuts".
- *`ground_truth.csv` → `items_visible`* — **never used.** Verified: that column
  is a verbatim copy of the detector's own predictions at conf 0.4 (76/76 rows
  match). Only `expected_kcal` was hand-corrected. Training a classifier on it and
  then comparing against the detector would be circular.

**2.2 Model capacity — why EfficientNetB0, not ResNet50.**

```
YOLOv8n        =  3.0M parameters   (Track A's detector, verified from best.pt)
EfficientNetB0 =  5.3M parameters   <- our baseline
ResNet50       = 25.6M parameters
```

The claim being tested is a *structural* limitation, not a feature-quality one.
Giving the baseline ~1.8x more capacity than the detector it is compared against
removes the "you handicapped the baseline" objection in advance. ResNet50 at 25.6M
parameters on 1593 training images overfits badly.

**2.3 Training.** Two-phase fine-tune (3 epochs frozen-backbone head warm-up, then
20 epochs full fine-tune at low LR), cosine LR decay, AdamW, class-weighted
cross-entropy (train distribution is skewed 7.6x, Lo-Mein 204 → Fried-Egg 27).
**Best checkpoint selected on validation accuracy, never on test.**

## 3. Headline results (76 test images)

### M1 — top-1 dish accuracy

| Subset | n | Classifier | Detector |
|---|---|---|---|
| All | 76 | **0.724** | 0.658 |
| Single-item | 51 | **0.941** | 0.824 |
| Multi-item | 25 | 0.280 | 0.320 |

### M2 — image-level multi-label F1 (set of classes present)

| Subset | n | clf P | clf R | clf F1 | det P | det R | det F1 |
|---|---|---|---|---|---|---|---|
| All | 76 | 0.882 | 0.383 | 0.534 | 0.860 | 0.846 | 0.853 |
| Single-item | 51 | 0.941 | 0.941 | **0.941** | 0.729 | 0.843 | 0.782 |
| Multi-item | 25 | 0.760 | **0.153** | 0.255 | 0.929 | 0.847 | **0.886** |

### M3 — downstream calorie MAE (both models → Track B's calorie table)

| Subset | n | clf MAE | clf MAPE | det MAE | det MAPE |
|---|---|---|---|---|---|
| All | 76 | 267.1 | 34.2% | **110.7** | 16.7% |
| Single-item | 51 | 130.5 | **17.3%** | 116.3 | 18.5% |
| Multi-item | 25 | **545.7** | 68.7% | 99.2 | 13.0% |

**Validity check:** the detector column of M3 reproduces Track B's published
110.7 kcal / 16.7% MAPE exactly. This confirms Track C's harness agrees with
Brandon's pipeline; without it the classifier column could not be trusted.

## 4. The framing — use this, not "the classifier was worse"

The data supports a sharper claim, and one that is much harder to attack:

> The single-label classifier is **better than the detector at what classifiers
> do** — 94.1% vs 82.4% top-1 on single-item plates, F1 0.941 vs 0.782 — and
> reaches **parity** with the full detection pipeline on downstream calorie error
> for single-dish photos (MAE 130.5 vs 116.3, but MAPE 17.3% vs 18.5%; the two
> metrics disagree, so neither approach wins). It is nonetheless unusable for this
> task: on multi-item plates its calorie error is **5.5x** the detection
> pipeline's (545.7 vs 99.2 kcal). The failure is not inaccuracy. It is having one
> output slot for a plate with six things on it.

Why this framing matters: because the baseline **wins its home turf**, the
multi-item collapse cannot be attributed to a weak or under-trained model. The
structural explanation is the only one left standing. A comparison where the
baseline loses everywhere invites the objection that it was built to lose.

**Secondary conclusion worth stating:** the detection architecture earns its
complexity *only* on multi-item plates. If the product scope were one dish per
photo, the simpler classifier would have been the correct engineering choice.
That is more useful than a blanket "detection is better."

## 5. Supporting numbers

- Classifier test top-1 **0.7237**, top-3 **0.8816** (76 images).
- Best validation accuracy **0.7315**. Validation set is 149 images, so **1 image
  = 0.67 percentage points**; differences under ~0.02 are fewer than 3 images and
  should not be over-interpreted.
- Test-set composition, derived from **ground-truth boxes** (not model output):
  **51 single-item, 25 multi-item**, 0 empty.
- `Chicken-Rendang` and `Fried-Egg` never appear as a top-1 ground-truth label in
  the test split, so only **15 of 17** classes are scoreable on M1.
- Overfitting onset visible from phase-2 epoch 6: train–valid gap widened 0.082 →
  0.231; validation **loss** began rising (1.196 → 1.249) while validation
  **accuracy** kept improving (0.705 → 0.732). The model became overconfident on
  the examples it still got wrong — cross-entropy penalises confident errors,
  accuracy does not. This does not affect M1/M2/M3, which consume top-1 only, but
  it does mean the confidence percentages in the demo are **not calibrated
  probabilities** and should not be presented as such.

## 6. Error decomposition — a second, independent finding

`src/error_decomposition.py` splits Track B's calorie error into the part a
perfect detector would still incur (the fixed-portion lookup table) and the part
detection contributes.

| Subset | n | Actual MAE | Perfect-detector MAE | Headroom |
|---|---|---|---|---|
| All | 76 | 110.7 | 57.3 | 48.2% |
| Single-item | 51 | 116.3 | 43.2 | **62.9%** |
| Multi-item | 25 | 99.2 | 86.1 | **13.2%** |

Even with flawless detection, multi-item MAE would only improve 99.2 → 86.1. About
**87% of the error on multi-item plates is the lookup table**, locked in before
detection happens.

It also explains why the detector's headline MAE *looks* better on multi-item
plates (99.2) than single-item (116.3), which is counter-intuitive:

| Subset | gross det. error | net after cancelling | cancelled |
|---|---|---|---|
| Single-item | 251.8 | 101.5 | 59.7% |
| Multi-item | 284.5 | 131.1 | 53.9% |

On both gross (FP+FN kcal) and net, the detector performs **worse** on multi-item
plates. The lower headline MAE is an artifact of (a) a fatter tail of catastrophic
single-item failures — one miss is the entire image, with nothing else in the sum
to dilute it — and (b) a larger denominator, since multi-item plates hold more
calories (759.1 vs 597.7 mean expected kcal).

**Worked example — image `fried34_jpg.rf.37744e...jpg`:**

```
TRUE      Anchovies, Boiled-Egg, Fried-Chicken, Peanuts, Rice, Rice, Sambal  -> 1178 kcal
DETECTED  Boiled-Egg, Chicken-Rendang, Peanuts, Rice, Roti-Canai             -> 1018 kcal
expected_kcal (Brandon's hand-corrected truth)                                ->  988 kcal

missed:    Anchovies, Fried-Chicken, one Rice, Sambal   =  710 kcal lost
invented:  Chicken-Rendang, Roti-Canai                  =  550 kcal added
                                                           net 160 kcal
REPORTED ERROR = |1018 - 988| = 30 kcal
```

The detector got four items wrong and hallucinated two dishes, and Track B's
reported error for that image is **30 kcal**. The misses and hallucinations
cancelled. This is a genuine methodological limitation of MAE on a summed
pipeline and is worth stating plainly in the evaluation section.

**Relationship to Part B §5.3.** Brandon concluded "dominant errors are detection
errors, not portion errors." This decomposition supports that overall (48%
headroom) and strongly on single-item plates (63%), but **refines it for
multi-item plates (13% headroom)** — there the table is the ceiling, which is
precisely why portion refinement could not help. The two analyses are compatible:
Brandon's evidence is the worst-5 tail, this is the aggregate.

## 7. Limitations to state

- Multi-item subset is **n = 25**. Report plainly; claim no statistical significance.
- The image-level label rule (largest annotated box) is a **design choice, not a
  ground-truth fact**. On multi-item plates M1 is a weak metric **for both models**
  (0.280 / 0.320) because the single "correct" label is a convention. M2 and M3 are
  the informative metrics on that subset.
- `expected_kcal` was hand-corrected by a group member, so "table error" in §6
  bundles genuine portion variation with annotation judgment.
- 17 classes only. Anything else is undetectable, not misdetected.
- Calorie figures are estimates. Ground truth exists for the 76 test images only.

**Demo behaviour on phone photos — measured, not assumed.** The whole pipeline was
replayed headlessly on real test images at a range of framings. The limitation is
**not resolution**; it is **how much of the frame the food occupies**.

```
plate scaled and pasted into a 4032x3024 frame, detections at conf 0.40
subject area     fried34   fried27   plain24
   75%              6        10         9        healthy
   54%              3         9         8
   37%              2         9         7
   23%              1        10         7
   12%              1         1         7        degrading
    7%              0         0         0        total failure
```

Pure upscaling is harmless — a 320x320 image enlarged to 3024x3024 returns the
same 5 detections, because YOLO resizes to 640 internally. Two things *do* break
it: **aspect-ratio distortion** (stretching to 4:3 dropped 5 detections to 3;
letterboxing instead preserved all 6) and **standing back from the plate**. Below
roughly 10% of frame area, detection returns nothing at all.

Operational guidance for the demo: **fill the frame with the plate.** The
practical cause is the training distribution — every training image is a tightly
cropped food photo, so a wide table shot is far outside it.

## 8. Figures available

- `outputs/figures/fig_c1_calorie_error.png` — M3 calorie MAE, classifier vs
  detector, split all / single / multi. **The headline figure.**
- `outputs/figures/fig_c2_multilabel.png` — M2 multi-label F1, same split.
- Suggested third figure: actual vs perfect-detector MAE from §6.
- Suggested table: §3 M3, plus §6 decomposition.

Figures generated from a `--simulate-classifier` run are automatically watermarked
"SIMULATED PROXY — NOT FOR THE REPORT" and the title is prefixed. If you see that
stamp, the numbers are placeholders; re-run `compare.py` without the flag.

## 9. Detector provenance — verified, and one caveat for the team

`best.pt` was verified against `predictions.json` before the demo was built:

```
train_args.name = baseline3          correct run, not a CLAHE run
model = yolov8n.pt, 3.01M params     correct architecture
17 classes, exact order match        against config.CLASS_NAMES
saved metrics != final-epoch metrics  confirms best.pt, not last.pt
re-ran on all 76 test images         76/76 identical class sets vs predictions.json
                                     (max confidence drift 0.0035, CUDA vs CPU)
```

**Caveat worth raising:** `test_results.json` shows `baseline3` is the **weakest of
the three baseline seeds**:

```
baseline3   mAP50 0.8718   <- predictions.json comes from this run
baseline_s1       0.8863
baseline_s2       0.8871
3-seed mean       0.8817
```

Every detector number in Tracks B and C therefore derives from a below-average
detector run. This makes the detector's results a conservative estimate, which
works in the report's favour, but it should be stated rather than discovered.

It also means the report must pick one convention: **0.872 (single run) or 0.882
(3-seed mean)**. Track A's README uses the former and `fig1_overall.png` uses the
latter; as written they contradict each other.

## 10. Reproduce

```bash
pip install -r requirements.txt

python src/derive_labels.py          # boxes -> one label per image + manifests
python src/train_classifier.py       # EfficientNetB0 fine-tune (~30 min on MPS)
python src/predict_classifier.py     # classifier output on the test split
python src/compare.py                # M1 + M2 + M3, the comparison table
python src/error_decomposition.py    # table error vs detector error
python src/make_figures.py           # report figures

streamlit run app.py                 # the demo
```

Requires `best.pt` at the repo root and the Roboflow dataset unzipped anywhere in
the repo root (`config.py` locates it by finding `data.yaml`). Both are gitignored.
`outputs/` is gitignored and fully regenerable from the above.
