# Track C (Baseline Classifier + Demo) — everything the report needs

Written for whoever drafts the Part II report. All numbers below are final and
reproducible from this folder. Format follows Part B's `REPORT_NOTES.md`.

## 1. What Track C does

Builds the obvious simpler alternative to the detection pipeline — a single-label
image classifier — and scores it against Track A's detector on axes both models
can occupy. The purpose is to establish whether the detection architecture was the
right engineering choice, and to quantify the cost of choosing wrong.

Classifier: **EfficientNetB0**, ImageNet-pretrained, fine-tuned to 17 classes.

This is not an optional extra. Section 9.3 of our literature review recommends,
as its second implementation recommendation:

> "we should use detection instead of classification. Malaysian meals usually have
> many items at once on one plate. This means that a classifier that only labels
> the image would not work well. A YOLO-based detector with bounding boxes, like
> UECFood, would probably prove a better fit [21], [22]."

The review **asserted** this from the literature. Track C **tests** it on Malaysian
food and reports the magnitude. Section 12 maps every review commitment to where
it is discharged.

## 2. Design decisions (these ARE the experiment)

**2.1 Deriving image labels from box labels.** The Roboflow dataset labels boxes,
not images. A classifier needs one label per image, so a collapsing rule was
required, and the choice of rule determines the result.

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
YOLOv8n         = 3.01M parameters   (Track A's detector, counted from best.pt)
EfficientNetB0  = 4.03M parameters   <- our baseline, 17-class head
ResNet50        = 23.5M parameters   (17-class head, for comparison)
```

All three counted with a 17-class head so the comparison is like-for-like. The
widely quoted "EfficientNetB0 = 5.3M" is the stock ImageNet model with a
1000-class head; swapping that for 17 classes removes 1.26M parameters.

The claim being tested is a *structural* limitation, not a feature-quality one.
Giving the baseline ~1.34x more capacity than the detector it is compared against
removes the "you handicapped the baseline" objection in advance. ResNet50 at 23.5M
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

> The single-label classifier is **better than the detector at what classifiers
> do** — 94.1% vs 82.4% top-1 on single-item plates, F1 0.941 vs 0.782 — and
> reaches **parity** with the full detection pipeline on downstream calorie error
> for single-dish photos (MAE 130.5 vs 116.3, but MAPE 17.3% vs 18.5%; the two
> metrics disagree, so neither approach wins). It is nonetheless unusable for this
> task: on multi-item plates its calorie error is **5.5x** the detection
> pipeline's (545.7 vs 99.2 kcal). The failure is not inaccuracy. It is having one
> output slot for a plate with six things on it.

Because the baseline **wins its home turf**, the multi-item collapse cannot be
attributed to a weak or under-trained model. The structural explanation is the
only one left standing. A comparison where the baseline loses everywhere invites
the objection that it was built to lose.

**Secondary conclusion:** the detection architecture earns its complexity *only*
on multi-item plates. If the product scope were one dish per photo, the simpler
classifier would have been the correct engineering choice.

## 5. Per-class diagnostics — Macro-F1 and the confusion matrix

Section 9.3 of the review commits us to reporting "Top-1 accuracy, Macro-F1 (for
rare dishes/Malaysian foods), and a confusion matrix". This section discharges
both remaining commitments. Produced by `src/classifier_diagnostics.py`.

```
Top-1 accuracy (= Micro-F1)   0.7237
Top-3 accuracy                0.8816
Weighted-F1                   0.7301
Macro-F1                      0.5544
Macro-Micro gap               0.1692
```

For single-label multi-class, micro-precision = micro-recall = micro-F1 =
accuracy, because every mistake is simultaneously one false positive and one false
negative. Macro-F1 is therefore the only one of the two that carries new
information, and the **0.169 gap is a direct, quantified confirmation of the
warning our own review records** in Section 7.2, citing Niu et al. [23]: that
accuracy "can conceal weak results on infrequent classes."

### Per-class F1 (15 of 17 classes appear in the test split)

| Class | n | Precision | Recall | F1 |
|---|---|---|---|---|
| Lo-Mein | 10 | 1.000 | 1.000 | **1.000** |
| Roti-Canai | 6 | 1.000 | 1.000 | **1.000** |
| Mee-Siam | 5 | 1.000 | 1.000 | **1.000** |
| Curry-Puff | 7 | 0.875 | 1.000 | 0.933 |
| Fried-Rice | 8 | 1.000 | 0.750 | 0.857 |
| Hokkien-Mee | 6 | 0.750 | 1.000 | 0.857 |
| Char-Kuey-Teow | 8 | 1.000 | 0.625 | 0.769 |
| Mee-Rebus | 7 | 0.667 | 0.857 | 0.750 |
| Slices-Cucumber | 6 | 1.000 | 0.333 | 0.500 |
| Fried-Chicken | 2 | 0.333 | 0.500 | 0.400 |
| Peanuts | 3 | 0.200 | 0.333 | 0.250 |
| Anchovies | 4 | 0.000 | 0.000 | **0.000** |
| Rice | 2 | 0.000 | 0.000 | **0.000** |
| Boiled-Egg | 1 | 0.000 | 0.000 | **0.000** |
| Sambal | 1 | 0.000 | 0.000 | **0.000** |

`Chicken-Rendang` and `Fried-Egg` never appear as a top-1 ground-truth label in
this split and are excluded from the macro average.

### The finding this exposes

Read the table by dish type rather than by rank. Every class the classifier scores
1.000 on is a **standalone noodle or bread dish** — Lo-Mein, Mee-Siam, Roti-Canai,
Hokkien-Mee, Curry-Puff, Fried-Rice, Char-Kuey-Teow. Every class it scores
**0.000** on is a **nasi lemak component** — Anchovies, Rice, Boiled-Egg, Sambal —
with Peanuts (0.250) and Slices-Cucumber (0.500) the next two worst.

That split is not a coincidence and it is not about class frequency alone. When
the largest annotated box on a composite plate is a *component*, the classifier
still sees the whole plate and has no mechanism to isolate the labelled item. It
answers with whichever component it finds most salient. The confusion matrix shows
this directly: Boiled-Egg is predicted as Sambal 100% of the time, Rice as Sambal
50%, Sambal itself as Fried-Chicken 100%.

**This is the same structural limitation as M2 and M3, arriving from a completely
independent direction.** M2 measures it as capped recall; M3 prices it in
kilocalories; the confusion matrix shows *which specific Malaysian dishes* the
architecture cannot handle. Use all three — they corroborate each other.

### Most-confused pairs

```
2x  Slices-Cucumber -> Peanuts          nasi lemak garnishes, small and similar
2x  Char-Kuey-Teow  -> Hokkien-Mee      fine-grained: two dark fried noodle dishes
2x  Anchovies       -> Peanuts          nasi lemak garnishes again
1x  Sambal          -> Fried-Chicken
1x  Slices-Cucumber -> Chicken-Rendang
1x  Slices-Cucumber -> Boiled-Egg
```

The Char-Kuey-Teow → Hokkien-Mee confusion is a textbook fine-grained recognition
failure, the problem Section 8.2 of the review describes: two visually similar
dark stir-fried noodle dishes separated mainly by cooking style. Everything else
is nasi lemak components being confused with one another.

Section 7.3 of our review notes that **none of the ten papers surveyed publishes a
full confusion matrix**, leaving "the question of which foods get confused with
which largely unanswered." Figure `fig_c3_confusion_matrix.png` answers it for
Malaysian food on the classification side; Track A's matrix answers it on the
detection side. Present them as a pair — together they are a stated contribution
of this project, not supporting material.

## 6. Supporting numbers

- Best validation accuracy **0.7315** at epoch 12 of 23. The final 11 epochs
  produced no improvement.
- Validation set is 149 images, so **1 image = 0.67 percentage points**;
  differences under ~0.02 are fewer than 3 images and should not be
  over-interpreted.
- Test-set composition, derived from **ground-truth boxes** (not model output):
  **51 single-item, 25 multi-item**, 0 empty.
- Overfitting onset visible from phase-2 epoch 6: train–valid gap widened 0.082 →
  0.231; validation **loss** began rising (1.196 → 1.249) while validation
  **accuracy** kept improving (0.705 → 0.732). The model became overconfident on
  the examples it still got wrong — cross-entropy penalises confident errors,
  accuracy does not. This does not affect M1/M2/M3, which consume top-1 only, but
  it does mean the confidence percentages in the demo are **not calibrated
  probabilities** and should not be presented as such.
- Top-3 rather than Top-5 is reported. With 17 classes, Top-5 covers 29% of the
  label space and stops discriminating; Top-3 (18%) is the informative analogue of
  the Top-5 used on 101- and 2000-class benchmarks in the review's Table 2.

## 7. Error decomposition — a second, independent finding

`src/error_decomposition.py` splits Track B's calorie error into the part a
perfect detector would still incur (the fixed-portion lookup table) and the part
detection contributes.

| Subset | n | Actual MAE | Perfect-detector MAE | Detection headroom |
|---|---|---|---|---|
| All | 76 | 110.7 | 57.3 | 48.2% |
| Single-item | 51 | 116.3 | 43.2 | **62.9%** |
| Multi-item | 25 | 99.2 | 86.1 | **13.2%** |

Even with flawless detection, multi-item MAE would only improve 99.2 → 86.1. About
**87% of the error on multi-item plates is the calorie table**, locked in before
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
TRUE      Anchovies, Boiled-Egg, Fried-Chicken, Peanuts, Rice, Rice, Sambal -> 1178 kcal
DETECTED  Boiled-Egg, Chicken-Rendang, Peanuts, Rice, Roti-Canai            -> 1018 kcal
expected_kcal (hand-corrected ground truth)                                  ->  988 kcal

missed:    Anchovies, Fried-Chicken, one Rice, Sambal   =  710 kcal lost
invented:  Chicken-Rendang, Roti-Canai                  =  550 kcal added
REPORTED ERROR = |1018 - 988| = 30 kcal
```

The detector got four items wrong and hallucinated two dishes, and Track B's
reported error for that image is **30 kcal**. The misses and hallucinations
cancelled. This is a genuine methodological limitation of MAE on a summed
pipeline and belongs in the evaluation section.

**Relationship to Part B §5.3.** Brandon concluded "dominant errors are detection
errors, not portion errors." This decomposition supports that overall (48%
headroom) and strongly on single-item plates (63%), but **refines it for
multi-item plates (13% headroom)** — there the table is the ceiling, which is
precisely why portion refinement could not help. The two analyses are compatible:
Brandon's evidence is the worst-5 tail, this is the aggregate.

## 8. Inference cost

Section 8.5 of the review names deployment cost as an unresolved research gap, and
Section 9.3's fifth recommendation says a phone-based tracker "will not be used if
it runs slowly." Parameter count is a proxy for size, not speed, so this is
measured directly by `src/efficiency_benchmark.py`.

Reference run (2-thread x86 CPU; **re-run on the machine you quote in the report,
latency does not transfer between machines**):

| Model | Params | File size | ms / image | img / s |
|---|---|---|---|---|
| EfficientNetB0 (classifier) | 4,029,325 | 15.65 MB | 74.9 | 13.4 |
| YOLOv8n (detector) | 3,008,963 | 5.97 MB | 152.6 | 6.6 |

Two things worth stating:

- **The detector has fewer parameters but is 2.04x slower.** EfficientNetB0's
  depthwise separable convolutions give it a low parameter count relative to its
  actual compute. Reporting parameters alone would have implied the opposite
  conclusion — exactly the "metric that measures something merely adjacent to the
  task" that Section 7.3 warns against, citing Lee [25].
- **The detector's file is 2.6x smaller** (5.97 MB vs 15.65 MB), which matters
  more than latency for app download size.

The deployment trade is therefore: **2.04x latency and 0.38x file size, in
exchange for 5.5x lower calorie error on multi-item plates.** On those terms
detection is clearly worth it, and that is now a measured statement rather than an
assumed one.

## 9. Limitations to state

- Multi-item subset is **n = 25**. State it; claim no statistical significance.
- The image-level label rule (class of the largest annotated box) is a **design
  choice, not ground truth**. On multi-item plates top-1 accuracy is a weak metric
  **for both models** (0.280 / 0.320) because the "correct" label is a convention.
  M2, M3 and the confusion matrix are the informative measures there.
- Macro-F1 excludes the two classes with zero test support. Four of the fifteen
  scored classes have n ≤ 2, so their per-class F1 values are extremely noisy —
  Sambal's 0.000 rests on a single image. Report the pattern, not the individual
  values.
- `expected_kcal` was hand-assigned by a group member using standard portion
  sizes, not measured. "Table error" in §7 therefore bundles genuine portion
  variation with annotation judgment.
- 17 classes only. Anything else is undetectable, not misdetected.
- Classifier confidences in the demo are **not calibrated** (see §6).
- Demo failure mode is **framing, not resolution** — measured, table below.
- Latency figures are hardware-specific; quote the device.

**Demo behaviour on phone photos — measured, not assumed.**

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

Operational guidance for the demo: **fill the frame with the plate.** The cause is
the training distribution — every training image is a tightly cropped food photo,
so a wide table shot is far outside it.

## 10. Figures available

- `outputs/figures/fig_c1_calorie_error.png` — M3 calorie MAE, classifier vs
  detector, split all / single / multi. **The headline figure.**
- `outputs/figures/fig_c2_multilabel.png` — M2 multi-label F1, same split.
- `outputs/figures/fig_c3_confusion_matrix.png` — **classifier confusion matrix,
  row-normalised.** Pair it with Track A's detection matrix.
- Suggested table: §3 M3, §5 per-class F1, §7 decomposition, §8 inference cost.

Figures generated from a `--simulate-classifier` run are automatically watermarked
"SIMULATED PROXY — NOT FOR THE REPORT" and the title is prefixed. If you see that
stamp, the numbers are placeholders; re-run `compare.py` without the flag.

## 11. Detector provenance — verified, and one caveat for the team

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

## 12. Traceability to the literature review

Every claim Track C makes traces to a section of Part I. Cite these explicitly in
the report — an implementation that visibly answers its own review scores better
than one that merely runs.

| Review section | What it says | Where Track C discharges it |
|---|---|---|
| 9.3 rec. 2 | "use detection instead of classification… a classifier that only labels the image would not work well" | §3 M2/M3 — asserted by the review, **measured** here: 5.5x calorie error on multi-item plates |
| 9.3 rec. 4 | report "Top-1 accuracy, Macro-F1 (for rare dishes/Malaysian foods), and a confusion matrix" | §5 — all three |
| 7.3 / 9.1 | "not one of the ten papers reports a full confusion matrix… which foods get confused with which goes largely unanswered" | §5, `fig_c3` — the classification-side answer for Malaysian food |
| 7.2 (Niu et al. [23]) | accuracy "can conceal weak results on infrequent classes" | §5 — Macro-F1 0.554 vs Top-1 0.724, a 0.169 gap, exactly as predicted |
| 8.2 | fine-grained recognition fails on "similar looking items" and cooking styles | §5 — Char-Kuey-Teow → Hokkien-Mee, two dark fried noodle dishes |
| 2.4.3 / 8.4 | mixed dishes are "a significant challenge for conventional food classification"; single-label models struggle where "multiple food items can appear in one image" | §3, §5 — the classifier scores 0.000 F1 on every nasi lemak component |
| 8.5 / 9.3 rec. 5 | deployment cost; "will not be used if it runs slowly" | §8 — measured latency and file size, not parameter proxies |
| 7.3 (Lee [25]) | a metric earns its place only if it measures what the task cares about | §8 — parameters would have implied the wrong conclusion; latency was measured instead |
| 9.1 | "the biggest research gap is in culture. None of the ten reviewed papers use a dataset that contains Malaysian or Southeast Asian food" | All of Track C — every number here is on Malaysian food |

## 13. Reproduce

```bash
cd Track_C_Baseline_Demo
pip install -r requirements.txt

python src/derive_labels.py           # boxes -> one label per image
python src/train_classifier.py        # EfficientNetB0 fine-tune (~10 min on MPS)
python src/predict_classifier.py      # classifier output on the test split
python src/compare.py                 # M1 + M2 + M3, the comparison table
python src/classifier_diagnostics.py  # Macro-F1, per-class, confusion matrix
python src/error_decomposition.py     # table error vs detector error
python src/efficiency_benchmark.py    # latency, file size, throughput
python src/make_figures.py            # report figures

streamlit run app.py                  # the demo
```

Requires `best.pt` at the repo root and the Roboflow dataset unzipped anywhere in
the repo root (`config.py` locates it by finding `data.yaml`). Both are gitignored.
`outputs/` is gitignored and fully regenerable from the above.
