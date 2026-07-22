# Part B (Calorie Estimation) — everything the report needs

Written for whoever drafts the Part II report. All numbers below are final and
reproducible from this folder.

## 1. What Part B does

Takes Track A's detections (`from_A/predictions.json`, 76 test images, conf ≥ 0.4
used → 217 detections), maps each detected item to a calorie value from a cited
lookup table, scales portion by OpenCV-measured food area, sums per image, and
evaluates against human-annotated ground truth.

## 2. The modification (novelty claim)

Naive baseline: portion ∝ bounding-box area (every box counted at standard
portion, factor 1.0). Our modification: segment the actual food region inside
each box with OpenCV — GrabCut (`cv2.grabCut`) initialised from the box,
falling back to HSV-saturation + Otsu thresholding (`cv2.threshold` +
`cv2.morphologyEx` + `cv2.connectedComponentsWithStats`) when GrabCut
degenerates — and scale the portion by measured food fraction.
This targets the criticism of box-area portion estimation in our lit review §8.3.

## 3. Headline results (76 images)

| Method | MAE (kcal) | MAPE (%) | Bias (kcal) |
|---|---|---|---|
| Box area (baseline) | 110.7 | 16.7 | +25.8 |
| Segmented area (ref=0.65) | 169.8 | 26.5 | −63.6 |
| Segmented area, calibrated (ref=0.574) | 174.0 | 28.4 | +8.4 |

**Segmentation did NOT improve MAE. Report this as a negative result** — the
brief explicitly allows it if analysed. The analysis (§5) is the interesting part.

## 4. Supporting numbers

- Segmentation method usage over 217 detections: GrabCut 105, HSV/Otsu fallback
  110, degenerate-mask box fallback 2. (GrabCut fails ~half the time because
  test crops are small — images are 320×320.)
- Median food fraction inside a detection box: 0.574 (p25 0.421, p75 0.669).
  So ~43% of a typical box is not food — the premise of the modification is
  real, boxes do overestimate food area.
- Calibrating the reference fraction from 0.65 → 0.574 removed the systematic
  bias (−63.6 → +8.4) but did not reduce MAE.
- Per-image errors: `outputs/real/per_image_errors.csv`.

## 5. Why segmentation didn't win (use this analysis)

1. **Ground truth favours the baseline by construction.** GT was built by
   counting visible items at standard portion sizes (true weights are
   unobservable from photos). The box method also counts items at standard
   portions, so it matches the GT methodology; segmentation adds an area signal
   the GT cannot reward.
2. **Segmentation noise at low resolution.** At 320×320, per-box crops are
   small; GrabCut degenerated on ~half. The area signal carries noise that
   directly multiplies the calorie estimate.
3. **Dominant errors are detection errors, not portion errors.** Worst 5 images
   (from per_image_errors.csv): missed curry puffs (GT 12, detected 7 → −648),
   overlapping roti canai undercounted (GT 3, detected 1 → −615), out-of-vocab
   maggi goreng detected as two noodle dishes (+604), fried rice image with no
   confident detection (−576), hokkien mee misclassified as boiled egg (−444).
   No portion method can fix identification errors. (This is also useful
   feedback for Track A's error analysis.)

Honest-limitations paragraph for the report: GT is estimated by a human rater
using standard portion sizes (cited databases), not measured weights; the
calibrated reference fraction was tuned on the test set because Track A's
validation-set predictions/weights were not available (state this); two test
images contain dishes outside the 17-class vocabulary (bihun goreng, maggi
goreng), for which GT used closest cited values.

## 6. Calorie table + citations (data/calorie_table.csv)

Every class has value, portion description, and source. Main sources to cite:

- KKM (Ministry of Health Malaysia) BeSS "Senarai Kalori Bagi Makanan dan
  Minuman": https://ecentral.my/wp-content/uploads/2022/12/BeSS_Bank-Calorie-Bahagian-Pemakanan.pdf
  (rendang chicken 250/plate, fried chicken 270/piece, curry puff 130, roti canai 300)
- Singapore HPB iDAT hawker values (via danielfooddiary.com/2015/09/28/calories/):
  char kuey teow 745, mee siam 694, mee rebus 571, hokkien mee 522, fried rice 576
- HPB nasi lemak component breakdown (ofnoah.sg/blog/calories-in-nasi-lemak):
  anchovies 60, peanuts 90, sambal 80
- Weakest value: Lo-Mein 450 (sources range 310–490) — acknowledge in report.
- Hokkien Mee varies by style (SG 522 vs KL dark style ~700–1000) — acknowledge.

## 7. Ground truth methodology (for the report's evaluation section)

76 test images. Ground truth kcal assigned by group member (B) by viewing each
image: item lists initialised from detector output, then manually corrected
(28/76 images corrected — wrong dish identity, wrong counts, items already
included in dish, out-of-vocab dishes); kcal = sum of per-item cited standard
portions, adjusted for visibly non-standard portions. Correction log is in
`data/ground_truth.csv` (`portion_notes` + `remakrs` columns).

## 8. Figures available

- `outputs/real/overlays_segment/*.jpg` — segmentation overlays (green = measured
  food area, red = detection box + kcal label). Good picks:
  `fried27_*.jpg` (nasi lemak, many components), `21_jpg.rf.9c0a*` (curry puff
  miscount), `36_jpg*` (out-of-vocab maggi goreng).
- Suggested table: the 3-method comparison in §3.
- Suggested figure: histogram of food_frac (data in `outputs/real/detail_segment.json`).

## 9. Reproduce

```
pip install -r requirements.txt
# images: Roboflow "Malaysian Food Recognition 2" v7, test/images -> dataset/test/images
python src/calorie_estimator.py --predictions from_A/predictions.json --images dataset/test/images --out outputs/real --method box
python src/calorie_estimator.py --predictions from_A/predictions.json --images dataset/test/images --out outputs/real --method segment --save-overlays
python src/evaluate.py --ground-truth data/ground_truth.csv --estimates outputs/real/estimates_box.csv outputs/real/estimates_segment.csv
```
