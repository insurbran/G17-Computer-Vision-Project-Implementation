# Track B — Calorie Estimation (Malaysian Food Detector, G17)

Reads Track A's detector output (`from_A/predictions.json`), segments the
actual food region inside each detection box with OpenCV, maps items to cited
calorie values scaled by measured food area, sums per image, and evaluates
MAE/MAPE against human-annotated ground truth.

## Method

- Baseline: each detected item counted at standard portion (box area, factor 1.0).
- Modification: OpenCV food segmentation inside each box — `cv2.grabCut`
  initialised from the box, HSV-saturation + Otsu fallback
  (`cv2.threshold`, `cv2.morphologyEx`, `cv2.connectedComponentsWithStats`)
  when GrabCut degenerates — portion scaled by measured food fraction.

## Results (76 test images)

| Method | MAE (kcal) | MAPE (%) | Bias |
|---|---|---|---|
| Box area (baseline) | 110.7 | 16.7 | +25.8 |
| Segmented area | 169.8 | 26.5 | −63.6 |
| Segmented area, calibrated | 174.0 | 28.4 | +8.4 |

Segmentation is a reported negative result: median food fraction inside boxes
is 0.574 (boxes really do over-cover), but the dominant errors are detection
errors (miscounts, misclassifications), which portion refinement cannot fix,
and the ground-truth methodology (items × standard portions) structurally
matches the baseline. Full analysis in `REPORT_NOTES.md` §5.

## Files

- `src/calorie_estimator.py` — main pipeline (`--method box|segment`, `--ref-frac`, `--save-overlays`)
- `src/segmentation.py` — GrabCut/HSV food-area measurement
- `src/evaluate.py` — MAE/MAPE + worst-case analysis
- `src/make_fake_predictions.py` — synthetic data generator used before Track A delivered
- `data/calorie_table.csv` — 17-class kcal lookup with per-value citations (KKM BeSS, HPB)
- `data/ground_truth.csv` — 76-image human-annotated ground truth incl. correction log
- `from_A/` — Track A's predictions.json + format notes
- `schema/predictions.schema.json` — the A↔B/C interface contract
- `outputs/real/` — estimates, per-image errors, segmentation overlay figures
- `REPORT_NOTES.md` — everything the report writer needs (numbers, analysis, citations)

## Reproduce

```
pip install -r requirements.txt
# dataset (gitignored): Roboflow "Malaysian Food Recognition 2" v7 (CC BY 4.0),
# put test/images into dataset/test/images
python src/calorie_estimator.py --predictions from_A/predictions.json --images dataset/test/images --out outputs/real --method box
python src/calorie_estimator.py --predictions from_A/predictions.json --images dataset/test/images --out outputs/real --method segment --save-overlays
python src/evaluate.py --ground-truth data/ground_truth.csv --estimates outputs/real/estimates_box.csv outputs/real/estimates_segment.csv
```
