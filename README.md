# Malaysian Food Detection & Calorie Estimation

CSC3014 Computer Vision — Project Part II, Group 17.

A computer vision pipeline that detects 17 Malaysian food classes in a photo,
segments the actual food region inside each detection with OpenCV, and estimates
the total calorie content of the plate. A single-label classifier baseline is
trained on the same data to demonstrate why a detection architecture is
structurally necessary for multi-item Malaysian plates such as nasi lemak.

```
photo --> [Track A] YOLOv8n detector (+ CLAHE experiment) --> predictions.json
                                                                   |
                                                                   v
          [Track B] OpenCV GrabCut/Otsu food segmentation --> per-item kcal --> total
                    (evaluated against box-area baseline, MAE/MAPE)
                                                                   |
          [Track C] EfficientNetB0 classifier baseline + Streamlit demo
                    (detector-vs-classifier comparison, error decomposition)
```

## Project structure

```
├── Track_A_Detection/            YOLOv8n training, CLAHE experiment, prediction export
│   ├── clahe_prep.py             build CLAHE-preprocessed copy of the dataset
│   ├── emit_predictions.py       run detector, write predictions.json for B and C
│   ├── aggregate_runs.py         evaluate all runs, build comparison table
│   └── make_figures.py           report figures
├── Track_B_Calories/             calorie estimation with OpenCV portion segmentation
│   ├── src/segmentation.py       GrabCut + HSV/Otsu food-area measurement
│   ├── src/calorie_estimator.py  predictions.json -> per-image kcal (box | segment)
│   ├── src/evaluate.py           MAE / MAPE / per-image error analysis
│   ├── data/calorie_table.csv    17-class kcal lookup, per-value citations (KKM, HPB)
│   ├── data/ground_truth.csv     human-annotated kcal for the 76 test images
│   ├── from_A/predictions.json   frozen detector output (the A->B/C interface)
│   ├── schema/                   JSON schema of that interface
│   └── outputs/real/             final estimates, per-image errors, overlay figures
└── Track_C_Baseline_Demo/        classifier baseline + comparison + demo
    ├── src/train_classifier.py   fine-tune EfficientNetB0 (labels = largest box)
    ├── src/compare.py            classifier vs detector on the same test set
    ├── src/classifier_diagnostics.py / error_decomposition.py / efficiency_benchmark.py
    └── app.py                    Streamlit demo: upload photo -> boxes -> calories
```

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+. For the Track C classifier, install the torch build matching your
hardware first (https://pytorch.org/get-started/locally/) — the default pip
package is CPU-only.

## Data and weights (not in repo, both regenerable)

1. **Dataset** — Roboflow Universe "Malaysian Food Recognition 2" v7 (CC BY 4.0):
   https://universe.roboflow.com/malaysian-dishes-classification/malaysian-food-recognition-2-0kwwr/dataset/7
   Download in YOLOv8 format and unzip anywhere in the repo root. Track C
   auto-locates it (searches for a folder containing `data.yaml` and `test/images`);
   for Tracks A/B, pass the path on the command line as shown below.
   Splits: 1,593 train (augmented) / 150 valid / 76 test.
2. **Detector weights `best.pt`** — place in the repo root. Either retrain
   (Track A instructions below, ~30 min on GPU) or use the group's shared weights.
   Note: `Track_B_Calories/from_A/predictions.json` is the frozen detector output
   for the 76 test images, so Tracks B and C evaluations run WITHOUT the weights;
   the weights are only needed to process new photos (the demo).

## How to run

### Track A — detection

```
yolo detect train data=<dataset>/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 patience=25 seed=42 amp=False workers=0
python Track_A_Detection/emit_predictions.py --weights <path-to-best.pt> --source <dataset>/test/images --out predictions.json
```

`amp=False workers=0` avoids intermittent CUDA crashes on Windows. For the CLAHE
experiment, first build the preprocessed dataset copy with
`python Track_A_Detection/clahe_prep.py`, train on it with the same command, then
compare runs with `python Track_A_Detection/aggregate_runs.py`.

### Track B — calorie estimation (works out of the box, no weights needed)

```
python Track_B_Calories/src/calorie_estimator.py --predictions Track_B_Calories/from_A/predictions.json --images <dataset>/test/images --out Track_B_Calories/outputs/real --method box
python Track_B_Calories/src/calorie_estimator.py --predictions Track_B_Calories/from_A/predictions.json --images <dataset>/test/images --out Track_B_Calories/outputs/real --method segment --save-overlays
python Track_B_Calories/src/evaluate.py --ground-truth Track_B_Calories/data/ground_truth.csv --estimates Track_B_Calories/outputs/real/estimates_box.csv Track_B_Calories/outputs/real/estimates_segment.csv
```

`--method box` is the naive baseline (portion = standard serving per detected
item); `--method segment` scales portions by the OpenCV-measured food fraction.
Segmentation overlays are written to `Track_B_Calories/outputs/real/overlays_segment/`.

### Track C — baseline classifier and comparison

```
python Track_C_Baseline_Demo/src/train_classifier.py
python Track_C_Baseline_Demo/src/compare.py
python Track_C_Baseline_Demo/src/classifier_diagnostics.py
python Track_C_Baseline_Demo/src/error_decomposition.py
python Track_C_Baseline_Demo/src/efficiency_benchmark.py
```

### Demo (needs `best.pt` in the repo root)

```
streamlit run Track_C_Baseline_Demo/app.py
```

Upload a food photo; the app shows detection boxes, the per-item and total
calorie estimate, and the classifier's single-label answer on the same image
for comparison.

## Results summary (76-image test set)

| Component | Result |
|---|---|
| Detector mAP@50 | 0.872–0.882 across 3 seeds (published dataset benchmark: 0.871) |
| CLAHE modification | negative result: no mAP change, small consistent precision drop |
| Calorie MAE (box baseline) | 110.7 kcal (16.7% MAPE) |
| Calorie MAE (segmentation) | 169.8 kcal — negative result, analysed in the report |
| Classifier top-1 (single-item) | 94.1% (beats detector's 82.4%) |
| Classifier calorie MAE (multi-item) | 545.7 kcal vs detector's 99.2 kcal (5.5x worse) |

Both modifications are reported negative results with full analysis in the
project report — the multi-seed CLAHE evaluation and the error decomposition
show the failure causes (background-texture false positives; segmentation noise
on small crops plus detection errors dominating the calorie error).

## Team

- Track A (detection): Firaz Wianda Faric Oscar
- Track B (calorie estimation): Chong Ting Fung
- Track C (baseline + demo): Teo Shae-Nern

## Data sources

- Dataset: Roboflow Universe "Malaysian Food Recognition 2" v7, CC BY 4.0
- Calorie values: KKM BeSS "Senarai Kalori Bagi Makanan dan Minuman" (MOH
  Malaysia); Singapore HPB iDAT hawker values — per-value citations in
  `Track_B_Calories/data/calorie_table.csv`
