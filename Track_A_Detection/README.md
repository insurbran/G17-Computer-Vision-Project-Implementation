# Track A - YOLO Food Detector

Detects 17 Malaysian food classes using YOLOv8n (Ultralytics).

Classes: Anchovies, Boiled-Egg, Char-Kuey-Teow, Chicken-Rendang, Curry-Puff,
Fried-Chicken, Fried-Egg, Fried-Rice, Hokkien-Mee, Lo-Mein, Mee-Rebus, Mee-Siam,
Peanuts, Rice, Roti-Canai, Sambal, Slices-Cucumber.

## Scripts

- `clahe_prep.py` - build a CLAHE-preprocessed copy of the dataset (the modification)
- `emit_predictions.py` - run the detector, write predictions.json for Tracks B and C
- `aggregate_runs.py` - evaluate all runs on the test set, build the comparison table
- `make_figures.py` - generate the report figures from test_results.json

## Results

Baseline test mAP@50 = 0.872, precision 0.884, recall 0.717.
Roboflow's own model on this dataset reports 0.871 mAP@50, so the baseline matches
the published benchmark.

CLAHE modification, verified across 3 seeds per condition: no reliable change in
mAP, a small but consistent drop in precision, no reliable effect on recall. The
precision-for-recall trade seen at a single seed did not survive replication. A
reported negative result.

## Dataset

Roboflow Universe, "Malaysian Food Recognition 2", version 7, CC BY 4.0.
https://universe.roboflow.com/malaysian-dishes-classification/malaysian-food-recognition-2-0kwwr/dataset/7

757 source images, 17 classes. Splits: train 1593 (augmented), valid 150, test 76.
Augmentation is applied to the training set only, so nothing leaks across splits.

## How to reproduce

1. Download the dataset from the Roboflow link above (YOLOv8 format).
2. Fix the paths in its data.yaml to point at your local dataset folder.
3. Train: `yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 patience=25 seed=42 device=0 amp=False workers=0`
4. Build predictions: `python emit_predictions.py`

Note: `amp=False` and `workers=0` are required on Windows with the 40-series GPU
used here. Leaving them at defaults caused intermittent CUDA crashes mid-training.

## Not in this repo

Dataset, trained weights (best.pt), the CLAHE dataset copy, and the runs/ folders
are all excluded. They are large and fully regenerable from the scripts above.
