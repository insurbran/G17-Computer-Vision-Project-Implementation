"""
app.py -- Track C demo (G17)

Streamlit demo: upload a food photo, get boxes and a calorie total, and see the
single-label classifier baseline fail next to it on the same image.

    streamlit run app.py

Three panels, one upload:
    1. Detection      -- Track A's YOLOv8n, boxes + confidences
    2. Calories       -- Track B's OpenCV segmentation + calorie table
    3. Classifier     -- Track C's EfficientNetB0 baseline, one label only

Panel 3 is the argument. On a nasi lemak plate the detector enumerates rice,
egg, peanuts, sambal, anchovies and cucumber and adds them up; the classifier
says one word and can only ever cost out one item. Showing both on the same
photo, live, makes the point better than any table.

WHAT THIS DEMO DOES NOT CLAIM
-----------------------------
Calorie accuracy on new photos. Ground truth exists only for the 76 test
images. On an arbitrary phone photo this reports an ESTIMATE with no error bar,
and the UI says so.
"""

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import config as cfg

# Track B's segmentation lives in a sibling folder with flat imports, so its
# own directory has to be on sys.path before we can import from it.
sys.path.insert(0, str(cfg.PART_B_SRC))

st.set_page_config(page_title="G17 Malaysian Food Calorie Estimator",
                   page_icon=None, layout="wide")

# Track B's REF_FOOD_FRAC is a module-level global that its main() mutates via
# `global`. Importing it and relying on its value is fragile, so the demo keeps
# its own explicit copy of the calculation. Same formula, no hidden state.
DEFAULT_REF_FRAC = 0.65
MIN_FACTOR, MAX_FACTOR = 0.5, 1.6


def portion_factor(food_frac: float, ref_frac: float) -> float:
    return max(MIN_FACTOR, min(MAX_FACTOR, food_frac / max(1e-6, ref_frac)))


# --- loading (cached so a re-render does not reload a model) ---------------

@st.cache_resource(show_spinner="Loading detector...")
def load_detector():
    from ultralytics import YOLO
    return YOLO(str(cfg.WEIGHTS))


@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    import torch
    sys.path.insert(0, str(cfg.HERE / "src"))
    from dataset import pick_device
    from train_classifier import build_model

    device = pick_device()
    ckpt = torch.load(cfg.CLASSIFIER_CKPT, map_location=device, weights_only=False)
    model = build_model(cfg.NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, device, ckpt.get("valid_acc")


@st.cache_data
def load_calorie_table():
    import csv
    with open(cfg.CALORIE_TABLE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ({r["class_name"]: float(r["kcal_per_portion"]) for r in rows},
            {r["class_name"]: r["portion_desc"] for r in rows})


def classify(model, device, pil_image):
    import torch
    from dataset import build_transforms
    tensor = build_transforms(train=False)(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    top3 = probs.topk(k=min(3, cfg.NUM_CLASSES))
    return [(cfg.CLASS_NAMES[i], float(s))
            for i, s in zip(top3.indices.tolist(), top3.values.tolist())]


# --- UI --------------------------------------------------------------------

st.title("Malaysian Food Calorie Estimator")
st.caption("G17 · CSC3014 · Detection (Track A) → Calories (Track B) → "
           "Baseline & Demo (Track C)")

with st.sidebar:
    st.header("Settings")
    conf = st.slider(
        "Detection confidence", 0.05, 0.95, cfg.CONF_THRESHOLD, 0.05,
        help="0.40 is what Track B used for its reported numbers. Change it and "
             "the totals stop being comparable to the report.")
    method = st.radio(
        "Portion method", ["segment", "box"], index=0,
        help="box = every item at standard portion (Track B's baseline, "
             "MAE 110.7 kcal). segment = OpenCV-measured food area "
             "(MAE 169.8 kcal — worse; reported as a negative result).")
    ref_frac = st.slider(
        "Reference food fraction", 0.30, 0.95, DEFAULT_REF_FRAC, 0.005,
        disabled=(method == "box"),
        help="Measured median inside a detection box is 0.574.")
    show_classifier = st.checkbox("Show classifier baseline", value=True)

    st.divider()
    st.caption("Estimates only. Ground truth exists for the 76 test images "
               "and nowhere else — treat any number here as indicative.")

# Preflight: fail with instructions, not a stack trace.
missing = []
if not cfg.WEIGHTS.exists():
    missing.append(
        f"**Detector weights** not found at `{cfg.WEIGHTS}`. Ask Oscar for "
        "`runs/detect/baseline3/weights/best.pt`. `predictions.json` is not a "
        "substitute — it only holds answers for the 76 test images.")
if show_classifier and not cfg.CLASSIFIER_CKPT.exists():
    missing.append(
        f"**Classifier** not found at `{cfg.CLASSIFIER_CKPT}`. Run "
        "`python src/derive_labels.py` then `python src/train_classifier.py`, "
        "or untick *Show classifier baseline* in the sidebar.")
if missing:
    for message in missing:
        st.error(message)
    st.stop()

uploaded = st.file_uploader("Upload a photo of a meal",
                            type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded is None:
    st.info("Upload a photo to begin. Multi-item plates such as nasi lemak "
            "show the difference between the two approaches most clearly.")
    st.stop()

import cv2  # imported late so the missing-file errors above render fast
from segmentation import draw_overlay, segment_food

pil_image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
kcal_table, portion_desc = load_calorie_table()

results = load_detector().predict(bgr, conf=conf, verbose=False)[0]
names = results.names

detections = []
if results.boxes is not None:
    for box in results.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append({"class": names[int(box.cls.item())],
                           "conf": float(box.conf.item()),
                           "xyxy": [x1, y1, x2, y2]})
detections.sort(key=lambda d: d["conf"], reverse=True)

col_detect, col_calories = st.columns(2)

with col_detect:
    st.subheader("1 · Detection")
    boxed = bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["xyxy"])
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (165, 110, 59), 2)  # C_DETECT, BGR
        cv2.putText(boxed, f"{det['class']} {det['conf']:.2f}",
                    (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (165, 110, 59), 2)
    st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.metric("Items detected", len(detections))

if not detections:
    st.warning(
        f"Nothing detected above confidence {conf:.2f}. Either the dish is "
        "outside the 17 trained classes, or the photo is far from the training "
        "distribution — the model was trained on 320×320 images, so large "
        "phone photos can behave differently. Try lowering the confidence.")
    st.stop()

# --- calories --------------------------------------------------------------

overlay = bgr.copy()
line_items, total = [], 0.0
for det in detections:
    cls = det["class"]
    if cls not in kcal_table:
        continue
    if method == "segment":
        seg = segment_food(bgr, det["xyxy"])
        factor = portion_factor(seg.food_frac, ref_frac)
        frac, seg_method = seg.food_frac, seg.method
    else:
        seg, factor, frac, seg_method = None, 1.0, 1.0, "box"
    kcal = kcal_table[cls] * factor
    total += kcal
    line_items.append({
        "Item": cls,
        "Portion": portion_desc.get(cls, ""),
        "Confidence": round(det["conf"], 3),
        "Food fraction": round(frac, 3),
        "Portion factor": round(factor, 3),
        "kcal": round(kcal, 1),
    })
    if seg is not None:
        overlay = draw_overlay(overlay, det["xyxy"], seg, f"{cls} {kcal:.0f}kcal")

with col_calories:
    st.subheader("2 · Calories")
    if method == "segment":
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Green = OpenCV-measured food area. Red = detection box.")
    else:
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption("Box method: every detected item counted at standard portion.")
    st.metric("Estimated total", f"{total:.0f} kcal")

st.dataframe(line_items, use_container_width=True, hide_index=True)

# --- classifier baseline ---------------------------------------------------

if show_classifier:
    st.divider()
    st.subheader("3 · Single-label classifier baseline")

    model, device, valid_acc = load_classifier()
    top3 = classify(model, device, pil_image)
    top_class, top_conf = top3[0]
    clf_kcal = kcal_table.get(top_class, 0.0)

    left, right = st.columns([1, 2])
    with left:
        st.metric("Classifier says", top_class, f"{top_conf:.1%} confident")
        st.metric("Its calorie estimate", f"{clf_kcal:.0f} kcal")
        if valid_acc:
            st.caption(f"EfficientNetB0, validation accuracy {valid_acc:.1%}")
    with right:
        st.write("**Top 3 guesses**")
        st.dataframe(
            [{"Class": c, "Confidence": round(s, 4)} for c, s in top3],
            use_container_width=True, hide_index=True)

    distinct = len({d["class"] for d in detections})
    difference = total - clf_kcal
    if distinct > 1:
        st.error(
            f"**The structural limitation, on this photo.** The detector found "
            f"{distinct} distinct food types and totalled **{total:.0f} kcal**. "
            f"The classifier can only name one — it said *{top_class}* and can "
            f"cost out **{clf_kcal:.0f} kcal**, a gap of **{difference:+.0f} kcal**. "
            f"It is not that the classifier guessed wrong; it has one output slot "
            f"and this plate has {distinct} things on it.")
    else:
        st.success(
            f"**Single-item plate — the fair fight.** Only one food type here, so "
            f"the classifier is on equal footing: it said *{top_class}* "
            f"({clf_kcal:.0f} kcal) against the detector's {total:.0f} kcal. "
            f"This is the control case. The gap opens up on multi-item plates.")

st.divider()
st.caption(
    "Detector: YOLOv8n, test mAP@50 0.872 (Track A). Calories: cited lookup "
    "table, KKM BeSS and HPB, scaled by OpenCV-measured food area (Track B, "
    "box-method MAE 110.7 kcal). Baseline: EfficientNetB0 (Track C). "
    "Dataset: Roboflow 'Malaysian Food Recognition 2' v7, CC BY 4.0. "
    "17 classes only — anything else cannot be detected.")
