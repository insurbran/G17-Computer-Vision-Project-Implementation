"""OpenCV food-region segmentation inside a detection bounding box.

Novelty for Part B: portion size from *segmented food pixels*, not raw box
area. Primary method GrabCut (cv2.grabCut) initialised from the box; falls
back to HSV saturation + Otsu thresholding when GrabCut degenerates.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegResult:
    mask: np.ndarray          # uint8 {0,1}, same size as the box crop
    food_area_px: int
    box_area_px: int
    food_frac: float          # food_area / box_area
    method: str               # "grabcut" | "hsv_otsu" | "box"


def _clip_bbox(bbox, w, h):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2


def _grabcut_mask(crop: np.ndarray, iters: int = 5) -> np.ndarray:
    """GrabCut with a rect slightly inside the crop as 'probable foreground'."""
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    m = max(2, int(0.05 * min(h, w)))              # 5% margin = probable bg
    rect = (m, m, w - 2 * m, h - 2 * m)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def _hsv_otsu_mask(crop: np.ndarray) -> np.ndarray:
    """Fallback: food is usually more saturated than plate/table background."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, mask = cv2.threshold(sat, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def _clean(mask: np.ndarray) -> np.ndarray:
    """Morphological open/close, then keep only the largest connected blob."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def segment_food(image: np.ndarray, bbox, use_grabcut: bool = True) -> SegResult:
    """Segment the food region inside `bbox` ([x1,y1,x2,y2] absolute px)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    box_area = max(1, (x2 - x1) * (y2 - y1))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return SegResult(np.ones(crop.shape[:2], np.uint8), box_area, box_area, 1.0, "box")

    method = "hsv_otsu"
    mask = None
    if use_grabcut:
        try:
            mask = _clean(_grabcut_mask(crop))
            method = "grabcut"
        except cv2.error:
            mask = None
    frac = float(mask.mean()) if mask is not None else 0.0
    if mask is None or frac < 0.05 or frac > 0.98:   # degenerate -> fallback
        mask = _clean(_hsv_otsu_mask(crop))
        method = "hsv_otsu"
        frac = float(mask.mean())
    if frac < 0.05 or frac > 0.98:                   # still degenerate -> box
        mask = np.ones(crop.shape[:2], np.uint8)
        method = "box"
        frac = 1.0

    food_area = int(mask.sum())
    return SegResult(mask, food_area, box_area, food_area / box_area, method)


def draw_overlay(image: np.ndarray, bbox, seg: SegResult, label: str = "") -> np.ndarray:
    """Debug/report figure: green mask overlay + box + label."""
    out = image.copy()
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    roi = out[y1:y2, x1:x2]
    if seg.mask.shape[:2] == roi.shape[:2]:
        green = np.zeros_like(roi); green[:, :, 1] = 255
        m = seg.mask.astype(bool)
        roi[m] = cv2.addWeighted(roi, 0.5, green, 0.5, 0)[m]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if label:
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return out
