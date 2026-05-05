"""Histogram equalization on Y channel (color) or full image (gray)."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.validate import ensure_bgr_u8

DISPLAY_NAME = "Histogram equalization"


def apply(bgr: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
