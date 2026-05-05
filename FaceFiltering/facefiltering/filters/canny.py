"""
Canny edge detector (grayscale).

OpenCV implementation with tunable thresholds and aperture size.
Output is returned as BGR (for consistent UI pipeline).
"""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.validate import clamp_int, ensure_bgr_u8

DISPLAY_NAME = "Canny edge detection"


def apply(
    bgr: np.ndarray,
    *,
    t1: int = 80,
    t2: int = 160,
    aperture: int = 3,
    l2gradient: bool = False,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    t1 = clamp_int(t1, 0, 500)
    t2 = clamp_int(t2, 0, 500)
    if t2 < t1:
        t1, t2 = t2, t1

    ap = int(aperture)
    if ap not in (3, 5, 7):
        ap = 3

    g = to_gray_u8(bgr)
    edges = cv2.Canny(g, threshold1=t1, threshold2=t2, apertureSize=ap, L2gradient=bool(l2gradient))
    return to_bgr_from_gray(edges)

