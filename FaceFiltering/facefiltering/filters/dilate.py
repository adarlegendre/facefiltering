"""Morphological dilation with elliptical structuring element."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.validate import clamp_int, ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Morphological dilation"


def apply(bgr: np.ndarray, *, ksize: int = 5, iterations: int = 1) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = odd_ksize(ksize, minimum=3, maximum=31)
    it = clamp_int(iterations, 1, 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(bgr, kernel, iterations=it)
