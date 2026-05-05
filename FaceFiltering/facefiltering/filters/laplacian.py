"""Laplacian magnitude (absolute), normalized to uint8."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.validate import ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Laplacian (abs)"


def apply(bgr: np.ndarray, *, ksize: int = 3) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = odd_ksize(ksize, minimum=3, maximum=31)
    g = to_gray_u8(bgr)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=k)
    lap = np.abs(lap)
    if not np.isfinite(lap).all():
        lap = np.nan_to_num(lap, nan=0.0, posinf=0.0, neginf=0.0)
    lap_u8 = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    return to_bgr_from_gray(lap_u8.astype(np.uint8))
