"""Sobel gradient magnitude (grayscale), normalized to uint8."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.validate import ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Sobel (magnitude)"


def apply(bgr: np.ndarray, *, ksize: int = 3) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = odd_ksize(ksize, minimum=3, maximum=31)
    g = to_gray_u8(bgr)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    if not np.isfinite(mag).all():
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    if mag_u8 is None:
        mag_u8 = np.zeros_like(g, dtype=np.uint8)
    return to_bgr_from_gray(mag_u8.astype(np.uint8))
