"""
Gaussian blur (low-pass). Kernel (0,0) lets OpenCV derive size from sigma;
otherwise use a square odd kernel size.
"""
from __future__ import annotations

import cv2

from facefiltering.validate import clamp_float, ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Gaussian blur"


def apply(bgr: np.ndarray, *, sigma: float = 1.0, ksize: int = 0) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    sigma = clamp_float(sigma, 0.05, 50.0)
    k = int(ksize)
    if k <= 0:
        ksz = (0, 0)
    else:
        k = odd_ksize(k, minimum=3, maximum=31)
        ksz = (k, k)
    return cv2.GaussianBlur(bgr, ksz, sigmaX=sigma, sigmaY=sigma)
