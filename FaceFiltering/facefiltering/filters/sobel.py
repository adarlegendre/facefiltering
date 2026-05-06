"""Sobel gradient magnitude (grayscale), normalized to uint8."""
from __future__ import annotations

import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.ops import convolve_gray, normalize_to_u8
from facefiltering.validate import ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Sobel (magnitude)"


def apply(bgr: np.ndarray, *, ksize: int = 3) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = odd_ksize(ksize, minimum=3, maximum=31)
    g = to_gray_u8(bgr)
    if k == 3:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    else:
        # Scale Sobel kernels by smoothing with Pascal-like weights.
        d = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        s = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        for _ in range((k - 3) // 2):
            d = np.convolve(d, [1.0, 1.0])
            s = np.convolve(s, [1.0, 1.0])
        kx = np.outer(s, d)
        ky = np.outer(d, s)
    gx = convolve_gray(g, kx, pad_mode="edge")
    gy = convolve_gray(g, ky, pad_mode="edge")
    mag = np.hypot(gx, gy)
    if not np.isfinite(mag).all():
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
    return to_bgr_from_gray(normalize_to_u8(mag))
