"""Laplacian magnitude (absolute), normalized to uint8."""
from __future__ import annotations

import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.ops import convolve_gray, normalize_to_u8
from facefiltering.validate import ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Laplacian (abs)"


def apply(bgr: np.ndarray, *, ksize: int = 3) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = odd_ksize(ksize, minimum=3, maximum=31)
    g = to_gray_u8(bgr)
    if k <= 3:
        ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    else:
        ker = np.zeros((k, k), dtype=np.float64)
        c = k // 2
        ker[c, c] = -4.0
        ker[c - 1, c] = ker[c + 1, c] = 1.0
        ker[c, c - 1] = ker[c, c + 1] = 1.0
    lap = convolve_gray(g, ker, pad_mode="edge")
    lap = np.abs(lap)
    if not np.isfinite(lap).all():
        lap = np.nan_to_num(lap, nan=0.0, posinf=0.0, neginf=0.0)
    return to_bgr_from_gray(normalize_to_u8(lap))
