"""Median blur (edge-preserving smoothing) on BGR."""
from __future__ import annotations

import numpy as np

from facefiltering.ops import median_blur_bgr
from facefiltering.validate import ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Median"


def apply(bgr: np.ndarray, *, ksize: int = 5) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = odd_ksize(ksize, minimum=3, maximum=15)
    return median_blur_bgr(bgr, k)
