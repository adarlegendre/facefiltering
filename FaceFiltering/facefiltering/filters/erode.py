"""Morphological erosion with elliptical structuring element."""
from __future__ import annotations

import numpy as np

from facefiltering.ops import elliptical_kernel, morphology_bgr
from facefiltering.validate import clamp_int, ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Morphological erosion"


def apply(bgr: np.ndarray, *, ksize: int = 5, iterations: int = 1) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = odd_ksize(ksize, minimum=3, maximum=31)
    it = clamp_int(iterations, 1, 20)
    kernel = elliptical_kernel(k)
    return morphology_bgr(bgr, kernel, op="erode", iterations=it)

