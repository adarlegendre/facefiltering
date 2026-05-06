"""Unsharp masking: sharpen = original + amount * (original - GaussianBlur)."""
from __future__ import annotations

import numpy as np

from facefiltering.ops import convolve_bgr, gaussian_kernel
from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Unsharp mask"


def apply(bgr: np.ndarray, *, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    sigma = clamp_float(sigma, 0.05, 50.0)
    amount = clamp_float(amount, 0.0, 5.0)
    k = int(2 * round(3 * sigma) + 1)
    k = max(3, min(31, k | 1))
    blur = convolve_bgr(bgr, gaussian_kernel(k, sigma), pad_mode="edge")
    out = bgr.astype(np.float64) + amount * (bgr.astype(np.float64) - blur)
    return np.clip(out, 0, 255).astype(np.uint8)
