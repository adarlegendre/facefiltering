"""Bloom effect: threshold bright regions, blur, and add as glow."""
from __future__ import annotations

import numpy as np

from facefiltering.gray import to_gray_u8
from facefiltering.ops import convolve_bgr, gaussian_kernel
from facefiltering.validate import clamp_float, clamp_int, ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Bloom"


def apply(
    bgr: np.ndarray,
    *,
    threshold: int = 180,
    sigma: float = 2.5,
    intensity: float = 0.7,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    t = clamp_int(threshold, 0, 255)
    s = clamp_float(sigma, 0.1, 20.0)
    a = clamp_float(intensity, 0.0, 3.0)

    g = to_gray_u8(bgr)
    bright_mask = (g >= t).astype(np.float64)[:, :, None]
    bright = bgr.astype(np.float64) * bright_mask

    k = odd_ksize(int(2 * round(3 * s) + 1), minimum=3, maximum=51)
    glow = convolve_bgr(bright, gaussian_kernel(k, s), pad_mode="edge")
    out = bgr.astype(np.float64) + a * glow
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

