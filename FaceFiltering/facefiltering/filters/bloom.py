"""Bloom effect: threshold bright regions, blur, and add as glow."""
from __future__ import annotations  # newer type hints

import numpy as np  # pixel arrays / math

from facefiltering.gray import to_gray_u8  # grayscale for “bright” mask

from facefiltering.ops import convolve_bgr, gaussian_kernel  # Gaussian blur on color

from facefiltering.validate import (
    clamp_float,  # clamp sigma, intensity
    clamp_int,  # clamp threshold 0–255
    ensure_bgr_u8,  # valid BGR uint8 image
    odd_ksize,  # odd blur kernel size
)

DISPLAY_NAME = "Bloom"


def apply(
    bgr: np.ndarray,
    *,
    threshold: int = 180,
    sigma: float = 2.5,
    intensity: float = 0.7,
) -> np.ndarray:
    # bright mask → blur → add glow
    bgr = ensure_bgr_u8(bgr)  # validate image
    t = clamp_int(threshold, 0, 255)  # threshold range
    s = clamp_float(sigma, 0.1, 20.0)  # blur width
    a = clamp_float(intensity, 0.0, 3.0)  # glow strength

    g = to_gray_u8(bgr)  # luminance
    bright_mask = (g >= t).astype(np.float64)[:, :, None]  # bright = 1
    bright = bgr.astype(np.float64) * bright_mask  # drop dark pixels

    k = odd_ksize(int(2 * round(3 * s) + 1), minimum=3, maximum=51)  # kernel from sigma
    glow = convolve_bgr(bright, gaussian_kernel(k, s), pad_mode="edge")  # soften bright layer
    out = bgr.astype(np.float64) + a * glow  # add glow
    return np.clip(np.round(out), 0, 255).astype(np.uint8)  # to bytes 0–255

