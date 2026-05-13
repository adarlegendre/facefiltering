"""Orton effect: soft glow + contrast compression stylization."""
from __future__ import annotations  # newer type hints

import numpy as np  # pixel arrays / math

from facefiltering.ops import convolve_bgr, gaussian_kernel  # Gaussian blur on color

from facefiltering.validate import (
    clamp_float,  # clamp sigma, strength
    ensure_bgr_u8,  # valid BGR uint8 image
    odd_ksize,  # odd blur kernel size
)

DISPLAY_NAME = "Orton effect"


def apply(
    bgr: np.ndarray,
    *,
    sigma: float = 2.0,
    strength: float = 0.6,
) -> np.ndarray:
    # blur → screen blend → mix with original
    bgr = ensure_bgr_u8(bgr)  # validate image
    s = clamp_float(sigma, 0.1, 20.0)  # blur width
    a = clamp_float(strength, 0.0, 1.0)  # effect amount

    k = odd_ksize(int(2 * round(3 * s) + 1), minimum=3, maximum=51)  # kernel from sigma

    src = bgr.astype(np.float64)  # float pixels
    blur = convolve_bgr(src, gaussian_kernel(k, s), pad_mode="edge")  # blurred image

    screen = 255.0 - ((255.0 - src) * (255.0 - blur) / 255.0)  # screen combine
    out = (1.0 - a) * src + a * screen  # blend sharp vs glow
    return np.clip(np.round(out), 0, 255).astype(np.uint8)  # to bytes 0–255
