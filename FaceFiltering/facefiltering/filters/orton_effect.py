"""Orton effect: soft glow + contrast compression stylization."""
from __future__ import annotations

import numpy as np

from facefiltering.ops import convolve_bgr, gaussian_kernel
from facefiltering.validate import clamp_float, ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Orton effect"


def apply(
    bgr: np.ndarray,
    *,
    sigma: float = 2.0,
    strength: float = 0.6,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    s = clamp_float(sigma, 0.1, 20.0)
    a = clamp_float(strength, 0.0, 1.0)

    k = odd_ksize(int(2 * round(3 * s) + 1), minimum=3, maximum=51)

    src = bgr.astype(np.float64)
    blur = convolve_bgr(src, gaussian_kernel(k, s), pad_mode="edge")

    # Classic screen blend gives the characteristic soft glow.
    screen = 255.0 - ((255.0 - src) * (255.0 - blur) / 255.0)
    out = (1.0 - a) * src + a * screen
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
