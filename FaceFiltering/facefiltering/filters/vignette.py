"""Vignette effect: radial darkening from center to edges."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Vignette"


def apply(
    bgr: np.ndarray,
    *,
    strength: float = 0.6,
    radius_ratio: float = 0.9,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    a = clamp_float(strength, 0.0, 1.0)
    rr = clamp_float(radius_ratio, 0.2, 2.0)

    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5

    xn = (xx - cx) / max(cx, 1.0)
    yn = (yy - cy) / max(cy, 1.0)
    r2 = xn * xn + yn * yn

    sigma = max(rr, 1e-6)
    radial = np.exp(-r2 / (2.0 * sigma * sigma))
    mask = (1.0 - a) + a * radial

    out = bgr.astype(np.float64) * mask[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
