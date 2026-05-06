"""Diffuse effect via random local jitter and blending."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_float, clamp_int, ensure_bgr_u8

DISPLAY_NAME = "Diffuse"


def apply(bgr: np.ndarray, *, radius: int = 3, mix: float = 1.0) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    r = clamp_int(radius, 1, 20)
    a = clamp_float(mix, 0.0, 1.0)

    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(12345)  # deterministic output per run
    ox = rng.integers(-r, r + 1, size=(h, w))
    oy = rng.integers(-r, r + 1, size=(h, w))
    sx = np.clip(xx + ox, 0, w - 1)
    sy = np.clip(yy + oy, 0, h - 1)
    diff = bgr[sy, sx]
    out = (1.0 - a) * bgr.astype(np.float64) + a * diff.astype(np.float64)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

