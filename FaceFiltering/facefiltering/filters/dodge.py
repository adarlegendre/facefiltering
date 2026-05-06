"""Dodge effect (highlight boost) using a stable color-dodge style mapping."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Dodge"


def apply(bgr: np.ndarray, *, strength: float = 0.55) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    s = clamp_float(strength, 0.0, 0.95)
    x = bgr.astype(np.float64)
    # Stable dodge-style mapping: stronger boost near bright tones.
    denom = 255.0 - s * x
    out = (x * 255.0) / np.maximum(denom, 1.0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

