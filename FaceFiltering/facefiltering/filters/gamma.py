"""Gamma (power-law) correction via LUT."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Gamma"


def apply(bgr: np.ndarray, *, gamma: float = 1.0) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    g = max(clamp_float(gamma, 0.05, 5.0), 1e-6)
    inv = 1.0 / g
    table = ((np.arange(256, dtype=np.float64) / 255.0) ** inv * 255.0)
    table = np.clip(np.round(table), 0, 255).astype(np.uint8)
    return table[bgr]
