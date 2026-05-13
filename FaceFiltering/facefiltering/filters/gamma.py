"""Gamma (power-law) correction via LUT."""
from __future__ import annotations  # newer type hints

import numpy as np  # LUT math + table[bgr] lookup

from facefiltering.validate import (
    clamp_float,  # clamp gamma
    ensure_bgr_u8,  # valid BGR uint8 image
)

DISPLAY_NAME = "Gamma"


def apply(bgr: np.ndarray, *, gamma: float = 1.0) -> np.ndarray:
    # LUT: map each level 0–255 through gamma curve
    bgr = ensure_bgr_u8(bgr)  # validate image
    g = max(clamp_float(gamma, 0.05, 5.0), 1e-6)  # gamma > 0
    inv = 1.0 / g  # exponent 1/gamma
    table = ((np.arange(256, dtype=np.float64) / 255.0) ** inv * 255.0)  # curve samples
    table = np.clip(np.round(table), 0, 255).astype(np.uint8)  # uint8 LUT
    return table[bgr]  # apply LUT per channel
