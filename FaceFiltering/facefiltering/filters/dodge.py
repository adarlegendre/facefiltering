"""Dodge effect (highlight boost) using a stable color-dodge style mapping."""
from __future__ import annotations  # newer type hints

import numpy as np  # pixel arrays / math

from facefiltering.validate import (
    clamp_float,  # clamp strength (safe denom)
    ensure_bgr_u8,  # valid BGR uint8 image
)

DISPLAY_NAME = "Dodge"


def apply(bgr: np.ndarray, *, strength: float = 0.55) -> np.ndarray:
    # dodge curve: brighten highs more
    bgr = ensure_bgr_u8(bgr)  # validate image
    s = clamp_float(strength, 0.0, 0.95)  # effect strength
    x = bgr.astype(np.float64)  # float pixels
    denom = 255.0 - s * x  # divisor term
    out = (x * 255.0) / np.maximum(denom, 1.0)  # dodge formula
    return np.clip(np.round(out), 0, 255).astype(np.uint8)  # to bytes 0–255

