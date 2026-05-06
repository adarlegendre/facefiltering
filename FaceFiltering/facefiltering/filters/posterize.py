"""Posterize effect by quantizing each channel to a small number of levels."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_int, ensure_bgr_u8

DISPLAY_NAME = "Posterize"


def apply(bgr: np.ndarray, *, levels: int = 8) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    lv = clamp_int(levels, 2, 64)
    x = bgr.astype(np.float64) / 255.0
    q = np.round(x * (lv - 1.0)) / (lv - 1.0)
    out = np.clip(np.round(q * 255.0), 0, 255).astype(np.uint8)
    return out

