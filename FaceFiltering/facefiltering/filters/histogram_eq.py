"""Histogram equalization on Y channel (color) or full image (gray)."""
from __future__ import annotations

import numpy as np

from facefiltering.ops import equalize_hist_u8
from facefiltering.validate import ensure_bgr_u8

DISPLAY_NAME = "Histogram equalization"


def apply(bgr: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    x = bgr.astype(np.float64)
    # Approximate Y channel in BGR order, then apply equalization on luminance only.
    y = np.clip(np.round(0.114 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.299 * x[:, :, 2]), 0, 255).astype(np.uint8)
    y_eq = equalize_hist_u8(y).astype(np.float64)
    scale = y_eq / np.maximum(y.astype(np.float64), 1.0)
    out = np.clip(np.round(x * scale[:, :, None]), 0, 255).astype(np.uint8)
    return out
