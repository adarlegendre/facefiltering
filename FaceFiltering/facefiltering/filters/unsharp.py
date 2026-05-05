"""Unsharp masking: sharpen = original + amount * (original - GaussianBlur)."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Unsharp mask"


def apply(bgr: np.ndarray, *, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    sigma = clamp_float(sigma, 0.05, 50.0)
    amount = clamp_float(amount, 0.0, 5.0)
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)
