"""Aura glow: soft edge-focused glow composited over image."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Aura glow"


def apply(
    bgr: np.ndarray,
    *,
    sigma: float = 3.0,
    intensity: float = 0.8,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    s = clamp_float(sigma, 0.1, 20.0)
    a = clamp_float(intensity, 0.0, 3.0)

    src = bgr.astype(np.float64)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-12)

    k = max(3, int(2 * round(3 * s) + 1) | 1)
    edge_glow = cv2.GaussianBlur((mag * 255.0).astype(np.uint8), (k, k), s).astype(np.float64) / 255.0

    out = src + a * 255.0 * edge_glow[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
