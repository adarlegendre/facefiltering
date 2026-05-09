"""Fresnel-like glow approximation using radial rim weighting."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Fresnel glow"


def apply(
    bgr: np.ndarray,
    *,
    power: float = 2.0,
    intensity: float = 0.7,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    p = clamp_float(power, 0.2, 8.0)
    a = clamp_float(intensity, 0.0, 3.0)

    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    xn = (xx - cx) / max(cx, 1.0)
    yn = (yy - cy) / max(cy, 1.0)
    r = np.sqrt(xn * xn + yn * yn)
    r = np.clip(r / (np.max(r) + 1e-12), 0.0, 1.0)

    rim = np.power(r, p)
    rim = cv2.GaussianBlur(rim.astype(np.float64), (0, 0), 1.0)
    rim = rim / (rim.max() + 1e-12)

    src = bgr.astype(np.float64)
    out = src + a * 255.0 * rim[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
