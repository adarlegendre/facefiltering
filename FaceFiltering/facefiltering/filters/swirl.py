"""Swirl geometric warp around the image center."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Swirl"


def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w, c = img.shape
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    top = Ia * (1.0 - wx) + Ib * wx
    bottom = Ic * (1.0 - wx) + Id * wx
    return top * (1.0 - wy) + bottom * wy


def apply(bgr: np.ndarray, *, strength: float = 2.0, radius_ratio: float = 0.75) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    s = clamp_float(strength, -8.0, 8.0)
    rr = clamp_float(radius_ratio, 0.05, 1.0)

    h, w = bgr.shape[:2]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    radius = rr * min(h, w) * 0.5
    inside = r < radius
    amt = np.zeros_like(r, dtype=np.float64)
    amt[inside] = s * (radius - r[inside]) / max(radius, 1e-6)
    src_theta = theta - amt  # inverse mapping to avoid holes

    src_x = cx + r * np.cos(src_theta)
    src_y = cy + r * np.sin(src_theta)

    out = _bilinear_sample(bgr.astype(np.float64), src_x, src_y)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

