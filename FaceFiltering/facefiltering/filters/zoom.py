"""Zoom geometric warp around image center (bilinear sampling)."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Zoom"


def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
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


def apply(bgr: np.ndarray, *, factor: float = 1.2) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    f = clamp_float(factor, 0.2, 4.0)
    h, w = bgr.shape[:2]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    src_x = cx + (xx - cx) / f
    src_y = cy + (yy - cy) / f
    out = _bilinear_sample(bgr.astype(np.float64), src_x, src_y)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

