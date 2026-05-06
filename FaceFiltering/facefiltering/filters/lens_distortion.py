"""Radial lens distortion (barrel/pincushion) with bilinear sampling."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Lens distortion"


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


def apply(bgr: np.ndarray, *, strength: float = -0.25) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = clamp_float(strength, -0.8, 0.8)
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    xn = (xx - cx) / max(cx, 1.0)
    yn = (yy - cy) / max(cy, 1.0)
    r2 = xn * xn + yn * yn

    # Inverse mapping approximation for stable sampling.
    denom = np.maximum(1.0 + k * r2, 0.2)
    src_xn = xn / denom
    src_yn = yn / denom

    src_x = src_xn * max(cx, 1.0) + cx
    src_y = src_yn * max(cy, 1.0) + cy
    out = _bilinear_sample(bgr.astype(np.float64), src_x, src_y)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

