"""Hue rotation in HSV color space (manual NumPy conversion)."""
from __future__ import annotations

import numpy as np

from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Hue rotate"


def _bgr_to_hsv_u8(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = bgr.astype(np.float64) / 255.0
    b, g, r = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    m = delta > 1e-12
    rm = (cmax == r) & m
    gm = (cmax == g) & m
    bm = (cmax == b) & m
    h[rm] = (60.0 * ((g[rm] - b[rm]) / delta[rm]) + 360.0) % 360.0
    h[gm] = 60.0 * ((b[gm] - r[gm]) / delta[gm] + 2.0)
    h[bm] = 60.0 * ((r[bm] - g[bm]) / delta[bm] + 4.0)

    s = np.zeros_like(cmax)
    nz = cmax > 1e-12
    s[nz] = delta[nz] / cmax[nz]
    v = cmax
    return h, s, v


def _hsv_to_bgr_u8(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    c = v * s
    hp = (h % 360.0) / 60.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))
    z = np.zeros_like(h)

    rp = np.zeros_like(h)
    gp = np.zeros_like(h)
    bp = np.zeros_like(h)

    m0 = (0 <= hp) & (hp < 1)
    m1 = (1 <= hp) & (hp < 2)
    m2 = (2 <= hp) & (hp < 3)
    m3 = (3 <= hp) & (hp < 4)
    m4 = (4 <= hp) & (hp < 5)
    m5 = (5 <= hp) & (hp < 6)

    rp[m0], gp[m0], bp[m0] = c[m0], x[m0], z[m0]
    rp[m1], gp[m1], bp[m1] = x[m1], c[m1], z[m1]
    rp[m2], gp[m2], bp[m2] = z[m2], c[m2], x[m2]
    rp[m3], gp[m3], bp[m3] = z[m3], x[m3], c[m3]
    rp[m4], gp[m4], bp[m4] = x[m4], z[m4], c[m4]
    rp[m5], gp[m5], bp[m5] = c[m5], z[m5], x[m5]

    m = v - c
    r = rp + m
    g = gp + m
    b = bp + m
    out = np.stack([b, g, r], axis=-1)
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def apply(bgr: np.ndarray, *, degrees: float = 45.0) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    deg = clamp_float(degrees, -360.0, 360.0)
    h, s, v = _bgr_to_hsv_u8(bgr)
    h = (h + deg) % 360.0
    return _hsv_to_bgr_u8(h, s, v)

