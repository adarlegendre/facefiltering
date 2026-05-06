"""Crosshatch threshold stylization from luminance bands."""
from __future__ import annotations

import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.validate import clamp_int, ensure_bgr_u8

DISPLAY_NAME = "Crosshatch threshold"


def apply(bgr: np.ndarray, *, levels: int = 4, step: int = 8) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    lv = clamp_int(levels, 2, 8)
    st = clamp_int(step, 3, 40)

    g = to_gray_u8(bgr)
    band = np.floor(g.astype(np.float64) / (256.0 / lv)).astype(np.int32)
    darkness = (lv - 1) - np.clip(band, 0, lv - 1)

    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.full((h, w), 255, dtype=np.uint8)

    m1 = ((xx + yy) % st) == 0
    m2 = ((xx - yy) % st) == 0
    m3 = (yy % st) == 0
    m4 = (xx % st) == 0

    out[(darkness >= 1) & m1] = 0
    out[(darkness >= 2) & m2] = 0
    out[(darkness >= 3) & m3] = 0
    out[(darkness >= 4) & m4] = 0
    return to_bgr_from_gray(out)

