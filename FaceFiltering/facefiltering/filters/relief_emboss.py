"""Relief emboss stylization using directional convolution and bias."""
from __future__ import annotations

import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.ops import convolve_gray
from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "Relief emboss"


def apply(bgr: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    s = clamp_float(strength, 0.1, 4.0)
    g = to_gray_u8(bgr).astype(np.float64)
    kernel = np.array([[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]], dtype=np.float64) * s
    emb = convolve_gray(g, kernel, pad_mode="edge") + 128.0
    out = np.clip(np.round(emb), 0, 255).astype(np.uint8)
    return to_bgr_from_gray(out)

