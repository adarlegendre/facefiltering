"""
Shared input validation and parameter clamping for all filters.
"""
from __future__ import annotations

import numpy as np


class FilterInputError(ValueError):
    """Raised when an image cannot be processed as BGR uint8."""


def ensure_bgr_u8(img: np.ndarray, *, min_side: int = 3) -> np.ndarray:
    """
    Normalize to contiguous BGR uint8, shape (H, W, 3).

    Accepts:
    - uint8/float BGR or BGRA (alpha dropped)
    - uint8 grayscale (H, W) -> replicated to 3 channels
    - float in [0, 1] scaled to uint8
    """
    if img is None:
        raise FilterInputError("Image is None.")
    if not isinstance(img, np.ndarray):
        raise FilterInputError(f"Expected numpy array, got {type(img)}.")

    x = np.ascontiguousarray(img)

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)

    if x.ndim != 3:
        raise FilterInputError(f"Expected 2D or 3D array, got shape {x.shape}.")

    c = x.shape[2]
    if c == 4:
        x = x[:, :, :3]
        c = 3
    if c != 3:
        raise FilterInputError(f"Expected 1 or 3 channels after normalize, got {c}.")

    h, w = x.shape[0], x.shape[1]
    if h < min_side or w < min_side:
        raise FilterInputError(f"Image too small: {w}x{h} (min side {min_side}).")

    if x.dtype == np.uint8:
        out = x
    elif np.issubdtype(x.dtype, np.floating):
        xf = np.clip(x.astype(np.float64), 0.0, None)
        if xf.size and xf.max() <= 1.0 + 1e-6:
            xf = xf * 255.0
        out = np.clip(np.round(xf), 0, 255).astype(np.uint8)
    else:
        out = np.clip(x.astype(np.float64), 0, 255).astype(np.uint8)

    return np.ascontiguousarray(out)


def odd_ksize(k: int, *, minimum: int = 3, maximum: int = 31) -> int:
    k = int(round(k))
    k = max(minimum, min(maximum, k))
    if k % 2 == 0:
        k = min(maximum, k + 1)
    if k % 2 == 0:
        k = max(minimum, k - 1)
    return k


def clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def clamp_float(v: float, lo: float, hi: float) -> float:
    v = float(v)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v
