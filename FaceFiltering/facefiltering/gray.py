"""BGR <-> grayscale helpers (uint8), pure NumPy."""
from __future__ import annotations

import numpy as np


def to_gray_u8(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.astype(np.uint8, copy=False)
    x = bgr.astype(np.float64, copy=False)
    # BT.601 luma weights in BGR channel order.
    gray = 0.114 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.299 * x[:, :, 2]
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def to_bgr_from_gray(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.uint8, copy=False)
    return np.stack([g, g, g], axis=-1)
