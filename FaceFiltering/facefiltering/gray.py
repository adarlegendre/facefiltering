"""BGR <-> grayscale helpers (uint8)."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray_u8(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.astype(np.uint8, copy=False)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_bgr_from_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
