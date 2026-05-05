"""Global binary threshold on luminance."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.validate import clamp_int, ensure_bgr_u8

DISPLAY_NAME = "Binary threshold"


def apply(bgr: np.ndarray, *, thresh: int = 127) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    t = clamp_int(thresh, 0, 255)
    g = to_gray_u8(bgr)
    _, bw = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
    return to_bgr_from_gray(bw)
