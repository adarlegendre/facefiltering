"""Foreground extraction with GrabCut-based background removal."""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.validate import clamp_float, clamp_int, ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Foreground extraction"


def apply(
    bgr: np.ndarray,
    *,
    margin_ratio: float = 0.08,
    iterations: int = 5,
    smooth_ksize: int = 5,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    mr = clamp_float(margin_ratio, 0.01, 0.45)
    it = clamp_int(iterations, 1, 15)
    k = odd_ksize(smooth_ksize, minimum=1, maximum=31)

    h, w = bgr.shape[:2]
    mx = max(1, min(int(round(w * mr)), (w - 3) // 2))
    my = max(1, min(int(round(h * mr)), (h - 3) // 2))
    rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, it, cv2.GC_INIT_WITH_RECT)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float64)

    if k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg_u8 = (fg * 255.0).astype(np.uint8)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, ker, iterations=1)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, ker, iterations=1)
        fg = cv2.GaussianBlur(fg_u8.astype(np.float64) / 255.0, (k, k), 0)

    out = bgr.astype(np.float64) * fg[:, :, None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
