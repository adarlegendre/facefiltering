"""Canny edge detector (grayscale), pure NumPy implementation."""
from __future__ import annotations

import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.ops import convolve_gray, gaussian_kernel
from facefiltering.validate import clamp_int, ensure_bgr_u8

DISPLAY_NAME = "Canny edge detection"


def _non_max_suppression(mag: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    out = np.zeros_like(mag, dtype=np.float64)
    ang = (angle_deg + 180.0) % 180.0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = ang[y, x]
            m = mag[y, x]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = mag[y, x + 1], mag[y, x - 1]
            elif 22.5 <= a < 67.5:
                q, r = mag[y + 1, x - 1], mag[y - 1, x + 1]
            elif 67.5 <= a < 112.5:
                q, r = mag[y + 1, x], mag[y - 1, x]
            else:
                q, r = mag[y - 1, x - 1], mag[y + 1, x + 1]
            if m >= q and m >= r:
                out[y, x] = m
    return out


def _hysteresis(nms: np.ndarray, t1: float, t2: float) -> np.ndarray:
    strong = nms >= t2
    weak = (nms >= t1) & ~strong
    h, w = nms.shape
    out = np.zeros((h, w), dtype=np.uint8)
    out[strong] = 255
    stack = list(map(tuple, np.argwhere(strong)))
    while stack:
        y, x = stack.pop()
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        region_weak = weak[y0:y1, x0:x1]
        if not np.any(region_weak):
            continue
        weak_pos = np.argwhere(region_weak)
        for dy, dx in weak_pos:
            ny, nx = y0 + dy, x0 + dx
            weak[ny, nx] = False
            out[ny, nx] = 255
            stack.append((ny, nx))
    return out


def apply(
    bgr: np.ndarray,
    *,
    t1: int = 80,
    t2: int = 160,
    aperture: int = 3,
    l2gradient: bool = False,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    t1 = clamp_int(t1, 0, 500)
    t2 = clamp_int(t2, 0, 500)
    if t2 < t1:
        t1, t2 = t2, t1

    # Keep API-compatible aperture argument; use it as Sobel kernel size for gradients.
    ap = int(aperture)
    if ap not in (3, 5, 7):
        ap = 3

    g = to_gray_u8(bgr).astype(np.float64)
    blur = convolve_gray(g, gaussian_kernel(5, 1.0), pad_mode="edge")

    if ap == 3:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    elif ap == 5:
        kx = np.array(
            [[-1, -2, 0, 2, 1], [-4, -8, 0, 8, 4], [-6, -12, 0, 12, 6], [-4, -8, 0, 8, 4], [-1, -2, 0, 2, 1]],
            dtype=np.float64,
        )
        ky = kx.T
    else:
        base = np.array([-1, -4, -5, 0, 5, 4, 1], dtype=np.float64)
        smooth = np.array([1, 6, 15, 20, 15, 6, 1], dtype=np.float64)
        kx = np.outer(smooth, base)
        ky = np.outer(base, smooth)

    gx = convolve_gray(blur, kx, pad_mode="edge")
    gy = convolve_gray(blur, ky, pad_mode="edge")
    if l2gradient:
        mag = np.hypot(gx, gy)
    else:
        mag = np.abs(gx) + np.abs(gy)
    angle = np.degrees(np.arctan2(gy, gx))
    nms = _non_max_suppression(mag, angle)
    edges = _hysteresis(nms, float(t1), float(t2))
    return to_bgr_from_gray(edges)

