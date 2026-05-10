"""Pure NumPy helpers shared by spatial filters (Gaussian convolution)."""
from __future__ import annotations

import numpy as np


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size // 2)
    xx, yy = np.meshgrid(ax, ax)
    s2 = max(float(sigma) ** 2, 1e-12)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * s2))
    k /= k.sum() + 1e-12
    return k


def _convolve2d(gray: np.ndarray, kernel: np.ndarray, *, pad_mode: str = "edge") -> np.ndarray:
    src = np.asarray(gray, dtype=np.float64)
    ker = np.asarray(kernel, dtype=np.float64)
    kh, kw = ker.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(src, ((ph, ph), (pw, pw)), mode=pad_mode)
    out = np.zeros(src.shape, dtype=np.float64)
    for y in range(kh):
        ys = y
        ye = ys + src.shape[0]
        for x in range(kw):
            xs = x
            xe = xs + src.shape[1]
            out += ker[y, x] * pad[ys:ye, xs:xe]
    return out


def convolve_gray(gray: np.ndarray, kernel: np.ndarray, *, pad_mode: str = "edge") -> np.ndarray:
    return _convolve2d(gray, kernel, pad_mode=pad_mode)


def convolve_bgr(bgr: np.ndarray, kernel: np.ndarray, *, pad_mode: str = "edge") -> np.ndarray:
    src = np.asarray(bgr, dtype=np.float64)
    ch = [convolve_gray(src[:, :, i], kernel, pad_mode=pad_mode) for i in range(src.shape[2])]
    return np.stack(ch, axis=-1)
