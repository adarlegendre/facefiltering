"""Pure NumPy image operations used by filter implementations."""
from __future__ import annotations

import numpy as np


def normalize_to_u8(x: np.ndarray) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float64)
    if xf.size == 0:
        return np.zeros_like(xf, dtype=np.uint8)
    if not np.isfinite(xf).all():
        xf = np.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(xf.min())
    mx = float(xf.max())
    if mx - mn < 1e-12:
        return np.zeros(xf.shape, dtype=np.uint8)
    out = (xf - mn) * (255.0 / (mx - mn))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


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


def median_blur_bgr(bgr: np.ndarray, ksize: int) -> np.ndarray:
    k = int(ksize)
    r = k // 2
    src = np.asarray(bgr, dtype=np.uint8)
    out = np.empty_like(src)
    for c in range(src.shape[2]):
        pad = np.pad(src[:, :, c], ((r, r), (r, r)), mode="edge")
        sw = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
        out[:, :, c] = np.median(sw, axis=(-2, -1)).astype(np.uint8)
    return out


def elliptical_kernel(ksize: int) -> np.ndarray:
    k = int(ksize)
    r = k // 2
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    if r == 0:
        return np.ones((1, 1), dtype=bool)
    mask = (xx * xx + yy * yy) <= (r * r)
    return mask


def morphology_bgr(bgr: np.ndarray, kernel: np.ndarray, *, op: str, iterations: int = 1) -> np.ndarray:
    src = np.asarray(bgr, dtype=np.uint8)
    ker = np.asarray(kernel, dtype=bool)
    kh, kw = ker.shape
    ph, pw = kh // 2, kw // 2
    out = src.copy()
    iters = max(1, int(iterations))
    for _ in range(iters):
        nxt = np.empty_like(out)
        for c in range(out.shape[2]):
            pad = np.pad(out[:, :, c], ((ph, ph), (pw, pw)), mode="edge")
            sw = np.lib.stride_tricks.sliding_window_view(pad, (kh, kw))
            vals = np.where(ker[None, None, :, :], sw, 0 if op == "dilate" else 255)
            if op == "dilate":
                nxt[:, :, c] = vals.max(axis=(-2, -1)).astype(np.uint8)
            else:
                nxt[:, :, c] = vals.min(axis=(-2, -1)).astype(np.uint8)
        out = nxt
    return out


def equalize_hist_u8(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    nz = np.nonzero(cdf)[0]
    if nz.size == 0:
        return g.copy()
    cdf_min = cdf[nz[0]]
    denom = max(cdf[-1] - cdf_min, 1.0)
    lut = np.clip(np.round((cdf - cdf_min) * 255.0 / denom), 0, 255).astype(np.uint8)
    return lut[g]

