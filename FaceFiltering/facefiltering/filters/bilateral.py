"""Bilateral smoothing: edge-preserving denoising."""
from __future__ import annotations

import numpy as np

from facefiltering.gray import to_gray_u8
from facefiltering.validate import clamp_float, ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Bilateral"


def apply(
    bgr: np.ndarray,
    *,
    ksize: int = 7,
    sigma_space: float = 3.0,
    sigma_color: float = 25.0,
) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr)
    k = odd_ksize(ksize, minimum=3, maximum=21)
    ss = clamp_float(sigma_space, 0.1, 50.0)
    sc = clamp_float(sigma_color, 1.0, 255.0)

    src = bgr.astype(np.float64)
    g = to_gray_u8(bgr).astype(np.float64)
    h, w = g.shape
    r = k // 2

    pad_g = np.pad(g, ((r, r), (r, r)), mode="edge")
    pad_src = np.pad(src, ((r, r), (r, r), (0, 0)), mode="edge")

    ax = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    spatial = np.exp(-(xx * xx + yy * yy) / (2.0 * ss * ss))

    num = np.zeros_like(src, dtype=np.float64)
    den = np.zeros((h, w), dtype=np.float64)

    for y in range(k):
        ys = y
        ye = ys + h
        for x in range(k):
            xs = x
            xe = xs + w

            ng = pad_g[ys:ye, xs:xe]
            rg = np.exp(-((ng - g) ** 2) / (2.0 * sc * sc))
            wgt = spatial[y, x] * rg

            den += wgt
            num += pad_src[ys:ye, xs:xe, :] * wgt[:, :, None]

    out = num / (den[:, :, None] + 1e-12)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
