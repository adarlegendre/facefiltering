"""
High-pass emphasis in the Fourier domain (Gaussian high-pass mask on grayscale).
Uses float64 FFT for stability; crops back to original size.
"""
from __future__ import annotations

import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.ops import normalize_to_u8
from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "High-pass (Fourier)"


def apply(bgr: np.ndarray, *, cutoff_ratio: float = 0.08) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr, min_side=8)
    cr = clamp_float(cutoff_ratio, 0.01, 0.45)

    g = to_gray_u8(bgr).astype(np.float64) / 255.0
    rows, cols = g.shape
    d0 = cr * min(rows, cols) * 0.5
    d0 = max(d0, 1.0)

    pad_r = 1 << int(np.ceil(np.log2(max(rows, 1))))
    pad_c = 1 << int(np.ceil(np.log2(max(cols, 1))))
    gp = np.pad(g, ((0, pad_r - rows), (0, pad_c - cols)), mode="constant", constant_values=0.0)

    dft = np.fft.fft2(gp)
    dft_shift = np.fft.fftshift(dft)
    crow, ccol = pad_r // 2, pad_c // 2

    yy, xx = np.ogrid[:pad_r, :pad_c]
    dist = np.sqrt((yy - crow) ** 2 + (xx - ccol) ** 2).astype(np.float64)
    hp = 1.0 - np.exp(-(dist ** 2) / (2.0 * (d0 ** 2)))
    hp = hp.astype(np.complex128)

    filt = dft_shift * hp
    filt_ishift = np.fft.ifftshift(filt)
    img_back = np.fft.ifft2(filt_ishift)
    img_back = np.real(img_back[:rows, :cols])
    if not np.isfinite(img_back).all():
        img_back = np.nan_to_num(img_back, nan=0.0, posinf=0.0, neginf=0.0)
    return to_bgr_from_gray(normalize_to_u8(img_back))
