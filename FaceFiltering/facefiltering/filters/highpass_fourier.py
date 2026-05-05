"""
High-pass emphasis in the Fourier domain (Gaussian high-pass mask on grayscale).
Uses float64 FFT for stability; crops back to original size.
"""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.validate import clamp_float, ensure_bgr_u8

DISPLAY_NAME = "High-pass (Fourier)"


def apply(bgr: np.ndarray, *, cutoff_ratio: float = 0.08) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr, min_side=8)
    cr = clamp_float(cutoff_ratio, 0.01, 0.45)

    g = to_gray_u8(bgr).astype(np.float64) / 255.0
    rows, cols = g.shape
    crow, ccol = rows // 2, cols // 2
    d0 = cr * min(rows, cols) * 0.5
    d0 = max(d0, 1.0)

    pad_r = cv2.getOptimalDFTSize(rows)
    pad_c = cv2.getOptimalDFTSize(cols)
    gp = cv2.copyMakeBorder(g, 0, pad_r - rows, 0, pad_c - cols, cv2.BORDER_CONSTANT, value=0.0)

    dft = np.fft.fft2(gp)
    dft_shift = np.fft.fftshift(dft)

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
    img_u8 = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return to_bgr_from_gray(img_u8.astype(np.uint8))
