"""
Wiener deconvolution in frequency domain (grayscale), Gaussian PSF blur model.
Stable denominator: |H|^2 + K with K = noise_to_signal.
"""
from __future__ import annotations

import cv2
import numpy as np

from facefiltering.gray import to_bgr_from_gray, to_gray_u8
from facefiltering.validate import clamp_float, ensure_bgr_u8, odd_ksize

DISPLAY_NAME = "Wiener deconvolution"


def apply(bgr: np.ndarray, *, psf_size: int = 15, noise_to_signal: float = 1e-3) -> np.ndarray:
    bgr = ensure_bgr_u8(bgr, min_side=8)
    psf = odd_ksize(psf_size, minimum=3, maximum=51)
    ns = clamp_float(noise_to_signal, 1e-8, 1.0)

    g = to_gray_u8(bgr).astype(np.float64) / 255.0
    rows, cols = g.shape

    ax = np.arange(psf, dtype=np.float64) - (psf // 2)
    gx, gy = np.meshgrid(ax, ax)
    sigma = max(psf / 6.0, 1e-6)
    h = np.exp(-(gx * gx + gy * gy) / (2.0 * sigma * sigma))
    h /= h.sum() + 1e-12

    pad_r = cv2.getOptimalDFTSize(rows)
    pad_c = cv2.getOptimalDFTSize(cols)
    gp = cv2.copyMakeBorder(g, 0, pad_r - rows, 0, pad_c - cols, cv2.BORDER_REFLECT)

    hp = np.zeros_like(gp, dtype=np.float64)
    hp[:psf, :psf] = h
    hp = np.roll(hp, -psf // 2, axis=0)
    hp = np.roll(hp, -psf // 2, axis=1)

    G = np.fft.fft2(gp.astype(np.complex128))
    H = np.fft.fft2(hp.astype(np.complex128))
    H2 = (np.abs(H) ** 2).astype(np.float64)
    W = np.conj(H) / (H2 + ns)
    F_hat = G * W
    f = np.real(np.fft.ifft2(F_hat))[:rows, :cols]
    if not np.isfinite(f).all():
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.clip(f, 0.0, 1.0)
    out = (f * 255.0).astype(np.uint8)
    return to_bgr_from_gray(out)
