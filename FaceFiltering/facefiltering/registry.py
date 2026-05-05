"""
Central registry: filter display name -> implementation.
All filters receive validated BGR uint8 from apply_filter.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from facefiltering.filters import binary_threshold as f_bin
from facefiltering.filters import canny as f_canny
from facefiltering.filters import dilate as f_dilate
from facefiltering.filters import erode as f_erode
from facefiltering.filters import gamma as f_gamma
from facefiltering.filters import gaussian_blur as f_gauss
from facefiltering.filters import highpass_fourier as f_hp
from facefiltering.filters import histogram_eq as f_hist
from facefiltering.filters import laplacian as f_lap
from facefiltering.filters import median as f_median
from facefiltering.filters import sobel as f_sobel
from facefiltering.filters import unsharp as f_unsharp
from facefiltering.filters import wiener as f_wiener
from facefiltering.validate import ensure_bgr_u8

# Stable UI order
_FILTER_ORDER: List[Tuple[str, Callable[..., np.ndarray]]] = [
    (f_sobel.DISPLAY_NAME, f_sobel.apply),
    (f_lap.DISPLAY_NAME, f_lap.apply),
    (f_canny.DISPLAY_NAME, f_canny.apply),
    (f_unsharp.DISPLAY_NAME, f_unsharp.apply),
    (f_gauss.DISPLAY_NAME, f_gauss.apply),
    (f_bin.DISPLAY_NAME, f_bin.apply),
    (f_gamma.DISPLAY_NAME, f_gamma.apply),
    (f_hist.DISPLAY_NAME, f_hist.apply),
    (f_hp.DISPLAY_NAME, f_hp.apply),
    (f_median.DISPLAY_NAME, f_median.apply),
    (f_dilate.DISPLAY_NAME, f_dilate.apply),
    (f_erode.DISPLAY_NAME, f_erode.apply),
    (f_wiener.DISPLAY_NAME, f_wiener.apply),
]

FILTER_NAMES: List[str] = [n for n, _ in _FILTER_ORDER]
_REGISTRY: Dict[str, Callable[..., np.ndarray]] = dict(_FILTER_ORDER)

# Categories for UI grouping / classification
FILTER_CATEGORIES: Dict[str, str] = {
    f_sobel.DISPLAY_NAME: "Edge detection",
    f_lap.DISPLAY_NAME: "Edge detection",
    f_canny.DISPLAY_NAME: "Edge detection",
    f_unsharp.DISPLAY_NAME: "Sharpening / enhancement",
    f_gauss.DISPLAY_NAME: "Smoothing / denoising",
    f_median.DISPLAY_NAME: "Smoothing / denoising",
    f_bin.DISPLAY_NAME: "Intensity transforms",
    f_gamma.DISPLAY_NAME: "Intensity transforms",
    f_hist.DISPLAY_NAME: "Intensity transforms",
    f_hp.DISPLAY_NAME: "Frequency-domain filtering",
    f_wiener.DISPLAY_NAME: "Frequency-domain filtering",
    f_dilate.DISPLAY_NAME: "Morphology",
    f_erode.DISPLAY_NAME: "Morphology",
}

CATEGORY_NAMES: List[str] = [
    "Edge detection",
    "Sharpening / enhancement",
    "Smoothing / denoising",
    "Intensity transforms",
    "Frequency-domain filtering",
    "Morphology",
]

FILTERS_BY_CATEGORY: Dict[str, List[str]] = {
    cat: [name for name in FILTER_NAMES if FILTER_CATEGORIES.get(name) == cat] for cat in CATEGORY_NAMES
}

# Higher-level classification (as requested for the project UI)
FILTER_TYPES: Dict[str, str] = {
    # Feature / edge detection
    f_sobel.DISPLAY_NAME: "Feature",
    f_lap.DISPLAY_NAME: "Feature",
    f_canny.DISPLAY_NAME: "Feature",
    # Spatial domain filtering (direct pixel neighborhood operations)
    f_gauss.DISPLAY_NAME: "Spatial",
    f_median.DISPLAY_NAME: "Spatial",
    f_unsharp.DISPLAY_NAME: "Spatial",
    # Intensity transforms (pixel-by-pixel math)
    f_bin.DISPLAY_NAME: "Intensity",
    f_gamma.DISPLAY_NAME: "Intensity",
    # Color adjustments
    f_hist.DISPLAY_NAME: "Color",
    # Frequency-domain filtering
    f_hp.DISPLAY_NAME: "Frequency",
    # Morphological (shape manipulation)
    f_dilate.DISPLAY_NAME: "Morphological",
    f_erode.DISPLAY_NAME: "Morphological",
    # Restoration / deblurring
    f_wiener.DISPLAY_NAME: "Restoration",
}

TYPE_DESCRIPTIONS: Dict[str, str] = {
    "All": "show all filters",
    "Spatial": "direct pixel editing (neighborhood operations)",
    "Frequency": "edit via signal components (Fourier domain)",
    "Intensity": "pixel-by-pixel math (point transforms)",
    "Morphological": "shape manipulation (structuring elements)",
    "Color": "color / luminance adjustments",
    "Feature": "detect edges/details",
    "Restoration": "fix damaged/blurred images (deconvolution)",
    "Convolution": "linear filtering via convolution kernels",
}

TYPE_NAMES: List[str] = [
    "All",
    "Convolution",
    "Spatial",
    "Frequency",
    "Intensity",
    "Morphological",
    "Color",
    "Feature",
    "Restoration",
]

FILTERS_BY_TYPE: Dict[str, List[str]] = {
    t: [name for name in FILTER_NAMES if FILTER_TYPES.get(name) == t] for t in TYPE_NAMES
}

# Convolution capture (cross-cutting): filters that are (or are built from) linear convolution.
CONVOLUTION_FILTERS: List[str] = [
    f_sobel.DISPLAY_NAME,
    f_lap.DISPLAY_NAME,
    f_gauss.DISPLAY_NAME,
    f_unsharp.DISPLAY_NAME,
    f_hp.DISPLAY_NAME,  # implemented in Fourier domain, but equivalent to convolution in space
]
FILTERS_BY_TYPE["Convolution"] = [n for n in FILTER_NAMES if n in set(CONVOLUTION_FILTERS)]
FILTERS_BY_TYPE["All"] = FILTER_NAMES[:]

def categories_for_type(t: str) -> List[str]:
    """Return the set of existing categories that appear under a type."""
    if t == "All":
        return ["All"] + CATEGORY_NAMES
    names = FILTERS_BY_TYPE.get(t, FILTER_NAMES)
    cats = {FILTER_CATEGORIES.get(n, "Other") for n in names}
    # keep stable ordering based on CATEGORY_NAMES
    return ["All"] + ([c for c in CATEGORY_NAMES if c in cats] or CATEGORY_NAMES)

def filters_for_type_and_category(t: str, c: str) -> List[str]:
    names = FILTERS_BY_TYPE.get(t, FILTER_NAMES) if t != "All" else FILTER_NAMES
    if c == "All":
        return names
    return [n for n in names if FILTER_CATEGORIES.get(n) == c] or names


def apply_filter(name: str, bgr: np.ndarray, **kwargs) -> np.ndarray:
    """
    Run a registered filter by display name.
    kwargs match UI / CLI: ksize, sigma, amount, gauss_sigma, gauss_ksize,
    thresh, gamma, cutoff, iter, psf, ns, canny_t1, canny_t2, canny_ap, canny_l2,
    erode_ksize, erode_iter
    """
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown filter: {name!r}. Choose one of {FILTER_NAMES}.")

    img = ensure_bgr_u8(bgr)

    if name == f_sobel.DISPLAY_NAME:
        return fn(img, ksize=int(kwargs.get("ksize", 3)))
    if name == f_lap.DISPLAY_NAME:
        return fn(img, ksize=int(kwargs.get("ksize", 3)))
    if name == f_canny.DISPLAY_NAME:
        return fn(
            img,
            t1=int(kwargs.get("canny_t1", 80)),
            t2=int(kwargs.get("canny_t2", 160)),
            aperture=int(kwargs.get("canny_ap", 3)),
            l2gradient=bool(kwargs.get("canny_l2", False)),
        )
    if name == f_unsharp.DISPLAY_NAME:
        return fn(img, sigma=float(kwargs.get("sigma", 1.0)), amount=float(kwargs.get("amount", 1.5)))
    if name == f_gauss.DISPLAY_NAME:
        return fn(
            img,
            sigma=float(kwargs.get("gauss_sigma", 1.0)),
            ksize=int(kwargs.get("gauss_ksize", 0)),
        )
    if name == f_bin.DISPLAY_NAME:
        return fn(img, thresh=int(kwargs.get("thresh", 127)))
    if name == f_gamma.DISPLAY_NAME:
        return fn(img, gamma=float(kwargs.get("gamma", 1.0)))
    if name == f_hist.DISPLAY_NAME:
        return fn(img)
    if name == f_hp.DISPLAY_NAME:
        return fn(img, cutoff_ratio=float(kwargs.get("cutoff", 0.08)))
    if name == f_median.DISPLAY_NAME:
        return fn(img, ksize=int(kwargs.get("ksize", 5)))
    if name == f_dilate.DISPLAY_NAME:
        return fn(img, ksize=int(kwargs.get("ksize", 5)), iterations=int(kwargs.get("iter", 1)))
    if name == f_erode.DISPLAY_NAME:
        return fn(
            img,
            ksize=int(kwargs.get("erode_ksize", 5)),
            iterations=int(kwargs.get("erode_iter", 1)),
        )
    if name == f_wiener.DISPLAY_NAME:
        return fn(img, psf_size=int(kwargs.get("psf", 15)), noise_to_signal=float(kwargs.get("ns", 1e-3)))

    raise RuntimeError("Registry dispatch out of sync.")
