"""
Central registry: filter display name -> implementation.
All filters receive validated BGR uint8 from apply_filter.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from facefiltering.filters import binary_threshold as f_bin
from facefiltering.filters import bloom as f_bloom
from facefiltering.filters import canny as f_canny
from facefiltering.filters import crosshatch_threshold as f_ch
from facefiltering.filters import dilate as f_dilate
from facefiltering.filters import diffuse as f_diffuse
from facefiltering.filters import dodge as f_dodge
from facefiltering.filters import erode as f_erode
from facefiltering.filters import gamma as f_gamma
from facefiltering.filters import gaussian_blur as f_gauss
from facefiltering.filters import highpass_fourier as f_hp
from facefiltering.filters import histogram_eq as f_hist
from facefiltering.filters import hue_rotate as f_hue
from facefiltering.filters import laplacian as f_lap
from facefiltering.filters import lens_distortion as f_ld
from facefiltering.filters import median as f_median
from facefiltering.filters import posterize as f_poster
from facefiltering.filters import relief_emboss as f_emb
from facefiltering.filters import sobel as f_sobel
from facefiltering.filters import swirl as f_swirl
from facefiltering.filters import unsharp as f_unsharp
from facefiltering.filters import wiener as f_wiener
from facefiltering.filters import zoom as f_zoom
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
    (f_emb.DISPLAY_NAME, f_emb.apply),
    (f_dodge.DISPLAY_NAME, f_dodge.apply),
    (f_poster.DISPLAY_NAME, f_poster.apply),
    (f_ch.DISPLAY_NAME, f_ch.apply),
    (f_diffuse.DISPLAY_NAME, f_diffuse.apply),
    (f_hue.DISPLAY_NAME, f_hue.apply),
    (f_hp.DISPLAY_NAME, f_hp.apply),
    (f_median.DISPLAY_NAME, f_median.apply),
    (f_dilate.DISPLAY_NAME, f_dilate.apply),
    (f_erode.DISPLAY_NAME, f_erode.apply),
    (f_swirl.DISPLAY_NAME, f_swirl.apply),
    (f_zoom.DISPLAY_NAME, f_zoom.apply),
    (f_ld.DISPLAY_NAME, f_ld.apply),
    (f_bloom.DISPLAY_NAME, f_bloom.apply),
    (f_wiener.DISPLAY_NAME, f_wiener.apply),
]

FILTER_NAMES: List[str] = [n for n, _ in _FILTER_ORDER]
_REGISTRY: Dict[str, Callable[..., np.ndarray]] = dict(_FILTER_ORDER)

# Methodology = how the filter is computed.
FILTER_METHODOLOGIES: Dict[str, str] = {
    f_sobel.DISPLAY_NAME: "Spatial neighborhood",
    f_lap.DISPLAY_NAME: "Spatial neighborhood",
    f_canny.DISPLAY_NAME: "Spatial neighborhood",
    f_unsharp.DISPLAY_NAME: "Spatial neighborhood",
    f_gauss.DISPLAY_NAME: "Spatial neighborhood",
    f_median.DISPLAY_NAME: "Spatial neighborhood",
    f_bin.DISPLAY_NAME: "Point-wise intensity mapping",
    f_gamma.DISPLAY_NAME: "Point-wise intensity mapping",
    f_hist.DISPLAY_NAME: "Global histogram remapping",
    f_hp.DISPLAY_NAME: "Frequency-domain (Fourier)",
    f_wiener.DISPLAY_NAME: "Frequency-domain (Fourier)",
    f_dilate.DISPLAY_NAME: "Morphological (structuring element)",
    f_erode.DISPLAY_NAME: "Morphological (structuring element)",
    f_swirl.DISPLAY_NAME: "Geometric warping",
    f_bloom.DISPLAY_NAME: "Spatial neighborhood",
    f_dodge.DISPLAY_NAME: "Point-wise intensity mapping",
    f_poster.DISPLAY_NAME: "Point-wise intensity mapping",
    f_ch.DISPLAY_NAME: "Point-wise intensity mapping",
    f_hue.DISPLAY_NAME: "Point-wise intensity mapping",
    f_zoom.DISPLAY_NAME: "Geometric warping",
    f_ld.DISPLAY_NAME: "Geometric warping",
    f_emb.DISPLAY_NAME: "Spatial neighborhood",
    f_diffuse.DISPLAY_NAME: "Spatial neighborhood",
}

METHODOLOGY_DESCRIPTIONS: Dict[str, str] = {
    "All": "show all filters",
    "Spatial neighborhood": "operate on local pixel neighborhoods (kernels/windows)",
    "Point-wise intensity mapping": "map each pixel independently with a math transform",
    "Global histogram remapping": "remap intensities using image-wide histogram statistics",
    "Frequency-domain (Fourier)": "process spectral components after Fourier transform",
    "Morphological (structuring element)": "shape-based max/min operations with structuring elements",
    "Geometric warping": "reposition pixels by coordinate transforms",
}

METHODOLOGY_NAMES: List[str] = [
    "All",
    "Spatial neighborhood",
    "Point-wise intensity mapping",
    "Global histogram remapping",
    "Frequency-domain (Fourier)",
    "Morphological (structuring element)",
    "Geometric warping",
]

FILTERS_BY_METHODOLOGY: Dict[str, List[str]] = {
    m: [name for name in FILTER_NAMES if FILTER_METHODOLOGIES.get(name) == m] for m in METHODOLOGY_NAMES
}
FILTERS_BY_METHODOLOGY["All"] = FILTER_NAMES[:]

# Function = what the filter is used for / output effect.
FILTER_FUNCTIONS: Dict[str, str] = {
    f_sobel.DISPLAY_NAME: "Edge detection",
    f_lap.DISPLAY_NAME: "Edge detection",
    f_canny.DISPLAY_NAME: "Edge detection",
    f_unsharp.DISPLAY_NAME: "Sharpening / detail enhancement",
    f_gauss.DISPLAY_NAME: "Smoothing / denoising",
    f_median.DISPLAY_NAME: "Smoothing / denoising",
    f_bin.DISPLAY_NAME: "Segmentation / masking",
    f_gamma.DISPLAY_NAME: "Tone / brightness adjustment",
    f_hist.DISPLAY_NAME: "Contrast enhancement",
    f_hp.DISPLAY_NAME: "Sharpening / detail enhancement",
    f_wiener.DISPLAY_NAME: "Deblurring / restoration",
    f_dilate.DISPLAY_NAME: "Shape / region refinement",
    f_erode.DISPLAY_NAME: "Shape / region refinement",
    f_dodge.DISPLAY_NAME: "Tone / brightness adjustment",
    f_poster.DISPLAY_NAME: "Creative stylization",
    f_ch.DISPLAY_NAME: "Creative stylization",
    f_hue.DISPLAY_NAME: "Creative stylization",
    f_swirl.DISPLAY_NAME: "Creative stylization",
    f_zoom.DISPLAY_NAME: "Creative stylization",
    f_ld.DISPLAY_NAME: "Creative stylization",
    f_bloom.DISPLAY_NAME: "Creative stylization",
    f_emb.DISPLAY_NAME: "Creative stylization",
    f_diffuse.DISPLAY_NAME: "Creative stylization",
}

FUNCTION_NAMES: List[str] = [
    "Edge detection",
    "Sharpening / detail enhancement",
    "Smoothing / denoising",
    "Segmentation / masking",
    "Tone / brightness adjustment",
    "Contrast enhancement",
    "Deblurring / restoration",
    "Shape / region refinement",
    "Creative stylization",
]

FILTERS_BY_FUNCTION: Dict[str, List[str]] = {
    fn: [name for name in FILTER_NAMES if FILTER_FUNCTIONS.get(name) == fn] for fn in FUNCTION_NAMES
}

# Cross-cutting tag: linear convolution-based filters.
CONVOLUTION_FILTERS: List[str] = [
    f_sobel.DISPLAY_NAME,
    f_lap.DISPLAY_NAME,
    f_gauss.DISPLAY_NAME,
    f_unsharp.DISPLAY_NAME,
    f_hp.DISPLAY_NAME,  # Fourier implementation but equivalent to convolution in space.
]

def functions_for_methodology(methodology: str) -> List[str]:
    """Return the available function groups under a selected methodology."""
    if methodology == "All":
        return ["All"] + FUNCTION_NAMES
    names = FILTERS_BY_METHODOLOGY.get(methodology, FILTER_NAMES)
    funcs = {FILTER_FUNCTIONS.get(n, "Other") for n in names}
    return ["All"] + ([f for f in FUNCTION_NAMES if f in funcs] or FUNCTION_NAMES)

def filters_for_methodology_and_function(methodology: str, function: str) -> List[str]:
    names = FILTERS_BY_METHODOLOGY.get(methodology, FILTER_NAMES) if methodology != "All" else FILTER_NAMES
    if function == "All":
        return names
    return [n for n in names if FILTER_FUNCTIONS.get(n) == function] or names

# Backwards-compatible aliases used by older imports/UI names.
FILTER_TYPES = FILTER_METHODOLOGIES
TYPE_DESCRIPTIONS = METHODOLOGY_DESCRIPTIONS
TYPE_NAMES = METHODOLOGY_NAMES
FILTERS_BY_TYPE = FILTERS_BY_METHODOLOGY
FILTER_CATEGORIES = FILTER_FUNCTIONS
CATEGORY_NAMES = FUNCTION_NAMES
FILTERS_BY_CATEGORY = FILTERS_BY_FUNCTION
categories_for_type = functions_for_methodology
filters_for_type_and_category = filters_for_methodology_and_function


def apply_filter(name: str, bgr: np.ndarray, **kwargs) -> np.ndarray:
    """
    Run a registered filter by display name.
    kwargs match UI / CLI: ksize, sigma, amount, gauss_sigma, gauss_ksize,
    thresh, gamma, cutoff, iter, psf, ns, canny_t1, canny_t2, canny_ap, canny_l2,
    erode_ksize, erode_iter, dodge_strength, swirl_strength, swirl_radius,
    bloom_thresh, bloom_sigma, bloom_intensity, poster_levels, hatch_levels,
    hatch_step, zoom_factor, lens_strength, emboss_strength, diffuse_radius,
    diffuse_mix, hue_degrees
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
    if name == f_dodge.DISPLAY_NAME:
        return fn(img, strength=float(kwargs.get("dodge_strength", 0.55)))
    if name == f_swirl.DISPLAY_NAME:
        return fn(
            img,
            strength=float(kwargs.get("swirl_strength", 2.0)),
            radius_ratio=float(kwargs.get("swirl_radius", 0.75)),
        )
    if name == f_bloom.DISPLAY_NAME:
        return fn(
            img,
            threshold=int(kwargs.get("bloom_thresh", 180)),
            sigma=float(kwargs.get("bloom_sigma", 2.5)),
            intensity=float(kwargs.get("bloom_intensity", 0.7)),
        )
    if name == f_poster.DISPLAY_NAME:
        return fn(img, levels=int(kwargs.get("poster_levels", 8)))
    if name == f_ch.DISPLAY_NAME:
        return fn(
            img,
            levels=int(kwargs.get("hatch_levels", 4)),
            step=int(kwargs.get("hatch_step", 8)),
        )
    if name == f_zoom.DISPLAY_NAME:
        return fn(img, factor=float(kwargs.get("zoom_factor", 1.2)))
    if name == f_ld.DISPLAY_NAME:
        return fn(img, strength=float(kwargs.get("lens_strength", -0.25)))
    if name == f_emb.DISPLAY_NAME:
        return fn(img, strength=float(kwargs.get("emboss_strength", 1.0)))
    if name == f_diffuse.DISPLAY_NAME:
        return fn(
            img,
            radius=int(kwargs.get("diffuse_radius", 3)),
            mix=float(kwargs.get("diffuse_mix", 1.0)),
        )
    if name == f_hue.DISPLAY_NAME:
        return fn(img, degrees=float(kwargs.get("hue_degrees", 45.0)))

    raise RuntimeError("Registry dispatch out of sync.")
