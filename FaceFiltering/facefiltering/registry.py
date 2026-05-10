"""
Central registry: filter display name -> implementation.
All filters receive validated BGR uint8 from apply_filter.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from facefiltering.filters import bloom as f_bloom
from facefiltering.filters import dodge as f_dodge
from facefiltering.filters import gamma as f_gamma
from facefiltering.filters import orton_effect as f_orton
from facefiltering.validate import ensure_bgr_u8

# Stable UI order
_FILTER_ORDER: List[Tuple[str, Callable[..., np.ndarray]]] = [
    (f_bloom.DISPLAY_NAME, f_bloom.apply),
    (f_orton.DISPLAY_NAME, f_orton.apply),
    (f_gamma.DISPLAY_NAME, f_gamma.apply),
    (f_dodge.DISPLAY_NAME, f_dodge.apply),
]

FILTER_NAMES: List[str] = [n for n, _ in _FILTER_ORDER]
_REGISTRY: Dict[str, Callable[..., np.ndarray]] = dict(_FILTER_ORDER)

# Methodology = how the filter is computed.
FILTER_METHODOLOGIES: Dict[str, str] = {
    f_bloom.DISPLAY_NAME: "Spatial neighborhood",
    f_orton.DISPLAY_NAME: "Spatial neighborhood",
    f_gamma.DISPLAY_NAME: "Point-wise intensity mapping",
    f_dodge.DISPLAY_NAME: "Point-wise intensity mapping",
}

METHODOLOGY_DESCRIPTIONS: Dict[str, str] = {
    "All": "show all filters",
    "Spatial neighborhood": "operate on local pixel neighborhoods (kernels/windows)",
    "Point-wise intensity mapping": "map each pixel independently with a math transform",
}

METHODOLOGY_NAMES: List[str] = [
    "All",
    "Spatial neighborhood",
    "Point-wise intensity mapping",
]

FILTERS_BY_METHODOLOGY: Dict[str, List[str]] = {
    m: [name for name in FILTER_NAMES if FILTER_METHODOLOGIES.get(name) == m] for m in METHODOLOGY_NAMES
}
FILTERS_BY_METHODOLOGY["All"] = FILTER_NAMES[:]

# Function = what the filter is used for / output effect.
FILTER_FUNCTIONS: Dict[str, str] = {
    f_bloom.DISPLAY_NAME: "Creative stylization",
    f_orton.DISPLAY_NAME: "Creative stylization",
    f_gamma.DISPLAY_NAME: "Tone / brightness adjustment",
    f_dodge.DISPLAY_NAME: "Tone / brightness adjustment",
}

FUNCTION_NAMES: List[str] = [
    "Tone / brightness adjustment",
    "Creative stylization",
]

FILTERS_BY_FUNCTION: Dict[str, List[str]] = {
    fn: [name for name in FILTER_NAMES if FILTER_FUNCTIONS.get(name) == fn] for fn in FUNCTION_NAMES
}

# Cross-cutting tag: linear convolution-based filters.
CONVOLUTION_FILTERS: List[str] = [
    f_bloom.DISPLAY_NAME,
    f_orton.DISPLAY_NAME,
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
    kwargs match UI / CLI: gamma, dodge_strength, bloom_thresh, bloom_sigma,
    bloom_intensity, orton_sigma, orton_strength
    """
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown filter: {name!r}. Choose one of {FILTER_NAMES}.")

    img = ensure_bgr_u8(bgr)

    if name == f_bloom.DISPLAY_NAME:
        return fn(
            img,
            threshold=int(kwargs.get("bloom_thresh", 180)),
            sigma=float(kwargs.get("bloom_sigma", 2.5)),
            intensity=float(kwargs.get("bloom_intensity", 0.7)),
        )
    if name == f_orton.DISPLAY_NAME:
        return fn(
            img,
            sigma=float(kwargs.get("orton_sigma", 2.0)),
            strength=float(kwargs.get("orton_strength", 0.6)),
        )
    if name == f_gamma.DISPLAY_NAME:
        return fn(img, gamma=float(kwargs.get("gamma", 1.0)))
    if name == f_dodge.DISPLAY_NAME:
        return fn(img, strength=float(kwargs.get("dodge_strength", 0.55)))

    raise RuntimeError("Registry dispatch out of sync.")
