"""FaceFiltering: modular classical filters + registry."""

from facefiltering.registry import (
    CATEGORY_NAMES,
    FILTERS_BY_CATEGORY,
    FILTER_CATEGORIES,
    FILTER_NAMES,
    FILTERS_BY_TYPE,
    FILTER_TYPES,
    TYPE_DESCRIPTIONS,
    TYPE_NAMES,
    categories_for_type,
    filters_for_type_and_category,
    apply_filter,
)
from facefiltering.validate import FilterInputError, ensure_bgr_u8

__all__ = [
    "CATEGORY_NAMES",
    "FILTERS_BY_CATEGORY",
    "FILTER_CATEGORIES",
    "TYPE_NAMES",
    "TYPE_DESCRIPTIONS",
    "FILTER_TYPES",
    "FILTERS_BY_TYPE",
    "categories_for_type",
    "filters_for_type_and_category",
    "FILTER_NAMES",
    "apply_filter",
    "ensure_bgr_u8",
    "FilterInputError",
]
