"""Public compatibility entry point for the leaf-vein napari widget.

The implementation is split by responsibility under :mod:`.widgets`; napari
and existing imports continue to resolve ``CropWidget`` from this module.
"""

from .widgets.crop_widget import CropWidget
from .widgets.layers import (
    ORDER_LABEL_COLORS,
    ORDER_LABELS_LAYER_NAME,
    ORDER_UNCERTAIN_LABEL,
    ROLE_KEY,
    ROLE_ORDER_LABELS,
    ROLE_ROI_CROP,
    ROLE_ROI_SHAPES,
    ROLE_SMOOTHED_IMAGE,
    ROLE_VEIN_MASK,
    SMOOTHED_LAYER_NAME,
    SOURCE_CROP_KEY,
    SOURCE_IMAGE_KEY,
    SOURCE_MASK_KEY,
    VEIN_MASK_LAYER_NAME,
)

__all__ = [
    "CropWidget",
    "ORDER_LABEL_COLORS",
    "ORDER_LABELS_LAYER_NAME",
    "ORDER_UNCERTAIN_LABEL",
    "ROLE_KEY",
    "ROLE_ORDER_LABELS",
    "ROLE_ROI_CROP",
    "ROLE_ROI_SHAPES",
    "ROLE_SMOOTHED_IMAGE",
    "ROLE_VEIN_MASK",
    "SMOOTHED_LAYER_NAME",
    "SOURCE_CROP_KEY",
    "SOURCE_IMAGE_KEY",
    "SOURCE_MASK_KEY",
    "VEIN_MASK_LAYER_NAME",
]
