"""Public compatibility entry point for the leaf-vein napari widget.

The implementation is split by responsibility under :mod:`.widgets`; napari
and existing imports continue to resolve ``CropWidget`` from this module.
"""

from .widgets.crop_widget import CropWidget
from .calibration import (
    CALIBRATION_INPUT_UNIT_KEY,
    CALIBRATION_INPUT_VALUE_KEY,
    CALIBRATION_METHOD_KEY,
    CALIBRATION_MM_PER_PIXEL_KEY,
    CALIBRATION_SCALE_BAR_PIXEL_LENGTH_KEY,
)
from .widgets.layers import (
    ORDER_LABEL_COLORS,
    ORDER_LABELS_LAYER_NAME,
    ORDER_UNCERTAIN_LABEL,
    ROLE_KEY,
    ROLE_LEAF_MASK,
    LEAF_MASK_LAYER_NAME,
    ROLE_ORDER_LABELS,
    ROLE_ROI_CROP,
    ROLE_ROI_SHAPES,
    ROLE_SCALE_BAR,
    ROLE_SMOOTHED_IMAGE,
    ROLE_VEIN_MASK,
    SMOOTHED_LAYER_NAME,
    SOURCE_CROP_KEY,
    SOURCE_IMAGE_KEY,
    SOURCE_MASK_KEY,
    VEIN_MASK_LAYER_NAME,
)

__all__ = [
    "CALIBRATION_INPUT_UNIT_KEY",
    "CALIBRATION_INPUT_VALUE_KEY",
    "CALIBRATION_METHOD_KEY",
    "CALIBRATION_MM_PER_PIXEL_KEY",
    "CALIBRATION_SCALE_BAR_PIXEL_LENGTH_KEY",
    "CropWidget",
    "ORDER_LABEL_COLORS",
    "ORDER_LABELS_LAYER_NAME",
    "ORDER_UNCERTAIN_LABEL",
    "ROLE_KEY",
    "ROLE_LEAF_MASK",
    "LEAF_MASK_LAYER_NAME",
    "ROLE_ORDER_LABELS",
    "ROLE_ROI_CROP",
    "ROLE_ROI_SHAPES",
    "ROLE_SCALE_BAR",
    "ROLE_SMOOTHED_IMAGE",
    "ROLE_VEIN_MASK",
    "SMOOTHED_LAYER_NAME",
    "SOURCE_CROP_KEY",
    "SOURCE_IMAGE_KEY",
    "SOURCE_MASK_KEY",
    "VEIN_MASK_LAYER_NAME",
]
