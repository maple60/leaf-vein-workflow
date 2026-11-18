"""Pixel-to-metric calibration helpers independent of napari and Qt."""

from collections.abc import Mapping

import numpy as np


MM_PER_INCH = 25.4

CALIBRATION_MM_PER_PIXEL_KEY = "leaf_vein_workflow_mm_per_pixel"
CALIBRATION_METHOD_KEY = "leaf_vein_workflow_calibration_method"
CALIBRATION_INPUT_VALUE_KEY = "leaf_vein_workflow_calibration_input_value"
CALIBRATION_INPUT_UNIT_KEY = "leaf_vein_workflow_calibration_input_unit"
CALIBRATION_SCALE_BAR_PIXEL_LENGTH_KEY = (
    "leaf_vein_workflow_scale_bar_pixel_length"
)

CALIBRATION_METADATA_KEYS = (
    CALIBRATION_MM_PER_PIXEL_KEY,
    CALIBRATION_METHOD_KEY,
    CALIBRATION_INPUT_VALUE_KEY,
    CALIBRATION_INPUT_UNIT_KEY,
    CALIBRATION_SCALE_BAR_PIXEL_LENGTH_KEY,
)


def validate_positive_finite(value: float, name: str) -> float:
    """Return a finite positive float or raise a user-facing error."""
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a finite value greater than zero.")
    return converted


def dpi_to_mm_per_pixel(dpi: float) -> float:
    """Convert dots per inch to millimetres represented by one pixel."""
    return MM_PER_INCH / validate_positive_finite(dpi, "DPI")


def mm_per_pixel_to_dpi(mm_per_pixel: float) -> float:
    """Return the equivalent DPI represented by millimetres per pixel."""
    return MM_PER_INCH / millimetres_per_pixel(mm_per_pixel)


def micrometres_to_mm_per_pixel(micrometres_per_pixel: float) -> float:
    """Convert micrometres per pixel to millimetres per pixel."""
    value = validate_positive_finite(
        micrometres_per_pixel,
        "Micrometres per pixel",
    )
    return value / 1000.0


def millimetres_per_pixel(value: float) -> float:
    """Validate and return a direct millimetres-per-pixel calibration."""
    return validate_positive_finite(value, "Millimetres per pixel")


def line_length_pixels(vertices: np.ndarray) -> float:
    """Return the Euclidean pixel length of one two-point scale-bar line."""
    points = np.asarray(vertices, dtype=float)
    if points.shape != (2, 2):
        raise ValueError("Scale bar must be one 2D line with exactly two endpoints.")
    if not np.isfinite(points).all():
        raise ValueError("Scale bar endpoints must contain finite coordinates.")
    return validate_positive_finite(
        float(np.linalg.norm(points[1] - points[0])),
        "Scale bar pixel length",
    )


def scale_bar_to_mm_per_pixel(
    pixel_length: float,
    real_length: float,
    unit: str,
) -> float:
    """Convert a known scale-bar length to millimetres per pixel."""
    pixels = validate_positive_finite(pixel_length, "Scale bar pixel length")
    known_length = validate_positive_finite(real_length, "Scale bar real length")
    if unit == "um":
        length_mm = known_length / 1000.0
    elif unit == "mm":
        length_mm = known_length
    else:
        raise ValueError("Scale bar unit must be 'um' or 'mm'.")
    return length_mm / pixels


def make_calibration_metadata(
    *,
    mm_per_pixel: float,
    method: str,
    input_value: float,
    input_unit: str,
    scale_bar_pixel_length: float | None = None,
) -> dict[str, float | str]:
    """Build the normalized metadata stored on workflow layers."""
    metadata: dict[str, float | str] = {
        CALIBRATION_MM_PER_PIXEL_KEY: millimetres_per_pixel(mm_per_pixel),
        CALIBRATION_METHOD_KEY: str(method),
        CALIBRATION_INPUT_VALUE_KEY: validate_positive_finite(
            input_value,
            "Calibration input",
        ),
        CALIBRATION_INPUT_UNIT_KEY: str(input_unit),
    }
    if scale_bar_pixel_length is not None:
        metadata[CALIBRATION_SCALE_BAR_PIXEL_LENGTH_KEY] = (
            validate_positive_finite(
                scale_bar_pixel_length,
                "Scale bar pixel length",
            )
        )
    return metadata


def calibration_metadata_from(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Copy only scale-calibration fields from arbitrary layer metadata."""
    return {
        key: metadata[key]
        for key in CALIBRATION_METADATA_KEYS
        if key in metadata
    }


SCALE_BAR_CANDIDATES_MM = (10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1)


def choose_scale_bar_length_mm(
    image_width_px: float,
    mm_per_pixel: float,
    preferred_mm: float = 10.0,
    max_fraction: float = 0.4,
) -> float:
    """Pick a round display-bar length that fits within part of the image.

    The preferred length (10 mm) is used whenever it fits; smaller images such
    as microscope fields fall back to the next round length.
    """
    pixel_size = millimetres_per_pixel(mm_per_pixel)
    width = validate_positive_finite(image_width_px, "Image width")
    limit_mm = width * pixel_size * max_fraction
    candidates = [length for length in SCALE_BAR_CANDIDATES_MM if length <= preferred_mm]
    for length in candidates:
        if length <= limit_mm:
            return length
    return candidates[-1]


def scale_bar_segment(
    image_shape: tuple[int, int],
    mm_per_pixel: float,
    length_mm: float,
    margin_fraction: float = 0.04,
) -> np.ndarray:
    """Return (row, col) endpoints of a horizontal bar near the bottom-left."""
    height, width = (float(value) for value in image_shape)
    length_px = validate_positive_finite(length_mm, "Scale bar length") / (
        millimetres_per_pixel(mm_per_pixel)
    )
    margin = margin_fraction * min(height, width)
    row = height - margin
    return np.array([[row, margin], [row, margin + length_px]], dtype=float)


def format_length_mm(length_mm: float) -> str:
    """Format a display-bar length as mm, or µm below one millimetre."""
    if length_mm < 1.0:
        return f"{length_mm * 1000.0:g} µm"
    return f"{length_mm:g} mm"
