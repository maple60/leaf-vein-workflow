import numpy as np
import pytest

from leaf_vein_workflow.calibration import (
    CALIBRATION_INPUT_UNIT_KEY,
    CALIBRATION_METHOD_KEY,
    CALIBRATION_MM_PER_PIXEL_KEY,
    CALIBRATION_SCALE_BAR_PIXEL_LENGTH_KEY,
    calibration_metadata_from,
    dpi_to_mm_per_pixel,
    line_length_pixels,
    make_calibration_metadata,
    micrometres_to_mm_per_pixel,
    millimetres_per_pixel,
    mm_per_pixel_to_dpi,
    scale_bar_to_mm_per_pixel,
)


def test_numeric_calibration_methods_normalize_to_mm_per_pixel():
    assert dpi_to_mm_per_pixel(254) == pytest.approx(0.1)
    assert mm_per_pixel_to_dpi(0.1) == pytest.approx(254.0)
    assert micrometres_to_mm_per_pixel(5) == pytest.approx(0.005)
    assert millimetres_per_pixel(0.02) == pytest.approx(0.02)


def test_scale_bar_uses_two_point_pixel_length_and_real_unit():
    vertices = np.array([[1.0, 2.0], [4.0, 6.0]])

    pixel_length = line_length_pixels(vertices)

    assert pixel_length == pytest.approx(5.0)
    assert scale_bar_to_mm_per_pixel(pixel_length, 1.0, "mm") == pytest.approx(
        0.2
    )
    assert scale_bar_to_mm_per_pixel(pixel_length, 500.0, "um") == pytest.approx(
        0.1
    )


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_numeric_calibration_rejects_non_positive_or_non_finite_values(value):
    with pytest.raises(ValueError):
        millimetres_per_pixel(value)


def test_scale_bar_rejects_invalid_line_or_unit():
    with pytest.raises(ValueError, match="two endpoints"):
        line_length_pixels(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="greater than zero"):
        line_length_pixels(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unit"):
        scale_bar_to_mm_per_pixel(10, 1, "cm")


def test_calibration_metadata_can_be_copied_without_unrelated_fields():
    metadata = make_calibration_metadata(
        mm_per_pixel=0.01,
        method="scale_bar",
        input_value=1.0,
        input_unit="mm",
        scale_bar_pixel_length=100.0,
    )
    metadata["unrelated"] = "keep only on source"

    copied = calibration_metadata_from(metadata)

    assert copied[CALIBRATION_MM_PER_PIXEL_KEY] == pytest.approx(0.01)
    assert copied[CALIBRATION_METHOD_KEY] == "scale_bar"
    assert copied[CALIBRATION_INPUT_UNIT_KEY] == "mm"
    assert copied[CALIBRATION_SCALE_BAR_PIXEL_LENGTH_KEY] == pytest.approx(100.0)
    assert "unrelated" not in copied
