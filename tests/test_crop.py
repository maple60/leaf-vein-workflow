import numpy as np
import pytest

from leaf_vein_workflow.crop import crop_to_rectangle, rectangle_to_slices


def test_rectangle_to_slices_uses_floor_ceil_bounds():
    rectangle = np.array(
        [
            [1.2, 2.1],
            [1.2, 4.9],
            [3.7, 4.9],
            [3.7, 2.1],
        ]
    )

    y_slice, x_slice = rectangle_to_slices(rectangle, (5, 6))

    assert y_slice == slice(1, 4)
    assert x_slice == slice(2, 5)


def test_crop_clips_roi_to_image_bounds():
    image = np.arange(25).reshape(5, 5)
    rectangle = np.array(
        [
            [-2.0, -1.0],
            [-2.0, 3.0],
            [2.0, 3.0],
            [2.0, -1.0],
        ]
    )

    cropped = crop_to_rectangle(image, rectangle)

    np.testing.assert_array_equal(cropped, image[0:2, 0:3])


def test_crop_preserves_rgb_channels():
    image = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    rectangle = np.array(
        [
            [1.0, 1.0],
            [1.0, 4.0],
            [3.0, 4.0],
            [3.0, 1.0],
        ]
    )

    cropped = crop_to_rectangle(image, rectangle)

    assert cropped.shape == (2, 3, 3)
    np.testing.assert_array_equal(cropped, image[1:3, 1:4, :])


def test_empty_roi_errors():
    image = np.zeros((5, 5))

    with pytest.raises(ValueError, match="ROI is empty"):
        crop_to_rectangle(image, np.empty((0, 2)))


def test_outside_roi_errors():
    image = np.zeros((5, 5))
    rectangle = np.array(
        [
            [-5.0, -5.0],
            [-5.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -5.0],
        ]
    )

    with pytest.raises(ValueError, match="does not overlap"):
        crop_to_rectangle(image, rectangle)


def test_unsupported_image_shape_errors():
    image = np.zeros((3, 5, 5))
    rectangle = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [2.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="2D grayscale"):
        crop_to_rectangle(image, rectangle)
