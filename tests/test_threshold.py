import numpy as np
import pytest

from leaf_vein_workflow.threshold import (
    otsu_threshold,
    threshold_image,
    to_grayscale,
)


def test_to_grayscale_accepts_2d_image():
    image = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    grayscale = to_grayscale(image)

    assert grayscale.dtype == float
    np.testing.assert_array_equal(grayscale, image.astype(float))


def test_to_grayscale_converts_rgb_and_ignores_alpha():
    image = np.array(
        [
            [[10, 20, 30, 255], [40, 50, 60, 0]],
            [[70, 80, 90, 128], [100, 110, 120, 64]],
        ],
        dtype=np.uint8,
    )

    grayscale = to_grayscale(image)

    expected = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )
    np.testing.assert_allclose(grayscale, expected)


def test_threshold_image_uses_manual_threshold_for_bright_foreground():
    image = np.array([[0, 1], [2, 3]])

    mask = threshold_image(image, 1.5)

    np.testing.assert_array_equal(mask, np.array([[False, False], [True, True]]))


def test_threshold_image_can_invert_for_dark_foreground():
    image = np.array([[0, 1], [2, 3]])

    mask = threshold_image(image, 1.5, invert=True)

    np.testing.assert_array_equal(mask, np.array([[True, True], [False, False]]))


def test_otsu_threshold_separates_simple_two_level_image():
    image = np.vstack(
        [
            np.zeros((5, 10), dtype=float),
            np.full((5, 10), 10.0, dtype=float),
        ]
    )

    threshold = otsu_threshold(image)
    mask = threshold_image(image, threshold)

    assert 0 < threshold < 10
    np.testing.assert_array_equal(
        mask,
        np.vstack(
            [
                np.zeros((5, 10), dtype=bool),
                np.ones((5, 10), dtype=bool),
            ]
        ),
    )


def test_threshold_rejects_unsupported_shape():
    with pytest.raises(ValueError, match="2D grayscale"):
        threshold_image(np.zeros((2, 3, 5)), 0)
