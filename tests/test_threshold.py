import numpy as np
import pytest

from leaf_vein_workflow.threshold import (
    otsu_threshold,
    postprocess_mask,
    smooth_image,
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


def test_smooth_image_returns_2d_float_image():
    image = np.zeros((7, 7), dtype=float)
    image[3, 3] = 10

    smoothed = smooth_image(image, sigma=1.0)

    assert smoothed.dtype == float
    assert smoothed.shape == image.shape
    assert smoothed[3, 3] < image[3, 3]
    assert smoothed[3, 3] > smoothed[0, 0]


def test_smooth_image_rejects_negative_sigma():
    with pytest.raises(ValueError, match="non-negative"):
        smooth_image(np.zeros((3, 3)), sigma=-1)


def test_postprocess_mask_removes_small_objects_and_fills_holes():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[1:6, 1:6] = 1
    mask[3, 3] = 0
    mask[0, 0] = 1

    cleaned = postprocess_mask(
        mask,
        small_object_size=2,
        max_hole_size=2,
        closing_radius=0,
        opening_radius=0,
    )

    assert cleaned.dtype == np.uint8
    assert cleaned[0, 0] == 0
    assert cleaned[3, 3] == 1
    assert np.count_nonzero(cleaned) == 25


def test_postprocess_mask_uses_closing_radius_to_bridge_small_gaps():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 1:3] = 1
    mask[2:5, 4:6] = 1

    cleaned = postprocess_mask(
        mask,
        small_object_size=0,
        max_hole_size=0,
        closing_radius=1,
        opening_radius=0,
    )

    assert cleaned[3, 3] == 1


def test_postprocess_mask_rejects_negative_parameters():
    with pytest.raises(ValueError, match="Opening radius"):
        postprocess_mask(np.ones((3, 3)), opening_radius=-1)


def test_threshold_rejects_unsupported_shape():
    with pytest.raises(ValueError, match="2D grayscale"):
        threshold_image(np.zeros((2, 3, 5)), 0)


def _leaf_on_paper():
    """White paper, a gray leaf with a vein grid (enclosed areoles)."""
    image = np.full((240, 320), 0.95)
    image[20:220, 20:300] = 0.55
    for row in range(20, 220, 10):
        image[row:row + 2, 20:300] = 0.15
    for col in range(20, 300, 10):
        image[20:220, col:col + 2] = 0.15
    return image


def test_vein_threshold_ignores_background_and_picks_dark_veins():
    from leaf_vein_workflow.threshold import estimate_vein_threshold, threshold_image

    image = _leaf_on_paper()
    estimate = estimate_vein_threshold(image, exclude_background=True)

    assert estimate.leaf_mask is not None
    assert estimate.invert is True
    assert estimate.vein_side == "dark"
    vein = threshold_image(image, estimate.threshold, invert=True) & estimate.leaf_mask
    assert np.array_equal(vein, image < 0.3)
    assert estimate.vein_fraction < 0.5


def test_vein_threshold_without_background_uses_the_whole_roi():
    from leaf_vein_workflow.threshold import estimate_vein_threshold

    inside_blade = _leaf_on_paper()[25:215, 25:295]
    estimate = estimate_vein_threshold(inside_blade, exclude_background=True)

    # 葉脈で囲まれた葉身は背景とみなさず、1段階の Otsu になる。
    assert estimate.leaf_mask is None
    assert estimate.invert is True
    assert np.isclose(estimate.vein_fraction, np.mean(inside_blade < 0.3))


def test_reviewed_leaf_region_controls_second_otsu():
    from leaf_vein_workflow.threshold import estimate_vein_threshold, estimate_leaf_mask

    image = _leaf_on_paper()
    _, candidate = estimate_leaf_mask(image)
    assert candidate is not None
    reviewed = np.zeros(image.shape, dtype=bool)
    reviewed[30:90, 30:90] = True
    estimate = estimate_vein_threshold(image, leaf_mask=reviewed)
    np.testing.assert_array_equal(estimate.leaf_mask, reviewed)
    assert estimate.invert
    assert 0.15 < estimate.threshold < 0.55


@pytest.mark.parametrize("leaf", [np.zeros((2, 3), bool), np.ones((3, 2), bool)])
def test_second_otsu_rejects_empty_or_mismatched_leaf_region(leaf):
    from leaf_vein_workflow.threshold import estimate_vein_threshold

    with pytest.raises(ValueError, match="Leaf mask"):
        estimate_vein_threshold(np.arange(6).reshape(2, 3), leaf_mask=leaf)
def test_largest_leaf_component_preserves_holes_and_diagonal_connections():
    from leaf_vein_workflow.threshold import largest_leaf_component
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True
    mask[2, 2] = False
    mask[4, 4] = True
    mask[8, 8] = True
    kept = largest_leaf_component(mask)
    assert kept.sum() == 9
    assert kept[4, 4]
    assert not kept[2, 2]
    assert not kept[8, 8]
    assert not largest_leaf_component(np.zeros((2, 2))).any()
