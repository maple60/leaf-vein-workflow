import numpy as np
import pytest

from leaf_vein_workflow.skeleton import skeletonize_mask


def test_skeletonize_mask_creates_skeleton_from_2d_mask():
    mask = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    skeleton = skeletonize_mask(mask)

    assert skeleton.dtype == np.uint8
    assert skeleton.shape == mask.shape
    assert np.any(skeleton == 1)
    assert np.count_nonzero(skeleton) < np.count_nonzero(mask)


def test_skeletonize_mask_treats_positive_labels_as_foreground():
    mask = np.array(
        [
            [0, 0, 0],
            [2, 3, 4],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    skeleton = skeletonize_mask(mask)

    np.testing.assert_array_equal(
        skeleton,
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
    )


def test_skeletonize_mask_rejects_empty_mask():
    with pytest.raises(ValueError, match="empty"):
        skeletonize_mask(np.zeros((0, 2), dtype=np.uint8))


def test_skeletonize_mask_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        skeletonize_mask(np.zeros((2, 3, 1), dtype=np.uint8))
