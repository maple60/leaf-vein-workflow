import numpy as np
import pytest

from leaf_vein_workflow.skeleton import (
    create_skeleton_helpers,
    make_branch_table,
    make_node_table,
    skeletonize_mask,
)


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


def test_create_skeleton_helpers_builds_diameter_branch_and_node_tables():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 1:6] = True
    mask[1:4, 3] = True

    helpers = create_skeleton_helpers(mask, spur_length_px=99, display_radius=2)

    assert helpers.skeleton.dtype == np.uint8
    assert helpers.skeleton.shape == mask.shape
    assert helpers.skeleton_display.shape == mask.shape
    assert helpers.diameter_on_skeleton.shape == mask.shape
    assert np.isfinite(helpers.diameter_on_skeleton[helpers.skeleton > 0]).all()
    assert "branch_id" in helpers.branch_table.columns
    assert "branch-distance" in helpers.branch_table.columns
    assert "spur-candidate" in helpers.branch_table.columns
    assert len(helpers.branch_table) > 0
    assert helpers.endpoint_coordinates.shape[1] == 2
    assert helpers.branchpoint_coordinates.shape[1] == 2


def test_spur_candidate_threshold_is_applied():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 1:6] = True
    mask[1:4, 3] = True

    permissive = create_skeleton_helpers(mask, spur_length_px=99, display_radius=0)
    strict = create_skeleton_helpers(mask, spur_length_px=0, display_radius=0)

    assert permissive.branch_table["spur-candidate"].any()
    assert not strict.branch_table["spur-candidate"].any()


def test_branch_and_node_table_helpers_accept_existing_skeleton():
    skeleton = np.zeros((5, 5), dtype=np.uint8)
    skeleton[2, 1:4] = 1
    diameter = np.where(skeleton > 0, 2.0, 0.0)

    branch_table = make_branch_table(skeleton, diameter, spur_length_px=20)
    node_table = make_node_table(skeleton)

    assert list(branch_table["branch_id"]) == [0]
    assert branch_table.loc[0, "branch-distance"] == 2.0
    assert set(node_table["node-type"]) == {"endpoint", "path"}
