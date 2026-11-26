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


def test_metric_calibration_adds_mm_columns_without_changing_pixel_columns():
    skeleton = np.zeros((5, 5), dtype=np.uint8)
    skeleton[2, 1:4] = 1
    diameter = np.where(skeleton > 0, 2.0, 0.0)

    branch_table = make_branch_table(
        skeleton,
        diameter,
        spur_length_px=20,
        mm_per_pixel=0.01,
    )
    node_table = make_node_table(skeleton, mm_per_pixel=0.01)

    assert branch_table.loc[0, "branch-distance"] == 2.0
    assert branch_table.loc[0, "branch-distance-mm"] == pytest.approx(0.02)
    assert branch_table.loc[0, "mean-diameter-mm"] == pytest.approx(
        branch_table.loc[0, "mean-diameter-px"] * 0.01
    )
    np.testing.assert_allclose(
        node_table["coord-y-mm"],
        node_table["coord-y-px"] * 0.01,
    )


def test_uncalibrated_tables_remain_pixel_only():
    skeleton = np.zeros((5, 5), dtype=np.uint8)
    skeleton[2, 1:4] = 1
    diameter = np.where(skeleton > 0, 2.0, 0.0)

    branch_table = make_branch_table(skeleton, diameter)
    node_table = make_node_table(skeleton)

    assert "branch-distance-mm" not in branch_table
    assert "coord-y-mm" not in node_table


def test_metric_calibration_rejects_non_finite_pixel_size():
    skeleton = np.zeros((5, 5), dtype=np.uint8)
    skeleton[2, 1:4] = 1
    diameter = np.where(skeleton > 0, 2.0, 0.0)

    with pytest.raises(ValueError, match="finite"):
        make_branch_table(skeleton, diameter, mm_per_pixel=np.nan)


def _bumpy_thick_vein():
    """A 40 px wide horizontal vein, 400 px long, with small bumps on its edges."""
    mask = np.zeros((120, 480), bool)
    mask[40:80, 40:440] = True
    for col in range(70, 420, 30):
        mask[34:40, col:col + 6] = True  # 上の縁のでこぼこ
        mask[80:86, col + 15:col + 21] = True  # 下の縁のでこぼこ
    return mask


def _length(skeleton):
    from skan import Skeleton
    from skan.csr import summarize

    return summarize(Skeleton(skeleton.astype(np.uint8)), separator="-")["branch-distance"].sum()


def test_prune_spurs_removes_bumps_on_a_thick_vein():
    from skimage.morphology import skeletonize

    from leaf_vein_workflow.skeleton import prune_spurs

    mask = _bumpy_thick_vein()
    skeleton = skeletonize(mask)
    pruned = prune_spurs(skeleton, mask)

    # 骨格の中心線はおよそ 400 - 40 = 360 px（両端は半径ぶん内側で終わる）
    assert _length(skeleton) > 1.2 * 360
    assert _length(pruned) == pytest.approx(360, rel=0.1)


def test_prune_spurs_keeps_real_branches_and_thin_veins():
    from skimage.morphology import skeletonize

    from leaf_vein_workflow.skeleton import prune_spurs

    mask = _bumpy_thick_vein()
    mask[80:200, 238:244] = False
    mask = np.pad(mask, ((0, 100), (0, 0)))
    mask[80:200, 238:244] = True  # 太い脈から出る長さ 120 px の本物の枝（幅 6 px）
    skeleton = skeletonize(mask)
    pruned = prune_spurs(skeleton, mask)
    assert pruned[150:190, 230:250].any()

    thin = np.zeros((50, 300), bool)
    thin[24:27, 10:290] = True
    thin_skeleton = skeletonize(thin)
    assert _length(prune_spurs(thin_skeleton, thin)) == pytest.approx(_length(thin_skeleton), rel=0.02)
