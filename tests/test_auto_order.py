import numpy as np
import pytest
from skimage.draw import ellipse, line

from leaf_vein_workflow.auto_order import (
    PRIMARY,
    SECONDARY,
    UNCERTAIN,
    auto_assign_orders,
)


def _thick_line(image, r0, c0, r1, c1, width):
    rr, cc = line(r0, c0, r1, c1)
    for dr in range(-(width // 2), width - width // 2):
        for dc in range(-(width // 2), width - width // 2):
            r = np.clip(rr + dr, 0, image.shape[0] - 1)
            c = np.clip(cc + dc, 0, image.shape[1] - 1)
            image[r, c] = True


def _synthetic_leaf():
    """Craspedodromous leaf: thick midrib, secondaries running into the margin,
    and thin veins connecting them (like the konara scan)."""
    shape = (400, 700)
    leaf = np.zeros(shape, bool)
    rr, cc = ellipse(200, 350, 170, 330, shape=shape)
    leaf[rr, cc] = True

    vein = np.zeros(shape, bool)
    _thick_line(vein, 200, 30, 200, 670, 14)  # 1°
    secondaries = []
    for c in range(120, 600, 110):
        for direction in (-1, 1):
            # 葉縁の外まで引き、葉の輪郭で切る（2° が鋸歯に入る型）。
            end_r = 200 + direction * 190
            _thick_line(vein, 200, c, end_r, c + 114, 8)  # 2°
            secondaries.append((200 + direction * 75, c + 45))
            secondaries.append((200 + direction * 140, c + 84))
    # 3°：隣り合う 2° をつなぐ横木と、1° から出て 2° に合流する細い脈
    tertiaries = []
    for c in range(120, 490, 110):
        for direction in (-1, 1):
            r = 200 + direction * 110
            _thick_line(vein, r, c + 66, r, c + 110 + 66, 3)
            tertiaries.append((r, c + 66 + 55))
            _thick_line(vein, 200, c + 55, 200 + direction * 60, c + 110 + 36, 3)
    vein &= leaf
    # 葉の輪郭で切られた所の確認点は除く。
    secondaries = [point for point in secondaries if vein[point]]
    return vein, leaf, secondaries, tertiaries


def test_auto_order_finds_primary_and_secondaries():
    vein, leaf, secondaries, tertiaries = _synthetic_leaf()

    result = auto_assign_orders(
        # 合成画像の葉脈間隔に合う解像度（0.02 mm/px ≈ 1270 dpi）にする。
        vein, (200, 40), (200, 660), leaf_mask=leaf, mm_per_pixel=0.02
    )
    labels = result.labels

    assert labels[200, 350] == PRIMARY
    assert labels[200, 60] == PRIMARY
    for r, c in secondaries:
        assert labels[r, c] == SECONDARY, (r, c)
    for r, c in tertiaries:
        assert labels[r, c] == UNCERTAIN, (r, c)
    assert result.n_secondary == 10
    assert np.all(labels[~vein] == 0)
    assert np.all(labels[vein] > 0)


def test_auto_order_rejects_disconnected_apex():
    vein = np.zeros((50, 100), bool)
    vein[20:24, 5:40] = True
    vein[20:24, 60:95] = True

    with pytest.raises(ValueError, match="not connected"):
        auto_assign_orders(vein, (22, 6), (22, 94), mm_per_pixel=0.1)


def test_holes_in_the_leaf_outline_are_not_treated_as_margin():
    """Pale specks inside the blade leave holes in the Otsu leaf area."""
    vein, leaf, secondaries, _ = _synthetic_leaf()
    holed = leaf.copy()
    # 2° のすぐ脇（葉縁の帯の幅より近い所）に小さな穴を開ける
    for c in (230, 340, 450):
        for direction in (-1, 1):
            r, col = 200 + direction * 80, c + 48 + 9
            holed[r - 4:r + 4, col - 4:col + 4] = False
    holed |= vein  # 葉脈の上には開けない

    result = auto_assign_orders(
        vein, (200, 40), (200, 660), leaf_mask=holed, mm_per_pixel=0.02
    )

    assert result.n_secondary == 10
    for r, c in secondaries:
        assert result.labels[r, c] == SECONDARY, (r, c)
