"""Widget integration tests: order labels."""

from types import SimpleNamespace

import numpy as np
import pytest

from widget_helpers import (
    _advance_to_mask,
    _has_qt_backend,
    _patch_napari_dirs,
    _wait_for_delayed_mask_update,
)


def test_paint_order_picker_sets_the_painted_label(widget_flow):
    flow = widget_flow
    _advance_to_mask(flow)
    combo = flow.widget.paint_order_combo
    combo.setCurrentIndex(combo.findData(2))

    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    assert order_layer.mode == "paint"
    assert order_layer.selected_label == 2

    combo.setCurrentIndex(combo.findData(3))
    assert order_layer.selected_label == 3
    assert flow.viewer.layers.selection.active is order_layer

    # napari 側でラベルを変えると、dock の選択も追従する。
    order_layer.selected_label = 1
    assert combo.currentData() == 1


def test_order_labels_restrict_edits_to_confirmed_mask(widget_flow):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)

    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]

    assert order_layer.metadata[flow.module.ROLE_KEY] == flow.module.ROLE_ORDER_LABELS
    assert mask_layer.visible is False
    assert order_layer.colormap is not mask_layer.colormap
    assert set(flow.module.ORDER_LABEL_COLORS).issubset(
        order_layer.colormap.color_dict
    )
    np.testing.assert_array_equal(
        order_layer.data,
        np.array([[0, 4, 0], [0, 0, 0]], dtype=np.uint8),
    )
    np.testing.assert_allclose(order_layer.translate, [1.0, 1.0])
    assert flow.viewer.layers.selection.active is order_layer
    assert order_layer.mode == "paint"

    order_layer.brush_size = 1
    order_layer.paint((0, 0), 2)
    flow.app.processEvents()
    assert int(np.asarray(order_layer.data)[0, 0]) == 0

    order_layer.paint((0, 1), 2)
    flow.app.processEvents()
    assert int(np.asarray(order_layer.data)[0, 1]) == 2

    order_layer.data = np.array([[1, 2, 0], [3, 0, 4]], dtype=np.uint8)
    flow.app.processEvents()
    np.testing.assert_array_equal(
        order_layer.data,
        np.array([[0, 2, 0], [0, 0, 0]], dtype=np.uint8),
    )


def test_create_order_labels_reuses_layer_and_tracks_mask_changes(widget_flow):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    order_layer.paint((0, 1), 2)
    flow.app.processEvents()

    flow.widget.create_order_labels()

    assert flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME] is order_layer
    assert int(np.asarray(order_layer.data)[0, 1]) == 2

    mask_layer.data = np.zeros_like(mask_layer.data)
    flow.app.processEvents()
    assert np.count_nonzero(order_layer.data) == 0


def test_vein_added_to_mask_after_order_labels_becomes_uncertain(widget_flow):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    order_layer.data = np.array([[0, 2, 0], [0, 0, 0]], dtype=np.uint8)
    flow.app.processEvents()

    mask_layer.data = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    flow.app.processEvents()

    np.testing.assert_array_equal(
        order_layer.data,
        np.array([[0, 2, 4], [0, 0, 0]], dtype=np.uint8),
    )


def test_mask_and_order_label_display_defaults(widget_flow):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)
    assert mask_layer.opacity == pytest.approx(0.5)
    assert flow.module.CropWidget is not None

    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    assert order_layer.opacity == pytest.approx(0.6)


def test_auto_assign_orders_end_to_end_keeps_hand_painting(monkeypatch, tmp_path):
    """Synthetic leaf scan: Otsu mask, two clicks, then 1°/2° fill uncertain pixels."""
    if not _has_qt_backend():
        pytest.skip("No Qt backend is installed.")
    _patch_napari_dirs(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

    import leaf_vein_workflow._widget as widget_module
    from test_auto_order import _synthetic_leaf

    vein, leaf, secondaries, tertiaries = _synthetic_leaf()
    # 紙 0.95、葉身 0.55、葉脈 0.15 の透過スキャンを模した画像
    image = np.where(leaf, 0.55, 0.95)
    image[vein] = 0.15

    app = QApplication.instance() or QApplication([])
    viewer = ViewerModel()
    widget = widget_module.CropWidget(viewer, settings_path=tmp_path / "none.yaml")
    try:
        viewer.add_image(image, name="leaf")
        app.processEvents()
        # 0.02 mm/px（≈ 1270 dpi）で校正する
        widget.calibration_method_combo.setCurrentIndex(
            widget.calibration_method_combo.findData("mm_per_pixel")
        )
        widget.calibration_value_input.setText("0.02")
        widget.apply_scale()
        roi = viewer.layers["leaf ROI rectangle"]
        roi.add_rectangles(np.array([[[0, 0], [0, 699], [399, 699], [399, 0]]], float))
        app.processEvents()
        widget.add_roi()
        widget.gaussian_sigma_spinbox.setValue(0.0)
        widget.apply_gaussian_filter()
        widget.use_otsu_threshold()
        widget.confirm_leaf()
        _wait_for_delayed_mask_update(SimpleNamespace(app=app))
        widget.create_order_labels()
        order_layer = viewer.layers[widget_module.ORDER_LABELS_LAYER_NAME]
        offset = np.asarray(order_layer.translate, dtype=int)

        # 手で塗った 3° は自動振り分けで上書きされない
        hand_r, hand_c = np.array(tertiaries[0]) - offset
        painted = np.asarray(order_layer.data).copy()
        painted[hand_r, hand_c] = 3
        order_layer.data = painted

        seeds = widget.mark_base_and_apex()
        assert viewer.layers.selection.active is seeds and seeds.mode == "add"
        seeds.add(np.array([[200, 40], [200, 660]], float) - offset)
        app.processEvents()
        # 2点打つと order_labels に戻って塗れる状態になる（設定の既定）
        assert viewer.layers.selection.active is order_layer
        assert order_layer.mode == "paint"

        changed = widget.auto_assign_orders(background=False)
        labels = np.asarray(order_layer.data)

        assert changed > 0
        assert labels[200 - offset[0], 350 - offset[1]] == 1
        for r, c in secondaries:
            assert labels[r - offset[0], c - offset[1]] == 2, (r, c)
        assert labels[hand_r, hand_c] == 3
        record = widget._layers.processing(order_layer)["auto_order"]
        assert record["n_secondary"] == 10
        assert record["leaf_outline_from_otsu"] is True
        assert not seeds.visible
        np.testing.assert_array_equal(
            seeds.data, np.array([[200, 40], [200, 660]], float) - offset
        )
        assert widget.mark_base_and_apex() is seeds
        assert seeds.visible
    finally:
        widget.close()
        app.processEvents()


def test_marking_can_keep_the_points_layer_active(widget_flow):
    flow = widget_flow
    _advance_to_mask(flow)
    flow.widget.create_order_labels()
    flow.widget._controls.activate_labels_after_marking = False

    seeds = flow.widget.mark_base_and_apex()
    seeds.add(np.array([[0.0, 0.0], [1.0, 2.0]]))
    flow.app.processEvents()

    assert flow.viewer.layers.selection.active is seeds


def test_edit_mask_preserves_orders_and_syncs_paint_events(widget_flow):
    flow = widget_flow
    _, _, mask = _advance_to_mask(flow)
    mask.data = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    flow.widget.create_order_labels()
    order = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    order.data = np.array([[2, 3, 0], [0, 0, 0]], dtype=np.uint8)
    flow.widget.edit_mask_button.click()
    assert flow.viewer.layers.selection.active is mask
    assert mask.visible and not order.visible
    mask.data_setitem((np.array([0, 0]), np.array([0, 2])), np.array([0, 1]))
    flow.app.processEvents()
    assert order.data[0, 0] == 0
    assert order.data[0, 1] == 3
    assert order.data[0, 2] == 4
    flow.widget.create_order_labels()
    assert flow.viewer.layers.selection.active is order
    assert order.visible and not mask.visible
    assert order.data[0, 1] == 3
