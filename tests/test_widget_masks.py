"""Widget integration tests: masks."""

import numpy as np

from widget_helpers import (
    SOURCE_IMAGE,
    _advance_to_mask,
    _advance_to_smoothed_image,
    _draw_rectangle,
    _layer_names,
    _wait_for_delayed_mask_update,
)


def test_image_prepares_roi_and_add_roi_creates_crop(widget_flow):
    flow = widget_flow

    assert flow.viewer.layers.selection.active is flow.roi_layer
    assert flow.roi_layer.mode == "add_rectangle"

    _draw_rectangle(flow)
    assert "leaf ROI" not in _layer_names(flow)

    flow.widget.add_roi()
    crop_layer = flow.viewer.layers["leaf ROI"]

    assert flow.image_layer.visible is False
    assert crop_layer.data.shape == (2, 3)
    np.testing.assert_allclose(crop_layer.translate, [1.0, 1.0])
    np.testing.assert_array_equal(crop_layer.data, SOURCE_IMAGE[1:3, 1:4])
    assert flow.viewer.layers.selection.active is crop_layer
    assert flow.widget.gaussian_sigma_spinbox.value() == 1.0
    assert flow.widget.apply_gaussian_button.isEnabled()
    assert not flow.widget.otsu_button.isEnabled()
    assert flow.module.SMOOTHED_LAYER_NAME not in _layer_names(flow)
    assert flow.module.VEIN_MASK_LAYER_NAME not in _layer_names(flow)


def test_gaussian_enables_threshold_controls_without_creating_mask(widget_flow):
    flow = widget_flow
    crop_layer, smoothed_layer = _advance_to_smoothed_image(flow)

    assert flow.viewer.layers.selection.active is smoothed_layer
    assert (
        smoothed_layer.metadata[flow.module.ROLE_KEY]
        == flow.module.ROLE_SMOOTHED_IMAGE
    )
    assert smoothed_layer.data.shape == crop_layer.data.shape
    assert flow.widget.otsu_button.isEnabled()
    assert flow.widget.threshold_slider.isEnabled()
    assert not flow.widget.clean_mask_button.isEnabled()
    assert not flow.widget.create_order_labels_button.isEnabled()

    _wait_for_delayed_mask_update(flow)
    assert flow.module.VEIN_MASK_LAYER_NAME not in _layer_names(flow)
    assert flow.module.ORDER_LABELS_LAYER_NAME not in _layer_names(flow)

    flow.widget.threshold_slider.setValue(1000)
    _wait_for_delayed_mask_update(flow)
    assert flow.module.VEIN_MASK_LAYER_NAME not in _layer_names(flow)


def test_otsu_creates_editable_mask_and_threshold_updates_it(widget_flow):
    flow = widget_flow
    crop_layer, _, mask_layer = _advance_to_mask(flow)

    assert mask_layer.metadata[flow.module.ROLE_KEY] == flow.module.ROLE_VEIN_MASK
    assert mask_layer.data.shape == crop_layer.data.shape
    np.testing.assert_allclose(mask_layer.translate, [1.0, 1.0])
    assert np.any(mask_layer.data == 1)
    assert flow.viewer.layers.selection.active is mask_layer
    assert mask_layer.mode == "paint"
    assert flow.widget.clean_mask_button.isEnabled()
    assert flow.widget.create_order_labels_button.isEnabled()

    foreground_before = int(np.count_nonzero(mask_layer.data))
    # Otsu は少ない側（葉脈）を自動で選ぶ。その側が縮む向きにスライダーを動かす。
    inverted = flow.widget.invert_threshold_checkbox.isChecked()
    flow.widget.threshold_slider.setValue(0 if inverted else 1000)
    _wait_for_delayed_mask_update(flow)

    assert np.count_nonzero(mask_layer.data) <= foreground_before


def test_clean_mask_removes_small_object_and_keeps_paint_mode(widget_flow):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    flow.widget.min_object_size_spinbox.setValue(2)
    flow.widget.max_hole_size_spinbox.setValue(0)
    flow.widget.closing_radius_spinbox.setValue(0)
    flow.widget.opening_radius_spinbox.setValue(0)

    flow.widget.clean_mask()

    assert np.count_nonzero(mask_layer.data) == 0
    assert flow.viewer.layers.selection.active is mask_layer
    assert mask_layer.mode == "paint"


def test_recrop_reuses_crop_and_removes_derived_layers(widget_flow):
    flow = widget_flow
    crop_layer, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    flow.widget.create_order_labels()

    flow.viewer.layers.selection.active = flow.roi_layer
    flow.widget.add_roi()

    assert flow.viewer.layers["leaf ROI"] is crop_layer
    assert flow.viewer.layers.selection.active is crop_layer
    assert flow.image_layer.visible is False
    assert flow.module.SMOOTHED_LAYER_NAME not in _layer_names(flow)
    assert flow.module.VEIN_MASK_LAYER_NAME not in _layer_names(flow)
    assert flow.module.ORDER_LABELS_LAYER_NAME not in _layer_names(flow)
    assert not flow.widget.clean_mask_button.isEnabled()
    assert not flow.widget.create_order_labels_button.isEnabled()


def test_vein_mask_colors_vein_pixels_and_paints_vein_label(widget_flow):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)

    colors = mask_layer.colormap.color_dict
    assert colors[0][3] == 0
    assert colors[1][3] > 0
    assert mask_layer.selected_label == 1
