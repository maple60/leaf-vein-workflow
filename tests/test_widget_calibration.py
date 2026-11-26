"""Widget integration tests: calibration."""

import numpy as np
import pytest

from widget_helpers import (
    SOURCE_IMAGE,
    _advance_to_mask,
    _layer_names,
)


def test_dpi_calibration_is_inherited_by_derived_layers(widget_flow):
    flow = widget_flow
    dpi_index = flow.widget.calibration_method_combo.findData("dpi")
    flow.widget.calibration_method_combo.setCurrentIndex(dpi_index)
    flow.widget.calibration_value_input.setText("254.0")

    flow.widget.apply_scale()

    key = flow.module.CALIBRATION_MM_PER_PIXEL_KEY
    assert flow.image_layer.metadata[key] == pytest.approx(0.1)
    np.testing.assert_allclose(flow.image_layer.scale, [1.0, 1.0])

    crop_layer, smoothed_layer, mask_layer = _advance_to_mask(flow)
    for layer in (flow.roi_layer, crop_layer, smoothed_layer, mask_layer):
        assert layer.metadata[key] == pytest.approx(0.1)

    mm_index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(mm_index)
    flow.widget.calibration_value_input.setText("0.025")
    flow.widget.apply_scale()

    for layer in (flow.image_layer, flow.roi_layer, crop_layer, smoothed_layer, mask_layer):
        assert layer.metadata[key] == pytest.approx(0.025)


def test_cleanup_controls_show_metric_equivalents_and_keep_pixel_inputs(widget_flow):
    flow = widget_flow
    assert flow.widget.min_object_size_label.text() == "Small object area (px)"
    assert flow.widget.max_hole_size_label.text() == "Maximum hole area (px)"
    assert flow.widget.closing_radius_label.text() == "Closing radius (px)"
    assert flow.widget.opening_radius_label.text() == "Opening radius (px)"

    mm_index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(mm_index)
    flow.widget.calibration_value_input.setText("0.1")
    flow.widget.apply_scale()

    assert flow.widget.min_object_size_spinbox.value() == 100
    assert flow.widget.max_hole_size_spinbox.value() == 50
    assert flow.widget.min_object_size_label.text() == (
        "Small object area: ≈ 1 mm² (100 px)"
    )
    assert flow.widget.max_hole_size_label.text() == (
        "Maximum hole area: ≈ 0.5 mm² (50 px)"
    )
    assert flow.widget.closing_radius_label.text() == (
        "Closing radius: ≈ 100 µm (1 px)"
    )
    assert flow.widget.opening_radius_label.text() == (
        "Opening radius: ≈ 0 µm (0 px)"
    )

    flow.widget.min_object_size_spinbox.setValue(25)
    flow.widget.opening_radius_spinbox.setValue(2)
    assert flow.widget.min_object_size_label.text() == (
        "Small object area: ≈ 0.25 mm² (25 px)"
    )
    assert flow.widget.opening_radius_label.text() == (
        "Opening radius: ≈ 200 µm (2 px)"
    )

    flow.widget.calibration_value_input.setText("0.025")
    flow.widget.apply_scale()
    assert flow.widget.min_object_size_label.text() == (
        "Small object area: ≈ 0.015625 mm² (25 px)"
    )
    assert flow.widget.max_hole_size_label.text() == (
        "Maximum hole area: ≈ 0.03125 mm² (50 px)"
    )
    assert flow.widget.closing_radius_label.text() == (
        "Closing radius: ≈ 25 µm (1 px)"
    )
    assert flow.widget.opening_radius_label.text() == (
        "Opening radius: ≈ 50 µm (2 px)"
    )


def test_manual_scale_bar_calibration_and_method_replacement(widget_flow):
    flow = widget_flow
    crop_layer, _, mask_layer = _advance_to_mask(flow)
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    assert not flow.image_layer.visible

    scale_bar_index = flow.widget.calibration_method_combo.findData("scale_bar")
    flow.widget.calibration_method_combo.setCurrentIndex(scale_bar_index)
    flow.widget.draw_scale_bar_line()

    line_layer = flow.viewer.layers["leaf scale bar"]
    assert flow.image_layer.visible
    assert flow.viewer.layers.selection.active is line_layer
    line_layer.add_lines(np.array([[[0.0, 0.0], [0.0, 10.0]]]))
    flow.widget.calibration_value_input.setText("1.0")
    mm_index = flow.widget.calibration_unit_combo.findData("mm")
    flow.widget.calibration_unit_combo.setCurrentIndex(mm_index)

    flow.widget.apply_scale()

    key = flow.module.CALIBRATION_MM_PER_PIXEL_KEY
    bar_key = flow.module.CALIBRATION_SCALE_BAR_PIXEL_LENGTH_KEY
    assert line_layer.mode == "select"
    assert flow.image_layer.metadata[key] == pytest.approx(0.1)
    assert crop_layer.metadata[key] == pytest.approx(0.1)
    assert mask_layer.metadata[key] == pytest.approx(0.1)
    assert order_layer.metadata[key] == pytest.approx(0.1)
    assert flow.image_layer.metadata[bar_key] == pytest.approx(10.0)

    dpi_index = flow.widget.calibration_method_combo.findData("dpi")
    flow.widget.calibration_method_combo.setCurrentIndex(dpi_index)
    flow.widget.calibration_value_input.setText("254.0")
    flow.widget.apply_scale()

    assert flow.image_layer.metadata[key] == pytest.approx(0.1)
    for layer in (
        flow.image_layer,
        line_layer,
        crop_layer,
        mask_layer,
        order_layer,
    ):
        assert bar_key not in layer.metadata


def test_manual_scale_bar_requires_exactly_one_line(widget_flow):
    flow = widget_flow
    scale_bar_index = flow.widget.calibration_method_combo.findData("scale_bar")
    flow.widget.calibration_method_combo.setCurrentIndex(scale_bar_index)

    flow.widget.apply_scale()
    assert flow.module.CALIBRATION_MM_PER_PIXEL_KEY not in flow.image_layer.metadata

    flow.widget.draw_scale_bar_line()
    line_layer = flow.viewer.layers["leaf scale bar"]
    line_layer.add_lines(
        np.array(
            [
                [[0.0, 0.0], [0.0, 10.0]],
                [[1.0, 0.0], [1.0, 10.0]],
            ]
        )
    )
    flow.widget.calibration_value_input.setText("1.0")

    flow.widget.apply_scale()

    assert len(line_layer.data) == 2
    assert flow.module.CALIBRATION_MM_PER_PIXEL_KEY not in flow.image_layer.metadata


def test_calibration_status_tracks_the_active_source_image(widget_flow):
    flow = widget_flow
    mm_index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(mm_index)
    flow.widget.calibration_value_input.setText("0.1")
    flow.widget.apply_scale()

    second_image = flow.viewer.add_image(SOURCE_IMAGE.copy(), name="leaf2")
    flow.app.processEvents()
    second_roi = flow.viewer.layers["leaf2 ROI rectangle"]
    flow.widget.calibration_value_input.setText("0.02")
    flow.widget.apply_scale()

    flow.viewer.layers.selection.active = flow.roi_layer
    flow.app.processEvents()
    assert "Scale for leaf:" in flow.widget.calibration_status_label.text()
    assert "100 µm" in flow.widget.calibration_status_label.text()
    assert "equivalent DPI ≈ 254" in flow.widget.calibration_status_label.text()
    assert "≈ 1 mm² (100 px)" in flow.widget.min_object_size_label.text()

    flow.viewer.layers.selection.active = second_roi
    flow.app.processEvents()
    assert "Scale for leaf2:" in flow.widget.calibration_status_label.text()
    assert "20 µm" in flow.widget.calibration_status_label.text()
    assert "equivalent DPI ≈ 1270" in flow.widget.calibration_status_label.text()
    assert "≈ 0.04 mm² (100 px)" in flow.widget.min_object_size_label.text()
    assert second_image.metadata[
        flow.module.CALIBRATION_MM_PER_PIXEL_KEY
    ] == pytest.approx(0.02)


def test_apply_scale_draws_10_mm_display_bar_without_stealing_selection(widget_flow):
    flow = widget_flow
    image = flow.viewer.add_image(np.zeros((200, 300)), name="big")
    flow.app.processEvents()
    flow.viewer.layers.selection.active = image
    index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(index)
    flow.widget.calibration_value_input.setText("0.1")

    flow.widget.apply_scale()

    bar = flow.viewer.layers["big 10 mm bar"]
    assert bar.metadata[flow.module.ROLE_KEY] == "scale_indicator"
    start, end = np.asarray(bar.data[0])
    assert np.linalg.norm(end - start) == pytest.approx(100.0)
    assert flow.viewer.layers.selection.active is image

    flow.widget.calibration_value_input.setText("0.01")
    flow.widget.apply_scale()

    assert "big 10 mm bar" not in _layer_names(flow)
    bar = flow.viewer.layers["big 1 mm bar"]
    start, end = np.asarray(bar.data[0])
    assert np.linalg.norm(end - start) == pytest.approx(100.0)
    assert sum(1 for name in _layer_names(flow) if name.startswith("big ") and name.endswith(" bar")) == 1


def test_scale_bar_stays_above_layers_added_later(widget_flow):
    flow = widget_flow
    flow.widget.calibration_value_input.setText("300")
    flow.widget.apply_scale()
    bar = flow.viewer.layers["leaf 10 mm bar"] if "leaf 10 mm bar" in flow.viewer.layers else None
    if bar is None:
        bar = next(layer for layer in flow.viewer.layers if layer.name.endswith(" bar"))

    _advance_to_mask(flow)
    flow.app.processEvents()

    assert flow.viewer.layers[-1] is bar
