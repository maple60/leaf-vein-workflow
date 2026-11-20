"""Leaf confirmation, area provenance, and re-extraction boundaries."""

import numpy as np
import pytest

from test_widget_import import (
    widget_flow,
    _advance_to_smoothed_image,
    _wait_for_delayed_mask_update,
)


def _review_leaf(flow, *, calibrated=True):
    crop, _ = _advance_to_smoothed_image(flow)
    if calibrated:
        flow.widget.calibration_method_combo.setCurrentIndex(
            flow.widget.calibration_method_combo.findData("mm_per_pixel")
        )
        flow.widget.calibration_value_input.setText("0.1")
        flow.widget.apply_scale()
    flow.widget.use_otsu_threshold()
    leaf = flow.viewer.layers[flow.module.LEAF_MASK_LAYER_NAME]
    leaf.data = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.uint8)
    return crop, leaf


def _label_veins(flow):
    flow.widget.create_order_labels()
    order = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    order.data = (np.asarray(order.data) > 0).astype(np.uint8)
    return order


def test_otsu_waits_for_manual_leaf_confirmation(widget_flow):
    flow = widget_flow
    crop, leaf = _review_leaf(flow)
    assert flow.viewer.layers.selection.active is leaf
    assert leaf.mode == "paint"
    assert flow.module.VEIN_MASK_LAYER_NAME not in flow.viewer.layers
    assert not flow.widget.create_order_labels_button.isEnabled()
    flow.widget.threshold_slider.setValue(1000)
    _wait_for_delayed_mask_update(flow)
    assert flow.module.VEIN_MASK_LAYER_NAME not in flow.viewer.layers
    assert id(crop) not in flow.widget._layers.leaf_masks


def test_confirm_uses_edited_region_for_area_and_veins_and_export(widget_flow, tmp_path):
    from leaf_vein_workflow.provenance import array_digest
    import yaml
    import pandas as pd

    flow = widget_flow
    crop, leaf = _review_leaf(flow)
    expected = np.asarray(leaf.data) > 0
    flow.widget.confirm_leaf()
    vein = flow.viewer.layers[flow.module.VEIN_MASK_LAYER_NAME]
    assert not np.any(np.asarray(vein.data)[~expected])
    assert flow.widget._measurement_workflow.leaf_area_mm2() == pytest.approx(0.04)
    assert flow.widget.leaf_area_source_combo.currentData() == "leaf_mask"
    assert "0.04 mm²" in flow.widget.leaf_area_status_label.text()
    _label_veins(flow)
    rows = flow.widget.measure_veins()
    assert rows is not None
    assert all(row["leaf_area_mm2"] == pytest.approx(0.04) for row in rows)
    assert all(row["leaf_area_source"] == "leaf_mask" for row in rows)
    result = flow.widget.export_measurements(tmp_path / "export")
    np.testing.assert_array_equal(np.load(result.leaf_mask_npy), expected)
    record = yaml.safe_load(result.metadata_yaml.read_text(encoding="utf-8"))
    assert record["leaf_area"]["state"] == "specimen"
    assert record["leaf_area"]["scope"] == "current_roi"
    assert record["leaf_area"]["sha256"] == array_digest(expected.astype(np.uint8))
    exported = pd.read_csv(result.labels.measurements_csv)
    assert set(exported.leaf_area_source) == {"leaf_mask"}


def test_leaf_edits_require_reconfirmation_even_with_external_area(widget_flow, monkeypatch):
    flow = widget_flow
    _, leaf = _review_leaf(flow)
    flow.widget.confirm_leaf()
    order = _label_veins(flow)
    leaf.data = np.ones((2, 3), dtype=np.uint8)
    warnings = []
    monkeypatch.setattr(type(flow.widget._controls), "warn", lambda self, text: warnings.append(text))
    flow.widget.leaf_area_source_combo.setCurrentIndex(flow.widget.leaf_area_source_combo.findData("input"))
    flow.widget.leaf_area_input.setText("10")
    flow.widget._layers.activate(order)
    assert flow.widget.measure_veins() is None
    assert flow.widget.export_measurements() is None
    assert any("Confirm Leaf" in warning for warning in warnings)


def test_reconfirmation_can_cancel_or_replace_veins_and_orders(widget_flow, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    flow = widget_flow
    crop, leaf = _review_leaf(flow)
    flow.widget.confirm_leaf()
    order = _label_veins(flow)
    vein = flow.viewer.layers[flow.module.VEIN_MASK_LAYER_NAME]
    before = np.array(vein.data, copy=True)
    leaf.data = np.ones((2, 3), dtype=np.uint8)
    flow.widget._layers.activate(leaf)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    flow.widget.confirm_leaf()
    np.testing.assert_array_equal(vein.data, before)
    assert flow.widget._layers.is_present(order)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    flow.widget.confirm_leaf()
    assert not flow.widget._layers.is_present(order)
    assert not flow.widget._layers.is_present(vein)
    assert flow.widget._measurement_workflow.leaf_area_mm2() == pytest.approx(0.06)


def test_empty_leaf_is_not_confirmed(widget_flow, monkeypatch):
    flow = widget_flow
    crop, leaf = _review_leaf(flow)
    leaf.data = np.zeros((2, 3), dtype=np.uint8)
    warnings = []
    monkeypatch.setattr(type(flow.widget._controls), "warn", lambda self, text: warnings.append(text))
    flow.widget.confirm_leaf()
    assert warnings
    assert id(crop) not in flow.widget._layers.leaf_masks
    assert flow.module.VEIN_MASK_LAYER_NAME not in flow.viewer.layers


def test_unscaled_leaf_has_pixel_area_without_invented_mm_area(widget_flow):
    flow = widget_flow
    _, leaf = _review_leaf(flow, calibrated=False)
    flow.widget.confirm_leaf()
    _label_veins(flow)
    rows = flow.widget.measure_veins()
    assert rows is not None
    assert all(row["leaf_area_px"] == 4 for row in rows)
    assert all(row["leaf_area_mm2"] is None for row in rows)
    assert "4 px²" in flow.widget.leaf_area_status_label.text()


def test_recalibration_updates_confirmed_area(widget_flow):
    flow = widget_flow
    _, leaf = _review_leaf(flow)
    flow.widget.confirm_leaf()
    flow.widget.calibration_value_input.setText("0.2")
    flow.widget.apply_scale()
    assert flow.widget._measurement_workflow.leaf_area_mm2() == pytest.approx(0.16)
    assert leaf.metadata[flow.module.CALIBRATION_MM_PER_PIXEL_KEY] == pytest.approx(0.2)
    assert "0.16 mm²" in flow.widget.leaf_area_status_label.text()


def test_roi_replacement_removes_leaf_confirmation(widget_flow):
    flow = widget_flow
    crop, leaf = _review_leaf(flow)
    flow.widget.confirm_leaf()
    flow.widget._layers.activate(flow.roi_layer)
    flow.widget.add_roi()
    assert not flow.widget._layers.is_present(leaf)
    assert id(crop) not in flow.widget._layers.leaf_masks
    assert not flow.widget.confirm_leaf_button.isEnabled()


def test_cleanup_stays_inside_confirmed_leaf(widget_flow):
    flow = widget_flow
    _, leaf = _review_leaf(flow)
    expected = np.asarray(leaf.data) > 0
    flow.widget.confirm_leaf()
    vein = flow.viewer.layers[flow.module.VEIN_MASK_LAYER_NAME]
    vein.data = np.ones((2, 3), dtype=np.uint8)
    flow.widget.min_object_size_spinbox.setValue(0)
    flow.widget.max_hole_size_spinbox.setValue(0)
    flow.widget.closing_radius_spinbox.setValue(0)
    flow.widget.opening_radius_spinbox.setValue(0)
    flow.widget.clean_mask()
    np.testing.assert_array_equal(np.asarray(vein.data) > 0, expected)


def test_external_area_is_explicit_and_never_falls_back_when_empty(widget_flow):
    flow = widget_flow
    _review_leaf(flow)
    flow.widget.confirm_leaf()
    _label_veins(flow)
    flow.widget.leaf_area_source_combo.setCurrentIndex(flow.widget.leaf_area_source_combo.findData("input"))
    rows = flow.widget.measure_veins()
    assert all(row["leaf_area_mm2"] is None for row in rows)
    flow.widget.leaf_area_input.setText("5")
    rows = flow.widget.measure_veins()
    assert all(row["leaf_area_mm2"] == 5 for row in rows)
    assert all(row["leaf_area_source"] == "input" for row in rows)


def test_area_source_shows_measured_value_or_external_input(widget_flow):
    flow = widget_flow
    _review_leaf(flow)
    flow.widget.confirm_leaf()
    controls = flow.widget._controls
    assert not controls.leaf_area_input.isVisibleTo(flow.widget)
    assert not controls.leaf_area_unit_combo.isVisibleTo(flow.widget)
    assert "0.04 mm²" in controls.confirmed_leaf_area_label.text()
    assert not controls.confirmed_leaf_area_label.isHidden()
    controls.leaf_area_source_combo.setCurrentIndex(controls.leaf_area_source_combo.findData("input"))
    assert not controls.leaf_area_input.isHidden()
    assert not controls.leaf_area_unit_combo.isHidden()
    assert controls.confirmed_leaf_area_label.isHidden()
    controls.leaf_area_input.setText("12.3")
    controls.leaf_area_source_combo.setCurrentIndex(controls.leaf_area_source_combo.findData("leaf_mask"))
    assert controls.leaf_area_input.isHidden()
    controls.leaf_area_source_combo.setCurrentIndex(controls.leaf_area_source_combo.findData("input"))
    assert controls.leaf_area_input.text() == "12.3"
