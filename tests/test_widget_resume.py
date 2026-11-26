"""Widget integration tests: resume."""

import numpy as np
import pytest

from widget_helpers import (
    SOURCE_IMAGE,
    _advance_to_mask,
)


def test_reset_can_cancel_and_then_start_next_leaf(widget_flow, monkeypatch, tmp_path):
    from qtpy.QtWidgets import QMessageBox
    flow = widget_flow
    _advance_to_mask(flow)
    flow.widget.create_order_labels()
    saved = tmp_path / "existing-result.txt"
    saved.write_text("keep", encoding="utf-8")
    before = list(flow.viewer.layers)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    flow.widget.reset_analysis()
    assert list(flow.viewer.layers) == before
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    flow.widget.reset_analysis()
    flow.app.processEvents()
    assert len(flow.viewer.layers) == 0
    assert not flow.widget._layers.roi_to_image
    assert not flow.widget._layers.order_labels_to_mask
    assert saved.read_text(encoding="utf-8") == "keep"
    assert flow.widget._controls.sections["image"].is_expanded()
    flow.viewer.add_image(SOURCE_IMAGE.copy(), name="next leaf")
    flow.app.processEvents()
    assert "next leaf ROI rectangle" in flow.viewer.layers
    assert flow.widget.add_roi_button.isEnabled()


def test_resume_export_roundtrip_preserves_edits_area_scale_and_other_leaf(widget_flow, tmp_path):
    from skimage.io import imsave
    flow = widget_flow
    _, _, mask = _advance_to_mask(flow)
    mask.data = np.ones((2, 3), dtype=np.uint8)
    flow.widget.create_order_labels()
    original = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    original.data = np.array([[2, 3, 4], [4, 0, 3]], dtype=np.uint8)
    flow.widget._layers.record_processing(original, {
        "auto_order": {"base_rc": [0, 0], "apex_rc": [0, 2]},
    })
    flow.widget.calibration_method_combo.setCurrentIndex(flow.widget.calibration_method_combo.findData('mm_per_pixel'))
    flow.widget.calibration_value_input.setText('0.1')
    flow.widget.apply_scale()
    flow.widget.leaf_area_input.setText('12')
    first = flow.widget.export_measurements(tmp_path / 'exports')
    source = tmp_path / 'original.tif'
    imsave(source, SOURCE_IMAGE.astype(np.uint8), check_contrast=False)
    labels_before = original.data.copy()
    restored = flow.widget.load_export(first.metadata_yaml.parent, source_path=source)
    flow.app.processEvents()
    assert restored is not None and restored is not original
    np.testing.assert_array_equal(restored.data, labels_before)
    np.testing.assert_array_equal(original.data, labels_before)
    assert tuple(restored.translate) == (1, 1)
    assert restored.metadata[flow.module.CALIBRATION_MM_PER_PIXEL_KEY] == pytest.approx(0.1)
    assert flow.widget._measurement_workflow.leaf_area_mm2() == pytest.approx(12)
    assert flow.viewer.layers.selection.active is restored
    restored_mask = flow.widget._layers.order_labels_to_mask[id(restored)]
    restored_crop = flow.widget._layers.mask_to_crop[id(restored_mask)]
    restored_leaf = flow.widget._layers.crop_to_leaf[id(restored_crop)]
    restored_smoothed = flow.widget._layers.crop_to_smoothed[id(restored_crop)]
    assert restored.visible and restored_smoothed.visible
    assert not restored_mask.visible and not restored_leaf.visible
    seeds = flow.widget._auto_order_workflow.seeds_layer(restored)
    assert seeds is not None and not seeds.visible
    np.testing.assert_array_equal(seeds.data, [[0, 0], [0, 2]])
    restored.data_setitem((np.array([1]), np.array([0])), 3)
    second = flow.widget.export_measurements(tmp_path / 'exports')
    saved_again = np.load(second.labels.order_labels_npy)
    assert saved_again[1, 0] == 3
    np.testing.assert_array_equal(np.load(first.labels.order_labels_npy), labels_before)
    assert original.data[1, 0] == 4


def test_resume_restores_confirmed_leaf_area_without_reapplying_largest_filter(widget_flow, tmp_path):
    from skimage.io import imsave
    flow = widget_flow
    crop, _, _ = _advance_to_mask(flow)
    flow.widget.create_order_labels()
    flow.widget.leaf_area_source_combo.setCurrentIndex(flow.widget.leaf_area_source_combo.findData('leaf_mask'))
    first = flow.widget.export_measurements(tmp_path / 'exports')
    source = tmp_path / 'original.tif'
    imsave(source, SOURCE_IMAGE.astype(np.uint8), check_contrast=False)
    expected = np.load(first.leaf_mask_npy)
    restored = flow.widget.load_export(first.metadata_yaml, source_path=source)
    flow.app.processEvents()
    assert restored is not None
    new_mask = flow.widget._layers.order_labels_to_mask[id(restored)]
    new_crop = flow.widget._layers.mask_to_crop[id(new_mask)]
    np.testing.assert_array_equal(flow.widget._layers.confirmed_leaf_mask(new_crop), expected)
    assert flow.widget.leaf_area_source_combo.currentData() == 'leaf_mask'
    assert flow.widget.measure_veins() is not None
