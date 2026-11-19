import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE_IMAGE = np.arange(25).reshape(5, 5)
ROI_RECTANGLE = np.array(
    [
        [
            [1.0, 1.0],
            [1.0, 4.0],
            [3.0, 4.0],
            [3.0, 1.0],
        ]
    ]
)


def _has_qt_backend():
    qt_backend_names = ("PyQt5", "PyQt6", "PySide2", "PySide6")
    return any(importlib.util.find_spec(name) for name in qt_backend_names)


def _patch_napari_dirs(monkeypatch, tmp_path):
    import platformdirs

    def user_cache_dir(appname=None, appauthor=None, version=None, **kwargs):
        parts = ["cache"]
        if appname:
            parts.append(str(appname))
        if version:
            parts.append(str(version))
        return str(tmp_path.joinpath(*parts))

    def user_config_dir(appname=None, appauthor=None, version=None, **kwargs):
        parts = ["config"]
        if appname:
            parts.append(str(appname))
        if version:
            parts.append(str(version))
        return str(tmp_path.joinpath(*parts))

    monkeypatch.setattr(platformdirs, "user_cache_dir", user_cache_dir)
    monkeypatch.setattr(platformdirs, "user_config_dir", user_config_dir)


@pytest.fixture
def widget_flow(monkeypatch, tmp_path):
    if not _has_qt_backend():
        pytest.skip("No Qt backend is installed.")

    _patch_napari_dirs(monkeypatch, tmp_path)
    # Keep the widget from creating leaf_vein_workflow.yaml in the repository.
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

    import leaf_vein_workflow._widget as widget_module

    app = QApplication.instance() or QApplication([])
    viewer = ViewerModel()
    widget = widget_module.CropWidget(viewer)
    image_layer = viewer.add_image(SOURCE_IMAGE.copy(), name="leaf")
    app.processEvents()
    roi_layer = viewer.layers["leaf ROI rectangle"]

    try:
        yield SimpleNamespace(
            app=app,
            viewer=viewer,
            widget=widget,
            module=widget_module,
            image_layer=image_layer,
            roi_layer=roi_layer,
        )
    finally:
        widget.close()
        app.processEvents()


def _layer_names(flow):
    return {layer.name for layer in flow.viewer.layers}


def _draw_rectangle(flow):
    flow.roi_layer.add_rectangles(ROI_RECTANGLE.copy())
    flow.app.processEvents()


def _advance_to_crop(flow):
    _draw_rectangle(flow)
    flow.widget.add_roi()
    flow.app.processEvents()
    return flow.viewer.layers["leaf ROI"]


def _advance_to_smoothed_image(flow):
    crop_layer = _advance_to_crop(flow)
    flow.widget.apply_gaussian_filter()
    flow.app.processEvents()
    return crop_layer, flow.viewer.layers[flow.module.SMOOTHED_LAYER_NAME]


def _advance_to_mask(flow):
    crop_layer, smoothed_layer = _advance_to_smoothed_image(flow)
    flow.widget.use_otsu_threshold()
    _wait_for_delayed_mask_update(flow)
    mask_layer = flow.viewer.layers[flow.module.VEIN_MASK_LAYER_NAME]
    return crop_layer, smoothed_layer, mask_layer


def _wait_for_delayed_mask_update(flow):
    from qtpy.QtTest import QTest

    # 閾値変更は連続入力をまとめるQTimer経由なので、発火後まで待つ。
    QTest.qWait(250)
    flow.app.processEvents()


def test_workflow_controls_are_scrollable_and_collapsible(widget_flow):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QScrollArea, QToolButton

    flow = widget_flow
    scroll_area = flow.widget.findChild(QScrollArea, "workflow_scroll_area")
    assert scroll_area is not None
    assert scroll_area.widgetResizable()

    def button(name):
        found = flow.widget.findChild(QToolButton, f"workflow_section_{name}")
        assert found is not None
        return found

    calibration = button("scale_calibration")
    cleanup = button("mask_cleanup")
    assert calibration.isChecked()
    assert not cleanup.isChecked()
    assert cleanup.arrowType() == Qt.ArrowType.RightArrow
    assert cleanup.parent().content.isHidden()

    # 1つ開くと他は閉じる（アコーディオン）。
    cleanup.click()
    flow.app.processEvents()
    assert cleanup.isChecked()
    assert cleanup.arrowType() == Qt.ArrowType.DownArrow
    assert not calibration.isChecked()
    assert calibration.parent().content.isHidden()

    cleanup.click()
    flow.app.processEvents()
    assert not cleanup.isChecked()
    assert cleanup.parent().content.isHidden()


def test_completed_stages_open_the_next_section(widget_flow):
    from qtpy.QtWidgets import QToolButton

    flow = widget_flow

    def is_open(name):
        return flow.widget.findChild(
            QToolButton, f"workflow_section_{name}"
        ).isChecked()

    flow.widget.calibration_value_input.setText("300")
    flow.widget.apply_scale()
    assert is_open("roi") and not is_open("scale_calibration")

    _advance_to_crop(flow)
    assert is_open("mask_creation") and not is_open("roi")


def test_widget_module_reexports_public_api(monkeypatch, tmp_path):
    if not _has_qt_backend():
        pytest.skip("No Qt backend is installed.")

    _patch_napari_dirs(monkeypatch, tmp_path)
    pytest.importorskip("napari")

    import leaf_vein_workflow._widget as widget

    public_names = (
        "CALIBRATION_MM_PER_PIXEL_KEY",
        "CropWidget",
        "ORDER_LABEL_COLORS",
        "ORDER_LABELS_LAYER_NAME",
        "ROLE_KEY",
        "ROLE_ORDER_LABELS",
        "ROLE_SCALE_BAR",
        "ROLE_SMOOTHED_IMAGE",
        "ROLE_VEIN_MASK",
        "SMOOTHED_LAYER_NAME",
        "VEIN_MASK_LAYER_NAME",
    )
    assert all(getattr(widget, name, None) is not None for name in public_names)


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
    flow.widget.threshold_slider.setValue(1000)
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



def test_export_measurements_converts_leaf_area_units_and_uses_output_dir(
    widget_flow, tmp_path
):
    import csv
    from pathlib import Path

    flow = widget_flow
    assert Path(flow.widget.output_dir_input.text()) == Path.cwd() / "output"

    mm_index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(mm_index)
    flow.widget.calibration_value_input.setText("0.1")
    flow.widget.apply_scale()

    _, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.ones((2, 3), dtype=np.uint8)
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    order_layer.data = np.array([[1, 1, 1], [3, 0, 4]], dtype=np.uint8)
    flow.app.processEvents()

    assert flow.widget.export_button.isEnabled()
    flow.widget.leaf_area_input.setText("2")
    rows = flow.widget.measure_veins()
    by_name = {row["label_name"]: row for row in rows}
    assert by_name["primary"]["length_mm"] == pytest.approx(0.2)
    assert by_name["primary"]["vla_mm_per_mm2"] == pytest.approx(0.1)
    assert by_name["secondary"]["length_mm"] == 0.0
    assert by_name["major"]["length_mm"] == pytest.approx(
        by_name["primary"]["length_mm"] + by_name["tertiary"]["length_mm"]
    )

    # 0.04 cm² = 4 mm²
    flow.widget.leaf_area_input.setText("0.04")
    flow.widget.leaf_area_unit_combo.setCurrentIndex(
        flow.widget.leaf_area_unit_combo.findData("cm2")
    )
    flow.widget.output_dir_input.setText(str(tmp_path / "out"))
    status_before = flow.widget.status_label.text()

    result = flow.widget.export_measurements()
    assert result.paths is None
    paths = result.labels

    assert paths.measurements_csv.parent == tmp_path / "out"
    with paths.measurements_csv.open(encoding="utf-8") as file:
        exported = {row["label_name"]: row for row in csv.DictReader(file)}
    assert float(exported["primary"]["leaf_area_mm2"]) == pytest.approx(4.0)
    assert float(exported["primary"]["vla_mm_per_mm2"]) == pytest.approx(0.05)
    assert "tertiary" in exported
    assert paths.qc_overlay_png.exists()
    assert paths.branch_table_csv.exists()
    assert flow.widget.status_label.text() == status_before


def test_measure_veins_warns_without_order_labels(widget_flow):
    flow = widget_flow
    _advance_to_mask(flow)

    assert flow.widget.measure_veins() is None
    assert "order labels" in flow.widget.status_label.text()


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


def test_vein_mask_colors_vein_pixels_and_paints_vein_label(widget_flow):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)

    colors = mask_layer.colormap.color_dict
    assert colors[0][3] == 0
    assert colors[1][3] > 0
    assert mask_layer.selected_label == 1


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


def test_export_counts_mask_pixels_without_order_label(widget_flow, tmp_path):
    flow = widget_flow
    _, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    # 次数ラベルを消しゴムで0にした（maskには残っている）状態。
    order_layer.data = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    flow.app.processEvents()

    flow.widget.export_measurements(tmp_path)

    assert flow.widget._measurement_workflow.unlabeled_vein_pixels == 1


def test_vein_paths_take_selected_order_and_export_lengths(widget_flow, tmp_path):
    import csv

    flow = widget_flow
    image = flow.viewer.add_image(np.zeros((200, 300)), name="whole")
    flow.app.processEvents()
    flow.viewer.layers.selection.active = image
    index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(index)
    flow.widget.calibration_value_input.setText("0.1")
    flow.widget.apply_scale()
    flow.viewer.layers.selection.active = image

    paths_layer = flow.widget.draw_vein_paths()
    assert paths_layer.name == "whole vein paths"
    assert paths_layer.mode == "add_path"
    assert paths_layer.metadata[flow.module.CALIBRATION_MM_PER_PIXEL_KEY] == 0.1

    paths_layer.add_paths([np.array([[10.0, 10.0], [110.0, 10.0]])])
    flow.widget.path_order_combo.setCurrentIndex(
        flow.widget.path_order_combo.findData(2)
    )
    paths_layer.add_paths([np.array([[20.0, 10.0], [20.0, 40.0]])])
    paths_layer.add_paths([np.array([[30.0, 10.0], [30.0, 30.0]])])
    assert paths_layer.features["order"].tolist() == [1, 2, 2]

    # 選択したパスの次数を付け替える。
    paths_layer.selected_data = {2}
    flow.widget.path_order_combo.setCurrentIndex(
        flow.widget.path_order_combo.findData(1)
    )
    assert paths_layer.features["order"].tolist() == [1, 2, 1]

    flow.widget.leaf_area_input.setText("1")
    flow.widget.leaf_area_unit_combo.setCurrentIndex(
        flow.widget.leaf_area_unit_combo.findData("cm2")
    )
    result = flow.widget.export_measurements(tmp_path)

    assert result.labels is None
    with result.paths.measurements_csv.open(encoding="utf-8") as file:
        rows = {row["label_name"]: row for row in csv.DictReader(file)}
    assert float(rows["primary"]["length_mm"]) == pytest.approx(12.0)
    assert float(rows["secondary"]["length_mm"]) == pytest.approx(3.0)
    assert float(rows["primary_secondary"]["vla_mm_per_mm2"]) == pytest.approx(0.15)
    assert result.paths.vertices_csv.exists()


def test_settings_file_sets_widget_defaults_and_reset_restores_them(widget_flow, tmp_path):
    from pathlib import Path

    flow = widget_flow
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(
        "calibration:\n  method: mm_per_pixel\n  value: 0.05\n"
        "scale_bar:\n  length_mm: 5\n"
        "mask:\n  gaussian_sigma: 2.5\n  invert_threshold: true\n"
        "cleanup:\n  max_hole_px: 7\n"
        "measurement:\n  leaf_area_unit: cm2\n  output_dir: results\n",
        encoding="utf-8",
    )
    widget = flow.module.CropWidget(flow.viewer, settings_path=settings_path)
    try:
        assert widget.calibration_method_combo.currentData() == "mm_per_pixel"
        assert widget.calibration_value_input.text() == "0.05"
        assert widget._controls.scale_bar_length_mm == 5.0
        assert widget.gaussian_sigma_spinbox.value() == 2.5
        assert widget.invert_threshold_checkbox.isChecked()
        assert widget.max_hole_size_spinbox.value() == 7
        assert widget.min_object_size_spinbox.value() == 100
        assert widget.leaf_area_unit_combo.currentData() == "cm2"
        assert Path(widget.output_dir_input.text()) == Path.cwd() / "results"
        assert str(settings_path) in widget._controls.settings_path_label.text()

        widget._settings_workflow.reset_to_defaults()

        assert widget.calibration_method_combo.currentData() == "dpi"
        assert widget.calibration_value_input.text() == "1200"
        assert widget._controls.scale_bar_length_mm == 10.0
        assert not widget.invert_threshold_checkbox.isChecked()
        assert widget.max_hole_size_spinbox.value() == 50
        assert widget.leaf_area_unit_combo.currentData() == "mm2"
        assert (tmp_path / "config.yaml.bak").exists()
    finally:
        widget.close()


def test_export_metadata_records_actual_processing(widget_flow, tmp_path):
    import yaml

    flow = widget_flow
    dpi_index = flow.widget.calibration_method_combo.findData("dpi")
    flow.widget.calibration_method_combo.setCurrentIndex(dpi_index)
    flow.widget.calibration_value_input.setText("254")
    flow.widget.apply_scale()

    flow.widget.gaussian_sigma_spinbox.setValue(0.5)
    _, _, mask_layer = _advance_to_mask(flow)
    flow.widget.max_hole_size_spinbox.setValue(3)
    flow.widget.clean_mask()
    # Clean Mask の後に手で1画素足す。
    edited = np.asarray(mask_layer.data).copy()
    edited[0, 0] = 1
    edited[0, 1] = 1
    mask_layer.data = edited
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    labels = np.asarray(order_layer.data).copy()
    labels[labels == 4] = 1
    order_layer.data = labels
    flow.app.processEvents()
    flow.widget.leaf_area_input.setText("2.5")
    flow.widget.leaf_area_unit_combo.setCurrentIndex(
        flow.widget.leaf_area_unit_combo.findData("cm2")
    )

    result = flow.widget.export_measurements(tmp_path)

    meta = yaml.safe_load(result.metadata_yaml.read_text(encoding="utf-8"))
    assert meta["image_name"] == "leaf"
    assert meta["calibration"]["method"] == "dpi"
    assert meta["calibration"]["input_value"] == 254.0
    assert meta["leaf_area"] == {"input": "2.5", "unit": "cm2", "mm2": 250.0}
    processing = meta["order_labels"]["processing"]
    assert processing["roi"] == {
        "row_start": 1, "row_stop": 3, "col_start": 1, "col_stop": 4,
    }
    assert processing["gaussian_sigma"] == 0.5
    assert "otsu_threshold" in processing
    assert set(processing["threshold"]) == {"value", "invert"}
    assert len(processing["cleanup"]) == 1
    assert processing["cleanup"][0]["max_hole_px"] == 3
    assert processing["mask_edited_by_hand_after_last_step"] is True
    assert meta["order_labels"]["uncertain_px"] == 0
    assert "defaults_in_effect" in meta["settings"]
    assert "napari" in meta["software"]

    prefix = result.metadata_yaml.name.removesuffix("_metadata.yaml")
    assert result.vein_mask_npy.name == f"{prefix}_vein_mask.npy"
    np.testing.assert_array_equal(np.load(result.vein_mask_npy), edited)
    for name in meta["files"]:
        assert name.startswith(prefix)
        assert (tmp_path / name).exists()
