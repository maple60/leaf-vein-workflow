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
    flow.widget.calibration_value_spinbox.setValue(254.0)

    flow.widget.apply_scale()

    key = flow.module.CALIBRATION_MM_PER_PIXEL_KEY
    assert flow.image_layer.metadata[key] == pytest.approx(0.1)
    np.testing.assert_allclose(flow.image_layer.scale, [1.0, 1.0])

    crop_layer, smoothed_layer, mask_layer = _advance_to_mask(flow)
    for layer in (flow.roi_layer, crop_layer, smoothed_layer, mask_layer):
        assert layer.metadata[key] == pytest.approx(0.1)

    mm_index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(mm_index)
    flow.widget.calibration_value_spinbox.setValue(0.025)
    flow.widget.apply_scale()

    for layer in (flow.image_layer, flow.roi_layer, crop_layer, smoothed_layer, mask_layer):
        assert layer.metadata[key] == pytest.approx(0.025)


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
    flow.widget.calibration_value_spinbox.setValue(1.0)
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
    flow.widget.calibration_value_spinbox.setValue(254.0)
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
    flow.widget.calibration_value_spinbox.setValue(1.0)

    flow.widget.apply_scale()

    assert len(line_layer.data) == 2
    assert flow.module.CALIBRATION_MM_PER_PIXEL_KEY not in flow.image_layer.metadata


def test_calibration_status_tracks_the_active_source_image(widget_flow):
    flow = widget_flow
    mm_index = flow.widget.calibration_method_combo.findData("mm_per_pixel")
    flow.widget.calibration_method_combo.setCurrentIndex(mm_index)
    flow.widget.calibration_value_spinbox.setValue(0.1)
    flow.widget.apply_scale()

    second_image = flow.viewer.add_image(SOURCE_IMAGE.copy(), name="leaf2")
    flow.app.processEvents()
    second_roi = flow.viewer.layers["leaf2 ROI rectangle"]
    flow.widget.calibration_value_spinbox.setValue(0.02)
    flow.widget.apply_scale()

    flow.viewer.layers.selection.active = flow.roi_layer
    flow.app.processEvents()
    assert "Scale for leaf:" in flow.widget.calibration_status_label.text()
    assert "100 µm" in flow.widget.calibration_status_label.text()
    assert "equivalent DPI ≈ 254" in flow.widget.calibration_status_label.text()

    flow.viewer.layers.selection.active = second_roi
    flow.app.processEvents()
    assert "Scale for leaf2:" in flow.widget.calibration_status_label.text()
    assert "20 µm" in flow.widget.calibration_status_label.text()
    assert "equivalent DPI ≈ 1270" in flow.widget.calibration_status_label.text()
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
    mask_layer.data = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)

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
    mask_layer.data = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    flow.widget.create_order_labels()
    order_layer = flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME]
    order_layer.paint((0, 1), 2)
    flow.app.processEvents()

    flow.widget.create_order_labels()

    assert flow.viewer.layers[flow.module.ORDER_LABELS_LAYER_NAME] is order_layer
    assert int(np.asarray(order_layer.data)[0, 1]) == 2

    mask_layer.data = np.ones_like(mask_layer.data)
    flow.app.processEvents()
    assert np.count_nonzero(order_layer.data) == 0


def test_recrop_reuses_crop_and_removes_derived_layers(widget_flow):
    flow = widget_flow
    crop_layer, _, mask_layer = _advance_to_mask(flow)
    mask_layer.data = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8)
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
