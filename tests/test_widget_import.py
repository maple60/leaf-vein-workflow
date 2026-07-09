import importlib.util

import numpy as np
import pytest


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


def test_widget_module_imports_when_qt_backend_is_available(monkeypatch, tmp_path):
    if not _has_qt_backend():
        pytest.skip("No Qt backend is installed.")

    _patch_napari_dirs(monkeypatch, tmp_path)
    pytest.importorskip("napari")

    import leaf_vein_workflow._widget as widget

    assert widget.CropWidget is not None


def test_widget_runs_explicit_mask_refinement_flow(monkeypatch, tmp_path):
    if not _has_qt_backend():
        pytest.skip("No Qt backend is installed.")

    _patch_napari_dirs(monkeypatch, tmp_path)
    import napari
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QApplication

    from leaf_vein_workflow._widget import (
        CropWidget,
        ROLE_KEY,
        ROLE_SMOOTHED_IMAGE,
        ROLE_VEIN_MASK,
        SMOOTHED_LAYER_NAME,
        VEIN_MASK_LAYER_NAME,
    )

    app = QApplication.instance() or QApplication([])
    try:
        viewer = napari.Viewer(show=False)
    except Exception as exc:
        pytest.skip(f"napari Viewer is not available: {exc}")

    try:
        widget = CropWidget(viewer)
        viewer.add_image(np.arange(25).reshape(5, 5), name="leaf")
        app.processEvents()

        roi_layer = viewer.layers["leaf ROI rectangle"]
        assert viewer.layers.selection.active is roi_layer
        assert roi_layer.mode == "add_rectangle"

        roi_layer.add_rectangles(
            np.array(
                [
                    [
                        [1.0, 1.0],
                        [1.0, 4.0],
                        [3.0, 4.0],
                        [3.0, 1.0],
                    ]
                ]
            )
        )

        assert all(layer.name != "leaf ROI" for layer in viewer.layers)

        widget.add_roi()

        image_layer = viewer.layers["leaf"]
        crop_layer = viewer.layers["leaf ROI"]
        assert image_layer.visible is False
        assert crop_layer.data.shape == (2, 3)
        np.testing.assert_allclose(crop_layer.translate, [1.0, 1.0])
        np.testing.assert_array_equal(
            crop_layer.data,
            np.arange(25).reshape(5, 5)[1:3, 1:4],
        )
        assert viewer.layers.selection.active is crop_layer
        assert widget.gaussian_sigma_spinbox.value() == 1.0
        assert widget.apply_gaussian_button.isEnabled()
        assert not widget.otsu_button.isEnabled()
        assert all(layer.name != SMOOTHED_LAYER_NAME for layer in viewer.layers)
        assert all(layer.name != VEIN_MASK_LAYER_NAME for layer in viewer.layers)

        widget.apply_gaussian_filter()
        smoothed_layer = viewer.layers[SMOOTHED_LAYER_NAME]
        assert viewer.layers.selection.active is smoothed_layer
        assert smoothed_layer.metadata[ROLE_KEY] == ROLE_SMOOTHED_IMAGE
        assert smoothed_layer.data.shape == crop_layer.data.shape
        assert widget.otsu_button.isEnabled()
        assert widget.threshold_slider.isEnabled()
        assert not widget.clean_mask_button.isEnabled()
        QTest.qWait(250)
        app.processEvents()
        assert all(layer.name != VEIN_MASK_LAYER_NAME for layer in viewer.layers)

        widget.threshold_slider.setValue(1000)
        QTest.qWait(250)
        app.processEvents()
        assert all(layer.name != VEIN_MASK_LAYER_NAME for layer in viewer.layers)

        widget.use_otsu_threshold()
        QTest.qWait(250)
        app.processEvents()
        mask_layer = viewer.layers[VEIN_MASK_LAYER_NAME]
        assert mask_layer.metadata[ROLE_KEY] == ROLE_VEIN_MASK
        assert mask_layer.data.shape == (2, 3)
        np.testing.assert_allclose(mask_layer.translate, [1.0, 1.0])
        assert np.any(mask_layer.data == 1)
        assert viewer.layers.selection.active is mask_layer
        assert mask_layer.mode == "paint"
        assert widget.clean_mask_button.isEnabled()

        mask_layer.data = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.uint8)
        widget.min_object_size_spinbox.setValue(2)
        widget.max_hole_size_spinbox.setValue(0)
        widget.closing_radius_spinbox.setValue(0)
        widget.opening_radius_spinbox.setValue(0)
        widget.clean_mask()

        assert np.count_nonzero(mask_layer.data) == 0
        assert viewer.layers.selection.active is mask_layer
        assert mask_layer.mode == "paint"

        viewer.layers.selection.active = roi_layer
        widget.add_roi()
        assert viewer.layers.selection.active is crop_layer
        assert image_layer.visible is False
        assert all(layer.name != SMOOTHED_LAYER_NAME for layer in viewer.layers)
        assert all(layer.name != VEIN_MASK_LAYER_NAME for layer in viewer.layers)
        assert not widget.clean_mask_button.isEnabled()

        widget.apply_gaussian_filter()
        smoothed_layer = viewer.layers[SMOOTHED_LAYER_NAME]
        assert viewer.layers.selection.active is smoothed_layer
        widget.use_otsu_threshold()
        QTest.qWait(250)
        app.processEvents()
        mask_layer = viewer.layers[VEIN_MASK_LAYER_NAME]

        assert np.any(mask_layer.data == 1)
        assert widget.clean_mask_button.isEnabled()

        foreground_before = int(np.count_nonzero(mask_layer.data))
        widget.threshold_slider.setValue(1000)
        QTest.qWait(250)
        app.processEvents()
        assert np.count_nonzero(mask_layer.data) <= foreground_before
        assert widget is not None
    finally:
        viewer.close()
        del app
