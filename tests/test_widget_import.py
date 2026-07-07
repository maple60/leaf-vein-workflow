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


def test_widget_auto_creates_roi_layers_for_loaded_image(monkeypatch, tmp_path):
    if not _has_qt_backend():
        pytest.skip("No Qt backend is installed.")

    _patch_napari_dirs(monkeypatch, tmp_path)
    import napari
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QApplication

    from leaf_vein_workflow._widget import (
        CropWidget,
        ROLE_KEY,
        ROLE_ORDER_LABELS,
        ROLE_VEIN_MASK,
        ROLE_VEIN_SKELETON,
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
        assert viewer.layers.selection.active is crop_layer
        assert image_layer.visible is False
        assert crop_layer.data.shape == (2, 3)
        np.testing.assert_allclose(crop_layer.translate, [1.0, 1.0])
        np.testing.assert_array_equal(
            crop_layer.data,
            np.arange(25).reshape(5, 5)[1:3, 1:4],
        )

        QTest.qWait(250)
        app.processEvents()
        mask_layer = viewer.layers["leaf ROI vein mask"]
        assert mask_layer.metadata[ROLE_KEY] == ROLE_VEIN_MASK
        assert mask_layer.data.shape == (2, 3)
        np.testing.assert_allclose(mask_layer.translate, [1.0, 1.0])
        assert np.any(mask_layer.data == 1)

        viewer.layers.selection.active = mask_layer
        widget.create_skeleton()

        skeleton_layer = viewer.layers["leaf ROI vein mask skeleton"]
        order_layer = viewer.layers["leaf ROI vein mask order labels"]
        assert skeleton_layer.metadata[ROLE_KEY] == ROLE_VEIN_SKELETON
        assert order_layer.metadata[ROLE_KEY] == ROLE_ORDER_LABELS
        assert skeleton_layer.data.shape == mask_layer.data.shape
        assert order_layer.data.shape == mask_layer.data.shape
        np.testing.assert_allclose(skeleton_layer.translate, mask_layer.translate)
        np.testing.assert_allclose(order_layer.translate, mask_layer.translate)

        order_layer.data[0, 0] = 2
        viewer.layers.selection.active = mask_layer
        widget.create_skeleton()
        assert viewer.layers["leaf ROI vein mask order labels"] is order_layer
        assert order_layer.data[0, 0] == 2

        viewer.layers.selection.active = roi_layer
        widget.add_roi()
        assert viewer.layers.selection.active is crop_layer
        assert image_layer.visible is False

        widget.use_otsu_threshold()
        QTest.qWait(250)
        app.processEvents()

        assert np.any(mask_layer.data == 1)

        widget.threshold_spinbox.setValue(100.0)
        QTest.qWait(250)
        app.processEvents()
        assert not np.any(mask_layer.data == 1)
        assert widget is not None
    finally:
        viewer.close()
        del app
