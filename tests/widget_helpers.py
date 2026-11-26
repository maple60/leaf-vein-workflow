"""Small shared images and explicit workflow steps for GUI tests."""


import importlib.util
import numpy as np


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
    # These existing tests exercise vein edits/export, not leaf segmentation.
    flow.viewer.layers[flow.module.LEAF_MASK_LAYER_NAME].data = np.ones(
        crop_layer.data.shape[:2], dtype=np.uint8
    )
    flow.widget.confirm_leaf()
    flow.widget.leaf_area_source_combo.setCurrentIndex(flow.widget.leaf_area_source_combo.findData("input"))
    _wait_for_delayed_mask_update(flow)
    mask_layer = flow.viewer.layers[flow.module.VEIN_MASK_LAYER_NAME]
    return crop_layer, smoothed_layer, mask_layer


def _wait_for_delayed_mask_update(flow):
    from qtpy.QtTest import QTest

    # 閾値変更は連続入力をまとめるQTimer経由なので、発火後まで待つ。
    QTest.qWait(250)
    flow.app.processEvents()
