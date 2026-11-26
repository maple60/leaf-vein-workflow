"""Widget integration tests: import."""

import pytest

from widget_helpers import (
    _has_qt_backend,
    _patch_napari_dirs,
)


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
