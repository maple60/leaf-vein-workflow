"""GUI fixture: isolate settings/cache and create one fresh viewer per test."""

from types import SimpleNamespace

import numpy as np
import pytest

from widget_helpers import SOURCE_IMAGE, _has_qt_backend, _patch_napari_dirs


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
