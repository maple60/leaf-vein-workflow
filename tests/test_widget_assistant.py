"""Exercise the guide in a host window without changing analysis layers."""

import pytest

from widget_helpers import _has_qt_backend

if not _has_qt_backend():
    pytest.skip("No Qt backend is installed.", allow_module_level=True)

from qtpy.QtCore import QPoint, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QHBoxLayout, QWidget

from leaf_vein_workflow.widgets.assistant import STAGE_TIPS


def _show_in_window(flow):
    host = QWidget()
    host.resize(1000, 700)
    QHBoxLayout(host).addWidget(flow.widget)
    host.show()
    flow.app.processEvents()
    return host


def test_folipuka_drag_and_window_bounds(widget_flow):
    flow = widget_flow
    host = _show_in_window(flow)
    try:
        guide = flow.widget.folipuka_assistant
        assert guide.isVisible()
        assert guide.parentWidget() is host
        assert not guide.mascot.pixmap().isNull()
        assert guide.mascot.pixmap().width() <= 128
        assert guide.mascot.pixmap().height() <= 128
        assert guide.mascot.pixmap().hasAlphaChannel()
        assert "Folipuka" in flow.widget.folipuka_checkbox.text()
        start = guide.pos()
        point = QPoint(60, 60)
        QTest.mousePress(guide.mascot, Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(guide.mascot, point - QPoint(180, 120))
        QTest.mouseRelease(guide.mascot, Qt.MouseButton.LeftButton, pos=point)
        assert guide.pos() != start
        assert guide._drag_offset is None
        guide.move(900, 600)
        host.resize(700, 500)
        flow.app.processEvents()
        assert host.rect().contains(guide.geometry())
        # The transparent area beside the dolphin lets canvas clicks through.
        assert not guide.mask().contains(QPoint(5, guide.height() - 5))
    finally:
        host.close()


def test_folipuka_guide_toggle_and_dock_lifetime(widget_flow):
    flow = widget_flow
    host = _show_in_window(flow)
    try:
        guide = flow.widget.folipuka_assistant
        flow.widget._controls.open_section("orders")
        assert guide.message.text() == STAGE_TIPS["orders"]
        guide.dismissed.emit()
        assert not flow.widget.folipuka_checkbox.isChecked()
        assert not guide.isVisible()
        flow.widget.folipuka_checkbox.setChecked(True)
        assert guide.isVisible()
        layers_before = list(flow.viewer.layers)
        flow.widget.hide()
        assert not guide.isVisible()
        flow.widget.show()
        assert guide.isVisible()
        assert list(flow.viewer.layers) == layers_before
    finally:
        host.close()
