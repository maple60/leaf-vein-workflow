"""Standalone guide tests, independent of the image-analysis workflow."""

import importlib.util

import pytest

if not any(importlib.util.find_spec(name) for name in ("PyQt5", "PyQt6", "PySide2", "PySide6")):
    pytest.skip("No Qt backend is installed.", allow_module_level=True)

from qtpy.QtCore import QPoint, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QWidget

from leaf_vein_workflow.widgets.assistant import FolipukaAssistant, STAGE_TIPS


@pytest.fixture
def guide_window():
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    host.resize(900, 700)
    guide = FolipukaAssistant(host)
    guide.dismissed.connect(guide.hide)
    host.show()
    guide.show_on(host)
    app.processEvents()
    try:
        yield app, host, guide
    finally:
        host.close()
        host.deleteLater()
        app.processEvents()


def test_drag_resize_and_transparent_space(guide_window):
    app, host, guide = guide_window
    assert guide.isVisible()
    assert not guide.mascot.pixmap().isNull()
    start = guide.pos()
    point = QPoint(60, 60)
    QTest.mousePress(guide.mascot, Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(guide.mascot, point - QPoint(120, 80))
    QTest.mouseRelease(guide.mascot, Qt.MouseButton.LeftButton, pos=point)
    assert guide.pos() != start
    host.resize(600, 500)
    app.processEvents()
    assert host.rect().contains(guide.geometry())
    assert not guide.mask().contains(QPoint(5, guide.height() - 5))


def test_stage_dismiss_and_reparent(guide_window):
    app, host, guide = guide_window
    guide.set_stage("orders")
    assert guide.message.text() == STAGE_TIPS["orders"]
    guide.dismissed.emit()
    assert not guide.isVisible()
    other = QWidget()
    other.resize(900, 700)
    try:
        other.show()
        guide.show_on(other)
        app.processEvents()
        assert guide.isVisible()
        assert guide.parentWidget() is other
        guide.show_on(host)
    finally:
        other.close()
        other.deleteLater()
