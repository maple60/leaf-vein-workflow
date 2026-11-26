"""Widget integration tests: controls."""


from widget_helpers import (
    _advance_to_crop,
    _advance_to_mask,
)


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

    calibration = button("image")
    cleanup = button("export")
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
    assert is_open("roi") and not is_open("image")

    _advance_to_crop(flow)
    assert is_open("mask") and not is_open("roi")

    _advance_to_mask(flow)
    flow.app.processEvents()
    assert is_open("vein_mask") and not is_open("mask")
    flow.widget._controls.open_section("orders")
    flow.widget.create_order_labels()
    # 塗り分けはこれからなので、手順5は開いたまま。
    assert is_open("orders")


def test_dock_shows_the_seven_steps_in_order(widget_flow):
    from qtpy.QtWidgets import QToolButton

    flow = widget_flow
    titles = [
        section.toggle_button.text()
        for section in flow.widget._controls.sections.values()
    ]
    assert titles == [
        "1. Image & scale",
        "2. ROI",
        "3. Leaf mask",
        "4. Vein mask",
        "5. Paint vein orders",
        "6. Leaf area & VLA",
        "7. Export",
        "Settings",
    ]
    # パス計測は今は表示しない。
    assert flow.widget.findChild(QToolButton, "workflow_section_vein_paths") is None
    assert not flow.widget.draw_vein_paths_button.isVisibleTo(flow.widget)


def test_leaf_and_vein_controls_belong_to_separate_stages(widget_flow):
    flow = widget_flow
    sections = flow.widget._controls.sections
    assert sections["mask"].content.isAncestorOf(flow.widget.confirm_leaf_button)
    for control in (flow.widget.threshold_slider, flow.widget.clean_mask_button):
        assert sections["vein_mask"].content.isAncestorOf(control)
        assert not sections["mask"].content.isAncestorOf(control)
    assert flow.widget.clean_mask_button.text() == "Clean Vein Mask"
    flow.widget._controls.open_section("vein_mask")
    assert "Clean Vein Mask" in flow.widget.folipuka_assistant.message.text()


def test_failed_leaf_confirmation_does_not_advance(widget_flow):
    flow = widget_flow
    flow.widget._controls.open_section("mask")
    flow.widget.confirm_leaf()
    assert flow.widget._controls.sections["mask"].is_expanded()
    assert not flow.widget._controls.sections["vein_mask"].is_expanded()


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


def test_progress_reporter_delivers_on_the_main_thread(widget_flow):
    from leaf_vein_workflow.widgets.progress import ProgressReporter

    received = []
    reporter = ProgressReporter()
    reporter.progressed.connect(lambda fraction, message: received.append((fraction, message)))
    reporter(0.5, "Halfway")
    widget_flow.app.processEvents()
    assert received == [(0.5, "Halfway")]

    controls = widget_flow.widget._controls
    controls.place_progress("export")
    controls.show_progress(0.33, "Tracing")
    assert controls.progress_bar.isVisibleTo(widget_flow.widget)
    assert "33%" in controls.progress_bar.format()
    assert controls.progress_label.text() == "Tracing"
    assert controls.progress_label.wordWrap()
    assert controls.progress_label.isVisibleTo(widget_flow.widget)
    controls.hide_progress()
    assert not controls.progress_bar.isVisibleTo(widget_flow.widget)
    assert not controls.progress_label.isVisibleTo(widget_flow.widget)


def test_progress_bar_follows_its_action(widget_flow):
    controls = widget_flow.widget._controls
    for section, button in (("orders", controls.auto_orders_button), ("export", controls.export_button)):
        controls.place_progress(section)
        layout = controls.sections[section].content.layout()
        assert layout.indexOf(controls.progress_bar) == layout.indexOf(button) + 1
        assert layout.indexOf(controls.progress_label) == layout.indexOf(button) + 2
        assert controls.sections[section].is_expanded()
