"""Qt controls for the leaf-vein workflow dock widget.

This module only creates controls, arranges them, and changes their enabled
state.  Layer operations and image processing belong to the workflow modules.
"""

from dataclasses import dataclass, field
from pathlib import Path

from napari.utils.notifications import show_warning
from qtpy.QtCore import QLocale, Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def make_number_input(text: str = "", placeholder: str = "") -> QLineEdit:
    """Return a plain text box that only accepts non-negative decimals.

    A line edit avoids spin-box arrows, accidental wheel changes, and fixed
    decimal padding such as ``300.000000000``.
    """
    line_edit = QLineEdit(text)
    validator = QDoubleValidator(0.0, 1e12, 12)
    validator.setNotation(QDoubleValidator.Notation.StandardNotation)
    # 小数点を常に "." として扱い、OSのロケールに左右されないようにする。
    validator.setLocale(QLocale.c())
    line_edit.setValidator(validator)
    line_edit.setPlaceholderText(placeholder)
    return line_edit


def parse_number_input(line_edit: QLineEdit, name: str) -> float:
    """Read a number from a text box or raise a user-facing ValueError."""
    text = line_edit.text().strip()
    if not text:
        raise ValueError(f"Enter a value for {name}.")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


class CollapsibleSection(QWidget):
    """A compact workflow section whose contents can be shown or hidden."""

    def __init__(self, title: str, content_layout, *, expanded: bool = True):
        super().__init__()

        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName(
            f"workflow_section_{title.lower().replace(' ', '_')}"
        )
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.content = QWidget()
        self.content.setLayout(content_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)

        self.toggle_button.toggled.connect(self._set_expanded)
        self._set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        """Open or close the section as if its header had been clicked."""
        self.toggle_button.setChecked(expanded)

    def is_expanded(self) -> bool:
        """Return whether the section content is shown."""
        return self.toggle_button.isChecked()

    def _set_expanded(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        arrow = (
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggle_button.setArrowType(arrow)


@dataclass(slots=True)
class WorkflowControls:
    """Named references to every control shown by :class:`CropWidget`."""

    status_label: QLabel
    calibration_status_label: QLabel
    calibration_method_label: QLabel
    calibration_method_combo: QComboBox
    calibration_value_label: QLabel
    calibration_value_input: QLineEdit
    calibration_unit_combo: QComboBox
    draw_scale_bar_button: QPushButton
    apply_scale_button: QPushButton
    path_order_combo: QComboBox
    draw_vein_paths_button: QPushButton
    vein_paths_hint_label: QLabel
    add_roi_button: QPushButton
    apply_gaussian_button: QPushButton
    otsu_button: QPushButton
    invert_threshold_checkbox: QCheckBox
    clean_mask_button: QPushButton
    create_order_labels_button: QPushButton
    leaf_area_label: QLabel
    leaf_area_input: QLineEdit
    leaf_area_unit_combo: QComboBox
    output_dir_input: QLineEdit
    change_output_dir_button: QPushButton
    export_button: QPushButton
    settings_path_label: QLabel
    open_settings_button: QPushButton
    reload_settings_button: QPushButton
    reset_settings_button: QPushButton
    gaussian_sigma_label: QLabel
    gaussian_sigma_spinbox: QDoubleSpinBox
    threshold_label: QLabel
    threshold_spinbox: QDoubleSpinBox
    threshold_slider: QSlider
    min_object_size_label: QLabel
    min_object_size_spinbox: QSpinBox
    max_hole_size_label: QLabel
    max_hole_size_spinbox: QSpinBox
    closing_radius_label: QLabel
    closing_radius_spinbox: QSpinBox
    opening_radius_label: QLabel
    opening_radius_spinbox: QSpinBox
    sections: dict[str, CollapsibleSection] = field(default_factory=dict)
    # 設定ファイル由来で、対応する入力欄を持たない値。
    scale_bar_length_mm: float = 10.0

    @classmethod
    def create(cls) -> "WorkflowControls":
        """Create controls with the workflow's default values."""
        status_label = QLabel(
            "Open an image with napari. A rectangle ROI layer will be created "
            "automatically."
        )
        status_label.setWordWrap(True)

        calibration_status_label = QLabel("Scale: not calibrated (pixel units)")
        calibration_status_label.setWordWrap(True)

        calibration_method_combo = QComboBox()
        calibration_method_combo.addItem("DPI", "dpi")
        calibration_method_combo.addItem("µm per pixel", "um_per_pixel")
        calibration_method_combo.addItem("mm per pixel", "mm_per_pixel")
        calibration_method_combo.addItem("Scale bar", "scale_bar")

        calibration_value_input = make_number_input("300", "e.g. 600")

        calibration_unit_combo = QComboBox()
        calibration_unit_combo.addItem("µm", "um")
        calibration_unit_combo.addItem("mm", "mm")
        calibration_unit_combo.setEnabled(False)

        settings_path_label = QLabel("Settings: built-in defaults")
        settings_path_label.setWordWrap(True)
        settings_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        path_order_combo = QComboBox()
        path_order_combo.addItem("1° primary", 1)
        path_order_combo.addItem("2° secondary", 2)

        vein_paths_hint_label = QLabel(
            "Trace on the whole leaf: click vertices, double-click to finish. "
            "Select paths and change the order to relabel them."
        )
        vein_paths_hint_label.setWordWrap(True)

        leaf_area_unit_combo = QComboBox()
        leaf_area_unit_combo.addItem("mm²", "mm2")
        leaf_area_unit_combo.addItem("cm²", "cm2")

        # 既定の出力先は napari を起動したフォルダ直下の output。
        output_dir_input = QLineEdit(str(Path.cwd() / "output"))
        output_dir_input.setReadOnly(True)

        gaussian_sigma_spinbox = QDoubleSpinBox()
        gaussian_sigma_spinbox.setRange(0.0, 50.0)
        gaussian_sigma_spinbox.setDecimals(3)
        gaussian_sigma_spinbox.setSingleStep(0.1)
        gaussian_sigma_spinbox.setValue(1.0)

        threshold_spinbox = QDoubleSpinBox()
        threshold_spinbox.setDecimals(6)
        threshold_spinbox.setRange(0.0, 1.0)
        threshold_spinbox.setSingleStep(0.01)
        threshold_spinbox.setValue(0.5)

        threshold_slider = QSlider(Qt.Orientation.Horizontal)
        threshold_slider.setRange(0, 1000)
        threshold_slider.setValue(500)

        min_object_size_spinbox = QSpinBox()
        min_object_size_spinbox.setRange(0, 1_000_000)
        min_object_size_spinbox.setSingleStep(10)
        min_object_size_spinbox.setValue(100)

        max_hole_size_spinbox = QSpinBox()
        max_hole_size_spinbox.setRange(0, 1_000_000)
        max_hole_size_spinbox.setSingleStep(10)
        # 葉脈が前景なので、大きな穴埋めは小さな網目（areole）を潰してしまう。
        max_hole_size_spinbox.setValue(50)

        closing_radius_spinbox = QSpinBox()
        closing_radius_spinbox.setRange(0, 50)
        closing_radius_spinbox.setValue(1)

        opening_radius_spinbox = QSpinBox()
        opening_radius_spinbox.setRange(0, 50)
        opening_radius_spinbox.setValue(0)

        controls = cls(
            status_label=status_label,
            calibration_status_label=calibration_status_label,
            calibration_method_label=QLabel("Scale calibration method"),
            calibration_method_combo=calibration_method_combo,
            calibration_value_label=QLabel("DPI"),
            calibration_value_input=calibration_value_input,
            calibration_unit_combo=calibration_unit_combo,
            draw_scale_bar_button=QPushButton("Draw Scale Bar Line"),
            apply_scale_button=QPushButton("Apply Scale"),
            path_order_combo=path_order_combo,
            draw_vein_paths_button=QPushButton("Draw Vein Paths"),
            vein_paths_hint_label=vein_paths_hint_label,
            add_roi_button=QPushButton("Add ROI"),
            apply_gaussian_button=QPushButton("Apply Gaussian"),
            otsu_button=QPushButton("Apply Otsu"),
            invert_threshold_checkbox=QCheckBox("Invert threshold"),
            clean_mask_button=QPushButton("Clean Mask"),
            create_order_labels_button=QPushButton("Create Order Labels"),
            leaf_area_label=QLabel("Leaf area"),
            leaf_area_input=make_number_input("", "measured before preparation"),
            leaf_area_unit_combo=leaf_area_unit_combo,
            output_dir_input=output_dir_input,
            change_output_dir_button=QPushButton("Change..."),
            export_button=QPushButton("Export Measurements"),
            settings_path_label=settings_path_label,
            open_settings_button=QPushButton("Open Settings File"),
            reload_settings_button=QPushButton("Reload"),
            reset_settings_button=QPushButton("Reset to Defaults"),
            gaussian_sigma_label=QLabel("Gaussian sigma"),
            gaussian_sigma_spinbox=gaussian_sigma_spinbox,
            threshold_label=QLabel("Mask threshold"),
            threshold_spinbox=threshold_spinbox,
            threshold_slider=threshold_slider,
            min_object_size_label=QLabel("Small object area (px)"),
            min_object_size_spinbox=min_object_size_spinbox,
            max_hole_size_label=QLabel("Maximum hole area (px)"),
            max_hole_size_spinbox=max_hole_size_spinbox,
            closing_radius_label=QLabel("Closing radius (px)"),
            closing_radius_spinbox=closing_radius_spinbox,
            opening_radius_label=QLabel("Opening radius (px)"),
            opening_radius_spinbox=opening_radius_spinbox,
        )
        controls.invert_threshold_checkbox.setToolTip(
            "Off: pixels brighter than the threshold become vein (1). "
            "On: pixels darker than the threshold become vein."
        )
        controls.set_calibration_enabled(False)
        controls.add_roi_button.setEnabled(False)
        controls.set_gaussian_enabled(False)
        controls.set_threshold_enabled(False)
        controls.set_cleanup_enabled(False)
        controls.set_measurement_enabled(False)
        return controls

    def create_layout(self) -> QVBoxLayout:
        """Return collapsible workflow stages inside a vertical scroll area."""
        calibration_method_layout = QHBoxLayout()
        calibration_method_layout.addWidget(self.calibration_method_label)
        calibration_method_layout.addWidget(self.calibration_method_combo)

        calibration_value_layout = QHBoxLayout()
        calibration_value_layout.addWidget(self.calibration_value_label)
        calibration_value_layout.addWidget(self.calibration_value_input)
        calibration_value_layout.addWidget(self.calibration_unit_combo)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)

        calibration_layout = QVBoxLayout()
        calibration_layout.addLayout(calibration_method_layout)
        calibration_layout.addLayout(calibration_value_layout)
        calibration_layout.addWidget(self.draw_scale_bar_button)
        calibration_layout.addWidget(self.apply_scale_button)
        calibration_layout.addWidget(self.calibration_status_label)

        path_order_layout = QHBoxLayout()
        path_order_layout.addWidget(QLabel("Order"))
        path_order_layout.addWidget(self.path_order_combo, 1)

        vein_paths_layout = QVBoxLayout()
        vein_paths_layout.addLayout(path_order_layout)
        vein_paths_layout.addWidget(self.draw_vein_paths_button)
        vein_paths_layout.addWidget(self.vein_paths_hint_label)

        roi_layout = QVBoxLayout()
        roi_layout.addWidget(self.add_roi_button)

        mask_creation_layout = QVBoxLayout()
        gaussian_layout = QHBoxLayout()
        gaussian_layout.addWidget(self.gaussian_sigma_label)
        gaussian_layout.addWidget(self.gaussian_sigma_spinbox, 1)
        mask_creation_layout.addLayout(gaussian_layout)
        mask_creation_layout.addWidget(self.apply_gaussian_button)
        mask_creation_layout.addWidget(self.otsu_button)
        mask_creation_layout.addWidget(self.invert_threshold_checkbox)
        mask_creation_layout.addWidget(self.threshold_label)
        mask_creation_layout.addLayout(threshold_layout)

        mask_cleanup_layout = QVBoxLayout()
        mask_cleanup_layout.addWidget(self.min_object_size_label)
        mask_cleanup_layout.addWidget(self.min_object_size_spinbox)
        mask_cleanup_layout.addWidget(self.max_hole_size_label)
        mask_cleanup_layout.addWidget(self.max_hole_size_spinbox)
        mask_cleanup_layout.addWidget(self.closing_radius_label)
        mask_cleanup_layout.addWidget(self.closing_radius_spinbox)
        mask_cleanup_layout.addWidget(self.opening_radius_label)
        mask_cleanup_layout.addWidget(self.opening_radius_spinbox)
        mask_cleanup_layout.addWidget(self.clean_mask_button)

        order_labels_layout = QVBoxLayout()
        order_labels_layout.addWidget(self.create_order_labels_button)

        leaf_area_layout = QHBoxLayout()
        leaf_area_layout.addWidget(self.leaf_area_label)
        leaf_area_layout.addWidget(self.leaf_area_input, 1)
        leaf_area_layout.addWidget(self.leaf_area_unit_combo)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output"))
        output_dir_layout.addWidget(self.output_dir_input, 1)
        output_dir_layout.addWidget(self.change_output_dir_button)

        measurement_layout = QVBoxLayout()
        measurement_layout.addLayout(leaf_area_layout)
        measurement_layout.addLayout(output_dir_layout)
        measurement_layout.addWidget(self.export_button)

        settings_buttons_layout = QHBoxLayout()
        settings_buttons_layout.addWidget(self.reload_settings_button)
        settings_buttons_layout.addWidget(self.reset_settings_button)

        settings_layout = QVBoxLayout()
        settings_layout.addWidget(self.settings_path_label)
        settings_layout.addWidget(self.open_settings_button)
        settings_layout.addLayout(settings_buttons_layout)

        scroll_content = QWidget()
        stages_layout = QVBoxLayout(scroll_content)
        stages_layout.setContentsMargins(0, 0, 0, 0)
        stage_layouts = (
            ("calibration", "Scale calibration", calibration_layout),
            ("vein_paths", "Vein paths (1°/2°)", vein_paths_layout),
            ("roi", "ROI", roi_layout),
            ("mask_creation", "Mask creation", mask_creation_layout),
            ("mask_cleanup", "Mask cleanup", mask_cleanup_layout),
            ("order_labels", "Vein order labels", order_labels_layout),
            ("measurement", "Measurement", measurement_layout),
            ("settings", "Settings", settings_layout),
        )
        for key, title, stage_layout in stage_layouts:
            section = CollapsibleSection(
                title,
                stage_layout,
                expanded=(key == "calibration"),
            )
            section.toggle_button.toggled.connect(
                lambda checked, key=key: self._on_section_toggled(key, checked)
            )
            self.sections[key] = section
            stages_layout.addWidget(section)
        stages_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setObjectName("workflow_scroll_area")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setWidget(scroll_content)

        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        layout.addWidget(self.status_label)
        return layout

    def _on_section_toggled(self, key: str, expanded: bool) -> None:
        """Keep one stage open at a time so the dock rarely needs scrolling."""
        if not expanded:
            return
        for other_key, section in self.sections.items():
            if other_key != key and section.is_expanded():
                section.set_expanded(False)

    def open_section(self, key: str) -> None:
        """Open one workflow stage and close the others."""
        section = self.sections.get(key)
        if section is not None:
            section.set_expanded(True)

    def set_calibration_enabled(self, enabled: bool) -> None:
        """Enable calibration controls after at least one source image exists."""
        self.calibration_method_label.setEnabled(enabled)
        self.calibration_method_combo.setEnabled(enabled)
        self.calibration_value_label.setEnabled(enabled)
        self.calibration_value_input.setEnabled(enabled)
        self.apply_scale_button.setEnabled(enabled)
        method = self.calibration_method_combo.currentData()
        is_scale_bar = enabled and method == "scale_bar"
        self.calibration_unit_combo.setEnabled(is_scale_bar)
        self.calibration_unit_combo.setVisible(is_scale_bar)
        self.draw_scale_bar_button.setEnabled(is_scale_bar)
        self.path_order_combo.setEnabled(enabled)
        self.draw_vein_paths_button.setEnabled(enabled)
        self.set_measurement_enabled(enabled)

    def set_calibration_method(self, method: str) -> None:
        """Adapt the shared numeric field to the selected calibration method."""
        labels = {
            "dpi": "DPI",
            "um_per_pixel": "µm / pixel",
            "mm_per_pixel": "mm / pixel",
            "scale_bar": "Scale bar length",
        }
        self.calibration_value_label.setText(labels.get(method, "Scale value"))
        placeholders = {
            "dpi": "e.g. 600",
            "um_per_pixel": "e.g. 42.3",
            "mm_per_pixel": "e.g. 0.0423",
            "scale_bar": "e.g. 10",
        }
        self.calibration_value_input.setPlaceholderText(
            placeholders.get(method, "")
        )
        is_scale_bar = method == "scale_bar"
        enabled = self.calibration_method_combo.isEnabled()
        self.calibration_unit_combo.setEnabled(enabled and is_scale_bar)
        self.calibration_unit_combo.setVisible(is_scale_bar)
        self.draw_scale_bar_button.setEnabled(enabled and is_scale_bar)

    def set_calibration_status(self, message: str) -> None:
        """Show the current physical calibration separately from workflow status."""
        self.calibration_status_label.setText(message)

    def set_gaussian_enabled(self, enabled: bool) -> None:
        """Enable controls for the explicit Gaussian step."""
        self.gaussian_sigma_label.setEnabled(enabled)
        self.gaussian_sigma_spinbox.setEnabled(enabled)
        self.apply_gaussian_button.setEnabled(enabled)

    def set_threshold_enabled(self, enabled: bool) -> None:
        """Enable controls used after a smoothed layer exists."""
        self.otsu_button.setEnabled(enabled)
        self.invert_threshold_checkbox.setEnabled(enabled)
        self.threshold_label.setEnabled(enabled)
        self.threshold_spinbox.setEnabled(enabled)
        self.threshold_slider.setEnabled(enabled)

    def set_cleanup_enabled(self, enabled: bool) -> None:
        """Enable controls that require an explicitly created mask."""
        self.min_object_size_label.setEnabled(enabled)
        self.min_object_size_spinbox.setEnabled(enabled)
        self.max_hole_size_label.setEnabled(enabled)
        self.max_hole_size_spinbox.setEnabled(enabled)
        self.closing_radius_label.setEnabled(enabled)
        self.closing_radius_spinbox.setEnabled(enabled)
        self.opening_radius_label.setEnabled(enabled)
        self.opening_radius_spinbox.setEnabled(enabled)
        self.clean_mask_button.setEnabled(enabled)
        self.create_order_labels_button.setEnabled(enabled)

    def set_measurement_enabled(self, enabled: bool) -> None:
        """Enable measurement once a mask (and so order labels) can exist."""
        self.leaf_area_label.setEnabled(enabled)
        self.leaf_area_input.setEnabled(enabled)
        self.leaf_area_unit_combo.setEnabled(enabled)
        self.export_button.setEnabled(enabled)

    def set_cleanup_unit_labels(self, mm_per_pixel: float | None) -> None:
        """Show physical equivalents while keeping cleanup inputs in pixels."""
        if mm_per_pixel is None:
            self.min_object_size_label.setText("Small object area (px)")
            self.max_hole_size_label.setText("Maximum hole area (px)")
            self.closing_radius_label.setText("Closing radius (px)")
            self.opening_radius_label.setText("Opening radius (px)")
            return

        pixel_size_mm = float(mm_per_pixel)
        pixel_area_mm2 = pixel_size_mm**2
        small_object_area = self.min_object_size_spinbox.value() * pixel_area_mm2
        maximum_hole_area = self.max_hole_size_spinbox.value() * pixel_area_mm2
        closing_length = self._format_metric_length(
            self.closing_radius_spinbox.value() * pixel_size_mm
        )
        opening_length = self._format_metric_length(
            self.opening_radius_spinbox.value() * pixel_size_mm
        )

        self.min_object_size_label.setText(
            "Small object area: "
            f"≈ {small_object_area:.6g} mm² "
            f"({self.min_object_size_spinbox.value()} px)"
        )
        self.max_hole_size_label.setText(
            "Maximum hole area: "
            f"≈ {maximum_hole_area:.6g} mm² "
            f"({self.max_hole_size_spinbox.value()} px)"
        )
        self.closing_radius_label.setText(
            f"Closing radius: ≈ {closing_length} "
            f"({self.closing_radius_spinbox.value()} px)"
        )
        self.opening_radius_label.setText(
            f"Opening radius: ≈ {opening_length} "
            f"({self.opening_radius_spinbox.value()} px)"
        )

    @staticmethod
    def _format_metric_length(length_mm: float) -> str:
        """Choose µm for sub-millimetre radii and mm otherwise."""
        if length_mm < 1.0:
            return f"{length_mm * 1000.0:.6g} µm"
        return f"{length_mm:.6g} mm"

    def set_status(self, message: str) -> None:
        """Show workflow guidance without raising a notification."""
        self.status_label.setText(message)

    def warn(self, message: str) -> None:
        """Show the same warning in both the dock and napari notifications."""
        self.set_status(message)
        show_warning(message)

