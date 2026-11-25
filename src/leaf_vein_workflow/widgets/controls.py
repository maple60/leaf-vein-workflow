"""Qt controls for the leaf-vein workflow dock widget.

This module only creates controls, arranges them, and changes their enabled
state.  Layer operations and image processing belong to the workflow modules.
"""

from dataclasses import dataclass, field
from pathlib import Path

from napari.utils.notifications import show_warning
from qtpy.QtCore import QLocale, Qt
from qtpy.QtGui import QColor, QDoubleValidator, QIcon, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .layers import ORDER_LABEL_COLORS


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

    def __init__(
        self,
        title: str,
        content_layout,
        *,
        expanded: bool = True,
        key: str | None = None,
    ):
        super().__init__()

        self.toggle_button = QToolButton()
        name = key or title.lower().replace(' ', '_')
        self.toggle_button.setObjectName(f"workflow_section_{name}")
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
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.content.setLayout(content_layout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
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
    progress_bar: QProgressBar
    progress_label: QLabel
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
    confirm_leaf_button: QPushButton
    leaf_area_status_label: QLabel
    confirmed_leaf_area_label: QLabel
    leaf_area_source_combo: QComboBox
    invert_threshold_checkbox: QCheckBox
    exclude_background_checkbox: QCheckBox
    largest_leaf_checkbox: QCheckBox
    clean_mask_button: QPushButton
    create_order_labels_button: QPushButton
    edit_mask_button: QPushButton
    next_leaf_button: QPushButton
    resume_button: QPushButton
    paint_order_combo: QComboBox
    mark_seeds_button: QPushButton
    auto_orders_button: QPushButton
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
    activate_labels_after_marking: bool = True
    auto_order_options: dict = field(default_factory=dict)

    @classmethod
    def create(cls) -> "WorkflowControls":
        """Create controls with the workflow's default values."""
        status_label = QLabel(
            "Open an image with napari. A rectangle ROI layer will be created "
            "automatically."
        )
        status_label.setWordWrap(True)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 1000)
        progress_bar.setTextVisible(True)
        progress_bar.setVisible(False)
        progress_label = QLabel()
        progress_label.setWordWrap(True)
        progress_label.setMinimumWidth(0)
        progress_label.setVisible(False)

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

        # 塗る次数。色見本はラベル層の色と同じにする。
        paint_order_combo = QComboBox()
        for text, value in (
            ("1° primary", 1),
            ("2° secondary", 2),
            ("3° tertiary", 3),
            ("Uncertain (not 4°)", 4),
        ):
            paint_order_combo.addItem(_color_icon(ORDER_LABEL_COLORS[value]), text, value)

        leaf_area_unit_combo = QComboBox()
        leaf_area_unit_combo.addItem("mm²", "mm2")
        leaf_area_unit_combo.addItem("cm²", "cm2")
        leaf_area_source_combo = QComboBox()
        leaf_area_source_combo.addItem("Confirmed leaf mask (current ROI)", "leaf_mask")
        leaf_area_source_combo.addItem("External area (manual input)", "input")
        leaf_area_source_combo.setCurrentIndex(1)

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
            progress_bar=progress_bar,
            progress_label=progress_label,
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
            confirm_leaf_button=QPushButton("Confirm Leaf && Extract Veins"),
            leaf_area_status_label=QLabel("Leaf area: not confirmed"),
            confirmed_leaf_area_label=QLabel("Leaf area: not confirmed"),
            leaf_area_source_combo=leaf_area_source_combo,
            invert_threshold_checkbox=QCheckBox("Invert (dark pixels are vein)"),
            exclude_background_checkbox=QCheckBox("ROI includes background (paper)"),
            largest_leaf_checkbox=QCheckBox("Keep largest leaf component"),
            clean_mask_button=QPushButton("Clean Vein Mask"),
            create_order_labels_button=QPushButton("Create Order Labels"),
            edit_mask_button=QPushButton("Edit Vein Mask"),
            next_leaf_button=QPushButton("Next Leaf / Reset Analysis"),
            resume_button=QPushButton("Load Export / Resume"),
            paint_order_combo=paint_order_combo,
            mark_seeds_button=QPushButton("Mark Base && Apex"),
            auto_orders_button=QPushButton("Auto-assign 1°/2° + 3° guess"),
            leaf_area_label=QLabel("Leaf area"),
            leaf_area_input=make_number_input("", "external area (optional)"),
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
            "On: pixels darker than the threshold become vein. "
            "Apply Otsu sets this automatically."
        )
        controls.largest_leaf_checkbox.setChecked(True)
        controls.largest_leaf_checkbox.setToolTip(
            "Before confirmation, keep only the largest connected leaf region. "
            "Turn off for intentionally separated blade pieces. Petioles or debris "
            "connected to the blade still need manual erasing."
        )
        controls.exclude_background_checkbox.setToolTip(
            "On: Apply Otsu first separates the leaf from the background around "
            "it, then finds veins inside the leaf. Turn off when the ROI lies "
            "entirely inside the leaf blade."
        )
        controls.set_calibration_enabled(False)
        controls.add_roi_button.setEnabled(False)
        controls.set_gaussian_enabled(False)
        controls.set_threshold_enabled(False)
        controls.confirm_leaf_button.setEnabled(False)
        controls.leaf_area_status_label.setWordWrap(True)
        controls.confirmed_leaf_area_label.setWordWrap(True)
        controls.leaf_area_source_combo.currentIndexChanged.connect(
            lambda *_: controls.update_leaf_area_fields()
        )
        controls.update_leaf_area_fields()
        controls.set_cleanup_enabled(False)
        controls.set_measurement_enabled(False)
        return controls

    def create_layout(self) -> QVBoxLayout:
        """Return the six workflow steps as collapsible sections in a scroll area.

        Sections follow the order of the work: image and scale, ROI, vein mask,
        painting vein orders, leaf area, and export.  Path tracing for 1°/2°
        veins is implemented but not shown for now.
        """
        calibration_method_layout = QHBoxLayout()
        calibration_method_layout.addWidget(self.calibration_method_label)
        calibration_method_layout.addWidget(self.calibration_method_combo)

        calibration_value_layout = QHBoxLayout()
        calibration_value_layout.addWidget(self.calibration_value_label)
        calibration_value_layout.addWidget(self.calibration_value_input)
        calibration_value_layout.addWidget(self.calibration_unit_combo)

        image_layout = QVBoxLayout()
        image_layout.addWidget(
            _hint(
                "Open the leaf scan (File > Open, or drag it into napari). "
                "Enter the scan resolution, then press Apply Scale."
            )
        )
        image_layout.addLayout(calibration_method_layout)
        image_layout.addLayout(calibration_value_layout)
        image_layout.addWidget(self.draw_scale_bar_button)
        image_layout.addWidget(self.apply_scale_button)
        image_layout.addWidget(self.calibration_status_label)
        image_layout.addWidget(self.resume_button)

        roi_layout = QVBoxLayout()
        roi_layout.addWidget(
            _hint(
                "Drag the rectangle over the area to analyze, then press Add ROI."
            )
        )
        roi_layout.addWidget(self.add_roi_button)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)

        gaussian_layout = QHBoxLayout()
        gaussian_layout.addWidget(self.gaussian_sigma_label)
        gaussian_layout.addWidget(self.gaussian_sigma_spinbox, 1)

        mask_layout = QVBoxLayout()
        mask_layout.addWidget(
            _hint(
                "Apply Gaussian, then Apply Otsu to review leaf_mask. Paint or "
                "erase the blade boundary and remove the petiole and debris. "
                "Confirm Leaf calculates area and extracts veins inside it. "
                "For a local ROI, the area is that region, not the whole leaf."
            )
        )
        mask_layout.addLayout(gaussian_layout)
        mask_layout.addWidget(self.apply_gaussian_button)
        mask_layout.addWidget(self.exclude_background_checkbox)
        mask_layout.addWidget(self.otsu_button)
        mask_layout.addWidget(self.largest_leaf_checkbox)
        mask_layout.addWidget(self.confirm_leaf_button)
        mask_layout.addWidget(self.leaf_area_status_label)
        vein_mask_layout = QVBoxLayout()
        vein_mask_layout.addWidget(_hint(
            "Review the extracted vein_mask. Adjust the vein threshold if needed, "
            "then optionally clean small objects and holes. Paint or erase on "
            "vein_mask for manual corrections before assigning vein orders."
        ))
        vein_mask_layout.addWidget(self.invert_threshold_checkbox)
        vein_mask_layout.addWidget(self.threshold_label)
        vein_mask_layout.addLayout(threshold_layout)
        vein_mask_layout.addWidget(_subheading("Clean up (optional)"))
        for label, spinbox in (
            (self.min_object_size_label, self.min_object_size_spinbox),
            (self.max_hole_size_label, self.max_hole_size_spinbox),
            (self.closing_radius_label, self.closing_radius_spinbox),
            (self.opening_radius_label, self.opening_radius_spinbox),
        ):
            # 校正後は単位付きの長い表示になるので、折り返して横にはみ出さない。
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            row = QHBoxLayout()
            row.addWidget(label, 1)
            row.addWidget(spinbox)
            vein_mask_layout.addLayout(row)
        vein_mask_layout.addWidget(self.clean_mask_button)

        paint_order_layout = QHBoxLayout()
        paint_order_layout.addWidget(QLabel("Paint as"))
        paint_order_layout.addWidget(self.paint_order_combo, 1)

        orders_layout = QVBoxLayout()
        orders_layout.addWidget(
            _hint(
                "Press Create Order Labels: every vein pixel starts as "
                "Uncertain (purple). Choose an order and drag over veins in the "
                "image; only vein pixels change color. Set the brush size in "
                "napari's layer controls (top left)."
            )
        )
        orders_layout.addWidget(self.create_order_labels_button)
        orders_layout.addWidget(self.edit_mask_button)
        orders_layout.addLayout(paint_order_layout)
        orders_layout.addWidget(_subheading("Auto 1°/2° + 3° guess (optional)"))
        orders_layout.addWidget(
            _hint(
                "Mark the leaf base and the apex of the 1°, then auto-assign. "
                "Only Uncertain pixels change, so your painting is kept. "
                "A full leaf at 3200 dpi takes about a minute."
            )
        )
        # 狭い dock でもはみ出さないよう、縦に並べる。
        orders_layout.addWidget(self.mark_seeds_button)
        orders_layout.addWidget(self.auto_orders_button)

        leaf_area_layout = QHBoxLayout()
        leaf_area_layout.addWidget(self.leaf_area_label)
        leaf_area_layout.addWidget(self.leaf_area_input, 1)
        leaf_area_layout.addWidget(self.leaf_area_unit_combo)

        leaf_area_section_layout = QVBoxLayout()
        leaf_area_section_layout.addWidget(
            _hint(
                "Use the confirmed specimen mask for VLA in the same ROI. "
                "External area is an explicit override; match it to the region "
                "and preparation state of the measured veins."
            )
        )
        leaf_area_section_layout.addWidget(self.leaf_area_source_combo)
        leaf_area_section_layout.addWidget(self.confirmed_leaf_area_label)
        leaf_area_section_layout.addLayout(leaf_area_layout)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output"))
        output_dir_layout.addWidget(self.output_dir_input, 1)
        output_dir_layout.addWidget(self.change_output_dir_button)

        export_layout = QVBoxLayout()
        export_layout.addWidget(
            _hint(
                "Writes per-order length/VLA, branch diameter (mean/median/max), "
                "labels, QC, and metadata into a new folder for each export."
            )
        )
        export_layout.addLayout(output_dir_layout)
        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.progress_bar)
        export_layout.addWidget(self.progress_label)
        export_layout.addWidget(self.next_leaf_button)

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
            ("image", "1. Image & scale", image_layout),
            ("roi", "2. ROI", roi_layout),
            ("mask", "3. Leaf mask", mask_layout),
            ("vein_mask", "4. Vein mask", vein_mask_layout),
            ("orders", "5. Paint vein orders", orders_layout),
            ("leaf_area", "6. Leaf area & VLA", leaf_area_section_layout),
            ("export", "7. Export", export_layout),
            ("settings", "Settings", settings_layout),
        )
        for key, title, stage_layout in stage_layouts:
            section = CollapsibleSection(
                title,
                stage_layout,
                expanded=(key == "image"),
                key=key,
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
        layout.addWidget(self.status_label)
        layout.addWidget(scroll_area)
        return layout

    def place_progress(self, section: str) -> None:
        """Place the shared progress bar immediately below the running action."""
        for widget in (self.progress_bar, self.progress_label):
            parent = widget.parentWidget()
            if parent is not None and parent.layout() is not None:
                parent.layout().removeWidget(widget)
        layout = self.sections[section].content.layout()
        button = self.auto_orders_button if section == "orders" else self.export_button
        layout.insertWidget(layout.indexOf(button) + 1, self.progress_bar)
        layout.insertWidget(layout.indexOf(button) + 2, self.progress_label)
        self.open_section(section)

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
        self.exclude_background_checkbox.setEnabled(True)
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
        self.edit_mask_button.setEnabled(enabled)
        self.create_order_labels_button.setEnabled(enabled)
        self.paint_order_combo.setEnabled(enabled)
        self.mark_seeds_button.setEnabled(enabled)
        self.auto_orders_button.setEnabled(enabled)

    def set_leaf_area_status(self, message: str) -> None:
        """Show the same confirmed area at the mask and measurement steps."""
        self.leaf_area_status_label.setText(message)
        self.confirmed_leaf_area_label.setText(message)

    def update_leaf_area_fields(self, *args) -> None:
        """Only show manual area controls when external area is selected."""
        external = self.leaf_area_source_combo.currentData() == "input"
        for control in (self.leaf_area_label, self.leaf_area_input, self.leaf_area_unit_combo):
            control.setVisible(external)
        self.confirmed_leaf_area_label.setVisible(not external)

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

    def show_progress(self, fraction: float, message: str) -> None:
        """Show how far a long step has got, with what it is doing now."""
        self.progress_bar.setVisible(True)
        fraction = max(0.0, min(1.0, fraction))
        self.progress_bar.setValue(int(round(fraction * 1000)))
        self.progress_bar.setFormat(f"{fraction:.0%}")
        self.progress_label.setText(message)
        self.progress_label.setVisible(True)

    def hide_progress(self) -> None:
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(False)
        self.progress_label.clear()

    def set_status(self, message: str) -> None:
        """Show workflow guidance without raising a notification."""
        self.status_label.setText(message)

    def warn(self, message: str) -> None:
        """Show the same warning in both the dock and napari notifications."""
        self.set_status(message)
        show_warning(message)



def _hint(text: str) -> QLabel:
    """Return a short, wrapped instruction shown at the top of a step."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignTop)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
    return label


def _subheading(text: str) -> QLabel:
    """Return a small bold label that separates parts of one step."""
    label = QLabel(text)
    label.setStyleSheet("font-weight: bold;")
    return label


def _color_icon(color: str) -> QIcon:
    """Return a square swatch so each order matches its label color."""
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)
