"""Qt controls for the leaf-vein workflow dock widget.

This module only creates controls, arranges them, and changes their enabled
state.  Layer operations and image processing belong to the workflow modules.
"""

from dataclasses import dataclass

from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)


@dataclass(slots=True)
class WorkflowControls:
    """Named references to every control shown by :class:`CropWidget`."""

    status_label: QLabel
    calibration_status_label: QLabel
    calibration_method_label: QLabel
    calibration_method_combo: QComboBox
    calibration_value_label: QLabel
    calibration_value_spinbox: QDoubleSpinBox
    calibration_unit_combo: QComboBox
    draw_scale_bar_button: QPushButton
    apply_scale_button: QPushButton
    add_roi_button: QPushButton
    apply_gaussian_button: QPushButton
    otsu_button: QPushButton
    invert_threshold_checkbox: QCheckBox
    clean_mask_button: QPushButton
    create_order_labels_button: QPushButton
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

        calibration_value_spinbox = QDoubleSpinBox()
        calibration_value_spinbox.setDecimals(9)
        calibration_value_spinbox.setRange(0.0, 1_000_000_000.0)
        calibration_value_spinbox.setSingleStep(1.0)
        calibration_value_spinbox.setValue(300.0)

        calibration_unit_combo = QComboBox()
        calibration_unit_combo.addItem("µm", "um")
        calibration_unit_combo.addItem("mm", "mm")
        calibration_unit_combo.setEnabled(False)

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
        max_hole_size_spinbox.setValue(500)

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
            calibration_value_spinbox=calibration_value_spinbox,
            calibration_unit_combo=calibration_unit_combo,
            draw_scale_bar_button=QPushButton("Draw Scale Bar Line"),
            apply_scale_button=QPushButton("Apply Scale"),
            add_roi_button=QPushButton("Add ROI"),
            apply_gaussian_button=QPushButton("Apply Gaussian"),
            otsu_button=QPushButton("Apply Otsu"),
            invert_threshold_checkbox=QCheckBox("Invert threshold"),
            clean_mask_button=QPushButton("Clean Mask"),
            create_order_labels_button=QPushButton("Create Order Labels"),
            gaussian_sigma_label=QLabel("Gaussian sigma"),
            gaussian_sigma_spinbox=gaussian_sigma_spinbox,
            threshold_label=QLabel("Mask threshold"),
            threshold_spinbox=threshold_spinbox,
            threshold_slider=threshold_slider,
            min_object_size_label=QLabel("Small object size"),
            min_object_size_spinbox=min_object_size_spinbox,
            max_hole_size_label=QLabel("Hole size"),
            max_hole_size_spinbox=max_hole_size_spinbox,
            closing_radius_label=QLabel("Closing radius"),
            closing_radius_spinbox=closing_radius_spinbox,
            opening_radius_label=QLabel("Opening radius"),
            opening_radius_spinbox=opening_radius_spinbox,
        )
        controls.set_calibration_enabled(False)
        controls.add_roi_button.setEnabled(False)
        controls.set_gaussian_enabled(False)
        controls.set_threshold_enabled(False)
        controls.set_cleanup_enabled(False)
        return controls

    def create_layout(self) -> QVBoxLayout:
        """Return the single-column workflow layout."""
        calibration_method_layout = QHBoxLayout()
        calibration_method_layout.addWidget(self.calibration_method_label)
        calibration_method_layout.addWidget(self.calibration_method_combo)

        calibration_value_layout = QHBoxLayout()
        calibration_value_layout.addWidget(self.calibration_value_label)
        calibration_value_layout.addWidget(self.calibration_value_spinbox)
        calibration_value_layout.addWidget(self.calibration_unit_combo)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)

        layout = QVBoxLayout()
        layout.addLayout(calibration_method_layout)
        layout.addLayout(calibration_value_layout)
        layout.addWidget(self.draw_scale_bar_button)
        layout.addWidget(self.apply_scale_button)
        layout.addWidget(self.calibration_status_label)
        layout.addWidget(self.add_roi_button)
        layout.addWidget(self.gaussian_sigma_label)
        layout.addWidget(self.gaussian_sigma_spinbox)
        layout.addWidget(self.apply_gaussian_button)
        layout.addWidget(self.otsu_button)
        layout.addWidget(self.invert_threshold_checkbox)
        layout.addWidget(self.threshold_label)
        layout.addLayout(threshold_layout)
        layout.addWidget(self.min_object_size_label)
        layout.addWidget(self.min_object_size_spinbox)
        layout.addWidget(self.max_hole_size_label)
        layout.addWidget(self.max_hole_size_spinbox)
        layout.addWidget(self.closing_radius_label)
        layout.addWidget(self.closing_radius_spinbox)
        layout.addWidget(self.opening_radius_label)
        layout.addWidget(self.opening_radius_spinbox)
        layout.addWidget(self.clean_mask_button)
        layout.addWidget(self.create_order_labels_button)
        layout.addWidget(self.status_label)
        return layout

    def set_calibration_enabled(self, enabled: bool) -> None:
        """Enable calibration controls after at least one source image exists."""
        self.calibration_method_label.setEnabled(enabled)
        self.calibration_method_combo.setEnabled(enabled)
        self.calibration_value_label.setEnabled(enabled)
        self.calibration_value_spinbox.setEnabled(enabled)
        self.apply_scale_button.setEnabled(enabled)
        method = self.calibration_method_combo.currentData()
        is_scale_bar = enabled and method == "scale_bar"
        self.calibration_unit_combo.setEnabled(is_scale_bar)
        self.calibration_unit_combo.setVisible(is_scale_bar)
        self.draw_scale_bar_button.setEnabled(is_scale_bar)

    def set_calibration_method(self, method: str) -> None:
        """Adapt the shared numeric field to the selected calibration method."""
        labels = {
            "dpi": "DPI",
            "um_per_pixel": "µm / pixel",
            "mm_per_pixel": "mm / pixel",
            "scale_bar": "Scale bar length",
        }
        self.calibration_value_label.setText(labels.get(method, "Scale value"))
        settings = {
            "dpi": (3, 1.0),
            "um_per_pixel": (6, 0.1),
            "mm_per_pixel": (9, 0.001),
            "scale_bar": (6, 1.0),
        }
        decimals, single_step = settings.get(method, (6, 1.0))
        self.calibration_value_spinbox.setDecimals(decimals)
        self.calibration_value_spinbox.setSingleStep(single_step)
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

    def set_status(self, message: str) -> None:
        """Show workflow guidance without raising a notification."""
        self.status_label.setText(message)

    def warn(self, message: str) -> None:
        """Show the same warning in both the dock and napari notifications."""
        self.set_status(message)
        show_warning(message)

