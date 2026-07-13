"""Qt controls for the leaf-vein workflow dock widget.

This module only creates controls, arranges them, and changes their enabled
state.  Layer operations and image processing belong to the workflow modules.
"""

from dataclasses import dataclass

from napari.utils.notifications import show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
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
        controls.add_roi_button.setEnabled(False)
        controls.set_gaussian_enabled(False)
        controls.set_threshold_enabled(False)
        controls.set_cleanup_enabled(False)
        return controls

    def create_layout(self) -> QVBoxLayout:
        """Return the unchanged, single-column workflow layout."""
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)

        layout = QVBoxLayout()
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

