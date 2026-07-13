"""Composition root for the single leaf-vein napari dock widget.

This module wires Qt signals to explicit workflow objects.  Stage-specific
layer operations live in ``roi.py``, ``mask.py``, and ``order_labels.py``.
"""

from qtpy.QtWidgets import QWidget

from .calibration import CalibrationWorkflow
from .controls import WorkflowControls
from .layers import WorkflowLayers
from .mask import MaskWorkflow
from .order_labels import OrderLabelsWorkflow
from .roi import RoiWorkflow


class CropWidget(QWidget):
    """Single dock widget for ROI, mask refinement, and order labeling."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self._controls = WorkflowControls.create()
        self._layers = WorkflowLayers(napari_viewer)
        self._calibration_workflow = CalibrationWorkflow(
            self._layers,
            self._controls,
        )
        self._roi_workflow = RoiWorkflow(self._layers, self._controls)
        self._mask_workflow = MaskWorkflow(self._layers, self._controls, self)
        self._order_labels_workflow = OrderLabelsWorkflow(
            self._layers,
            self._controls,
        )

        # 既存利用コードが同じ属性名でQt controlへアクセスできるよう明示的に公開する。
        self.status_label = self._controls.status_label
        self.calibration_status_label = self._controls.calibration_status_label
        self.calibration_method_combo = self._controls.calibration_method_combo
        self.calibration_value_spinbox = self._controls.calibration_value_spinbox
        self.calibration_unit_combo = self._controls.calibration_unit_combo
        self.draw_scale_bar_button = self._controls.draw_scale_bar_button
        self.apply_scale_button = self._controls.apply_scale_button
        self.add_roi_button = self._controls.add_roi_button
        self.apply_gaussian_button = self._controls.apply_gaussian_button
        self.otsu_button = self._controls.otsu_button
        self.invert_threshold_checkbox = self._controls.invert_threshold_checkbox
        self.clean_mask_button = self._controls.clean_mask_button
        self.create_order_labels_button = self._controls.create_order_labels_button
        self.gaussian_sigma_label = self._controls.gaussian_sigma_label
        self.gaussian_sigma_spinbox = self._controls.gaussian_sigma_spinbox
        self.threshold_label = self._controls.threshold_label
        self.threshold_spinbox = self._controls.threshold_spinbox
        self.threshold_slider = self._controls.threshold_slider
        self.min_object_size_label = self._controls.min_object_size_label
        self.min_object_size_spinbox = self._controls.min_object_size_spinbox
        self.max_hole_size_label = self._controls.max_hole_size_label
        self.max_hole_size_spinbox = self._controls.max_hole_size_spinbox
        self.closing_radius_label = self._controls.closing_radius_label
        self.closing_radius_spinbox = self._controls.closing_radius_spinbox
        self.opening_radius_label = self._controls.opening_radius_label
        self.opening_radius_spinbox = self._controls.opening_radius_spinbox

        # TimerのQt parentはCropWidgetのままにして、dock破棄時の寿命を揃える。
        self._mask_update_timer = self._mask_workflow.update_timer

        # 旧実装で保持していた対応表も同じobjectへのaliasとして残す。
        self._roi_to_image = self._layers.roi_to_image
        self._roi_to_crop = self._layers.roi_to_crop
        self._scale_bar_to_image = self._layers.scale_bar_to_image
        self._image_to_scale_bar = self._layers.image_to_scale_bar
        self._crop_to_smoothed = self._layers.crop_to_smoothed
        self._smoothed_to_crop = self._layers.smoothed_to_crop
        self._crop_to_mask = self._layers.crop_to_mask
        self._mask_to_crop = self._layers.mask_to_crop
        self._mask_to_order_labels = self._layers.mask_to_order_labels
        self._order_labels_to_mask = self._layers.order_labels_to_mask
        self._mask_updates_enabled = self._layers.mask_updates_enabled

        self.setLayout(self._controls.create_layout())
        self._connect_signals()

        for layer in list(self.viewer.layers):
            self._prepare_image_layer(layer)

    def _connect_signals(self) -> None:
        """Wire UI and viewer events at one visible composition point."""
        self.calibration_method_combo.currentIndexChanged.connect(
            self._calibration_workflow.on_method_changed
        )
        self.draw_scale_bar_button.clicked.connect(self.draw_scale_bar_line)
        self.apply_scale_button.clicked.connect(self.apply_scale)
        self.add_roi_button.clicked.connect(self.add_roi)
        self.apply_gaussian_button.clicked.connect(self.apply_gaussian_filter)
        self.otsu_button.clicked.connect(self.use_otsu_threshold)
        self.clean_mask_button.clicked.connect(self.clean_mask)
        self.create_order_labels_button.clicked.connect(self.create_order_labels)
        self.invert_threshold_checkbox.toggled.connect(self._schedule_mask_update)
        self.threshold_spinbox.valueChanged.connect(
            self._on_threshold_spinbox_changed
        )
        self.threshold_slider.valueChanged.connect(
            self._on_threshold_slider_changed
        )
        self._mask_update_timer.timeout.connect(self._update_mask_from_controls)
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.selection.events.active.connect(
            self._calibration_workflow.on_active_layer_changed
        )

    def draw_scale_bar_line(self) -> None:
        """Create or select the manual scale-bar line for the active image."""
        self._calibration_workflow.draw_scale_bar_line()

    def apply_scale(self) -> None:
        """Apply the selected pixel-to-metric calibration method."""
        self._calibration_workflow.apply_scale()

    def add_roi(self) -> None:
        """Delegate explicit ROI confirmation to the ROI workflow."""
        self._roi_workflow.add_roi()

    def apply_gaussian_filter(self) -> None:
        """Delegate explicit smoothing to the mask workflow."""
        self._mask_workflow.apply_gaussian_filter()

    def use_otsu_threshold(self) -> None:
        """Delegate explicit mask creation to the mask workflow."""
        self._mask_workflow.use_otsu_threshold()

    def clean_mask(self) -> None:
        """Delegate explicit morphology cleanup to the mask workflow."""
        self._mask_workflow.clean_mask()

    def create_order_labels(self) -> None:
        """Delegate manual order-label setup to the labeling workflow."""
        self._order_labels_workflow.create_order_labels()

    def _on_layer_inserted(self, event) -> None:
        self._prepare_image_layer(event.value)

    def _prepare_image_layer(self, layer) -> None:
        self._calibration_workflow.prepare_image_layer(layer)
        self._roi_workflow.prepare_image_layer(layer)

    def _schedule_mask_update(self, *args) -> None:
        self._mask_workflow.schedule_mask_update(*args)

    def _on_threshold_spinbox_changed(self, value: float) -> None:
        self._mask_workflow.on_threshold_spinbox_changed(value)

    def _on_threshold_slider_changed(self, value: int) -> None:
        self._mask_workflow.on_threshold_slider_changed(value)

    def _update_mask_from_controls(self) -> None:
        self._mask_workflow.update_mask_from_controls()

