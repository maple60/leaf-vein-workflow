"""Composition root for the single leaf-vein napari dock widget.

This module wires Qt signals to explicit workflow objects.  Stage-specific
layer operations live in ``roi.py``, ``mask.py``, and ``order_labels.py``.
"""

from qtpy.QtWidgets import QFileDialog, QWidget

from .calibration import CalibrationWorkflow
from .controls import WorkflowControls
from .layers import WorkflowLayers
from .mask import MaskWorkflow
from .measurement import MeasurementWorkflow
from .order_labels import OrderLabelsWorkflow
from .roi import RoiWorkflow
from .settings import SettingsWorkflow
from .vein_paths import VeinPathsWorkflow


class CropWidget(QWidget):
    """Single dock widget for ROI, mask refinement, and order labeling."""

    def __init__(self, napari_viewer, settings_path=None):
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
        self._vein_paths_workflow = VeinPathsWorkflow(self._layers, self._controls)
        self._settings_workflow = SettingsWorkflow(self._controls, settings_path)
        self._measurement_workflow = MeasurementWorkflow(
            self._layers,
            self._controls,
            self._vein_paths_workflow,
        )

        self._measurement_workflow.settings = self._settings_workflow

        # 既存利用コードが同じ属性名でQt controlへアクセスできるよう明示的に公開する。
        self.status_label = self._controls.status_label
        self.calibration_status_label = self._controls.calibration_status_label
        self.calibration_method_combo = self._controls.calibration_method_combo
        self.calibration_value_input = self._controls.calibration_value_input
        self.calibration_unit_combo = self._controls.calibration_unit_combo
        self.draw_scale_bar_button = self._controls.draw_scale_bar_button
        self.apply_scale_button = self._controls.apply_scale_button
        self.path_order_combo = self._controls.path_order_combo
        self.draw_vein_paths_button = self._controls.draw_vein_paths_button
        self.add_roi_button = self._controls.add_roi_button
        self.apply_gaussian_button = self._controls.apply_gaussian_button
        self.otsu_button = self._controls.otsu_button
        self.invert_threshold_checkbox = self._controls.invert_threshold_checkbox
        self.clean_mask_button = self._controls.clean_mask_button
        self.create_order_labels_button = self._controls.create_order_labels_button
        self.leaf_area_input = self._controls.leaf_area_input
        self.leaf_area_unit_combo = self._controls.leaf_area_unit_combo
        self.output_dir_input = self._controls.output_dir_input
        self.change_output_dir_button = self._controls.change_output_dir_button
        self.export_button = self._controls.export_button
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
        # 起動時に設定ファイルを読み、各入力欄の既定値に反映する。
        self._settings_workflow.load_and_apply()

        for layer in list(self.viewer.layers):
            self._prepare_image_layer(layer)

    def _connect_signals(self) -> None:
        """Wire UI and viewer events at one visible composition point."""
        self.calibration_method_combo.currentIndexChanged.connect(
            self._calibration_workflow.on_method_changed
        )
        self.draw_scale_bar_button.clicked.connect(self.draw_scale_bar_line)
        self.apply_scale_button.clicked.connect(self.apply_scale)
        self.draw_vein_paths_button.clicked.connect(self.draw_vein_paths)
        self._controls.open_settings_button.clicked.connect(
            self._settings_workflow.open_settings_file
        )
        self._controls.reload_settings_button.clicked.connect(
            self._settings_workflow.reload
        )
        self._controls.reset_settings_button.clicked.connect(
            self._settings_workflow.reset_to_defaults
        )
        self.path_order_combo.currentIndexChanged.connect(
            self._vein_paths_workflow.on_order_changed
        )
        self.add_roi_button.clicked.connect(self.add_roi)
        self.apply_gaussian_button.clicked.connect(self.apply_gaussian_filter)
        self.otsu_button.clicked.connect(self.use_otsu_threshold)
        self.clean_mask_button.clicked.connect(self.clean_mask)
        self.create_order_labels_button.clicked.connect(self.create_order_labels)
        self.change_output_dir_button.clicked.connect(self.choose_output_dir)
        self.export_button.clicked.connect(self.export_measurements)
        self.invert_threshold_checkbox.toggled.connect(self._schedule_mask_update)
        self.threshold_spinbox.valueChanged.connect(
            self._on_threshold_spinbox_changed
        )
        self.threshold_slider.valueChanged.connect(
            self._on_threshold_slider_changed
        )
        for spinbox in (
            self.min_object_size_spinbox,
            self.max_hole_size_spinbox,
            self.closing_radius_spinbox,
            self.opening_radius_spinbox,
        ):
            spinbox.valueChanged.connect(
                self._calibration_workflow.refresh_cleanup_unit_labels
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
        source_image = self._layers.active_source_image()
        before = None if source_image is None else dict(source_image.metadata)
        self._calibration_workflow.apply_scale()
        if source_image is not None and dict(source_image.metadata) != before:
            self._controls.open_section("roi")

    def draw_vein_paths(self):
        """Create or select the 1°/2° path layer for the active image."""
        return self._vein_paths_workflow.draw_vein_paths()

    def add_roi(self) -> None:
        """Delegate explicit ROI confirmation to the ROI workflow."""
        self._roi_workflow.add_roi()
        if self._layers.active_crop_layer() is not None:
            self._controls.open_section("mask_creation")

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
        if self._measurement_workflow.active_order_labels_layer() is not None:
            self._controls.open_section("measurement")

    def measure_veins(self):
        """Return per-order vein length and VLA rows without exporting."""
        return self._measurement_workflow.measure()

    def choose_output_dir(self) -> None:
        """Let the user pick a different export folder."""
        current = self._measurement_workflow.output_dir()
        start = current if current.exists() else current.parent
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose output folder",
            str(start),
        )
        if selected:
            self._measurement_workflow.set_output_dir(selected)

    def export_measurements(self, output_dir=None):
        """Export to the output folder shown in the dock (or ``output_dir``)."""
        if output_dir:
            self._measurement_workflow.set_output_dir(output_dir)
        return self._measurement_workflow.export()

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

