"""Composition root for the single leaf-vein napari dock widget.

This module wires Qt signals to explicit workflow objects.  Stage-specific
layer operations live in ``roi.py``, ``mask.py``, and ``order_labels.py``.
"""

from napari.qt.threading import create_worker
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QFileDialog, QWidget

from ..auto_order import auto_assign_orders
from .auto_order import AutoOrderWorkflow
from .calibration import CalibrationWorkflow
from .controls import WorkflowControls
from .layers import WorkflowLayers
from .mask import MaskWorkflow
from .measurement import MeasurementWorkflow, run_export_job, show_export_messages
from .order_labels import OrderLabelsWorkflow
from .progress import ProgressReporter
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
        self._auto_order_workflow = AutoOrderWorkflow(self._layers, self._controls)
        self._auto_order_worker = None
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
        self.confirm_leaf_button = self._controls.confirm_leaf_button
        self.leaf_area_source_combo = self._controls.leaf_area_source_combo
        self.leaf_area_status_label = self._controls.leaf_area_status_label
        self.invert_threshold_checkbox = self._controls.invert_threshold_checkbox
        self.clean_mask_button = self._controls.clean_mask_button
        self.create_order_labels_button = self._controls.create_order_labels_button
        self.paint_order_combo = self._controls.paint_order_combo
        self.mark_seeds_button = self._controls.mark_seeds_button
        self.auto_orders_button = self._controls.auto_orders_button
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
        self.confirm_leaf_button.clicked.connect(self.confirm_leaf)
        self.clean_mask_button.clicked.connect(self.clean_mask)
        self.create_order_labels_button.clicked.connect(self.create_order_labels)
        self.paint_order_combo.currentIndexChanged.connect(
            self._order_labels_workflow.on_paint_order_changed
        )
        self.mark_seeds_button.clicked.connect(self.mark_base_and_apex)
        self.auto_orders_button.clicked.connect(
            lambda *_: self.auto_assign_orders(background=True)
        )
        self.change_output_dir_button.clicked.connect(self.choose_output_dir)
        self.export_button.clicked.connect(
            lambda *_: self.export_measurements(background=True)
        )
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
        self.viewer.layers.selection.events.active.connect(
            self._mask_workflow.refresh_leaf_status
        )

    def draw_scale_bar_line(self) -> None:
        """Create or select the manual scale-bar line for the active image."""
        self._calibration_workflow.draw_scale_bar_line()

    def apply_scale(self) -> None:
        """Apply the selected pixel-to-metric calibration method."""
        source_image = self._layers.active_source_image()
        before = None if source_image is None else dict(source_image.metadata)
        self._calibration_workflow.apply_scale()
        self._mask_workflow.refresh_leaf_status()
        if source_image is not None and dict(source_image.metadata) != before:
            self._controls.open_section("roi")

    def draw_vein_paths(self):
        """Create or select the 1°/2° path layer for the active image."""
        return self._vein_paths_workflow.draw_vein_paths()

    def add_roi(self) -> None:
        """Delegate explicit ROI confirmation to the ROI workflow."""
        self._roi_workflow.add_roi()
        if self._layers.active_crop_layer() is not None:
            self._controls.open_section("mask")

    def apply_gaussian_filter(self) -> None:
        """Delegate explicit smoothing to the mask workflow."""
        self._mask_workflow.apply_gaussian_filter()

    def use_otsu_threshold(self) -> None:
        """Delegate explicit mask creation to the mask workflow."""
        self._mask_workflow.use_otsu_threshold()

    def clean_mask(self) -> None:
        """Delegate explicit morphology cleanup to the mask workflow."""
        self._mask_workflow.clean_mask()

    def confirm_leaf(self) -> None:
        """Confirm the edited leaf region and extract its vein mask."""
        self._mask_workflow.confirm_leaf()

    def create_order_labels(self) -> None:
        """Delegate manual order-label setup to the labeling workflow."""
        # 塗り分けはこの後に行うので、セクションは開いたままにする。
        self._order_labels_workflow.create_order_labels()

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

    def export_measurements(self, output_dir=None, background=False):
        """Export to the output folder shown in the dock (or ``output_dir``).

        With ``background=True`` (the Export button) the measuring and file
        writing run in a worker thread so napari stays responsive; the worker
        is returned.  Otherwise the export runs here and its result is returned.
        """
        if output_dir:
            self._measurement_workflow.set_output_dir(output_dir)
        if not background:
            return self._measurement_workflow.export()

        job = self._measurement_workflow.prepare_export()
        if job is None:
            return None
        self.export_button.setEnabled(False)
        self.export_button.setText("Exporting...")
        self._controls.set_status(
            "Exporting in the background. Large scans can take about a minute; "
            "napari stays usable."
        )
        worker = create_worker(run_export_job, job, progress=self._new_progress_reporter())
        worker.returned.connect(self._on_export_finished)
        worker.errored.connect(self._on_export_failed)
        worker.finished.connect(self._reset_export_button)
        self._export_worker = worker
        worker.start()
        return worker

    def _on_export_finished(self, result) -> None:
        show_export_messages(result)
        self._controls.set_status(f"Exported to {self._measurement_workflow.output_dir()}")

    def _on_export_failed(self, error) -> None:
        self._controls.warn(f"Export failed: {error}")

    def mark_base_and_apex(self):
        """Create or select the Points layer for the base and apex clicks."""
        return self._auto_order_workflow.mark_base_and_apex()

    def auto_assign_orders(self, background=False):
        """Auto-assign 1°/2° into uncertain pixels of the order labels.

        With ``background=True`` (the button) the algorithm runs in a worker
        thread and the worker is returned; otherwise the number of changed
        pixels is returned.
        """
        job = self._auto_order_workflow.prepare()
        if job is None:
            return None
        if not background:
            try:
                result = auto_assign_orders(**job.kwargs())
            except ValueError as exc:
                self._controls.warn(str(exc))
                return None
            return self._auto_order_workflow.apply(job, result)

        self.auto_orders_button.setEnabled(False)
        self.auto_orders_button.setText("Assigning...")
        self._controls.set_status(
            "Auto-assigning 1°/2° in the background (about a minute for a full leaf)."
        )
        reporter = self._new_progress_reporter()
        worker = create_worker(auto_assign_orders, **job.kwargs(), progress=reporter)
        worker.returned.connect(lambda result: self._auto_order_workflow.apply(job, result))
        worker.errored.connect(lambda error: self._controls.warn(f"Auto-assign failed: {error}"))
        worker.finished.connect(self._reset_auto_order_button)
        self._auto_order_worker = worker
        worker.start()
        return worker

    def _new_progress_reporter(self) -> ProgressReporter:
        """Progress callback for a worker; Qt delivers it on the main thread."""
        reporter = ProgressReporter(self)
        # controls は slots の dataclass で弱参照できないので、widget のメソッドを挟む
        reporter.progressed.connect(self._show_progress)
        self._controls.show_progress(0.0, "Starting")
        return reporter

    def _show_progress(self, fraction: float, message: str) -> None:
        self._controls.show_progress(fraction, message)

    def _reset_auto_order_button(self) -> None:
        self._controls.hide_progress()
        self.auto_orders_button.setText("Auto-assign 1°/2°")
        self.auto_orders_button.setEnabled(True)
        self._auto_order_worker = None

    def _reset_export_button(self) -> None:
        self._controls.hide_progress()
        self.export_button.setText("Export Measurements")
        self.export_button.setEnabled(True)
        self._export_worker = None

    def _on_layer_inserted(self, event) -> None:
        self._prepare_image_layer(event.value)
        # 後から追加されたレイヤーの下にスケールバーが隠れないようにする。
        QTimer.singleShot(0, self._layers.raise_scale_indicators)

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

