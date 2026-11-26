"""Composition root for the single leaf-vein napari dock widget.

This module wires Qt signals to explicit workflow objects.  Stage-specific
layer operations live in ``roi.py``, ``mask.py``, and ``order_labels.py``.
"""

from napari.qt.threading import create_worker
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QCheckBox, QFileDialog, QMessageBox, QWidget

from ..auto_order import auto_assign_orders
from ..resume import MissingSourceImage, read_saved_analysis
from .auto_order import AutoOrderWorkflow
from .assistant import FolipukaAssistant
from .calibration import CalibrationWorkflow
from .controls import WorkflowControls
from .layers import WorkflowLayers
from .mask import MaskWorkflow
from .measurement import MeasurementWorkflow, run_export_job, show_export_messages
from .order_labels import OrderLabelsWorkflow
from .progress import ProgressReporter
from .roi import RoiWorkflow
from .resume import ResumeWorkflow
from .settings import SettingsWorkflow
from .vein_paths import VeinPathsWorkflow


class CropWidget(QWidget):
    """Compose the workflows and connect user actions to them.

    ここは操作の入口。計算はパッケージ直下の関数、レイヤーの変更は各工程の
    Workflow へ委譲する。ボタンから処理を探すときは _connect_signals を読む。
    CropWidget という名前と公開属性は、以前の利用コードとの互換性のために残す。
    """

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
        self._resume_workflow = ResumeWorkflow(
            layers=self._layers, controls=self._controls,
            roi=self._roi_workflow, mask=self._mask_workflow,
            orders=self._order_labels_workflow, auto_order=self._auto_order_workflow,
        )
        self._auto_order_worker = None
        self._export_worker = None
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
        self.edit_mask_button = self._controls.edit_mask_button
        self.next_leaf_button = self._controls.next_leaf_button
        self.resume_button = self._controls.resume_button
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
        self.folipuka_checkbox = QCheckBox("フォリプカちゃんを表示 / Show Folipuka")
        self.folipuka_checkbox.setChecked(True)
        self.layout().insertWidget(1, self.folipuka_checkbox)
        self.folipuka_assistant = FolipukaAssistant(self)
        self.destroyed.connect(self.folipuka_assistant.deleteLater)
        self.folipuka_checkbox.toggled.connect(self._toggle_folipuka)
        self.folipuka_assistant.dismissed.connect(lambda: self.folipuka_checkbox.setChecked(False))
        for key, section in self._controls.sections.items():
            section.toggle_button.toggled.connect(
                lambda expanded, key=key: self.folipuka_assistant.set_stage(key, expanded)
            )
        self._connect_signals()
        # 起動時に設定ファイルを読み、各入力欄の既定値に反映する。
        self._settings_workflow.load_and_apply()

        for layer in list(self.viewer.layers):
            self._prepare_image_layer(layer)

    def _toggle_folipuka(self, visible):
        if visible and self.isVisible():
            self.folipuka_assistant.show_on(self.window())
        else:
            self.folipuka_assistant.hide()

    def showEvent(self, event):
        super().showEvent(event)
        self._toggle_folipuka(self.folipuka_checkbox.isChecked())

    def hideEvent(self, event):
        self.folipuka_assistant.hide()
        super().hideEvent(event)

    def _connect_signals(self) -> None:
        """Wire UI and viewer events at one visible composition point."""
        # 操作ボタンの接続。クリックによる処理開始は、ここにまとめて見えるようにする。
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
        self.edit_mask_button.clicked.connect(self._order_labels_workflow.edit_vein_mask)
        self.next_leaf_button.clicked.connect(self.reset_analysis)
        self.resume_button.clicked.connect(lambda *_: self.load_export())
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
        # 編集中の入力値はタイマーでまとめて反映する。マスクが手修正段階に入った
        # 後は MaskWorkflow 側の gate が自動再計算を止め、描いた内容を保護する。
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
        # 選択レイヤーが変わったときだけ、校正値や葉面積の表示を対応するROIへ切り替える。
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
        if self._mask_workflow.clean_mask():
            self._controls.open_section("orders")

    def reset_analysis(self) -> None:
        """Start the next leaf after explicit confirmation of unsaved edits."""
        if self._auto_order_worker is not None or self._export_worker is not None:
            self._controls.warn("Wait for auto-assignment or export before resetting.")
            return
        source = self._layers.active_source_image()
        active = self.viewer.layers.selection.active
        if source is None and active is not None:
            source = self._layers.by_name(active.metadata.get("source_image"))
        if source is None:
            self._controls.open_section("image")
            return
        answer = QMessageBox.question(
            self, "Start next leaf?",
            "Remove this image and its workflow layers from the viewer? "
            "Export any edits you need first. Original files and exported results "
            "are kept on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._mask_update_timer.stop()
        self._layers.reset_source(source)
        self.leaf_area_input.clear()
        self._controls.set_leaf_area_status("Leaf area: not confirmed")
        self._controls.leaf_area_source_combo.setCurrentIndex(
            self._controls.leaf_area_source_combo.findData("input")
        )
        self._controls.confirm_leaf_button.setEnabled(False)
        self._controls.set_gaussian_enabled(False)
        self._controls.set_threshold_enabled(False)
        self._controls.set_cleanup_enabled(False)
        self._controls.set_measurement_enabled(False)
        self._controls.set_calibration_enabled(bool(self._layers.roi_to_image))
        self._controls.set_calibration_status("Scale: select an image to calibrate")
        self._controls.set_cleanup_unit_labels(None)
        self.add_roi_button.setEnabled(bool(self._layers.roi_to_image))
        self._controls.hide_progress()
        self._controls.open_section("image")
        self._controls.set_status("Ready for the next leaf. Open its scan with File > Open.")

    def confirm_leaf(self) -> None:
        """Confirm the edited leaf region and extract its vein mask."""
        if self._mask_workflow.confirm_leaf():
            self._controls.open_section("vein_mask")

    def load_export(self, path=None, source_path=None):
        """Resume an exported folder or metadata file without rerunning Otsu."""
        from pathlib import Path
        import yaml

        if self._auto_order_worker is not None or self._export_worker is not None:
            self._controls.warn("Wait for the current operation before loading an export.")
            return None
        interactive = path is None
        if interactive:
            path = QFileDialog.getExistingDirectory(self, "Choose export folder", str(self._measurement_workflow.output_dir()))
            if not path:
                return None
            if len(list(Path(path).glob("*_metadata.yaml"))) > 1:
                path, _ = QFileDialog.getOpenFileName(self, "Choose the saved analysis", path, "Export metadata (*_metadata.yaml)")
                if not path:
                    return None
        try:
            try:
                saved = read_saved_analysis(path, source_path)
            except MissingSourceImage:
                if not interactive:
                    raise
                source_path, _ = QFileDialog.getOpenFileName(self, "Locate the original leaf scan")
                if not source_path:
                    return None
                saved = read_saved_analysis(path, source_path)
            self._mask_update_timer.stop()
            return self._resume_workflow.restore(saved)
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            self._controls.warn(f"Cannot resume export: {exc}")
            return None

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
        if self._auto_order_worker is not None or self._export_worker is not None:
            self._controls.warn("Wait for the current operation to finish.")
            return None
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
        folder = result.metadata_yaml.parent if result.metadata_yaml else self._measurement_workflow.output_dir()
        self._controls.set_status(f"Exported to {folder}")

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
        if self._auto_order_worker is not None or self._export_worker is not None:
            self._controls.warn("Wait for the current operation to finish.")
            return None
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
        reporter = self._new_progress_reporter("orders")
        worker = create_worker(auto_assign_orders, **job.kwargs(), progress=reporter)
        worker.returned.connect(lambda result: self._auto_order_workflow.apply(job, result))
        worker.errored.connect(lambda error: self._controls.warn(f"Auto-assign failed: {error}"))
        worker.finished.connect(self._reset_auto_order_button)
        self._auto_order_worker = worker
        worker.start()
        return worker

    def _new_progress_reporter(self, section="export") -> ProgressReporter:
        """Progress callback for a worker; Qt delivers it on the main thread."""
        reporter = ProgressReporter(self)
        self._controls.place_progress(section)
        # controls は slots の dataclass で弱参照できないので、widget のメソッドを挟む
        reporter.progressed.connect(self._show_progress)
        self._controls.show_progress(0.0, "Starting")
        return reporter

    def _show_progress(self, fraction: float, message: str) -> None:
        self._controls.show_progress(fraction, message)

    def _reset_auto_order_button(self) -> None:
        self._controls.hide_progress()
        self.auto_orders_button.setText("Auto-assign 1°/2° + 3° guess")
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

