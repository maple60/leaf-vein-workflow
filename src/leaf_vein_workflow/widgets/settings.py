"""Apply YAML settings to the dock controls and manage the settings file."""

from pathlib import Path
from typing import Any

from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices

from ..settings import (
    LoadedSettings,
    find_settings_path,
    load_settings,
    reset_settings_file,
    resolve_output_dir,
    write_default_settings,
)
from .controls import WorkflowControls


class SettingsWorkflow:
    """Load, reload, reset, and apply user defaults."""

    def __init__(
        self,
        controls: WorkflowControls,
        settings_path: str | Path | None = None,
        start_dir: str | Path | None = None,
    ):
        self.controls = controls
        self.start_dir = Path(start_dir) if start_dir is not None else Path.cwd()
        self.path = (
            Path(settings_path)
            if settings_path is not None
            else find_settings_path(self.start_dir)
        )
        self.loaded: LoadedSettings | None = None

    def load_and_apply(self) -> LoadedSettings:
        """Read the settings file (if any) and push values into the controls."""
        self.loaded = load_settings(self.path)
        self.apply(self.loaded.values)
        source = (
            str(self.path)
            if self.path.is_file()
            else f"built-in defaults ({self.path} not created yet)"
        )
        self.controls.settings_path_label.setText(f"Settings: {source}")
        for message in self.loaded.warnings:
            show_warning(f"Settings: {message}")
        return self.loaded

    def reload(self) -> None:
        """Re-read the settings file after the user edits it."""
        loaded = self.load_and_apply()
        if not loaded.warnings:
            show_info("Reloaded settings.")

    def reset_to_defaults(self) -> Path | None:
        """Restore the recommended file and values, keeping a .bak copy."""
        backup = reset_settings_file(self.path)
        self.load_and_apply()
        if backup is not None:
            show_info(f"Restored recommended settings. Previous file: {backup.name}")
        else:
            show_info("Restored recommended settings.")
        return backup

    def open_settings_file(self) -> None:
        """Create the file with recommended values if needed, then open it."""
        if not self.path.is_file():
            write_default_settings(self.path)
            self.load_and_apply()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path)))

    def apply(self, values: dict[str, dict[str, Any]]) -> None:
        """Set control defaults from merged settings values."""
        controls = self.controls

        calibration = values["calibration"]
        index = controls.calibration_method_combo.findData(calibration["method"])
        if index >= 0:
            controls.calibration_method_combo.setCurrentIndex(index)
        controls.set_calibration_method(calibration["method"])
        controls.calibration_value_input.setText(f"{calibration['value']:g}")

        controls.scale_bar_length_mm = float(values["scale_bar"]["length_mm"])

        mask = values["mask"]
        controls.gaussian_sigma_spinbox.setValue(float(mask["gaussian_sigma"]))
        controls.invert_threshold_checkbox.setChecked(bool(mask["invert_threshold"]))

        cleanup = values["cleanup"]
        controls.min_object_size_spinbox.setValue(int(cleanup["small_object_px"]))
        controls.max_hole_size_spinbox.setValue(int(cleanup["max_hole_px"]))
        controls.closing_radius_spinbox.setValue(int(cleanup["closing_radius_px"]))
        controls.opening_radius_spinbox.setValue(int(cleanup["opening_radius_px"]))

        measurement = values["measurement"]
        unit_index = controls.leaf_area_unit_combo.findData(
            measurement["leaf_area_unit"]
        )
        if unit_index >= 0:
            controls.leaf_area_unit_combo.setCurrentIndex(unit_index)
        controls.output_dir_input.setText(
            str(resolve_output_dir(measurement["output_dir"], self.start_dir))
        )
