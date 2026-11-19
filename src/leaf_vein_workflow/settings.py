"""User-editable default settings stored as YAML, independent of napari.

Built-in recommended values live in :data:`DEFAULT_SETTINGS`.  A YAML file only
overrides the keys it contains, so a missing or invalid key falls back to the
recommended value.  :func:`reset_settings_file` restores the recommended file
and keeps the previous one as a ``.bak`` copy.

Lookup order:

1. ``leaf_vein_workflow.yaml`` in the folder napari was started from.  When it
   is missing but ``leaf_vein_workflow.example.yaml`` (committed in the
   repository) is there, the example is copied to create it, so each user edits
   their own untracked copy.
2. ``config.yaml`` in the per-user config folder.
"""

from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import platformdirs
import yaml

APP_NAME = "leaf-vein-workflow"
PROJECT_SETTINGS_NAME = "leaf_vein_workflow.yaml"
EXAMPLE_SETTINGS_NAME = "leaf_vein_workflow.example.yaml"
USER_SETTINGS_NAME = "config.yaml"

CALIBRATION_METHODS = ("dpi", "um_per_pixel", "mm_per_pixel", "scale_bar")
LEAF_AREA_UNITS = ("mm2", "cm2")

DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "calibration": {
        "method": "dpi",
        "value": 1200.0,
    },
    "scale_bar": {
        "length_mm": 10.0,
    },
    "mask": {
        "gaussian_sigma": 1.0,
        "invert_threshold": False,
    },
    "cleanup": {
        "small_object_px": 100,
        "max_hole_px": 50,
        "closing_radius_px": 1,
        "opening_radius_px": 0,
    },
    "measurement": {
        "leaf_area_unit": "mm2",
        "output_dir": "output",
    },
}

DEFAULT_SETTINGS_YAML = """\
# leaf-vein-workflow settings
# Only the keys written here override the built-in recommended values.
# Delete a key (or use "Reset to Defaults" in the widget) to go back to the
# recommended value. In the repository, copying leaf_vein_workflow.example.yaml
# over leaf_vein_workflow.yaml also restores every recommended value.

calibration:
  # dpi, um_per_pixel, mm_per_pixel, or scale_bar
  method: dpi
  # Scanner resolution. 1200 dpi (about 21 um/px) resolves 2nd-order vein width.
  value: 1200

scale_bar:
  # Display bar drawn after Apply Scale. Shorter round lengths are used when
  # this does not fit within 40% of the image width.
  length_mm: 10

mask:
  gaussian_sigma: 1.0
  # false: pixels brighter than the threshold become vein.
  # true: pixels darker than the threshold become vein.
  invert_threshold: false

cleanup:
  # Remove vein fragments up to this many pixels.
  small_object_px: 100
  # Fill holes inside veins up to this many pixels (large values fill areoles).
  max_hole_px: 50
  closing_radius_px: 1
  opening_radius_px: 0

measurement:
  # mm2 or cm2
  leaf_area_unit: mm2
  # Relative paths are resolved against the folder napari was started from.
  output_dir: output
"""


@dataclass(slots=True)
class LoadedSettings:
    """Merged settings plus where they came from and what was ignored."""

    values: dict[str, dict[str, Any]]
    path: Path | None
    warnings: list[str] = field(default_factory=list)


def default_settings() -> dict[str, dict[str, Any]]:
    """Return a fresh copy of the recommended settings."""
    return copy.deepcopy(DEFAULT_SETTINGS)


def user_settings_path() -> Path:
    """Return the per-user settings file path (it may not exist yet)."""
    return Path(platformdirs.user_config_dir(APP_NAME, appauthor=False)) / USER_SETTINGS_NAME


def find_settings_path(start_dir: str | Path | None = None) -> Path:
    """Return the project settings file, else the user file path.

    If only the committed example exists in ``start_dir``, it is copied to
    ``leaf_vein_workflow.yaml`` first and that copy is returned.
    """
    base = Path(start_dir) if start_dir is not None else Path.cwd()
    project_path = base / PROJECT_SETTINGS_NAME
    if project_path.is_file():
        return project_path
    example_path = base / EXAMPLE_SETTINGS_NAME
    if example_path.is_file():
        try:
            shutil.copyfile(example_path, project_path)
            return project_path
        except OSError:
            pass
    return user_settings_path()


def load_settings(path: str | Path | None) -> LoadedSettings:
    """Merge a YAML file over the recommended values, validating each key."""
    values = default_settings()
    if path is None:
        return LoadedSettings(values=values, path=None)

    settings_path = Path(path)
    if not settings_path.is_file():
        return LoadedSettings(values=values, path=settings_path)

    try:
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return LoadedSettings(
            values=values,
            path=settings_path,
            warnings=[f"Could not read {settings_path.name}: {exc}. Using defaults."],
        )

    warnings: list[str] = []
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return LoadedSettings(
            values=values,
            path=settings_path,
            warnings=[f"{settings_path.name} must contain sections. Using defaults."],
        )

    for section, section_values in raw.items():
        if section not in values:
            warnings.append(f"Unknown section '{section}' ignored.")
            continue
        if not isinstance(section_values, dict):
            warnings.append(f"Section '{section}' must be a mapping; ignored.")
            continue
        for key, value in section_values.items():
            if key not in values[section]:
                warnings.append(f"Unknown key '{section}.{key}' ignored.")
                continue
            try:
                values[section][key] = _validate(section, key, value)
            except ValueError as exc:
                warnings.append(f"{section}.{key}: {exc} Using the default.")

    return LoadedSettings(values=values, path=settings_path, warnings=warnings)


def write_default_settings(path: str | Path) -> Path:
    """Write the commented recommended settings file and return its path."""
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(DEFAULT_SETTINGS_YAML, encoding="utf-8")
    return settings_path


def reset_settings_file(path: str | Path) -> Path | None:
    """Restore the recommended file, keeping the old one as ``<name>.bak``."""
    settings_path = Path(path)
    backup: Path | None = None
    if settings_path.is_file():
        backup = settings_path.with_name(settings_path.name + ".bak")
        shutil.copy2(settings_path, backup)
    write_default_settings(settings_path)
    return backup


def resolve_output_dir(value: str, start_dir: str | Path | None = None) -> Path:
    """Resolve a configured output folder against the launch folder."""
    output = Path(value).expanduser()
    if output.is_absolute():
        return output
    base = Path(start_dir) if start_dir is not None else Path.cwd()
    return base / output


def _validate(section: str, key: str, value: Any) -> Any:
    default = DEFAULT_SETTINGS[section][key]

    if section == "calibration" and key == "method":
        if value not in CALIBRATION_METHODS:
            raise ValueError(f"must be one of {', '.join(CALIBRATION_METHODS)}.")
        return value
    if section == "measurement" and key == "leaf_area_unit":
        if value not in LEAF_AREA_UNITS:
            raise ValueError(f"must be one of {', '.join(LEAF_AREA_UNITS)}.")
        return value
    if section == "measurement" and key == "output_dir":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty folder path.")
        return value

    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError("must be true or false.")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("must be a non-negative integer.")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number.")
        number = float(value)
        allow_zero = section == "mask" and key == "gaussian_sigma"
        if number < 0 or (number == 0 and not allow_zero):
            raise ValueError("must be greater than zero.")
        return number
    return value
