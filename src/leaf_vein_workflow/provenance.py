"""Reproducibility metadata written next to exported measurements.

Records what is needed to repeat or audit an export: the input file (with a
SHA-256 checksum), the parameters actually applied at each processing step,
whether the mask was edited by hand afterwards, the leaf area used, software
versions, and the settings defaults in effect.
"""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

PACKAGES = ("leaf-vein-workflow", "napari", "numpy", "scikit-image", "skan", "pandas")


def array_digest(array: np.ndarray) -> str:
    """Return a short content hash used to detect manual edits of a mask."""
    values = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str | None:
    """Return the SHA-256 of a file, or ``None`` when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def software_versions() -> dict[str, str]:
    """Return Python and key package versions."""
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def source_file_info(path: str | Path | None) -> dict[str, Any] | None:
    """Describe the input image file, if napari knows where it came from."""
    if not path:
        return None
    file_path = Path(path)
    return {
        "path": str(file_path),
        "sha256": file_sha256(file_path),
    }


def write_metadata(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Write a metadata record as human-readable YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    full_record = {
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **_plain(record),
        "software": software_versions(),
    }
    output.write_text(
        yaml.safe_dump(full_record, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def _plain(value: Any) -> Any:
    """Convert numpy scalars, paths, and tuples into YAML-safe values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
