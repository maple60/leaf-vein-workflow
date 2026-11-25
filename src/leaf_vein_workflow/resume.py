"""Read and validate an existing label export before changing the viewer."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from skimage.io import imread

from .calibration import CALIBRATION_METADATA_KEYS, millimetres_per_pixel, validate_positive_finite
from .provenance import array_digest, file_sha256


class MissingSourceImage(ValueError):
    """The original image must be located again by the user."""


@dataclass
class SavedAnalysis:
    metadata_path: Path
    record: dict
    source_path: Path
    image: np.ndarray
    labels: np.ndarray
    vein: np.ndarray
    leaf: np.ndarray
    roi: tuple[int, int, int, int]
    calibration: dict
    sigma: float


def read_saved_analysis(path: str | Path, source_path: str | Path | None = None) -> SavedAnalysis:
    """Load current and legacy flat exports; never infer leaf area from veins."""
    metadata_path = Path(path)
    if metadata_path.is_dir():
        candidates = sorted(metadata_path.glob("*_metadata.yaml"))
        if len(candidates) != 1:
            raise ValueError("Select one *_metadata.yaml file in the export folder.")
        metadata_path = candidates[0]
    if not metadata_path.name.endswith("_metadata.yaml"):
        raise ValueError("Select an exported *_metadata.yaml file.")
    record = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("order_labels"), dict):
        raise ValueError("This export has no resumable order labels (path-only exports are not supported).")
    prefix = metadata_path.name.removesuffix("_metadata.yaml")

    def load_array(suffix, allowed):
        file = metadata_path.parent / f"{prefix}_{suffix}.npy"
        if not file.is_file():
            raise ValueError(f"Missing {file.name}; keep the masks and metadata together.")
        values = np.load(file, allow_pickle=False)
        if values.ndim != 2 or not values.size or not np.isin(values, allowed).all():
            raise ValueError(f"Invalid values or shape in {file.name}.")
        return values.astype(np.uint8)

    labels = load_array("vein_order_labels", (0, 1, 2, 3, 4))
    vein = load_array("vein_mask", (0, 1))
    leaf = load_array("leaf_mask", (0, 1))
    if labels.shape != vein.shape or labels.shape != leaf.shape:
        raise ValueError("Saved mask and label shapes do not match.")
    if not leaf.any() or np.any((labels > 0) & (vein == 0)) or np.any((vein > 0) & (leaf == 0)):
        raise ValueError("Saved labels or veins extend outside their mask, or the leaf is empty.")
    processing = record["order_labels"].get("processing") or {}
    leaf_record = processing.get("leaf_mask") or {}
    expected = leaf_record.get("sha256") or (record.get("leaf_area") or {}).get("sha256")
    if expected and array_digest(leaf) != expected:
        raise ValueError("Saved leaf mask does not match the metadata checksum.")
    bounds = processing.get("roi") or {}
    keys = ("row_start", "row_stop", "col_start", "col_stop")
    if any(not isinstance(bounds.get(key), int) for key in keys):
        raise ValueError("ROI bounds are missing from the export metadata.")
    r0, r1, c0, c1 = (bounds[key] for key in keys)
    if min(r0, c0) < 0 or (r1-r0, c1-c0) != labels.shape:
        raise ValueError("ROI bounds do not match the saved labels.")
    calibration = {}
    exported_scale = record.get("calibration") or {}
    for key in CALIBRATION_METADATA_KEYS:
        short = key.removeprefix("leaf_vein_workflow_").removeprefix("calibration_")
        if short in exported_scale:
            calibration[key] = exported_scale[short]
    if "mm_per_pixel" in exported_scale:
        millimetres_per_pixel(exported_scale["mm_per_pixel"])
    area = record.get("leaf_area") or {}
    if area.get("mm2") is not None:
        validate_positive_finite(area["mm2"], "Saved leaf area")
    sigma = float(processing.get("gaussian_sigma", 1.0))
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Invalid saved Gaussian sigma.")
    original = record.get("source_image") or {}
    source = Path(source_path or original.get("path") or "")
    if not source.is_file():
        raise MissingSourceImage("Locate the original scan to resume this export.")
    if original.get("sha256") and file_sha256(source) != original["sha256"]:
        raise ValueError("The selected scan does not match the original image checksum.")
    image = imread(source)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] not in (3, 4)):
        raise ValueError("The original scan must be a 2D grayscale or RGB/RGBA image.")
    if original.get("shape") and tuple(original["shape"]) != image.shape:
        raise ValueError("Original image shape does not match the metadata.")
    if r1 > image.shape[0] or c1 > image.shape[1]:
        raise ValueError("Saved ROI extends outside the original scan.")
    return SavedAnalysis(metadata_path, record, source, image, labels, vein, leaf,
                         (r0, r1, c0, c1), calibration, sigma)
