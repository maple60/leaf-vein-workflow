"""Read saved arrays and metadata without depending on napari or Qt.

読み込みは「ファイル選択 → 配列とROIの検証 → 元画像の照合」の順に行う。
この段階では画面を変更しない。検証が済んだ SavedAnalysis だけを、
widgets/resume.py が編集可能なレイヤーへ戻す。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from skimage.io import imread

from .calibration import (
    CALIBRATION_METADATA_KEYS,
    millimetres_per_pixel,
    validate_positive_finite,
)
from .provenance import array_digest, file_sha256


class MissingSourceImage(ValueError):
    """The original scan needs to be located again; saved labels remain usable."""


@dataclass
class SavedAnalysis:
    """Validated session data, independent of any viewer.

    labels は 0=背景、1/2/3=次数、4=未確定。leaf と vein は 0/1 の uint8。
    roi は元画像に対する (row_start, row_stop, col_start, col_stop) で、
    stop は NumPy の slice と同じく範囲に含まない。
    """

    metadata_path: Path
    record: dict[str, Any]
    source_path: Path
    image: np.ndarray
    labels: np.ndarray
    vein: np.ndarray
    leaf: np.ndarray
    roi: tuple[int, int, int, int]
    calibration: dict[str, Any]
    sigma: float

    @property
    def processing(self) -> dict[str, Any]:
        return self.record["order_labels"].get("processing") or {}

    @property
    def auto_order(self) -> dict[str, Any]:
        return self.record["order_labels"].get("auto_order") or {}

    @property
    def leaf_area(self) -> dict[str, Any]:
        return self.record.get("leaf_area") or {}


def read_saved_analysis(
    path: str | Path, source_path: str | Path | None = None,
) -> SavedAnalysis:
    """Read a current export folder or a legacy flat export's metadata file."""
    metadata_path = _metadata_path(path)
    record = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("order_labels"), dict):
        raise ValueError("This export has no resumable order labels (path-only exports are not supported).")

    labels = _read_label_array(metadata_path, "vein_order_labels", (0, 1, 2, 3, 4))
    vein = _read_label_array(metadata_path, "vein_mask", (0, 1))
    leaf = _read_label_array(metadata_path, "leaf_mask", (0, 1))
    processing = record["order_labels"].get("processing") or {}
    _validate_masks(labels, vein, leaf, processing, record.get("leaf_area") or {})
    roi = _read_roi(processing, labels.shape)
    calibration, sigma = _read_measurement_settings(record, processing)
    source, image = _read_source_image(record.get("source_image") or {}, source_path, roi)
    return SavedAnalysis(
        metadata_path=metadata_path, record=record, source_path=source,
        image=image, labels=labels, vein=vein, leaf=leaf,
        roi=roi, calibration=calibration, sigma=sigma,
    )


def _metadata_path(path: str | Path) -> Path:
    metadata_path = Path(path)
    if metadata_path.is_dir():
        candidates = sorted(metadata_path.glob("*_metadata.yaml"))
        if len(candidates) != 1:
            raise ValueError("Select one *_metadata.yaml file in the export folder.")
        metadata_path = candidates[0]
    if not metadata_path.name.endswith("_metadata.yaml"):
        raise ValueError("Select an exported *_metadata.yaml file.")
    return metadata_path


def _read_label_array(metadata_path: Path, suffix: str, allowed: tuple[int, ...]) -> np.ndarray:
    # YAML 内の任意のパスではなく、同じ接頭辞の兄弟ファイルだけを読む。
    # 古い「複数回分を一つのフォルダへ出力する形式」でも対応を取り違えない。
    prefix = metadata_path.name.removesuffix("_metadata.yaml")
    file = metadata_path.parent / f"{prefix}_{suffix}.npy"
    if not file.is_file():
        raise ValueError(f"Missing {file.name}; keep the masks and metadata together.")
    values = np.load(file, allow_pickle=False)
    if values.ndim != 2 or not values.size or not np.isin(values, allowed).all():
        raise ValueError(f"Invalid values or shape in {file.name}.")
    return values.astype(np.uint8)


def _validate_masks(labels, vein, leaf, processing, area) -> None:
    """Check the containment order: annotations ⊆ vein mask ⊆ leaf mask."""
    if labels.shape != vein.shape or labels.shape != leaf.shape:
        raise ValueError("Saved mask and label shapes do not match.")
    if not leaf.any() or np.any((labels > 0) & (vein == 0)) or np.any((vein > 0) & (leaf == 0)):
        raise ValueError("Saved labels or veins extend outside their mask, or the leaf is empty.")
    # 葉脈の占有面積から葉面積を推定すると VLA の分母が変わってしまう。
    # 保存された葉身マスクを照合し、そのまま復元する。
    leaf_record = processing.get("leaf_mask") or {}
    expected = leaf_record.get("sha256") or area.get("sha256")
    if expected and array_digest(leaf) != expected:
        raise ValueError("Saved leaf mask does not match the metadata checksum.")


def _read_roi(processing: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    bounds = processing.get("roi") or {}
    keys = ("row_start", "row_stop", "col_start", "col_stop")
    if any(not isinstance(bounds.get(key), int) for key in keys):
        raise ValueError("ROI bounds are missing from the export metadata.")
    r0, r1, c0, c1 = (bounds[key] for key in keys)
    if min(r0, c0) < 0 or (r1-r0, c1-c0) != shape:
        raise ValueError("ROI bounds do not match the saved labels.")
    return r0, r1, c0, c1


def _read_measurement_settings(record: dict, processing: dict) -> tuple[dict, float]:
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
    return calibration, sigma


def _read_source_image(original: dict, source_path, roi) -> tuple[Path, np.ndarray]:
    source = Path(source_path or original.get("path") or "")
    if not source.is_file():
        raise MissingSourceImage("Locate the original scan to resume this export.")
    # 画像を移動した場合も、記録されたハッシュがあれば同じ画像か確認できる。
    if original.get("sha256") and file_sha256(source) != original["sha256"]:
        raise ValueError("The selected scan does not match the original image checksum.")
    image = imread(source)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] not in (3, 4)):
        raise ValueError("The original scan must be a 2D grayscale or RGB/RGBA image.")
    if original.get("shape") and tuple(original["shape"]) != image.shape:
        raise ValueError("Original image shape does not match the metadata.")
    _, row_stop, _, col_stop = roi
    if row_stop > image.shape[0] or col_stop > image.shape[1]:
        raise ValueError("Saved ROI extends outside the original scan.")
    return source, image
