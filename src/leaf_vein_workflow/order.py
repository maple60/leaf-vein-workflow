"""Order-label validation, measurement, and export helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from skan import Skeleton
from skan.csr import summarize
from skimage.io import imsave
from skimage.morphology import skeletonize


ORDER_DEFINITIONS = (
    ("primary", "1", (1,)),
    ("secondary", "2", (2,)),
    ("minor", "3", (3,)),
    ("uncertain", "4", (4,)),
    ("major", "1+2", (1, 2)),
)

VALID_LABEL_VALUES = {0, 1, 2, 3, 4}

MEASUREMENT_COLUMNS = [
    "image_name",
    "label_name",
    "label_value",
    "area_px",
    "skeleton_pixels",
    "length_px",
    "leaf_area_px",
    "vla_px_per_px2",
    "pixel_size_mm",
    "length_mm",
    "leaf_area_mm2",
    "vla_mm_per_mm2",
]


@dataclass(slots=True)
class LabelValidationResult:
    """QC counts for a vein-order Labels layer."""

    outside_mask_pixels: int
    unlabeled_vein_pixels: int
    label_pixel_counts: dict[int, int]
    invalid_label_values: list[int]


@dataclass(slots=True)
class ExportedOrderPaths:
    """Paths written by an order-label export."""

    measurements_csv: Path
    order_labels_npy: Path
    qc_overlay_png: Path
    branch_table_csv: Path


def create_empty_order_labels(mask_shape: Sequence[int]) -> np.ndarray:
    """Return an empty uint8 order-label image matching a 2D mask shape."""
    if len(mask_shape) != 2:
        raise ValueError("Order labels require a 2D mask shape.")
    return np.zeros(tuple(mask_shape), dtype=np.uint8)


def validate_order_labels(
    order_labels: np.ndarray,
    vein_mask: np.ndarray,
) -> LabelValidationResult:
    """Validate order-label values and their relationship to a vein mask."""
    labels = _as_2d_array(order_labels, "order_labels")
    mask = _as_2d_array(vein_mask, "vein_mask") > 0
    if labels.shape != mask.shape:
        raise ValueError("order_labels and vein_mask must have the same shape.")

    positive_labels = labels > 0
    outside_mask_pixels = int(np.count_nonzero(positive_labels & ~mask))
    unlabeled_vein_pixels = int(np.count_nonzero(mask & (labels == 0)))

    unique_values, counts = np.unique(labels, return_counts=True)
    label_pixel_counts = {int(value): int(count) for value, count in zip(unique_values, counts)}
    for value in range(5):
        label_pixel_counts.setdefault(value, 0)

    invalid_label_values = [
        int(value)
        for value in unique_values
        if int(value) not in VALID_LABEL_VALUES
    ]

    return LabelValidationResult(
        outside_mask_pixels=outside_mask_pixels,
        unlabeled_vein_pixels=unlabeled_vein_pixels,
        label_pixel_counts=dict(sorted(label_pixel_counts.items())),
        invalid_label_values=invalid_label_values,
    )


def measure_order_labels(
    order_labels: np.ndarray,
    leaf_mask: np.ndarray | None = None,
    pixel_size_mm: float | None = None,
    image_name: str = "",
    label_layer_name: str = "vein_order_labels",
) -> list[dict[str, Any]]:
    """Measure skeleton length and VLA for each order-label class."""
    labels = _as_2d_array(order_labels, "order_labels")
    if pixel_size_mm is not None and pixel_size_mm <= 0:
        raise ValueError("pixel_size_mm must be positive when provided.")

    leaf_area_px: int | None = None
    if leaf_mask is not None:
        leaf = _as_2d_array(leaf_mask, "leaf_mask") > 0
        if leaf.shape != labels.shape:
            raise ValueError("leaf_mask and order_labels must have the same shape.")
        leaf_area_px = int(np.count_nonzero(leaf))

    rows: list[dict[str, Any]] = []
    for label_name, label_value, values in ORDER_DEFINITIONS:
        order_mask = np.isin(labels, values)
        skeleton, length_px = _skeleton_and_length(order_mask)
        area_px = int(np.count_nonzero(order_mask))
        skeleton_pixels = int(np.count_nonzero(skeleton))

        length_mm = (
            float(length_px) * float(pixel_size_mm)
            if pixel_size_mm is not None
            else None
        )
        leaf_area_mm2 = (
            float(leaf_area_px) * float(pixel_size_mm) ** 2
            if leaf_area_px is not None and pixel_size_mm is not None
            else None
        )
        vla_px_per_px2 = (
            float(length_px) / float(leaf_area_px)
            if leaf_area_px
            else None
        )
        vla_mm_per_mm2 = (
            float(length_mm) / float(leaf_area_mm2)
            if leaf_area_mm2
            else None
        )

        rows.append(
            {
                "image_name": image_name,
                "label_layer_name": label_layer_name,
                "label_name": label_name,
                "label_value": label_value,
                "area_px": area_px,
                "skeleton_pixels": skeleton_pixels,
                "length_px": float(length_px),
                "leaf_area_px": leaf_area_px,
                "vla_px_per_px2": vla_px_per_px2,
                "pixel_size_mm": pixel_size_mm,
                "length_mm": length_mm,
                "leaf_area_mm2": leaf_area_mm2,
                "vla_mm_per_mm2": vla_mm_per_mm2,
            }
        )

    return rows


def make_qc_overlay(
    base_image: np.ndarray | None,
    vein_mask: np.ndarray,
    skeleton: np.ndarray,
    order_labels: np.ndarray,
) -> np.ndarray:
    """Create an RGB overlay for export-time QC."""
    labels = _as_2d_array(order_labels, "order_labels")
    mask = _as_2d_array(vein_mask, "vein_mask") > 0
    skeleton_mask = _as_2d_array(skeleton, "skeleton") > 0

    if mask.shape != labels.shape or skeleton_mask.shape != labels.shape:
        raise ValueError("QC overlay inputs must have the same 2D shape.")

    rgb = _base_rgb(base_image, labels.shape)
    rgb[mask] = (0.75 * rgb[mask]) + np.array([0.18, 0.18, 0.18])

    label_colors = {
        1: np.array([1.0, 0.1, 0.1]),
        2: np.array([0.1, 0.45, 1.0]),
        3: np.array([0.1, 0.75, 0.25]),
        4: np.array([1.0, 0.85, 0.1]),
    }
    for label_value, color in label_colors.items():
        selected = labels == label_value
        rgb[selected] = (0.45 * rgb[selected]) + (0.55 * color)

    rgb[skeleton_mask] = np.array([1.0, 0.0, 0.0])
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def export_order_results(
    output_dir: str | Path,
    measurements: Sequence[Mapping[str, Any]],
    order_labels: np.ndarray,
    branch_table: pd.DataFrame,
    qc_overlay: np.ndarray,
    prefix: str | None = None,
) -> ExportedOrderPaths:
    """Write measurements, labels, QC overlay, and branch table to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_prefix = _safe_prefix(prefix)
    measurements_csv = output_path / f"{safe_prefix}_measurements.csv"
    order_labels_npy = output_path / f"{safe_prefix}_vein_order_labels.npy"
    qc_overlay_png = output_path / f"{safe_prefix}_qc_overlay.png"
    branch_table_csv = output_path / f"{safe_prefix}_branch_table.csv"

    with measurements_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MEASUREMENT_COLUMNS)
        writer.writeheader()
        for row in measurements:
            writer.writerow({column: row.get(column) for column in MEASUREMENT_COLUMNS})

    np.save(order_labels_npy, np.asarray(order_labels, dtype=np.uint8))
    imsave(qc_overlay_png, np.asarray(qc_overlay, dtype=np.uint8), check_contrast=False)
    branch_table.to_csv(branch_table_csv, index=False)

    return ExportedOrderPaths(
        measurements_csv=measurements_csv,
        order_labels_npy=order_labels_npy,
        qc_overlay_png=qc_overlay_png,
        branch_table_csv=branch_table_csv,
    )


def _skeleton_and_length(mask: np.ndarray) -> tuple[np.ndarray, float]:
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.uint8), 0.0

    skeleton = skeletonize(mask.astype(bool)).astype(np.uint8)
    if not np.any(skeleton):
        return skeleton, 0.0

    skeleton_graph = Skeleton(skeleton.astype(bool), spacing=1)
    branch_table = summarize(skeleton_graph, separator="-")
    if len(branch_table) == 0:
        return skeleton, 0.0

    return skeleton, float(branch_table["branch-distance"].sum())


def _as_2d_array(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if values.size == 0:
        raise ValueError(f"{name} is empty.")
    return values


def _base_rgb(base_image: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if base_image is None:
        return np.zeros((*shape, 3), dtype=float)

    base = np.asarray(base_image)
    if base.shape[:2] != shape:
        return np.zeros((*shape, 3), dtype=float)

    if base.ndim == 2:
        normalized = _normalize_float(base)
        return np.repeat(normalized[..., np.newaxis], 3, axis=2)

    if base.ndim == 3 and base.shape[2] in (3, 4):
        rgb = base[..., :3].astype(float, copy=False)
        if np.issubdtype(base.dtype, np.integer):
            max_value = np.iinfo(base.dtype).max
            return np.clip(rgb / max_value, 0.0, 1.0)
        max_value = float(np.nanmax(rgb)) if np.any(np.isfinite(rgb)) else 0.0
        if max_value > 1.0:
            rgb = rgb / max_value
        return np.clip(rgb, 0.0, 1.0)

    return np.zeros((*shape, 3), dtype=float)


def _normalize_float(values: np.ndarray) -> np.ndarray:
    numeric = values.astype(float, copy=False)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return np.zeros(numeric.shape, dtype=float)

    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if max_value == min_value:
        return np.zeros(numeric.shape, dtype=float)

    return np.clip((numeric - min_value) / (max_value - min_value), 0.0, 1.0)


def _safe_prefix(prefix: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not prefix:
        return timestamp

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("._-")
    if not normalized:
        return timestamp

    return f"{timestamp}_{normalized}"
