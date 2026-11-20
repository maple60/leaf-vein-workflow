"""Vein length from hand-traced paths, independent of napari.

This follows the whole-leaf protocol for major veins (Sack & Scoffoni 2013):
1° and 2° veins are traced as segmented lines and their summed length is
divided by leaf area.  Tertiary veins are measured in sample boxes instead and
are not handled here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .calibration import millimetres_per_pixel, validate_positive_finite


PATH_ORDER_NAMES = {1: "primary", 2: "secondary"}

PATH_ROW_DEFINITIONS = (
    ("primary", "1", (1,)),
    ("secondary", "2", (2,)),
    ("primary_secondary", "1+2", (1, 2)),
)

PATH_MEASUREMENT_COLUMNS = [
    "image_name",
    "method",
    "label_name",
    "label_value",
    "n_paths",
    "length_px",
    "pixel_size_mm",
    "length_mm",
    "leaf_area_mm2",
    "leaf_area_source",
    "vla_mm_per_mm2",
]

PATH_VERTEX_COLUMNS = ["path_id", "order", "vertex", "y_px", "x_px"]


@dataclass(slots=True)
class ExportedPathFiles:
    """Paths written by a vein-path export."""

    measurements_csv: Path
    vertices_csv: Path


def path_length_px(vertices: np.ndarray) -> float:
    """Return the summed segment length of one 2D polyline in pixels."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("A vein path must be an (N, 2) array of vertices.")
    points = points[:, -2:]
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def measure_vein_paths(
    paths: Sequence[np.ndarray],
    orders: Sequence[int],
    pixel_size_mm: float | None = None,
    leaf_area_mm2: float | None = None,
    image_name: str = "",
) -> list[dict[str, Any]]:
    """Sum traced path lengths per vein order and compute VLA."""
    if len(paths) != len(orders):
        raise ValueError("Each vein path needs exactly one order.")

    invalid = sorted({int(order) for order in orders} - set(PATH_ORDER_NAMES))
    if invalid:
        raise ValueError(f"Vein paths support orders 1 and 2 only, got {invalid}.")

    if pixel_size_mm is not None:
        pixel_size_mm = millimetres_per_pixel(pixel_size_mm)
    area = (
        validate_positive_finite(leaf_area_mm2, "Leaf area (mm²)")
        if leaf_area_mm2 is not None
        else None
    )

    lengths = np.array([path_length_px(path) for path in paths], dtype=float)
    order_array = np.asarray([int(order) for order in orders], dtype=int)

    rows: list[dict[str, Any]] = []
    for label_name, label_value, values in PATH_ROW_DEFINITIONS:
        selected = np.isin(order_array, values)
        length_px = float(lengths[selected].sum())
        length_mm = length_px * pixel_size_mm if pixel_size_mm is not None else None
        rows.append(
            {
                "image_name": image_name,
                "method": "path",
                "label_name": label_name,
                "label_value": label_value,
                "n_paths": int(np.count_nonzero(selected)),
                "length_px": length_px,
                "pixel_size_mm": pixel_size_mm,
                "length_mm": length_mm,
                "leaf_area_mm2": area,
                "leaf_area_source": "input" if area is not None else None,
                "vla_mm_per_mm2": (
                    length_mm / area
                    if length_mm is not None and area is not None
                    else None
                ),
            }
        )
    return rows


def export_vein_paths(
    output_dir: str | Path,
    measurements: Sequence[dict[str, Any]],
    paths: Sequence[np.ndarray],
    orders: Sequence[int],
    prefix: str,
) -> ExportedPathFiles:
    """Write per-order path measurements and every traced vertex to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    measurements_csv = output_path / f"{prefix}_path_measurements.csv"
    vertices_csv = output_path / f"{prefix}_path_vertices.csv"

    with measurements_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PATH_MEASUREMENT_COLUMNS)
        writer.writeheader()
        for row in measurements:
            writer.writerow({column: row.get(column) for column in PATH_MEASUREMENT_COLUMNS})

    # 頂点も残し、後から同じパスを再現・検証できるようにする。
    with vertices_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(PATH_VERTEX_COLUMNS)
        for path_id, (path, order) in enumerate(zip(paths, orders)):
            points = np.asarray(path, dtype=float)[:, -2:]
            for vertex, (y, x) in enumerate(points):
                writer.writerow([path_id, int(order), vertex, float(y), float(x)])

    return ExportedPathFiles(measurements_csv=measurements_csv, vertices_csv=vertices_csv)
