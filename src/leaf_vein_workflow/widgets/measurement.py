"""Vein-length and VLA measurement from the manual order-label layer.

Lengths are measured once per click; changing the leaf area afterwards only
recomputes VLA from the stored lengths.  Export always re-measures so the files
match the labels as they are at export time.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from napari.layers import Image, Labels
from napari.utils.notifications import show_info

from ..calibration import CALIBRATION_MM_PER_PIXEL_KEY
from ..order import (
    ExportedOrderPaths,
    apply_leaf_area,
    export_order_results,
    make_qc_overlay,
    measure_order_labels,
    validate_order_labels,
)
from ..skeleton import create_skeleton_helpers
from .controls import WorkflowControls, parse_number_input
from .layers import (
    ORDER_LABELS_LAYER_NAME,
    ORDER_UNCERTAIN_LABEL,
    ROLE_KEY,
    ROLE_ORDER_LABELS,
    SOURCE_IMAGE_KEY,
    WorkflowLayers,
)

# 結果欄に出す行。uncertain は件数だけ別に警告する。
DISPLAY_ORDER = (
    ("primary", "1° primary"),
    ("secondary", "2° secondary"),
    ("tertiary", "3° tertiary"),
    ("primary_secondary", "1°+2°"),
    ("major", "major (1°–3°)"),
)


class MeasurementWorkflow:
    """Measure order labels, show per-order results, and export them."""

    def __init__(self, layers: WorkflowLayers, controls: WorkflowControls):
        self.layers = layers
        self.controls = controls
        self.rows: list[dict[str, Any]] = []
        self.uncertain_pixels = 0

    def active_order_labels_layer(self) -> Labels | None:
        """Resolve the selected workflow layer to its order-label layer."""
        layer = self.layers.viewer.layers.selection.active
        if isinstance(layer, Labels) and layer.metadata.get(ROLE_KEY) == ROLE_ORDER_LABELS:
            return layer

        mask_layer = self.layers.active_mask_layer()
        if mask_layer is not None:
            order_layer = self.layers.mask_to_order_labels.get(id(mask_layer))
            if order_layer is not None and self.layers.is_present(order_layer):
                return order_layer

        named = self.layers.by_name(ORDER_LABELS_LAYER_NAME)
        if isinstance(named, Labels):
            return named
        return None

    def leaf_area_mm2(self) -> float | None:
        """Return the typed leaf area, ``None`` when blank, or raise ValueError."""
        if not self.controls.leaf_area_input.text().strip():
            return None
        value = parse_number_input(self.controls.leaf_area_input, "Leaf area (mm²)")
        if value <= 0:
            raise ValueError("Leaf area (mm²) must be greater than zero.")
        return value

    def measure(self) -> list[dict[str, Any]] | None:
        """Skeletonize each order class and show lengths and VLA."""
        order_layer = self.active_order_labels_layer()
        if order_layer is None:
            self.controls.warn("Create order labels before measuring.")
            return None

        try:
            leaf_area = self.leaf_area_mm2()
            rows = self._measure_layer(order_layer, leaf_area)
        except ValueError as exc:
            self.controls.warn(str(exc))
            return None

        self.rows = rows
        self.controls.export_button.setEnabled(True)
        self._show_rows()
        self.controls.set_status(f"Measured {order_layer.name}.")
        return rows

    def on_leaf_area_changed(self, *args) -> None:
        """Recompute VLA from stored lengths when the leaf area changes."""
        if not self.rows:
            return
        try:
            leaf_area = self.leaf_area_mm2()
        except ValueError as exc:
            self.controls.set_status(str(exc))
            return
        self.rows = apply_leaf_area(self.rows, leaf_area, source="input")
        self._show_rows()

    def export(self, output_dir: str | Path) -> ExportedOrderPaths | None:
        """Re-measure current labels and write CSV, labels, QC image, branches."""
        order_layer = self.active_order_labels_layer()
        if order_layer is None:
            self.controls.warn("Create order labels before exporting.")
            return None

        try:
            leaf_area = self.leaf_area_mm2()
            rows = self._measure_layer(order_layer, leaf_area)
        except ValueError as exc:
            self.controls.warn(str(exc))
            return None

        labels = np.asarray(order_layer.data)
        vein = labels > 0
        mm_per_pixel = order_layer.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        if np.any(vein):
            helpers = create_skeleton_helpers(
                vein,
                display_radius=0,
                mm_per_pixel=float(mm_per_pixel) if mm_per_pixel else None,
            )
            branch_table = helpers.branch_table
            skeleton = helpers.skeleton
        else:
            branch_table = pd.DataFrame()
            skeleton = np.zeros(labels.shape, dtype=np.uint8)

        overlay = make_qc_overlay(
            self._base_image(order_layer, labels.shape),
            vein,
            skeleton,
            labels,
        )
        paths = export_order_results(
            output_dir,
            rows,
            labels,
            branch_table=branch_table,
            qc_overlay=overlay,
            prefix=self._image_name(order_layer),
        )

        self.rows = rows
        self._show_rows()
        self.controls.set_status(f"Exported measurements to {paths.measurements_csv}")
        show_info(f"Exported {paths.measurements_csv.name}")
        return paths

    def default_export_dir(self) -> str:
        """Suggest the folder of the source image file when napari knows it."""
        order_layer = self.active_order_labels_layer()
        name = self._image_name(order_layer) if order_layer is not None else ""
        source = self.layers.by_name(name) if name else None
        path = getattr(getattr(source, "source", None), "path", None)
        if path:
            return str(Path(path).parent)
        return str(Path.cwd())

    def _measure_layer(
        self,
        order_layer: Labels,
        leaf_area_mm2: float | None,
    ) -> list[dict[str, Any]]:
        labels = np.asarray(order_layer.data)
        if labels.ndim != 2:
            raise ValueError("Measurement requires 2D order labels.")

        validation = validate_order_labels(labels, labels > 0)
        if validation.invalid_label_values:
            raise ValueError(
                "Order labels contain unsupported values: "
                f"{validation.invalid_label_values}. Use 1-4 only."
            )
        self.uncertain_pixels = validation.label_pixel_counts.get(
            ORDER_UNCERTAIN_LABEL, 0
        )

        mm_per_pixel = order_layer.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        return measure_order_labels(
            labels,
            pixel_size_mm=float(mm_per_pixel) if mm_per_pixel else None,
            image_name=self._image_name(order_layer),
            label_layer_name=order_layer.name,
            leaf_area_mm2=leaf_area_mm2,
        )

    def _show_rows(self) -> None:
        by_name = {row["label_name"]: row for row in self.rows}
        lines: list[str] = []
        calibrated = any(row.get("length_mm") is not None for row in self.rows)
        if not calibrated:
            lines.append("Not calibrated: lengths in px, VLA unavailable.")

        for key, title in DISPLAY_ORDER:
            row = by_name.get(key)
            if row is None:
                continue
            if row.get("length_mm") is not None:
                length = f"{row['length_mm']:.4g} mm"
            else:
                length = f"{row['length_px']:.4g} px"
            vla = row.get("vla_mm_per_mm2")
            vla_text = f"{vla:.4g} mm/mm²" if vla is not None else "—"
            lines.append(f"{title}: L = {length}, VLA = {vla_text}")

        if calibrated and not any(
            row.get("leaf_area_mm2") for row in self.rows
        ):
            lines.append("Enter leaf area (mm²) to compute VLA.")
        if self.uncertain_pixels:
            lines.append(
                f"⚠ {self.uncertain_pixels} px still labelled uncertain (4); "
                "they are excluded from every order."
            )
        self.controls.set_measurement_result("\n".join(lines))

    def _image_name(self, order_layer: Labels) -> str:
        return str(order_layer.metadata.get(SOURCE_IMAGE_KEY) or order_layer.name)

    def _base_image(self, order_layer: Labels, shape: tuple[int, ...]):
        mask_layer = self.layers.order_labels_to_mask.get(id(order_layer))
        crop_layer = (
            self.layers.mask_to_crop.get(id(mask_layer))
            if mask_layer is not None
            else None
        )
        if isinstance(crop_layer, Image) and self.layers.is_present(crop_layer):
            data = np.asarray(crop_layer.data)
            if data.shape[:2] == tuple(shape):
                return data
        return None
