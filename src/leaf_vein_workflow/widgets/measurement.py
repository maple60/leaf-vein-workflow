"""Vein-length and VLA export from order labels and traced vein paths.

Export measures the order labels and/or the traced 1°/2° paths as they are at
that moment and writes the results to the chosen output folder.  Results are not shown in the dock (it is too
narrow); only a short notification confirms the export.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_warning

from ..calibration import CALIBRATION_MM_PER_PIXEL_KEY
from ..order import (
    ExportedOrderPaths,
    _safe_prefix,
    export_order_results,
    make_qc_overlay,
    measure_order_labels,
    validate_order_labels,
)
from ..paths import ExportedPathFiles, export_vein_paths, measure_vein_paths
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

MM2_PER_UNIT = {"mm2": 1.0, "cm2": 100.0}


@dataclass(slots=True)
class MeasurementExport:
    """Files written by one Export click; either part may be absent."""

    labels: ExportedOrderPaths | None = None
    paths: ExportedPathFiles | None = None


class MeasurementWorkflow:
    """Measure order labels and export per-order lengths and VLA."""

    def __init__(
        self,
        layers: WorkflowLayers,
        controls: WorkflowControls,
        vein_paths=None,
    ):
        self.layers = layers
        self.controls = controls
        self.vein_paths = vein_paths
        self.uncertain_pixels = 0
        self.unlabeled_vein_pixels = 0

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
        """Return the typed leaf area converted to mm², or ``None`` when blank."""
        if not self.controls.leaf_area_input.text().strip():
            return None
        value = parse_number_input(self.controls.leaf_area_input, "Leaf area")
        if value <= 0:
            raise ValueError("Leaf area must be greater than zero.")
        unit = str(self.controls.leaf_area_unit_combo.currentData())
        return value * MM2_PER_UNIT[unit]

    def output_dir(self) -> Path:
        """Return the folder shown in the output field."""
        return Path(self.controls.output_dir_input.text())

    def set_output_dir(self, output_dir: str | Path) -> None:
        """Change the export folder shown in the dock."""
        self.controls.output_dir_input.setText(str(output_dir))

    def measure(self) -> list[dict[str, Any]] | None:
        """Return per-order measurement rows for the active order labels."""
        order_layer = self.active_order_labels_layer()
        if order_layer is None:
            self.controls.warn("Create order labels before measuring.")
            return None
        try:
            return self._measure_layer(order_layer, self.leaf_area_mm2())
        except ValueError as exc:
            self.controls.warn(str(exc))
            return None

    def export(self, output_dir: str | Path | None = None) -> MeasurementExport | None:
        """Export traced paths and/or order labels for the active image."""
        target = Path(output_dir) if output_dir is not None else self.output_dir()
        order_layer = self.active_order_labels_layer()
        paths_layer = (
            self.vein_paths.active_paths_layer() if self.vein_paths is not None else None
        )
        has_paths = paths_layer is not None and len(paths_layer.data) > 0
        if order_layer is None and not has_paths:
            self.controls.warn("Trace vein paths or create order labels before exporting.")
            return None

        try:
            leaf_area = self.leaf_area_mm2()
        except ValueError as exc:
            self.controls.warn(str(exc))
            return None

        result = MeasurementExport()
        if has_paths:
            result.paths = self._export_paths(paths_layer, leaf_area, target)
            if result.paths is None:
                return None
        if order_layer is not None:
            result.labels = self._export_labels(order_layer, leaf_area, target)
            if result.labels is None and result.paths is None:
                return None

        if leaf_area is None:
            show_warning("Leaf area is empty: VLA columns were left blank.")
        return result

    def _export_paths(
        self,
        paths_layer,
        leaf_area: float | None,
        target: Path,
    ) -> ExportedPathFiles | None:
        paths, orders = self.vein_paths.collect(paths_layer)
        pixel_size = self.vein_paths.pixel_size_mm(paths_layer)
        image_name = str(paths_layer.metadata.get(SOURCE_IMAGE_KEY) or paths_layer.name)
        try:
            rows = measure_vein_paths(
                paths,
                orders,
                pixel_size_mm=pixel_size,
                leaf_area_mm2=leaf_area,
                image_name=image_name,
            )
        except ValueError as exc:
            self.controls.warn(str(exc))
            return None
        files = export_vein_paths(target, rows, paths, orders, _safe_prefix(image_name))
        show_info(f"Exported {files.measurements_csv.name}")
        if pixel_size is None:
            show_warning("Paths image is not calibrated: lengths are in px.")
        return files

    def _export_labels(
        self,
        order_layer: Labels,
        leaf_area: float | None,
        target: Path,
    ) -> ExportedOrderPaths | None:
        try:
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
            target,
            rows,
            labels,
            branch_table=branch_table,
            qc_overlay=overlay,
            prefix=self._image_name(order_layer),
        )

        show_info(f"Exported {paths.measurements_csv.name}")
        if self.unlabeled_vein_pixels:
            show_warning(
                f"{self.unlabeled_vein_pixels} vein_mask px have no order label "
                "(0) and were excluded. Paint them or remove them from the mask."
            )
        if self.uncertain_pixels:
            show_warning(
                f"{self.uncertain_pixels} px are still labelled uncertain (4) "
                "and were excluded from every order."
            )
        if mm_per_pixel is None:
            show_warning("Image is not calibrated: lengths are in px and VLA is empty.")
        return paths

    def _measure_layer(
        self,
        order_layer: Labels,
        leaf_area_mm2: float | None,
    ) -> list[dict[str, Any]]:
        labels = np.asarray(order_layer.data)
        if labels.ndim != 2:
            raise ValueError("Measurement requires 2D order labels.")

        # mask と照合し、mask にあるのに次数が付いていない葉脈を数える。
        mask_layer = self.layers.order_labels_to_mask.get(id(order_layer))
        vein_mask = labels > 0
        if mask_layer is not None and self.layers.is_present(mask_layer):
            candidate = np.asarray(mask_layer.data)
            if candidate.shape == labels.shape:
                vein_mask = candidate > 0
        validation = validate_order_labels(labels, vein_mask)
        self.unlabeled_vein_pixels = validation.unlabeled_vein_pixels
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
