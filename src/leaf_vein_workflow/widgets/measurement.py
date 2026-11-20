"""Vein-length and VLA export from order labels and traced vein paths.

Export measures the order labels and/or the traced 1°/2° paths as they are at
that moment and writes the results to the chosen output folder.  Results are
not shown in the dock (it is too narrow); only short notifications confirm the
export.

Export runs in two parts so the heavy work can leave the Qt thread:

1. :meth:`MeasurementWorkflow.prepare_export` reads the layers and controls on
   the main thread and returns an :class:`ExportJob` holding copies of the data.
2. :func:`run_export_job` skeletonizes, measures, and writes files.  It touches
   no napari layer or Qt object, so it can run in a worker thread.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from napari.layers import Image, Labels
from napari.utils.notifications import show_info, show_warning

from ..calibration import CALIBRATION_METADATA_KEYS, CALIBRATION_MM_PER_PIXEL_KEY
from ..order import (
    ExportedOrderPaths,
    _safe_prefix,
    export_order_results,
    make_qc_overlay,
    measure_order_labels,
    validate_order_labels,
)
from ..paths import ExportedPathFiles, export_vein_paths, measure_vein_paths
from ..provenance import array_digest, source_file_info, write_metadata
from ..skeleton import create_skeleton_helpers
from .controls import WorkflowControls, parse_number_input
from .layers import (
    ORDER_UNCERTAIN_LABEL,
    SOURCE_IMAGE_KEY,
    WorkflowLayers,
)

MM2_PER_UNIT = {"mm2": 1.0, "cm2": 100.0}


@dataclass(slots=True)
class MeasurementExport:
    """Files written by one Export click; either part may be absent."""

    labels: ExportedOrderPaths | None = None
    paths: ExportedPathFiles | None = None
    vein_mask_npy: Path | None = None
    metadata_yaml: Path | None = None
    # (level, text) notifications to show on the main thread.
    messages: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class LabelsSnapshot:
    """Copies of everything needed to measure one order-label layer."""

    labels: np.ndarray
    vein_mask: np.ndarray | None
    base_image: np.ndarray | None
    mm_per_pixel: float | None
    image_name: str
    layer_name: str


@dataclass(slots=True)
class PathsSnapshot:
    """Measured path rows plus the vertices to write."""

    rows: list[dict[str, Any]]
    paths: list[np.ndarray]
    orders: list[int]


@dataclass(slots=True)
class ExportJob:
    """Data captured on the main thread for one export."""

    target: Path
    prefix: str
    leaf_area_mm2: float | None
    labels: LabelsSnapshot | None
    paths: PathsSnapshot | None
    metadata: dict[str, Any]
    source_path: str | None
    messages: list[tuple[str, str]] = field(default_factory=list)


def run_export_job(job: ExportJob) -> MeasurementExport:
    """Measure and write every file of one export (safe in a worker thread)."""
    result = MeasurementExport()
    job.target.mkdir(parents=True, exist_ok=True)

    if job.paths is not None:
        result.paths = export_vein_paths(
            job.target, job.paths.rows, job.paths.paths, job.paths.orders, job.prefix
        )
        result.messages.append(("info", f"Exported {result.paths.measurements_csv.name}"))

    if job.labels is not None:
        result.labels = _write_label_results(job)
        result.messages.append(("info", f"Exported {result.labels.measurements_csv.name}"))
        if job.labels.vein_mask is not None:
            result.vein_mask_npy = job.target / f"{job.prefix}_vein_mask.npy"
            np.save(result.vein_mask_npy, job.labels.vein_mask.astype(np.uint8))

    record = dict(job.metadata)
    source = source_file_info(job.source_path) or {"path": None, "sha256": None}
    source.update(record.pop("source_layer", {}))
    record = {"image_name": record.pop("image_name"), "source_image": source, **record}
    record["files"] = _file_names(result)
    result.metadata_yaml = write_metadata(job.target / f"{job.prefix}_metadata.yaml", record)

    result.messages = job.messages + result.messages
    return result


def _write_label_results(job: ExportJob) -> ExportedOrderPaths:
    snapshot = job.labels
    labels = snapshot.labels
    rows = measure_order_labels(
        labels,
        pixel_size_mm=snapshot.mm_per_pixel,
        image_name=snapshot.image_name,
        label_layer_name=snapshot.layer_name,
        leaf_area_mm2=job.leaf_area_mm2,
    )

    vein = labels > 0
    if np.any(vein):
        helpers = create_skeleton_helpers(
            vein,
            display_radius=0,
            mm_per_pixel=snapshot.mm_per_pixel,
        )
        branch_table = helpers.branch_table
        skeleton = helpers.skeleton
    else:
        branch_table = pd.DataFrame()
        skeleton = np.zeros(labels.shape, dtype=np.uint8)

    overlay = make_qc_overlay(snapshot.base_image, vein, skeleton, labels)
    return export_order_results(
        job.target,
        rows,
        labels,
        branch_table=branch_table,
        qc_overlay=overlay,
        prefix=job.prefix,
        add_timestamp=False,
    )


def _file_names(result: MeasurementExport) -> list[str]:
    files: list[str] = []
    if result.paths is not None:
        files += [result.paths.measurements_csv.name, result.paths.vertices_csv.name]
    if result.labels is not None:
        files += [
            result.labels.measurements_csv.name,
            result.labels.order_labels_npy.name,
            result.labels.qc_overlay_png.name,
            result.labels.branch_table_csv.name,
        ]
    if result.vein_mask_npy is not None:
        files.append(result.vein_mask_npy.name)
    return files


def show_export_messages(result: MeasurementExport) -> None:
    """Show the notifications collected during an export."""
    for level, text in result.messages:
        (show_warning if level == "warning" else show_info)(text)


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
        # 設定ファイルの情報をメタデータに残すため、widget 側から差し込む。
        self.settings = None
        self.last_label_record: dict[str, Any] = {}
        self.uncertain_pixels = 0
        self.unlabeled_vein_pixels = 0

    def active_order_labels_layer(self) -> Labels | None:
        """Resolve the selected workflow layer to its order-label layer."""
        return self.layers.active_order_labels_layer()

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
            snapshot = self._labels_snapshot(order_layer)
            return measure_order_labels(
                snapshot.labels,
                pixel_size_mm=snapshot.mm_per_pixel,
                image_name=snapshot.image_name,
                label_layer_name=snapshot.layer_name,
                leaf_area_mm2=self.leaf_area_mm2(),
            )
        except ValueError as exc:
            self.controls.warn(str(exc))
            return None

    def export(self, output_dir: str | Path | None = None) -> MeasurementExport | None:
        """Export traced paths and/or order labels in the calling thread."""
        job = self.prepare_export(output_dir)
        if job is None:
            return None
        result = run_export_job(job)
        show_export_messages(result)
        return result

    def prepare_export(self, output_dir: str | Path | None = None) -> ExportJob | None:
        """Validate inputs and copy layer data for :func:`run_export_job`."""
        target = Path(output_dir) if output_dir is not None else self.output_dir()
        order_layer = self.active_order_labels_layer()
        paths_layer = (
            self.vein_paths.active_paths_layer() if self.vein_paths is not None else None
        )
        has_paths = paths_layer is not None and len(paths_layer.data) > 0
        if order_layer is None and not has_paths:
            self.controls.warn("Create order labels before exporting.")
            return None

        try:
            leaf_area = self.leaf_area_mm2()
        except ValueError as exc:
            self.controls.warn(str(exc))
            return None

        reference_layer = order_layer if order_layer is not None else paths_layer
        image_name = str(
            reference_layer.metadata.get(SOURCE_IMAGE_KEY) or reference_layer.name
        )
        # 1回の Export で書く全ファイルに同じ接頭辞を付け、対応を取りやすくする。
        prefix = _safe_prefix(image_name)
        messages: list[tuple[str, str]] = []

        paths_snapshot = None
        if has_paths:
            paths_snapshot = self._paths_snapshot(paths_layer, leaf_area)
            if paths_snapshot is None:
                return None
            if self.vein_paths.pixel_size_mm(paths_layer) is None:
                messages.append(("warning", "Paths image is not calibrated: lengths are in px."))

        labels_snapshot = None
        if order_layer is not None:
            try:
                labels_snapshot = self._labels_snapshot(order_layer)
            except ValueError as exc:
                self.controls.warn(str(exc))
                if paths_snapshot is None:
                    return None
            if labels_snapshot is not None:
                messages += self._label_warnings(labels_snapshot)

        if leaf_area is None:
            messages.append(("warning", "Leaf area is empty: VLA columns were left blank."))

        source_layer = self.layers.by_name(image_name)
        source_path = getattr(getattr(source_layer, "source", None), "path", None)
        metadata = self._metadata_record(
            image_name,
            reference_layer,
            source_layer,
            labels_snapshot is not None,
            paths_layer if paths_snapshot is not None else None,
            leaf_area,
        )
        return ExportJob(
            target=target,
            prefix=prefix,
            leaf_area_mm2=leaf_area,
            labels=labels_snapshot,
            paths=paths_snapshot,
            metadata=metadata,
            source_path=str(source_path) if source_path else None,
            messages=messages,
        )

    def _label_warnings(self, snapshot: LabelsSnapshot) -> list[tuple[str, str]]:
        messages: list[tuple[str, str]] = []
        if self.unlabeled_vein_pixels:
            messages.append((
                "warning",
                f"{self.unlabeled_vein_pixels} vein_mask px have no order label "
                "(0) and were excluded. Paint them or remove them from the mask.",
            ))
        if self.uncertain_pixels:
            messages.append((
                "warning",
                f"{self.uncertain_pixels} px are still labelled uncertain (4) "
                "and were excluded from every order.",
            ))
        if snapshot.mm_per_pixel is None:
            messages.append((
                "warning",
                "Image is not calibrated: lengths are in px and VLA is empty.",
            ))
        return messages

    def _paths_snapshot(self, paths_layer, leaf_area: float | None) -> PathsSnapshot | None:
        paths, orders = self.vein_paths.collect(paths_layer)
        image_name = str(paths_layer.metadata.get(SOURCE_IMAGE_KEY) or paths_layer.name)
        try:
            rows = measure_vein_paths(
                paths,
                orders,
                pixel_size_mm=self.vein_paths.pixel_size_mm(paths_layer),
                leaf_area_mm2=leaf_area,
                image_name=image_name,
            )
        except ValueError as exc:
            self.controls.warn(str(exc))
            return None
        return PathsSnapshot(
            rows=rows,
            paths=[np.array(path, copy=True) for path in paths],
            orders=list(orders),
        )

    def _labels_snapshot(self, order_layer: Labels) -> LabelsSnapshot:
        """Validate labels against the mask and copy what export needs."""
        labels = np.array(order_layer.data, copy=True)
        if labels.ndim != 2:
            raise ValueError("Measurement requires 2D order labels.")

        # mask と照合し、mask にあるのに次数が付いていない葉脈を数える。
        mask_layer = self.layers.order_labels_to_mask.get(id(order_layer))
        mask_data = None
        vein_mask = labels > 0
        if mask_layer is not None and self.layers.is_present(mask_layer):
            candidate = np.asarray(mask_layer.data)
            if candidate.shape == labels.shape:
                mask_data = np.array(candidate, copy=True)
                vein_mask = mask_data > 0
        validation = validate_order_labels(labels, vein_mask)
        self.unlabeled_vein_pixels = validation.unlabeled_vein_pixels

        mask_record: dict[str, Any] = {}
        if mask_data is not None:
            mask_record = self.layers.processing(mask_layer)
            auto_digest = mask_record.pop("auto_mask_sha256", None)
            mask_record["mask_edited_by_hand_after_last_step"] = (
                auto_digest is not None and array_digest(mask_data) != auto_digest
            )
        self.last_label_record = {
            "layer_name": order_layer.name,
            "processing": mask_record or None,
            "label_pixel_counts": {
                str(key): int(value)
                for key, value in validation.label_pixel_counts.items()
            },
            "uncertain_px": int(validation.label_pixel_counts.get(ORDER_UNCERTAIN_LABEL, 0)),
            "unlabeled_vein_px": int(validation.unlabeled_vein_pixels),
            "combined_rows": "sum of per-order skeleton lengths",
        }
        if validation.invalid_label_values:
            raise ValueError(
                "Order labels contain unsupported values: "
                f"{validation.invalid_label_values}. Use 1-4 only."
            )
        self.uncertain_pixels = validation.label_pixel_counts.get(
            ORDER_UNCERTAIN_LABEL, 0
        )

        mm_per_pixel = order_layer.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        return LabelsSnapshot(
            labels=labels,
            vein_mask=mask_data,
            base_image=self._base_image(order_layer, labels.shape),
            mm_per_pixel=float(mm_per_pixel) if mm_per_pixel else None,
            image_name=self._image_name(order_layer),
            layer_name=order_layer.name,
        )

    def _metadata_record(
        self,
        image_name: str,
        reference_layer,
        source_layer,
        has_labels: bool,
        paths_layer,
        leaf_area: float | None,
    ) -> dict[str, Any]:
        source_layer_info: dict[str, Any] = {}
        if source_layer is not None:
            source_layer_info = {
                "layer_name": source_layer.name,
                "shape": list(np.shape(source_layer.data)),
            }

        calibration = {
            key.removeprefix("leaf_vein_workflow_").removeprefix("calibration_"): (
                reference_layer.metadata[key]
            )
            for key in CALIBRATION_METADATA_KEYS
            if key in reference_layer.metadata
        }

        record: dict[str, Any] = {
            "image_name": image_name,
            "source_layer": source_layer_info,
            "calibration": calibration or None,
            "leaf_area": {
                "input": self.controls.leaf_area_input.text().strip() or None,
                "unit": str(self.controls.leaf_area_unit_combo.currentData()),
                "mm2": leaf_area,
            },
        }

        if has_labels:
            record["order_labels"] = self.last_label_record

        if paths_layer is not None:
            _, orders = self.vein_paths.collect(paths_layer)
            record["vein_paths"] = {
                "layer_name": paths_layer.name,
                "n_paths": {
                    "primary": orders.count(1),
                    "secondary": orders.count(2),
                },
                "length": "sum of straight segments between traced vertices",
            }

        if self.settings is not None:
            loaded = self.settings.loaded
            record["settings"] = {
                "file": str(self.settings.path) if self.settings.path.is_file() else None,
                "defaults_in_effect": loaded.values if loaded is not None else None,
            }
        return record

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
                # napari 側の編集と競合しないようにコピーを渡す。
                return np.array(data, copy=True)
        return None
