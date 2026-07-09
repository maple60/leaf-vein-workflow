"""napari dock widget for manual vein-order labeling and export."""

from pathlib import Path

import numpy as np
from napari.layers import Image, Labels, Points, Shapes
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._widget import (
    ROLE_KEY,
    SOURCE_CROP_KEY,
    SOURCE_IMAGE_KEY,
    SOURCE_MASK_KEY,
)
from .order import (
    create_empty_order_labels,
    export_order_results,
    make_qc_overlay,
    measure_order_labels,
    validate_order_labels,
)
from .skeleton import SkeletonHelperResult
from .skeleton import create_skeleton_helpers as build_skeleton_helpers


LAYER_VEIN_MASK = "vein_mask"
LAYER_SKELETON = "skeleton"
LAYER_DIAMETER = "diameter_on_skeleton"
LAYER_ENDPOINTS = "endpoints"
LAYER_BRANCHPOINTS = "branchpoints"
LAYER_SPURS = "spur_candidates"
LAYER_ORDER_LABELS = "vein_order_labels"

ROLE_ORDER_HELPER = "order_label_helper"
ROLE_ORDER_LABELS = "vein_order_labels"


class OrderLabelWidget(QWidget):
    """Create skeleton helpers and export manual vein-order measurements."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._helpers_by_mask_id: dict[int, SkeletonHelperResult] = {}

        self.status_label = QLabel("Select a vein mask layer to begin.")
        self.status_label.setWordWrap(True)

        self.vein_mask_combo = QComboBox()
        self.leaf_mask_combo = QComboBox()
        self.order_labels_combo = QComboBox()

        self.spur_length_spinbox = QDoubleSpinBox()
        self.spur_length_spinbox.setRange(0.0, 1_000_000.0)
        self.spur_length_spinbox.setDecimals(2)
        self.spur_length_spinbox.setSingleStep(1.0)
        self.spur_length_spinbox.setValue(20.0)

        self.display_radius_spinbox = QSpinBox()
        self.display_radius_spinbox.setRange(0, 50)
        self.display_radius_spinbox.setValue(3)

        self.pixel_size_spinbox = QDoubleSpinBox()
        self.pixel_size_spinbox.setRange(0.0, 1_000_000.0)
        self.pixel_size_spinbox.setDecimals(9)
        self.pixel_size_spinbox.setSingleStep(0.001)
        self.pixel_size_spinbox.setSpecialValueText("None")
        self.pixel_size_spinbox.setValue(0.0)

        self.output_dir_line_edit = QLineEdit(str(Path.cwd()))
        self.browse_output_button = QPushButton("Browse")

        self.create_helpers_button = QPushButton("Create skeleton helpers")
        self.create_order_labels_button = QPushButton("Create order label layer")
        self.validate_labels_button = QPushButton("Validate labels")
        self.export_button = QPushButton("Export measurements")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("vein_mask layer"))
        layout.addWidget(self.vein_mask_combo)
        layout.addWidget(QLabel("leaf_mask layer (optional)"))
        layout.addWidget(self.leaf_mask_combo)
        layout.addWidget(QLabel("vein_order_labels layer"))
        layout.addWidget(self.order_labels_combo)
        layout.addWidget(QLabel("spur_length_px"))
        layout.addWidget(self.spur_length_spinbox)
        layout.addWidget(QLabel("skeleton_display_radius"))
        layout.addWidget(self.display_radius_spinbox)
        layout.addWidget(QLabel("pixel_size_mm (optional)"))
        layout.addWidget(self.pixel_size_spinbox)
        layout.addWidget(QLabel("output directory"))

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_line_edit)
        output_layout.addWidget(self.browse_output_button)
        layout.addLayout(output_layout)

        layout.addWidget(self.create_helpers_button)
        layout.addWidget(self.create_order_labels_button)
        layout.addWidget(self.validate_labels_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.browse_output_button.clicked.connect(self.browse_output_directory)
        self.create_helpers_button.clicked.connect(self.create_skeleton_helpers)
        self.create_order_labels_button.clicked.connect(self.create_order_label_layer)
        self.validate_labels_button.clicked.connect(self.validate_labels)
        self.export_button.clicked.connect(self.export_measurements)
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices)

        self._refresh_layer_choices()

    def browse_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self.output_dir_line_edit.text(),
        )
        if directory:
            self.output_dir_line_edit.setText(directory)

    def create_skeleton_helpers(self) -> None:
        mask_layer = self._selected_vein_mask_layer()
        if mask_layer is None:
            self._warn("Select a vein_mask layer before creating helpers.")
            return

        try:
            helpers = build_skeleton_helpers(
                np.asarray(mask_layer.data),
                spur_length_px=self.spur_length_spinbox.value(),
                display_radius=self.display_radius_spinbox.value(),
            )
        except ValueError as exc:
            self._warn(str(exc))
            return

        self._helpers_by_mask_id[id(mask_layer)] = helpers
        self._upsert_image_layer(
            LAYER_SKELETON,
            helpers.skeleton_display,
            mask_layer,
            opacity=0.75,
            colormap="gray",
        )
        self._upsert_image_layer(
            LAYER_DIAMETER,
            helpers.diameter_display,
            mask_layer,
            opacity=0.85,
            colormap="magma",
        )
        self._upsert_points_layer(
            LAYER_ENDPOINTS,
            helpers.endpoint_coordinates,
            mask_layer,
            face_color="cyan",
        )
        self._upsert_points_layer(
            LAYER_BRANCHPOINTS,
            helpers.branchpoint_coordinates,
            mask_layer,
            face_color="magenta",
        )
        self._upsert_spur_shapes_layer(helpers.spur_paths, mask_layer)

        self._set_status(
            f"Created skeleton helpers from {mask_layer.name}: "
            f"{len(helpers.branch_table)} branches."
        )
        show_info(f"Created skeleton helpers from {mask_layer.name}.")
        self._refresh_layer_choices()

    def create_order_label_layer(self) -> None:
        mask_layer = self._selected_vein_mask_layer()
        if mask_layer is None:
            self._warn("Select a vein_mask layer before creating order labels.")
            return

        existing = self._layer_by_name(LAYER_ORDER_LABELS)
        if existing is not None:
            if not isinstance(existing, Labels):
                self._warn(f"{LAYER_ORDER_LABELS} exists but is not a Labels layer.")
                return
            if existing.data.shape != mask_layer.data.shape:
                self._warn(
                    f"{LAYER_ORDER_LABELS} exists but does not match the mask shape."
                )
                return
            order_layer = existing
            order_layer.translate = tuple(mask_layer.translate)
            order_layer.metadata.update(self._order_labels_metadata(mask_layer))
            show_info(f"Reusing existing {LAYER_ORDER_LABELS} layer.")
        else:
            try:
                order_labels = create_empty_order_labels(np.asarray(mask_layer.data).shape)
            except ValueError as exc:
                self._warn(str(exc))
                return

            order_layer = self.viewer.add_labels(
                order_labels,
                name=LAYER_ORDER_LABELS,
                translate=tuple(mask_layer.translate),
                opacity=0.7,
                metadata=self._order_labels_metadata(mask_layer),
            )
            show_info(f"Created {LAYER_ORDER_LABELS}.")

        self._activate_layer(order_layer)
        try:
            order_layer.mode = "paint"
        except ValueError:
            pass
        self._set_status(f"Ready to paint {LAYER_ORDER_LABELS}.")
        self._refresh_layer_choices(preferred_order_layer=order_layer.name)

    def validate_labels(self) -> None:
        mask_layer = self._selected_vein_mask_layer()
        order_layer = self._selected_order_labels_layer()
        if mask_layer is None or order_layer is None:
            self._warn("Select both vein_mask and vein_order_labels layers.")
            return

        try:
            result = validate_order_labels(order_layer.data, mask_layer.data)
        except ValueError as exc:
            self._warn(str(exc))
            return

        counts = ", ".join(
            f"{value}: {count}" for value, count in result.label_pixel_counts.items()
        )
        message = (
            f"Outside mask: {result.outside_mask_pixels}; "
            f"Unlabeled vein pixels: {result.unlabeled_vein_pixels}; "
            f"Label pixels: {counts}"
        )
        if result.invalid_label_values:
            message += f"; Invalid labels: {result.invalid_label_values}"

        self._set_status(message)
        if result.outside_mask_pixels or result.invalid_label_values:
            show_warning(message)
        else:
            show_info(message)

    def export_measurements(self) -> None:
        mask_layer = self._selected_vein_mask_layer()
        order_layer = self._selected_order_labels_layer()
        if mask_layer is None or order_layer is None:
            self._warn("Select both vein_mask and vein_order_labels layers.")
            return

        leaf_layer = self._selected_leaf_mask_layer()
        if leaf_layer is None:
            show_warning(
                "leaf_mask is not selected. Leaf area and VLA will be left blank."
            )

        helpers = self._helpers_by_mask_id.get(id(mask_layer))
        if helpers is None:
            try:
                helpers = build_skeleton_helpers(
                    np.asarray(mask_layer.data),
                    spur_length_px=self.spur_length_spinbox.value(),
                    display_radius=self.display_radius_spinbox.value(),
                )
            except ValueError as exc:
                self._warn(str(exc))
                return
            self._helpers_by_mask_id[id(mask_layer)] = helpers

        try:
            measurements = measure_order_labels(
                order_layer.data,
                leaf_mask=None if leaf_layer is None else leaf_layer.data,
                pixel_size_mm=self._pixel_size_mm(),
                image_name=self._image_name(mask_layer),
                label_layer_name=order_layer.name,
            )
            qc_overlay = make_qc_overlay(
                self._source_base_image(mask_layer),
                mask_layer.data,
                helpers.skeleton,
                order_layer.data,
            )
            paths = export_order_results(
                self.output_dir_line_edit.text(),
                measurements,
                order_layer.data,
                helpers.branch_table,
                qc_overlay,
                prefix=f"{self._image_name(mask_layer)}_{order_layer.name}",
            )
        except (OSError, ValueError) as exc:
            self._warn(str(exc))
            return

        self._set_status(f"Exported measurements to {paths.measurements_csv}.")
        show_info(f"Exported measurements to {paths.measurements_csv}.")

    def _refresh_layer_choices(self, *args, preferred_order_layer: str | None = None) -> None:
        mask_current = self.vein_mask_combo.currentData()
        leaf_current = self.leaf_mask_combo.currentData()
        order_current = preferred_order_layer or self.order_labels_combo.currentData()

        mask_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, (Image, Labels))
        ]
        label_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Labels)
        ]

        self._set_combo_items(
            self.vein_mask_combo,
            mask_names,
            include_none=True,
            preferred=mask_current or self._preferred_name(mask_names, LAYER_VEIN_MASK),
        )
        self._set_combo_items(
            self.leaf_mask_combo,
            mask_names,
            include_none=True,
            preferred=leaf_current,
        )
        self._set_combo_items(
            self.order_labels_combo,
            label_names,
            include_none=True,
            preferred=order_current
            or self._preferred_name(label_names, LAYER_ORDER_LABELS),
        )

    def _set_combo_items(
        self,
        combo: QComboBox,
        names: list[str],
        include_none: bool,
        preferred: str | None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        if include_none:
            combo.addItem("(none)", None)
        for name in names:
            combo.addItem(name, name)

        if preferred in names:
            combo.setCurrentIndex(combo.findData(preferred))
        combo.blockSignals(False)

    @staticmethod
    def _preferred_name(names: list[str], preferred: str) -> str | None:
        return preferred if preferred in names else None

    def _selected_vein_mask_layer(self) -> Image | Labels | None:
        layer = self._selected_layer(self.vein_mask_combo)
        if isinstance(layer, (Image, Labels)):
            return layer
        return None

    def _selected_leaf_mask_layer(self) -> Image | Labels | None:
        layer = self._selected_layer(self.leaf_mask_combo)
        if isinstance(layer, (Image, Labels)):
            return layer
        return None

    def _selected_order_labels_layer(self) -> Labels | None:
        layer = self._selected_layer(self.order_labels_combo)
        if isinstance(layer, Labels):
            return layer
        return None

    def _selected_layer(self, combo: QComboBox):
        name = combo.currentData()
        if name is None:
            return None
        return self._layer_by_name(str(name))

    def _layer_by_name(self, name: str):
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def _upsert_image_layer(
        self,
        name: str,
        data: np.ndarray,
        source_layer: Image | Labels,
        opacity: float,
        colormap: str,
    ) -> Image | None:
        existing = self._layer_by_name(name)
        metadata = self._helper_metadata(source_layer, name)
        contrast_limits = self._contrast_limits(data)

        if existing is not None:
            if not isinstance(existing, Image):
                self._warn(f"{name} exists but is not an Image layer.")
                return None
            if existing.metadata.get(ROLE_KEY) != ROLE_ORDER_HELPER:
                self._warn(f"{name} exists and was not created by this widget.")
                return None
            existing.data = data
            existing.translate = tuple(source_layer.translate)
            existing.opacity = opacity
            existing.metadata.update(metadata)
            if contrast_limits is not None:
                existing.contrast_limits = contrast_limits
            return existing

        kwargs = {
            "name": name,
            "translate": tuple(source_layer.translate),
            "opacity": opacity,
            "metadata": metadata,
            "blending": "additive",
            "colormap": colormap,
        }
        if contrast_limits is not None:
            kwargs["contrast_limits"] = contrast_limits
        return self.viewer.add_image(data, **kwargs)

    def _upsert_points_layer(
        self,
        name: str,
        coordinates: np.ndarray,
        source_layer: Image | Labels,
        face_color: str,
    ) -> Points | None:
        existing = self._layer_by_name(name)
        metadata = self._helper_metadata(source_layer, name)
        if existing is not None:
            if not isinstance(existing, Points):
                self._warn(f"{name} exists but is not a Points layer.")
                return None
            if existing.metadata.get(ROLE_KEY) != ROLE_ORDER_HELPER:
                self._warn(f"{name} exists and was not created by this widget.")
                return None
            existing.data = coordinates
            existing.translate = tuple(source_layer.translate)
            existing.metadata.update(metadata)
            return existing

        return self.viewer.add_points(
            coordinates,
            name=name,
            ndim=2,
            size=6,
            face_color=face_color,
            translate=tuple(source_layer.translate),
            metadata=metadata,
        )

    def _upsert_spur_shapes_layer(
        self,
        spur_paths: list[np.ndarray],
        source_layer: Image | Labels,
    ) -> Shapes | None:
        existing = self._layer_by_name(LAYER_SPURS)
        metadata = self._helper_metadata(source_layer, LAYER_SPURS)
        if existing is not None:
            if not isinstance(existing, Shapes):
                self._warn(f"{LAYER_SPURS} exists but is not a Shapes layer.")
                return None
            if existing.metadata.get(ROLE_KEY) != ROLE_ORDER_HELPER:
                self._warn(
                    f"{LAYER_SPURS} exists and was not created by this widget."
                )
                return None
            self.viewer.layers.remove(existing)

        if spur_paths:
            return self.viewer.add_shapes(
                spur_paths,
                shape_type=["path"] * len(spur_paths),
                name=LAYER_SPURS,
                edge_color="yellow",
                face_color=[1, 1, 0, 0],
                edge_width=2,
                translate=tuple(source_layer.translate),
                metadata=metadata,
            )

        return self.viewer.add_shapes(
            name=LAYER_SPURS,
            ndim=2,
            shape_type="path",
            edge_color="yellow",
            face_color=[1, 1, 0, 0],
            edge_width=2,
            translate=tuple(source_layer.translate),
            metadata=metadata,
        )

    def _helper_metadata(
        self,
        source_layer: Image | Labels,
        helper_name: str,
    ) -> dict[str, str | None]:
        return {
            ROLE_KEY: ROLE_ORDER_HELPER,
            "helper_name": helper_name,
            SOURCE_IMAGE_KEY: source_layer.metadata.get(SOURCE_IMAGE_KEY),
            SOURCE_CROP_KEY: source_layer.metadata.get(SOURCE_CROP_KEY),
            SOURCE_MASK_KEY: source_layer.name,
        }

    def _order_labels_metadata(self, mask_layer: Image | Labels) -> dict[str, str | None]:
        return {
            ROLE_KEY: ROLE_ORDER_LABELS,
            SOURCE_IMAGE_KEY: mask_layer.metadata.get(SOURCE_IMAGE_KEY),
            SOURCE_CROP_KEY: mask_layer.metadata.get(SOURCE_CROP_KEY),
            SOURCE_MASK_KEY: mask_layer.name,
            "label_0": "background/unlabeled",
            "label_1": "primary vein/midrib",
            "label_2": "secondary vein",
            "label_3": "minor/higher-order veins",
            "label_4": "uncertain",
        }

    @staticmethod
    def _contrast_limits(data: np.ndarray) -> tuple[float, float] | None:
        values = np.asarray(data)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None

        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        if minimum == maximum:
            maximum = minimum + 1.0
        return minimum, maximum

    def _source_base_image(self, mask_layer: Image | Labels) -> np.ndarray | None:
        mask_shape = np.asarray(mask_layer.data).shape[:2]
        for key in (SOURCE_CROP_KEY, SOURCE_IMAGE_KEY):
            source_name = mask_layer.metadata.get(key)
            if not source_name:
                continue
            source_layer = self._layer_by_name(str(source_name))
            if source_layer is None:
                continue
            source_data = np.asarray(source_layer.data)
            if source_data.shape[:2] == mask_shape:
                return source_data
        return None

    def _image_name(self, mask_layer: Image | Labels) -> str:
        return str(mask_layer.metadata.get(SOURCE_IMAGE_KEY) or mask_layer.name)

    def _pixel_size_mm(self) -> float | None:
        value = self.pixel_size_spinbox.value()
        if value <= 0:
            return None
        return float(value)

    def _activate_layer(self, layer) -> None:
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(layer)
        self.viewer.layers.selection.active = layer

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _warn(self, message: str) -> None:
        self._set_status(message)
        show_warning(message)
