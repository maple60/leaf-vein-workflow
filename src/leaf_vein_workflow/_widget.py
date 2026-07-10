"""napari dock widget for ROI crop and editable vein-mask creation."""

import numpy as np
from napari.layers import Image, Labels, Shapes
from napari.utils.notifications import show_info, show_warning
from napari.utils.colormaps import direct_colormap
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .crop import crop_to_rectangle, rectangle_to_slices
from .threshold import (
    otsu_threshold,
    postprocess_mask,
    smooth_image,
    threshold_image,
    to_grayscale,
)


ROLE_KEY = "leaf_vein_workflow_role"
ROLE_ROI_SHAPES = "roi_shapes"
ROLE_ROI_CROP = "roi_crop"
ROLE_SMOOTHED_IMAGE = "smoothed_image"
ROLE_VEIN_MASK = "vein_mask"
ROLE_ORDER_LABELS = "order_labels"
SOURCE_IMAGE_KEY = "source_image"
SOURCE_CROP_KEY = "source_crop"
SOURCE_MASK_KEY = "source_mask"

SMOOTHED_LAYER_NAME = "smoothed_image"
VEIN_MASK_LAYER_NAME = "vein_mask"
ORDER_LABELS_LAYER_NAME = "order_labels"
ORDER_UNCERTAIN_LABEL = 4
ORDER_LABEL_COLORS = {
    0: "transparent",
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#984ea3",
    None: "#ff7f00",
}


class CropWidget(QWidget):
    """Single-widget workflow from ROI crop to editable binary vein mask."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self._roi_to_image: dict[int, Image] = {}
        self._roi_to_crop: dict[int, Image] = {}
        self._crop_to_smoothed: dict[int, Image] = {}
        self._smoothed_to_crop: dict[int, Image] = {}
        self._crop_to_mask: dict[int, Labels] = {}
        self._mask_to_crop: dict[int, Image] = {}
        self._mask_to_order_labels: dict[int, Labels] = {}
        self._order_labels_to_mask: dict[int, Labels] = {}
        self._order_label_data_setitems: dict[int, object] = {}
        self._order_label_clip_connected: set[int] = set()
        self._mask_clip_connected: set[int] = set()
        self._clipping_order_labels: set[int] = set()
        self._mask_updates_enabled: set[int] = set()

        self._threshold_min = 0.0
        self._threshold_max = 1.0
        self._syncing_threshold_controls = False

        self.status_label = QLabel(
            "Open an image with napari. A rectangle ROI layer will be created "
            "automatically."
        )
        self.status_label.setWordWrap(True)

        self.add_roi_button = QPushButton("Add ROI")
        self.apply_gaussian_button = QPushButton("Apply Gaussian")
        self.otsu_button = QPushButton("Apply Otsu")
        self.invert_threshold_checkbox = QCheckBox("Invert threshold")
        self.clean_mask_button = QPushButton("Clean Mask")
        self.create_order_labels_button = QPushButton("Create Order Labels")

        self.gaussian_sigma_label = QLabel("Gaussian sigma")
        self.gaussian_sigma_spinbox = QDoubleSpinBox()
        self.gaussian_sigma_spinbox.setRange(0.0, 50.0)
        self.gaussian_sigma_spinbox.setDecimals(3)
        self.gaussian_sigma_spinbox.setSingleStep(0.1)
        self.gaussian_sigma_spinbox.setValue(1.0)

        self.threshold_label = QLabel("Mask threshold")
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setDecimals(6)
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.01)
        self.threshold_spinbox.setValue(0.5)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 1000)
        self.threshold_slider.setValue(500)

        self.min_object_size_label = QLabel("Small object size")
        self.min_object_size_spinbox = QSpinBox()
        self.min_object_size_spinbox.setRange(0, 1_000_000)
        self.min_object_size_spinbox.setSingleStep(10)
        self.min_object_size_spinbox.setValue(100)

        self.max_hole_size_label = QLabel("Hole size")
        self.max_hole_size_spinbox = QSpinBox()
        self.max_hole_size_spinbox.setRange(0, 1_000_000)
        self.max_hole_size_spinbox.setSingleStep(10)
        self.max_hole_size_spinbox.setValue(500)

        self.closing_radius_label = QLabel("Closing radius")
        self.closing_radius_spinbox = QSpinBox()
        self.closing_radius_spinbox.setRange(0, 50)
        self.closing_radius_spinbox.setValue(1)

        self.opening_radius_label = QLabel("Opening radius")
        self.opening_radius_spinbox = QSpinBox()
        self.opening_radius_spinbox.setRange(0, 50)
        self.opening_radius_spinbox.setValue(0)

        self._mask_update_timer = QTimer(self)
        self._mask_update_timer.setSingleShot(True)
        self._mask_update_timer.setInterval(150)

        self.add_roi_button.setEnabled(False)
        self._set_gaussian_controls_enabled(False)
        self._set_threshold_controls_enabled(False)
        self._set_cleanup_controls_enabled(False)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)

        layout = QVBoxLayout()
        layout.addWidget(self.add_roi_button)
        layout.addWidget(self.gaussian_sigma_label)
        layout.addWidget(self.gaussian_sigma_spinbox)
        layout.addWidget(self.apply_gaussian_button)
        layout.addWidget(self.otsu_button)
        layout.addWidget(self.invert_threshold_checkbox)
        layout.addWidget(self.threshold_label)
        layout.addLayout(threshold_layout)
        layout.addWidget(self.min_object_size_label)
        layout.addWidget(self.min_object_size_spinbox)
        layout.addWidget(self.max_hole_size_label)
        layout.addWidget(self.max_hole_size_spinbox)
        layout.addWidget(self.closing_radius_label)
        layout.addWidget(self.closing_radius_spinbox)
        layout.addWidget(self.opening_radius_label)
        layout.addWidget(self.opening_radius_spinbox)
        layout.addWidget(self.clean_mask_button)
        layout.addWidget(self.create_order_labels_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.add_roi_button.clicked.connect(self.add_roi)
        self.apply_gaussian_button.clicked.connect(self.apply_gaussian_filter)
        self.otsu_button.clicked.connect(self.use_otsu_threshold)
        self.clean_mask_button.clicked.connect(self.clean_mask)
        self.create_order_labels_button.clicked.connect(self.create_order_labels)
        self.invert_threshold_checkbox.toggled.connect(self._schedule_mask_update)
        self.threshold_spinbox.valueChanged.connect(self._on_threshold_spinbox_changed)
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider_changed)
        self._mask_update_timer.timeout.connect(self._update_mask_from_controls)
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)

        for layer in list(self.viewer.layers):
            self._prepare_image_layer(layer)

    def _on_layer_inserted(self, event) -> None:
        self._prepare_image_layer(event.value)

    def _prepare_image_layer(self, layer) -> None:
        if not isinstance(layer, Image):
            return

        if layer.metadata.get(ROLE_KEY) is not None:
            return

        if any(source is layer for source in self._roi_to_image.values()):
            return

        roi_layer = self.viewer.add_shapes(
            name=self._unique_layer_name(f"{layer.name} ROI rectangle"),
            ndim=2,
            edge_color="yellow",
            face_color=[1, 1, 0, 0.15],
            edge_width=2,
        )
        roi_layer.metadata[ROLE_KEY] = ROLE_ROI_SHAPES
        roi_layer.metadata[SOURCE_IMAGE_KEY] = layer.name
        self._roi_to_image[id(roi_layer)] = layer
        roi_layer.events.data.connect(
            lambda event, roi_layer=roi_layer: self._on_roi_data_changed(
                roi_layer
            )
        )

        self._activate_rectangle_mode(roi_layer)
        QTimer.singleShot(
            0,
            lambda roi_layer=roi_layer: self._activate_rectangle_mode(
                roi_layer
            )
            if self._layer_is_present(roi_layer)
            else None,
        )

        self.add_roi_button.setEnabled(True)
        self._set_status(
            f"Created ROI rectangle layer for {layer.name}. Draw one rectangle."
        )
        show_info(f"Draw one rectangle ROI on {layer.name}.")

    def _on_roi_data_changed(self, roi_layer: Shapes) -> None:
        image_layer = self._roi_to_image.get(id(roi_layer))
        if image_layer is None:
            return

        if len(roi_layer.data) != 1:
            self._set_status("Draw exactly one rectangle, then click Add ROI.")
            return

        if self._single_shape_type(roi_layer) != "rectangle":
            self._warn("The ROI must be one rectangle.")
            return

        self._set_status(f"Rectangle ROI ready for {image_layer.name}. Click Add ROI.")

    def add_roi(self) -> None:
        """Create or update an ROI crop for explicit mask refinement."""
        roi_layer = self._active_roi_layer()
        if roi_layer is None:
            self._warn("Select an ROI rectangle layer before clicking Add ROI.")
            return

        image_layer = self._roi_to_image.get(id(roi_layer))
        if image_layer is None:
            self._warn("The selected ROI rectangle is not linked to an image.")
            return

        if len(roi_layer.data) != 1:
            self._warn("Draw exactly one rectangle before clicking Add ROI.")
            return

        if self._single_shape_type(roi_layer) != "rectangle":
            self._warn("The ROI must be one rectangle.")
            return

        try:
            y_slice, x_slice = rectangle_to_slices(
                np.asarray(roi_layer.data[0]),
                np.asarray(image_layer.data).shape,
            )
            cropped = crop_to_rectangle(
                np.asarray(image_layer.data),
                np.asarray(roi_layer.data[0]),
            )
        except ValueError as exc:
            self._warn(str(exc))
            return

        existing_crop = self._roi_to_crop.get(id(roi_layer))
        if existing_crop is not None and self._layer_is_present(existing_crop):
            existing_crop.data = cropped
            existing_crop.translate = (y_slice.start, x_slice.start)
            crop_layer = existing_crop
        else:
            crop_layer = self.viewer.add_image(
                cropped,
                name=self._unique_layer_name(f"{image_layer.name} ROI"),
                rgb=cropped.ndim == 3 and cropped.shape[-1] in (3, 4),
                translate=(y_slice.start, x_slice.start),
                metadata={
                    ROLE_KEY: ROLE_ROI_CROP,
                    SOURCE_IMAGE_KEY: image_layer.name,
                },
            )
            self._roi_to_crop[id(roi_layer)] = crop_layer

        image_layer.visible = False
        self._move_layer_to_top(crop_layer)
        self._activate_layer(crop_layer)
        self._clear_mask_products(crop_layer)
        self._set_gaussian_controls_enabled(True)
        self._set_threshold_controls_enabled(False)
        self._set_status(
            f"Created ROI crop from {image_layer.name}. "
            "Click Apply Gaussian to create the smoothed_image layer."
        )
        show_info(f"Created ROI crop from {image_layer.name}.")

    def apply_gaussian_filter(self) -> None:
        """Smooth the selected crop with the current Gaussian sigma."""
        crop_layer = self._active_crop_layer()
        if crop_layer is None:
            self._warn("Create or select an ROI crop layer before smoothing.")
            return

        try:
            smoothed = smooth_image(
                np.asarray(crop_layer.data),
                self.gaussian_sigma_spinbox.value(),
            )
        except ValueError as exc:
            self._warn(str(exc))
            return

        self._mask_updates_enabled.discard(id(crop_layer))
        self._clear_mask_layer(crop_layer)
        smoothed_layer = self._create_or_update_smoothed_layer(crop_layer, smoothed)
        self._set_threshold_range_from_image(smoothed)
        self._set_threshold_controls_enabled(True)
        self._move_layer_to_top(smoothed_layer)
        self._activate_layer(smoothed_layer)
        self._set_status(
            f"Applied Gaussian sigma={self.gaussian_sigma_spinbox.value():.3g}. "
            "Click Apply Otsu, then tune the threshold slider."
        )
        show_info(f"Applied Gaussian smoothing to {crop_layer.name}.")

    def use_otsu_threshold(self) -> None:
        """Estimate Otsu threshold from the smoothed crop and update vein_mask."""
        intensity_layer = self._active_intensity_layer()
        if intensity_layer is None:
            self._warn("Apply Gaussian smoothing before Otsu thresholding.")
            return

        try:
            threshold = otsu_threshold(np.asarray(intensity_layer.data))
        except ValueError as exc:
            self._warn(str(exc))
            return

        self._set_threshold_range_from_image(np.asarray(intensity_layer.data))
        self._set_threshold_value(threshold, update_mask=False)
        crop_layer = self._active_crop_layer()
        if crop_layer is not None:
            self._mask_updates_enabled.add(id(crop_layer))
        self._update_mask_from_controls(force=True)
        self._set_status(
            f"Applied Otsu threshold {threshold:.6g}. "
            f"{VEIN_MASK_LAYER_NAME} is editable in paint mode."
        )

    def clean_mask(self) -> None:
        """Clean the current vein mask after explicit morphology settings."""
        mask_layer = self._active_mask_layer()
        if mask_layer is None:
            self._warn(f"Create or select {VEIN_MASK_LAYER_NAME} before cleaning.")
            return

        try:
            cleaned = postprocess_mask(
                np.asarray(mask_layer.data),
                small_object_size=self.min_object_size_spinbox.value(),
                max_hole_size=self.max_hole_size_spinbox.value(),
                closing_radius=self.closing_radius_spinbox.value(),
                opening_radius=self.opening_radius_spinbox.value(),
            )
        except ValueError as exc:
            self._warn(str(exc))
            return

        mask_layer.data = cleaned
        crop_layer = self._mask_to_crop.get(id(mask_layer))
        if crop_layer is not None:
            self._mask_updates_enabled.discard(id(crop_layer))

        self._move_layer_to_top(mask_layer)
        self._activate_layer(mask_layer)
        try:
            mask_layer.mode = "paint"
        except ValueError:
            pass

        self._set_status(
            f"Cleaned {VEIN_MASK_LAYER_NAME}. Paint on this Labels layer to make "
            "manual corrections."
        )
        show_info(f"Cleaned {VEIN_MASK_LAYER_NAME}.")

    def create_order_labels(self) -> None:
        """Create an empty order-label layer from the confirmed vein mask."""
        mask_layer = self._active_mask_layer()
        if mask_layer is None:
            self._warn(f"Create or select {VEIN_MASK_LAYER_NAME} before order labels.")
            return

        mask_data = np.asarray(mask_layer.data)
        if not np.any(mask_data == 0):
            self._warn(f"{VEIN_MASK_LAYER_NAME} has no white-out pixels.")
            return

        if mask_data.ndim != 2:
            self._warn("Order labels require a 2D vein mask.")
            return
        labels = np.where(mask_data == 0, ORDER_UNCERTAIN_LABEL, 0).astype(np.uint8)

        order_layer = self._create_or_select_order_labels_layer(mask_layer, labels)
        if order_layer is None:
            return

        crop_layer = self._mask_to_crop.get(id(mask_layer))
        if crop_layer is not None:
            self._mask_updates_enabled.discard(id(crop_layer))

        mask_layer.visible = False
        self._move_layer_to_top(order_layer)
        self._activate_layer(order_layer)
        try:
            order_layer.mode = "paint"
        except ValueError:
            pass

        self._set_status(
            f"Created {ORDER_LABELS_LAYER_NAME}. Paint labels inside the confirmed "
            f"{VEIN_MASK_LAYER_NAME}."
        )
        show_info(f"Created {ORDER_LABELS_LAYER_NAME}.")

    def _create_or_update_smoothed_layer(
        self,
        crop_layer: Image,
        smoothed: np.ndarray,
    ) -> Image:
        existing = self._crop_to_smoothed.get(id(crop_layer))
        if existing is not None and self._layer_is_present(existing):
            existing.data = smoothed
            existing.translate = tuple(crop_layer.translate)
            return existing

        smoothed_layer = self.viewer.add_image(
            smoothed,
            name=self._unique_layer_name(SMOOTHED_LAYER_NAME),
            translate=tuple(crop_layer.translate),
            opacity=1.0,
            metadata={
                ROLE_KEY: ROLE_SMOOTHED_IMAGE,
                SOURCE_IMAGE_KEY: crop_layer.metadata.get(SOURCE_IMAGE_KEY),
                SOURCE_CROP_KEY: crop_layer.name,
            },
        )
        self._crop_to_smoothed[id(crop_layer)] = smoothed_layer
        self._smoothed_to_crop[id(smoothed_layer)] = crop_layer
        return smoothed_layer

    def _update_mask_from_controls(self, *, force: bool = False) -> None:
        intensity_layer = self._active_intensity_layer()
        if intensity_layer is None:
            return

        crop_layer = self._active_crop_layer()
        if crop_layer is None:
            return

        if not force and id(crop_layer) not in self._mask_updates_enabled:
            return

        try:
            mask = threshold_image(
                np.asarray(intensity_layer.data),
                self.threshold_spinbox.value(),
                invert=self.invert_threshold_checkbox.isChecked(),
            ).astype(np.uint8)
        except ValueError as exc:
            self._warn(str(exc))
            return

        mask_layer = self._create_or_update_mask_layer(crop_layer, mask)
        if mask_layer is None:
            return

        self._set_cleanup_controls_enabled(True)
        self._move_layer_to_top(mask_layer)
        self._activate_layer(mask_layer)
        try:
            mask_layer.mode = "paint"
        except ValueError:
            pass

        self._set_status(
            f"Updated {VEIN_MASK_LAYER_NAME}. Paint on this Labels layer to fix "
            "remaining threshold errors."
        )

    def _create_or_update_mask_layer(
        self,
        crop_layer: Image,
        mask: np.ndarray,
    ) -> Labels | None:
        existing = self._crop_to_mask.get(id(crop_layer))
        if existing is None or not self._layer_is_present(existing):
            named_layer = self._layer_by_name(VEIN_MASK_LAYER_NAME)
            if named_layer is not None:
                if not isinstance(named_layer, Labels):
                    self._warn(f"{VEIN_MASK_LAYER_NAME} exists but is not a Labels layer.")
                    return None
                if named_layer.data.shape != mask.shape:
                    self._warn(
                        f"{VEIN_MASK_LAYER_NAME} exists but has a different shape."
                    )
                    return None
                existing = named_layer

        metadata = {
            ROLE_KEY: ROLE_VEIN_MASK,
            SOURCE_IMAGE_KEY: crop_layer.metadata.get(SOURCE_IMAGE_KEY),
            SOURCE_CROP_KEY: crop_layer.name,
        }

        if existing is not None and self._layer_is_present(existing):
            existing.data = mask
            existing.translate = tuple(crop_layer.translate)
            existing.opacity = 0.65
            existing.metadata.update(metadata)
            mask_layer = existing
        else:
            mask_layer = self.viewer.add_labels(
                mask,
                name=VEIN_MASK_LAYER_NAME,
                translate=tuple(crop_layer.translate),
                opacity=0.65,
                metadata=metadata,
            )

        self._crop_to_mask[id(crop_layer)] = mask_layer
        self._mask_to_crop[id(mask_layer)] = crop_layer
        return mask_layer

    def _create_or_select_order_labels_layer(
        self,
        mask_layer: Labels,
        labels: np.ndarray,
    ) -> Labels | None:
        existing = self._mask_to_order_labels.get(id(mask_layer))
        if existing is None or not self._layer_is_present(existing):
            named_layer = self._layer_by_name(ORDER_LABELS_LAYER_NAME)
            if named_layer is not None:
                if not isinstance(named_layer, Labels):
                    self._warn(
                        f"{ORDER_LABELS_LAYER_NAME} exists but is not a Labels layer."
                    )
                    return None
                if named_layer.data.shape != labels.shape:
                    self._warn(
                        f"{ORDER_LABELS_LAYER_NAME} exists but has a different shape."
                    )
                    return None
                existing = named_layer

        metadata = {
            ROLE_KEY: ROLE_ORDER_LABELS,
            SOURCE_IMAGE_KEY: mask_layer.metadata.get(SOURCE_IMAGE_KEY),
            SOURCE_CROP_KEY: mask_layer.metadata.get(SOURCE_CROP_KEY),
            SOURCE_MASK_KEY: mask_layer.name,
        }

        if existing is not None and self._layer_is_present(existing):
            existing.translate = tuple(mask_layer.translate)
            existing.opacity = 0.8
            existing.colormap = direct_colormap(ORDER_LABEL_COLORS)
            existing.metadata.update(metadata)
            order_layer = existing
        else:
            order_layer = self.viewer.add_labels(
                labels,
                name=ORDER_LABELS_LAYER_NAME,
                translate=tuple(mask_layer.translate),
                opacity=0.8,
                colormap=direct_colormap(ORDER_LABEL_COLORS),
                metadata=metadata,
            )

        self._mask_to_order_labels[id(mask_layer)] = order_layer
        self._order_labels_to_mask[id(order_layer)] = mask_layer
        self._install_masked_order_label_paint(order_layer)
        self._connect_order_label_mask_clip(mask_layer, order_layer)
        self._clip_order_labels_to_mask(order_layer)
        return order_layer

    def _install_masked_order_label_paint(self, order_layer: Labels) -> None:
        order_id = id(order_layer)
        if order_id in self._order_label_data_setitems:
            return

        original_data_setitem = order_layer.data_setitem
        self._order_label_data_setitems[order_id] = original_data_setitem

        def masked_data_setitem(indices, value, refresh=True):
            filtered = self._filter_order_label_indices(order_layer, indices, value)
            if filtered is None:
                return None
            filtered_indices, filtered_value = filtered
            return original_data_setitem(filtered_indices, filtered_value, refresh)

        order_layer.data_setitem = masked_data_setitem

    def _filter_order_label_indices(
        self,
        order_layer: Labels,
        indices,
        value,
    ):
        if self._is_background_label_value(value):
            return indices, value

        mask_layer = self._order_labels_to_mask.get(id(order_layer))
        if mask_layer is None or not self._layer_is_present(mask_layer):
            return indices, value

        mask = np.asarray(mask_layer.data) == 0
        try:
            keep = np.asarray(mask[indices], dtype=bool)
        except (IndexError, TypeError, ValueError):
            return indices, value

        if keep.shape == ():
            return (indices, value) if bool(keep) else None

        if np.all(keep):
            return indices, value

        if not np.any(keep):
            return None

        try:
            filtered_indices = tuple(np.asarray(axis_indices)[keep] for axis_indices in indices)
        except (TypeError, ValueError):
            return indices, value

        value_array = np.asarray(value)
        if value_array.shape == keep.shape:
            return filtered_indices, value_array[keep]
        return filtered_indices, value

    @staticmethod
    def _is_background_label_value(value) -> bool:
        value_array = np.asarray(value)
        return value_array.shape == () and int(value_array) == 0

    def _connect_order_label_mask_clip(
        self,
        mask_layer: Labels,
        order_layer: Labels,
    ) -> None:
        order_id = id(order_layer)
        if order_id not in self._order_label_clip_connected:
            order_layer.events.data.connect(
                lambda event, order_layer=order_layer: self._clip_order_labels_to_mask(
                    order_layer
                )
            )
            order_layer.events.labels_update.connect(
                lambda event, order_layer=order_layer: self._clip_order_labels_to_mask(
                    order_layer
                )
            )
            self._order_label_clip_connected.add(order_id)

        mask_id = id(mask_layer)
        if mask_id not in self._mask_clip_connected:
            mask_layer.events.data.connect(
                lambda event, mask_layer=mask_layer: self._clip_order_labels_for_mask(
                    mask_layer
                )
            )
            self._mask_clip_connected.add(mask_id)

    def _clip_order_labels_for_mask(self, mask_layer: Labels) -> None:
        order_layer = self._mask_to_order_labels.get(id(mask_layer))
        if order_layer is not None and self._layer_is_present(order_layer):
            self._clip_order_labels_to_mask(order_layer)

    def _clip_order_labels_to_mask(self, order_layer: Labels) -> None:
        order_id = id(order_layer)
        if order_id in self._clipping_order_labels:
            return

        mask_layer = self._order_labels_to_mask.get(order_id)
        if mask_layer is None or not self._layer_is_present(mask_layer):
            return

        labels = np.asarray(order_layer.data)
        mask = np.asarray(mask_layer.data) == 0
        if labels.shape != mask.shape:
            return

        outside_mask = ~mask
        if not np.any((labels > 0) & outside_mask):
            return

        clipped = labels.copy()
        clipped[outside_mask] = 0
        self._clipping_order_labels.add(order_id)
        try:
            order_layer.data = clipped
        finally:
            self._clipping_order_labels.discard(order_id)

    def _clear_mask_products(self, crop_layer: Image) -> None:
        self._mask_updates_enabled.discard(id(crop_layer))

        smoothed_layer = self._crop_to_smoothed.pop(id(crop_layer), None)
        if smoothed_layer is not None:
            self._smoothed_to_crop.pop(id(smoothed_layer), None)
            if self._layer_is_present(smoothed_layer):
                self.viewer.layers.remove(smoothed_layer)

        self._clear_mask_layer(crop_layer)

    def _clear_mask_layer(self, crop_layer: Image) -> None:
        mask_layer = self._crop_to_mask.pop(id(crop_layer), None)
        if mask_layer is not None:
            self._mask_to_crop.pop(id(mask_layer), None)
            self._clear_order_labels_layer(mask_layer)
            if self._layer_is_present(mask_layer):
                self.viewer.layers.remove(mask_layer)
        self._set_cleanup_controls_enabled(False)

    def _clear_order_labels_layer(self, mask_layer: Labels) -> None:
        order_layer = self._mask_to_order_labels.pop(id(mask_layer), None)
        if order_layer is not None:
            self._order_labels_to_mask.pop(id(order_layer), None)
            self._order_label_data_setitems.pop(id(order_layer), None)
            self._order_label_clip_connected.discard(id(order_layer))
            self._clipping_order_labels.discard(id(order_layer))
            if self._layer_is_present(order_layer):
                self.viewer.layers.remove(order_layer)
        self._mask_clip_connected.discard(id(mask_layer))

    def _schedule_mask_update(self, *args) -> None:
        if self._syncing_threshold_controls:
            return

        crop_layer = self._active_crop_layer()
        if crop_layer is None:
            return

        if id(crop_layer) not in self._mask_updates_enabled:
            return

        if self._active_intensity_layer() is None:
            return
        self._mask_update_timer.start()

    def _on_threshold_spinbox_changed(self, value: float) -> None:
        if self._syncing_threshold_controls:
            return

        self._syncing_threshold_controls = True
        self.threshold_slider.setValue(self._threshold_to_slider(value))
        self._syncing_threshold_controls = False
        self._schedule_mask_update()

    def _on_threshold_slider_changed(self, value: int) -> None:
        if self._syncing_threshold_controls:
            return

        threshold = self._slider_to_threshold(value)
        self._syncing_threshold_controls = True
        self.threshold_spinbox.setValue(threshold)
        self._syncing_threshold_controls = False
        self._schedule_mask_update()

    def _set_threshold_range_from_image(self, image: np.ndarray) -> None:
        grayscale = to_grayscale(image)
        minimum = float(np.min(grayscale))
        maximum = float(np.max(grayscale))
        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5

        self._threshold_min = minimum
        self._threshold_max = maximum
        was_syncing = self._syncing_threshold_controls
        self._syncing_threshold_controls = True
        self.threshold_spinbox.setRange(minimum, maximum)
        self.threshold_spinbox.setSingleStep((maximum - minimum) / 100.0)
        self._syncing_threshold_controls = was_syncing
        self._set_threshold_value(
            float(np.clip(self.threshold_spinbox.value(), minimum, maximum)),
            update_mask=False,
        )

    def _set_threshold_value(self, value: float, *, update_mask: bool) -> None:
        self._syncing_threshold_controls = True
        self.threshold_spinbox.setValue(value)
        self.threshold_slider.setValue(self._threshold_to_slider(value))
        self._syncing_threshold_controls = False
        if update_mask:
            self._schedule_mask_update()

    def _threshold_to_slider(self, value: float) -> int:
        if self._threshold_max <= self._threshold_min:
            return 0
        fraction = (float(value) - self._threshold_min) / (
            self._threshold_max - self._threshold_min
        )
        return int(np.clip(round(fraction * 1000), 0, 1000))

    def _slider_to_threshold(self, value: int) -> float:
        fraction = float(value) / 1000.0
        return self._threshold_min + fraction * (
            self._threshold_max - self._threshold_min
        )

    def _active_roi_layer(self) -> Shapes | None:
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Shapes) and layer.metadata.get(ROLE_KEY) == ROLE_ROI_SHAPES:
            return layer
        return None

    def _active_crop_layer(self) -> Image | None:
        layer = self.viewer.layers.selection.active

        if isinstance(layer, Image) and layer.metadata.get(ROLE_KEY) == ROLE_ROI_CROP:
            return layer

        if isinstance(layer, Image) and layer.metadata.get(ROLE_KEY) == ROLE_SMOOTHED_IMAGE:
            crop_layer = self._smoothed_to_crop.get(id(layer))
            if crop_layer is not None and self._layer_is_present(crop_layer):
                return crop_layer

        if isinstance(layer, Labels) and layer.metadata.get(ROLE_KEY) == ROLE_VEIN_MASK:
            crop_layer = self._mask_to_crop.get(id(layer))
            if crop_layer is not None and self._layer_is_present(crop_layer):
                return crop_layer

        if isinstance(layer, Labels) and layer.metadata.get(ROLE_KEY) == ROLE_ORDER_LABELS:
            mask_layer = self._order_labels_to_mask.get(id(layer))
            if mask_layer is not None and self._layer_is_present(mask_layer):
                crop_layer = self._mask_to_crop.get(id(mask_layer))
                if crop_layer is not None and self._layer_is_present(crop_layer):
                    return crop_layer

        if isinstance(layer, Shapes) and layer.metadata.get(ROLE_KEY) == ROLE_ROI_SHAPES:
            crop_layer = self._roi_to_crop.get(id(layer))
            if crop_layer is not None and self._layer_is_present(crop_layer):
                return crop_layer

        return None

    def _active_intensity_layer(self) -> Image | None:
        crop_layer = self._active_crop_layer()
        if crop_layer is None:
            return None

        smoothed_layer = self._crop_to_smoothed.get(id(crop_layer))
        if smoothed_layer is not None and self._layer_is_present(smoothed_layer):
            return smoothed_layer

        return crop_layer

    def _active_mask_layer(self) -> Labels | None:
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Labels) and layer.metadata.get(ROLE_KEY) == ROLE_VEIN_MASK:
            return layer

        if isinstance(layer, Labels) and layer.metadata.get(ROLE_KEY) == ROLE_ORDER_LABELS:
            mask_layer = self._order_labels_to_mask.get(id(layer))
            if mask_layer is not None and self._layer_is_present(mask_layer):
                return mask_layer

        crop_layer = self._active_crop_layer()
        if crop_layer is None:
            return None

        mask_layer = self._crop_to_mask.get(id(crop_layer))
        if mask_layer is not None and self._layer_is_present(mask_layer):
            return mask_layer

        return None

    def _activate_rectangle_mode(self, roi_layer: Shapes) -> None:
        self._activate_layer(roi_layer)
        roi_layer.mode = "add_rectangle"

    def _activate_layer(self, layer) -> None:
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(layer)
        self.viewer.layers.selection.active = layer

    def _unique_layer_name(self, base_name: str) -> str:
        existing = {layer.name for layer in self.viewer.layers}
        if base_name not in existing:
            return base_name

        index = 2
        while f"{base_name} {index}" in existing:
            index += 1
        return f"{base_name} {index}"

    def _layer_by_name(self, name: str):
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def _set_gaussian_controls_enabled(self, enabled: bool) -> None:
        self.gaussian_sigma_label.setEnabled(enabled)
        self.gaussian_sigma_spinbox.setEnabled(enabled)
        self.apply_gaussian_button.setEnabled(enabled)

    def _set_threshold_controls_enabled(self, enabled: bool) -> None:
        self.otsu_button.setEnabled(enabled)
        self.invert_threshold_checkbox.setEnabled(enabled)
        self.threshold_label.setEnabled(enabled)
        self.threshold_spinbox.setEnabled(enabled)
        self.threshold_slider.setEnabled(enabled)

    def _set_cleanup_controls_enabled(self, enabled: bool) -> None:
        self.min_object_size_label.setEnabled(enabled)
        self.min_object_size_spinbox.setEnabled(enabled)
        self.max_hole_size_label.setEnabled(enabled)
        self.max_hole_size_spinbox.setEnabled(enabled)
        self.closing_radius_label.setEnabled(enabled)
        self.closing_radius_spinbox.setEnabled(enabled)
        self.opening_radius_label.setEnabled(enabled)
        self.opening_radius_spinbox.setEnabled(enabled)
        self.clean_mask_button.setEnabled(enabled)
        self.create_order_labels_button.setEnabled(enabled)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _warn(self, message: str) -> None:
        self._set_status(message)
        show_warning(message)

    @staticmethod
    def _single_shape_type(layer: Shapes) -> str | None:
        shape_type = layer.shape_type
        if isinstance(shape_type, str):
            return shape_type
        if len(shape_type) != 1:
            return None
        return str(shape_type[0])

    def _layer_is_present(self, target_layer) -> bool:
        return any(layer is target_layer for layer in self.viewer.layers)

    def _move_layer_to_top(self, layer) -> None:
        current_index = self.viewer.layers.index(layer)
        top_index = len(self.viewer.layers) - 1
        if current_index != top_index:
            self.viewer.layers.move(current_index, top_index)
