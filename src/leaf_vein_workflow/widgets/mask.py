"""Gaussian, threshold, and cleanup stages for the dock workflow.

The mask is created only after the explicit Otsu action.  Slider and checkbox
signals may update an existing mask, but must not cross that stage boundary.
"""

from typing import Any

import numpy as np
from napari.layers import Image, Labels
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer

from ..threshold import (
    otsu_threshold,
    postprocess_mask,
    smooth_image,
    threshold_image,
    to_grayscale,
)
from .controls import WorkflowControls
from .layers import (
    ROLE_KEY,
    ROLE_SMOOTHED_IMAGE,
    ROLE_VEIN_MASK,
    SMOOTHED_LAYER_NAME,
    SOURCE_CROP_KEY,
    SOURCE_IMAGE_KEY,
    VEIN_MASK_LAYER_NAME,
    WorkflowLayers,
)


class MaskWorkflow:
    """Run explicit smoothing, thresholding, and mask cleanup actions."""

    def __init__(
        self,
        layers: WorkflowLayers,
        controls: WorkflowControls,
        qt_parent: Any,
    ):
        self.layers = layers
        self.controls = controls
        self.threshold_min = 0.0
        self.threshold_max = 1.0
        self.syncing_threshold_controls = False

        self.update_timer = QTimer(qt_parent)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(150)

    def apply_gaussian_filter(self) -> None:
        """Smooth the selected crop with the current Gaussian sigma."""
        crop_layer = self.layers.active_crop_layer()
        if crop_layer is None:
            self.controls.warn("Create or select an ROI crop layer before smoothing.")
            return

        try:
            smoothed = smooth_image(
                np.asarray(crop_layer.data),
                self.controls.gaussian_sigma_spinbox.value(),
            )
        except ValueError as exc:
            self.controls.warn(str(exc))
            return

        self.layers.mask_updates_enabled.discard(id(crop_layer))
        self.layers.clear_mask_layer(crop_layer)
        self.controls.set_cleanup_enabled(False)
        smoothed_layer = self.create_or_update_smoothed_layer(crop_layer, smoothed)
        self.set_threshold_range_from_image(smoothed)
        self.controls.set_threshold_enabled(True)
        self.layers.move_to_top(smoothed_layer)
        self.layers.activate(smoothed_layer)
        self.controls.set_status(
            "Applied Gaussian "
            f"sigma={self.controls.gaussian_sigma_spinbox.value():.3g}. "
            "Click Apply Otsu, then tune the threshold slider."
        )
        show_info(f"Applied Gaussian smoothing to {crop_layer.name}.")

    def use_otsu_threshold(self) -> None:
        """Estimate an Otsu threshold and explicitly create the vein mask."""
        intensity_layer = self.layers.active_intensity_layer()
        if intensity_layer is None:
            self.controls.warn("Apply Gaussian smoothing before Otsu thresholding.")
            return

        try:
            threshold = otsu_threshold(np.asarray(intensity_layer.data))
        except ValueError as exc:
            self.controls.warn(str(exc))
            return

        self.set_threshold_range_from_image(np.asarray(intensity_layer.data))
        self.set_threshold_value(threshold, update_mask=False)
        crop_layer = self.layers.active_crop_layer()
        if crop_layer is not None:
            self.layers.mask_updates_enabled.add(id(crop_layer))
        self.update_mask_from_controls(force=True)
        self.controls.set_status(
            f"Applied Otsu threshold {threshold:.6g}. "
            f"{VEIN_MASK_LAYER_NAME} is editable in paint mode."
        )

    def clean_mask(self) -> None:
        """Clean the current mask after explicit morphology settings."""
        mask_layer = self.layers.active_mask_layer()
        if mask_layer is None:
            self.controls.warn(
                f"Create or select {VEIN_MASK_LAYER_NAME} before cleaning."
            )
            return

        try:
            cleaned = postprocess_mask(
                np.asarray(mask_layer.data),
                small_object_size=self.controls.min_object_size_spinbox.value(),
                max_hole_size=self.controls.max_hole_size_spinbox.value(),
                closing_radius=self.controls.closing_radius_spinbox.value(),
                opening_radius=self.controls.opening_radius_spinbox.value(),
            )
        except ValueError as exc:
            self.controls.warn(str(exc))
            return

        mask_layer.data = cleaned
        crop_layer = self.layers.mask_to_crop.get(id(mask_layer))
        if crop_layer is not None:
            # 手修正後のmaskを閾値UIの遅延更新で上書きしない。
            self.layers.mask_updates_enabled.discard(id(crop_layer))

        self.layers.move_to_top(mask_layer)
        self.layers.activate(mask_layer)
        try:
            mask_layer.mode = "paint"
        except ValueError:
            pass

        self.controls.set_status(
            f"Cleaned {VEIN_MASK_LAYER_NAME}. Paint on this Labels layer to make "
            "manual corrections."
        )
        show_info(f"Cleaned {VEIN_MASK_LAYER_NAME}.")

    def create_or_update_smoothed_layer(
        self,
        crop_layer: Image,
        smoothed: np.ndarray,
    ) -> Image:
        """Create one smoothed image per crop, reusing it when possible."""
        existing = self.layers.crop_to_smoothed.get(id(crop_layer))
        if existing is not None and self.layers.is_present(existing):
            existing.data = smoothed
            existing.translate = tuple(crop_layer.translate)
            existing.metadata.update(self.layers.inherited_calibration(crop_layer))
            return existing

        smoothed_layer = self.layers.viewer.add_image(
            smoothed,
            name=self.layers.unique_name(SMOOTHED_LAYER_NAME),
            translate=tuple(crop_layer.translate),
            opacity=1.0,
            metadata={
                ROLE_KEY: ROLE_SMOOTHED_IMAGE,
                SOURCE_IMAGE_KEY: crop_layer.metadata.get(SOURCE_IMAGE_KEY),
                SOURCE_CROP_KEY: crop_layer.name,
                **self.layers.inherited_calibration(crop_layer),
            },
        )
        self.layers.crop_to_smoothed[id(crop_layer)] = smoothed_layer
        self.layers.smoothed_to_crop[id(smoothed_layer)] = crop_layer
        return smoothed_layer

    def update_mask_from_controls(self, *, force: bool = False) -> None:
        """Update a mask only after Otsu has enabled this crop."""
        intensity_layer = self.layers.active_intensity_layer()
        if intensity_layer is None:
            return

        crop_layer = self.layers.active_crop_layer()
        if crop_layer is None:
            return

        if not force and id(crop_layer) not in self.layers.mask_updates_enabled:
            return

        try:
            mask = threshold_image(
                np.asarray(intensity_layer.data),
                self.controls.threshold_spinbox.value(),
                invert=self.controls.invert_threshold_checkbox.isChecked(),
            ).astype(np.uint8)
        except ValueError as exc:
            self.controls.warn(str(exc))
            return

        mask_layer = self.create_or_update_mask_layer(crop_layer, mask)
        if mask_layer is None:
            return

        self.controls.set_cleanup_enabled(True)
        self.layers.move_to_top(mask_layer)
        self.layers.activate(mask_layer)
        try:
            mask_layer.mode = "paint"
        except ValueError:
            pass

        self.controls.set_status(
            f"Updated {VEIN_MASK_LAYER_NAME}. Paint on this Labels layer to fix "
            "remaining threshold errors."
        )

    def create_or_update_mask_layer(
        self,
        crop_layer: Image,
        mask: np.ndarray,
    ) -> Labels | None:
        """Create one editable mask per crop, with name-collision checks."""
        existing = self.layers.crop_to_mask.get(id(crop_layer))
        if existing is None or not self.layers.is_present(existing):
            named_layer = self.layers.by_name(VEIN_MASK_LAYER_NAME)
            if named_layer is not None:
                if not isinstance(named_layer, Labels):
                    self.controls.warn(
                        f"{VEIN_MASK_LAYER_NAME} exists but is not a Labels layer."
                    )
                    return None
                if named_layer.data.shape != mask.shape:
                    self.controls.warn(
                        f"{VEIN_MASK_LAYER_NAME} exists but has a different shape."
                    )
                    return None
                existing = named_layer

        metadata = {
            ROLE_KEY: ROLE_VEIN_MASK,
            SOURCE_IMAGE_KEY: crop_layer.metadata.get(SOURCE_IMAGE_KEY),
            SOURCE_CROP_KEY: crop_layer.name,
            **self.layers.inherited_calibration(crop_layer),
        }

        if existing is not None and self.layers.is_present(existing):
            existing.data = mask
            existing.translate = tuple(crop_layer.translate)
            existing.opacity = 0.65
            existing.metadata.update(metadata)
            mask_layer = existing
        else:
            mask_layer = self.layers.viewer.add_labels(
                mask,
                name=VEIN_MASK_LAYER_NAME,
                translate=tuple(crop_layer.translate),
                opacity=0.65,
                metadata=metadata,
            )

        self.layers.crop_to_mask[id(crop_layer)] = mask_layer
        self.layers.mask_to_crop[id(mask_layer)] = crop_layer
        return mask_layer

    def schedule_mask_update(self, *args) -> None:
        """Debounce threshold UI changes for an already enabled mask."""
        if self.syncing_threshold_controls:
            return

        crop_layer = self.layers.active_crop_layer()
        if crop_layer is None:
            return

        # Gaussianやrange初期化が発火するsignalからmaskを作らせないためのgate。
        if id(crop_layer) not in self.layers.mask_updates_enabled:
            return

        if self.layers.active_intensity_layer() is None:
            return
        self.update_timer.start()

    def on_threshold_spinbox_changed(self, value: float) -> None:
        """Synchronize the slider from a numeric threshold edit."""
        if self.syncing_threshold_controls:
            return

        self.syncing_threshold_controls = True
        self.controls.threshold_slider.setValue(self.threshold_to_slider(value))
        self.syncing_threshold_controls = False
        self.schedule_mask_update()

    def on_threshold_slider_changed(self, value: int) -> None:
        """Synchronize the numeric threshold from a slider edit."""
        if self.syncing_threshold_controls:
            return

        threshold = self.slider_to_threshold(value)
        self.syncing_threshold_controls = True
        self.controls.threshold_spinbox.setValue(threshold)
        self.syncing_threshold_controls = False
        self.schedule_mask_update()

    def set_threshold_range_from_image(self, image: np.ndarray) -> None:
        """Map the fixed 0..1000 slider onto one image's intensity range."""
        grayscale = to_grayscale(image)
        minimum = float(np.min(grayscale))
        maximum = float(np.max(grayscale))
        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5

        self.threshold_min = minimum
        self.threshold_max = maximum
        was_syncing = self.syncing_threshold_controls
        self.syncing_threshold_controls = True
        self.controls.threshold_spinbox.setRange(minimum, maximum)
        self.controls.threshold_spinbox.setSingleStep((maximum - minimum) / 100.0)
        self.syncing_threshold_controls = was_syncing
        self.set_threshold_value(
            float(
                np.clip(
                    self.controls.threshold_spinbox.value(),
                    minimum,
                    maximum,
                )
            ),
            update_mask=False,
        )

    def set_threshold_value(self, value: float, *, update_mask: bool) -> None:
        """Set both threshold controls without recursive signal handling."""
        self.syncing_threshold_controls = True
        self.controls.threshold_spinbox.setValue(value)
        self.controls.threshold_slider.setValue(self.threshold_to_slider(value))
        self.syncing_threshold_controls = False
        if update_mask:
            self.schedule_mask_update()

    def threshold_to_slider(self, value: float) -> int:
        """Convert an intensity value to the fixed slider scale."""
        if self.threshold_max <= self.threshold_min:
            return 0
        fraction = (float(value) - self.threshold_min) / (
            self.threshold_max - self.threshold_min
        )
        return int(np.clip(round(fraction * 1000), 0, 1000))

    def slider_to_threshold(self, value: int) -> float:
        """Convert the fixed slider scale back to an intensity value."""
        fraction = float(value) / 1000.0
        return self.threshold_min + fraction * (
            self.threshold_max - self.threshold_min
        )

