"""Leaf confirmation and vein extraction orchestration for the dock workflow.

Otsu creates a leaf candidate. Vein extraction starts only after leaf
confirmation; threshold signals must not cross that stage boundary.
Explicit vein-mask cleanup is implemented in mask_cleanup.py.
"""

from typing import Any

import numpy as np
from napari.layers import Image, Labels
from napari.utils.colormaps import direct_colormap
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from ..calibration import CALIBRATION_MM_PER_PIXEL_KEY
from ..provenance import array_digest
from ..threshold import (
    estimate_leaf_mask,
    estimate_vein_threshold,
    largest_leaf_component,
    smooth_image,
    threshold_image,
    to_grayscale,
)
from .controls import WorkflowControls
from .mask_cleanup import clean_vein_mask
from .layers import (
    ROLE_KEY,
    ROLE_LEAF_MASK,
    LEAF_MASK_LAYER_NAME,
    LEAF_MASK_COLORS,
    LEAF_MASK_OPACITY,
    ROLE_SMOOTHED_IMAGE,
    ROLE_VEIN_MASK,
    SMOOTHED_LAYER_NAME,
    SOURCE_CROP_KEY,
    SOURCE_IMAGE_KEY,
    VEIN_MASK_COLORS,
    VEIN_MASK_OPACITY,
    VEIN_MASK_LABEL,
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

    def refresh_leaf_status(self, *args) -> None:
        """Show the area for the selected ROI using its current calibration."""
        crop = self.layers.active_crop_layer()
        layer = self.layers.crop_to_leaf.get(id(crop)) if crop is not None else None
        self.controls.confirm_leaf_button.setEnabled(
            layer is not None and self.layers.is_present(layer)
        )
        if layer is None or not self.layers.is_present(layer):
            self.controls.set_leaf_area_status("Leaf area: not confirmed")
            return
        try:
            leaf = self.layers.confirmed_leaf_mask(crop)
        except ValueError:
            self.controls.set_leaf_area_status("Leaf area: review and confirm leaf_mask")
            return
        count = int(np.count_nonzero(leaf))
        mm = crop.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        area = (
            f"{count * float(mm) ** 2:.6g} mm²"
            if mm else f"{count} px² (apply scale for mm²)"
        )
        self.controls.set_leaf_area_status(f"Confirmed ROI leaf area: {area}")

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
        self.update_timer.stop()
        self.layers.clear_mask_products(crop_layer)
        self.controls.confirm_leaf_button.setEnabled(False)
        self.controls.set_leaf_area_status("Leaf area: not confirmed")
        self.controls.set_cleanup_enabled(False)
        smoothed_layer = self.create_or_update_smoothed_layer(crop_layer, smoothed)
        record = self.layers.processing(crop_layer)
        record["gaussian_sigma"] = float(self.controls.gaussian_sigma_spinbox.value())
        self.layers.record_processing(smoothed_layer, record)
        self.set_threshold_range_from_image(smoothed)
        self.controls.set_threshold_enabled(True)
        self.layers.move_to_top(smoothed_layer)
        self.layers.activate(smoothed_layer)
        self.controls.set_status(
            "Applied Gaussian "
            f"sigma={self.controls.gaussian_sigma_spinbox.value():.3g}. "
            "Click Apply Otsu, review leaf_mask, then confirm the leaf."
        )
        show_info(f"Applied Gaussian smoothing to {crop_layer.name}.")

    def use_otsu_threshold(self) -> None:
        """Create an editable first-stage leaf candidate, without extracting veins."""
        crop = self.layers.active_crop_layer()
        intensity = self.layers.active_intensity_layer()
        if crop is None or intensity is None:
            self.controls.warn("Create an ROI and apply Gaussian before Otsu.")
            return
        try:
            threshold, candidate = estimate_leaf_mask(np.asarray(intensity.data))
        except ValueError as exc:
            self.controls.warn(str(exc))
            return
        if not self.controls.exclude_background_checkbox.isChecked():
            candidate = None
        if candidate is None:
            candidate = np.ones(np.shape(intensity.data)[:2], dtype=bool)
            show_info(
                "No background excluded: review the full ROI as a local leaf "
                "region. This is not automatically a whole-leaf area."
            )

        self.update_timer.stop()
        self.layers.mask_updates_enabled.discard(id(crop))
        self.layers.pending_leaf_masks.add(id(crop))
        if self.controls.largest_leaf_checkbox.isChecked():
            candidate = largest_leaf_component(candidate)
        leaf_layer = self.layers.crop_to_leaf.get(id(crop))
        if leaf_layer is not None and self.layers.is_present(leaf_layer):
            leaf_layer.data = candidate.astype(np.uint8)
        else:
            leaf_layer = self.layers.viewer.add_labels(
                candidate.astype(np.uint8),
                name=self.layers.unique_name(LEAF_MASK_LAYER_NAME),
                translate=tuple(crop.translate),
                opacity=LEAF_MASK_OPACITY,
                colormap=direct_colormap(LEAF_MASK_COLORS),
                metadata={
                    ROLE_KEY: ROLE_LEAF_MASK,
                    SOURCE_IMAGE_KEY: crop.metadata.get(SOURCE_IMAGE_KEY),
                    SOURCE_CROP_KEY: crop.name,
                    **self.layers.inherited_calibration(crop),
                },
            )
            self.layers.register_leaf(crop, leaf_layer)
        record = self.layers.processing(intensity)
        record["leaf_background_threshold"] = float(threshold)
        self.layers.record_processing(leaf_layer, record)
        leaf_layer.selected_label = 1
        leaf_layer.visible = True
        self.layers.move_to_top(leaf_layer)
        self.layers.activate(leaf_layer)
        leaf_layer.mode = "paint"
        self.controls.confirm_leaf_button.setEnabled(True)
        self.controls.set_cleanup_enabled(False)
        self.controls.set_leaf_area_status("Leaf area: review and confirm leaf_mask")
        self.controls.set_status(
            "Paint or erase leaf_mask (blade only), then click "
            "Confirm Leaf & Extract Veins."
        )

    def confirm_leaf(self) -> bool | None:
        """Freeze the edited leaf, calculate its area, and run the second Otsu."""
        crop = self.layers.active_crop_layer()
        if crop is None:
            self.controls.warn("Select a leaf mask or its ROI before confirming.")
            return
        leaf_layer = self.layers.crop_to_leaf.get(id(crop))
        intensity = self.layers.active_intensity_layer()
        if leaf_layer is None or not self.layers.is_present(leaf_layer) or intensity is None:
            self.controls.warn("Click Apply Otsu to create a leaf mask first.")
            return
        leaf = np.asarray(leaf_layer.data) > 0
        original_area = int(np.count_nonzero(leaf))
        if self.controls.largest_leaf_checkbox.isChecked():
            leaf = largest_leaf_component(leaf)
        try:
            estimate = estimate_vein_threshold(np.asarray(intensity.data), leaf_mask=leaf)
        except ValueError as exc:
            self.controls.warn(str(exc))
            return
        existing = self.layers.crop_to_mask.get(id(crop))
        if existing is not None and self.layers.is_present(existing):
            answer = QMessageBox.question(
                None, "Extract veins again?",
                "This replaces the vein mask and removes its order labels, "
                "including manual edits. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.update_timer.stop()
        self.layers.clear_mask_layer(crop)
        leaf_layer.data = leaf.astype(np.uint8)
        self.layers.confirm_leaf_region(crop, leaf)
        record = self.layers.processing(leaf_layer)
        record["leaf_mask"] = {
            "confirmed": True,
            "area_px": int(np.count_nonzero(leaf)),
            "sha256": array_digest(leaf.astype(np.uint8)),
            "scope": "current_roi",
            "largest_component_only": self.controls.largest_leaf_checkbox.isChecked(),
            "removed_disconnected_px": original_area - int(np.count_nonzero(leaf)),
        }
        record["otsu_threshold"] = float(estimate.threshold)
        record["otsu_vein_side"] = estimate.vein_side
        record["background_excluded"] = True
        self.layers.record_processing(intensity, record)
        self.layers.record_processing(leaf_layer, record)
        mm = crop.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        area_px = int(np.count_nonzero(leaf))
        area = f"{area_px * float(mm) ** 2:.6g} mm²" if mm else f"{area_px} px² (apply scale for mm²)"
        self.controls.set_leaf_area_status(f"Confirmed ROI leaf area: {area}")
        self.controls.leaf_area_source_combo.setCurrentIndex(
            self.controls.leaf_area_source_combo.findData("leaf_mask")
        )
        self.set_threshold_range_from_image(np.asarray(intensity.data))
        checkbox = self.controls.invert_threshold_checkbox
        checkbox.blockSignals(True)
        checkbox.setChecked(estimate.invert)
        checkbox.blockSignals(False)
        self.set_threshold_value(estimate.threshold, update_mask=False)
        leaf_layer.visible = False
        self.layers.mask_updates_enabled.add(id(crop))
        self.update_mask_from_controls(force=True)
        show_info(f"Confirmed leaf area: {area}. Veins extracted inside the same region.")
        return True

    def clean_mask(self) -> bool | None:
        """Delegate vein-mask cleanup to its separate processing stage."""
        return clean_vein_mask(self.layers, self.controls)

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
        self.layers.register_smoothed(crop_layer, smoothed_layer)
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
            vein = threshold_image(
                np.asarray(intensity_layer.data),
                self.controls.threshold_spinbox.value(),
                invert=self.controls.invert_threshold_checkbox.isChecked(),
            )
        except ValueError as exc:
            self.controls.warn(str(exc))
            return
        try:
            leaf = self.layers.confirmed_leaf_mask(crop_layer)
        except ValueError as exc:
            self.layers.mask_updates_enabled.discard(id(crop_layer))
            self.controls.warn(str(exc))
            return
        if leaf is not None and leaf.shape == vein.shape:
            vein = vein & leaf
        mask = vein.astype(np.uint8)

        mask_layer = self.create_or_update_mask_layer(crop_layer, mask)
        if mask_layer is None:
            return
        # 閾値から作り直した mask なので、cleanup 履歴はここでリセットする。
        record = self.layers.processing(intensity_layer)
        record["threshold"] = {
            "value": float(self.controls.threshold_spinbox.value()),
            "invert": bool(self.controls.invert_threshold_checkbox.isChecked()),
            "inside_leaf_only": leaf is not None,
        }
        record["cleanup"] = []
        record["auto_mask_sha256"] = array_digest(mask)
        self.layers.record_processing(mask_layer, record)

        self.controls.set_cleanup_enabled(True)
        self.layers.move_to_top(mask_layer)
        self.layers.activate(mask_layer)
        try:
            mask_layer.mode = "paint"
        except ValueError:
            pass

        self.controls.set_status(vein_coverage_message(mask, leaf))

    def create_or_update_mask_layer(
        self,
        crop_layer: Image,
        mask: np.ndarray,
    ) -> Labels | None:
        """Create one editable mask per crop, with name-collision checks."""
        existing = self.layers.crop_to_mask.get(id(crop_layer))
        if existing is None or not self.layers.is_present(existing):
            named_layer = self.layers.by_name(VEIN_MASK_LAYER_NAME)
            owner = self.layers.mask_to_crop.get(id(named_layer))
            if named_layer is not None and (owner is None or owner is crop_layer):
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
            existing.opacity = VEIN_MASK_OPACITY
            existing.colormap = direct_colormap(VEIN_MASK_COLORS)
            existing.metadata.update(metadata)
            mask_layer = existing
        else:
            mask_layer = self.layers.viewer.add_labels(
                mask,
                name=self.layers.unique_name(VEIN_MASK_LAYER_NAME),
                translate=tuple(crop_layer.translate),
                opacity=VEIN_MASK_OPACITY,
                colormap=direct_colormap(VEIN_MASK_COLORS),
                metadata=metadata,
            )
        mask_layer.selected_label = VEIN_MASK_LABEL

        self.layers.register_vein_mask(crop_layer, mask_layer)
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



# 葉脈が葉（または ROI）に占める割合がこれを超えたら、葉身や背景を選んでいる疑い。
VEIN_COVERAGE_WARNING = 0.45


def vein_coverage_message(mask: np.ndarray, leaf: np.ndarray | None) -> str:
    """Tell the user how much is colored as vein and whether it looks right."""
    vein_px = int(np.count_nonzero(mask))
    area_px = int(np.count_nonzero(leaf)) if leaf is not None else int(mask.size)
    fraction = vein_px / area_px if area_px else 0.0
    where = "of the leaf" if leaf is not None else "of the ROI"
    message = f"Colored = vein: {fraction:.0%} {where}."
    if fraction > VEIN_COVERAGE_WARNING:
        message += (
            " That is more than veins usually cover: if the leaf blade or the "
            "background is colored, toggle Invert or check the background option."
        )
    else:
        message += " Paint or erase on vein_mask to fix spots."
    return message
