"""ROI preparation and crop creation for the dock workflow.

This module owns the first explicit workflow stage.  Mask processing and
order-label editing are intentionally handled elsewhere.
"""

import numpy as np
from napari.layers import Image, Shapes
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer

from ..crop import crop_to_rectangle, rectangle_to_slices
from .controls import WorkflowControls
from .layers import (
    PROCESSING_KEY,
    ROLE_KEY,
    ROLE_ROI_CROP,
    ROLE_ROI_SHAPES,
    SOURCE_IMAGE_KEY,
    WorkflowLayers,
)


class RoiWorkflow:
    """Prepare rectangle layers and create confirmed ROI crops."""

    def __init__(self, layers: WorkflowLayers, controls: WorkflowControls):
        self.layers = layers
        self.controls = controls

    def on_layer_inserted(self, event) -> None:
        """Prepare a rectangle ROI when napari inserts a source image."""
        self.prepare_image_layer(event.value)

    def prepare_image_layer(self, layer) -> None:
        """Attach one editable rectangle layer to a new source image."""
        if not isinstance(layer, Image):
            return

        if layer.metadata.get(ROLE_KEY) is not None:
            return

        if any(source is layer for source in self.layers.roi_to_image.values()):
            return

        roi_layer = self.layers.viewer.add_shapes(
            name=self.layers.unique_name(f"{layer.name} ROI rectangle"),
            ndim=2,
            edge_color="yellow",
            face_color=[1, 1, 0, 0.15],
            edge_width=2,
        )
        roi_layer.metadata[ROLE_KEY] = ROLE_ROI_SHAPES
        roi_layer.metadata[SOURCE_IMAGE_KEY] = layer.name
        roi_layer.metadata.update(self.layers.inherited_calibration(layer))
        self.layers.roi_to_image[id(roi_layer)] = layer
        roi_layer.events.data.connect(
            lambda event, roi_layer=roi_layer: self.on_roi_data_changed(roi_layer)
        )

        self.layers.activate_rectangle_mode(roi_layer)
        # napari側の挿入処理後にselectionが変わることがあるため、次のevent loopでも再選択する。
        QTimer.singleShot(
            0,
            lambda roi_layer=roi_layer: self.layers.activate_rectangle_mode(roi_layer)
            if self.layers.is_present(roi_layer)
            else None,
        )

        self.controls.add_roi_button.setEnabled(True)
        self.controls.set_status(
            f"Created ROI rectangle layer for {layer.name}. Draw one rectangle."
        )
        show_info(f"Draw one rectangle ROI on {layer.name}.")

    def on_roi_data_changed(self, roi_layer: Shapes) -> None:
        """Report whether an edited ROI is ready for explicit confirmation."""
        image_layer = self.layers.roi_to_image.get(id(roi_layer))
        if image_layer is None:
            return

        if len(roi_layer.data) != 1:
            self.controls.set_status("Draw exactly one rectangle, then click Add ROI.")
            return

        if self.layers.single_shape_type(roi_layer) != "rectangle":
            self.controls.warn("The ROI must be one rectangle.")
            return

        self.controls.set_status(
            f"Rectangle ROI ready for {image_layer.name}. Click Add ROI."
        )

    def add_roi(self) -> None:
        """Create or update an ROI crop for explicit mask refinement."""
        roi_layer = self.layers.active_roi_layer()
        if roi_layer is None:
            self.controls.warn("Select an ROI rectangle layer before clicking Add ROI.")
            return

        image_layer = self.layers.roi_to_image.get(id(roi_layer))
        if image_layer is None:
            self.controls.warn("The selected ROI rectangle is not linked to an image.")
            return

        if len(roi_layer.data) != 1:
            self.controls.warn("Draw exactly one rectangle before clicking Add ROI.")
            return

        if self.layers.single_shape_type(roi_layer) != "rectangle":
            self.controls.warn("The ROI must be one rectangle.")
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
            self.controls.warn(str(exc))
            return

        existing_crop = self.layers.roi_to_crop.get(id(roi_layer))
        crop_metadata = {
            ROLE_KEY: ROLE_ROI_CROP,
            SOURCE_IMAGE_KEY: image_layer.name,
            PROCESSING_KEY: {
                "roi": {
                    "row_start": int(y_slice.start),
                    "row_stop": int(y_slice.stop),
                    "col_start": int(x_slice.start),
                    "col_stop": int(x_slice.stop),
                }
            },
            **self.layers.inherited_calibration(image_layer),
        }
        if existing_crop is not None and self.layers.is_present(existing_crop):
            existing_crop.data = cropped
            existing_crop.translate = (y_slice.start, x_slice.start)
            existing_crop.metadata.update(crop_metadata)
            crop_layer = existing_crop
        else:
            crop_layer = self.layers.viewer.add_image(
                cropped,
                name=self.layers.unique_name(f"{image_layer.name} ROI"),
                rgb=cropped.ndim == 3 and cropped.shape[-1] in (3, 4),
                translate=(y_slice.start, x_slice.start),
                metadata=crop_metadata,
            )
            self.layers.roi_to_crop[id(roi_layer)] = crop_layer

        image_layer.visible = False
        self.layers.move_to_top(crop_layer)
        self.layers.activate(crop_layer)
        self.layers.clear_mask_products(crop_layer)
        self.controls.set_gaussian_enabled(True)
        self.controls.set_threshold_enabled(False)
        self.controls.confirm_leaf_button.setEnabled(False)
        self.controls.set_leaf_area_status("Leaf area: not confirmed")
        self.controls.set_cleanup_enabled(False)
        self.controls.set_status(
            f"Created ROI crop from {image_layer.name}. "
            "Click Apply Gaussian to create the smoothed_image layer."
        )
        show_info(f"Created ROI crop from {image_layer.name}.")

