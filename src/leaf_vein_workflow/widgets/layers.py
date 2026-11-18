"""Shared napari layer state for the leaf-vein workflow.

The workflow modules use this class instead of keeping separate, implicit
layer mappings.  Image processing and Qt control updates do not belong here.
"""

from typing import Any

from napari.layers import Image, Labels, Shapes

from ..calibration import CALIBRATION_METADATA_KEYS, calibration_metadata_from


ROLE_KEY = "leaf_vein_workflow_role"
ROLE_ROI_SHAPES = "roi_shapes"
ROLE_ROI_CROP = "roi_crop"
ROLE_SMOOTHED_IMAGE = "smoothed_image"
ROLE_VEIN_MASK = "vein_mask"
ROLE_ORDER_LABELS = "order_labels"
ROLE_SCALE_BAR = "scale_bar"
ROLE_SCALE_INDICATOR = "scale_indicator"
ROLE_VEIN_PATHS = "vein_paths"
VEIN_PATH_ORDER_KEY = "order"
SOURCE_IMAGE_KEY = "source_image"
SOURCE_CROP_KEY = "source_crop"
SOURCE_MASK_KEY = "source_mask"

SMOOTHED_LAYER_NAME = "smoothed_image"
VEIN_MASK_LAYER_NAME = "vein_mask"
ORDER_LABELS_LAYER_NAME = "order_labels"
ORDER_UNCERTAIN_LABEL = 4
# vein_mask は 1 = 葉脈（着色）、0 = 背景（透明）。label 1 で塗ると葉脈を足せる。
VEIN_MASK_LABEL = 1
VEIN_MASK_COLORS = {
    0: "transparent",
    1: "#00d5ff",
    None: "#00d5ff",
}
ORDER_LABEL_COLORS = {
    0: "transparent",
    1: "#ff4b00",
    2: "#005aff",
    3: "#03af7a",
    4: "#990099",
    None: "#f6aa00",
}


class WorkflowLayers:
    """Own layer relationships and common layer-list operations."""

    def __init__(self, viewer: Any):
        self.viewer = viewer

        self.roi_to_image: dict[int, Image] = {}
        self.roi_to_crop: dict[int, Image] = {}
        self.scale_bar_to_image: dict[int, Image] = {}
        self.image_to_scale_bar: dict[int, Shapes] = {}
        # 校正結果を目で確かめるための表示用バー（計測には使わない）。
        self.image_to_scale_indicator: dict[int, Shapes] = {}
        self.image_to_vein_paths: dict[int, Shapes] = {}
        self.vein_paths_to_image: dict[int, Image] = {}
        self.crop_to_smoothed: dict[int, Image] = {}
        self.smoothed_to_crop: dict[int, Image] = {}
        self.crop_to_mask: dict[int, Labels] = {}
        self.mask_to_crop: dict[int, Image] = {}
        self.mask_to_order_labels: dict[int, Labels] = {}
        self.order_labels_to_mask: dict[int, Labels] = {}

        # napariの描画フックとevent接続を重複登録しないため、layer idを保持する。
        self.order_label_data_setitems: dict[int, object] = {}
        self.order_label_clip_connected: set[int] = set()
        self.mask_clip_connected: set[int] = set()
        self.clipping_order_labels: set[int] = set()

        # Otsuを明示的に実行したcropだけ、閾値UIからmask更新を許可する。
        self.mask_updates_enabled: set[int] = set()

    def active_source_image(self) -> Image | None:
        """Resolve the selected workflow layer back to its source image."""
        layer = self.viewer.layers.selection.active

        if isinstance(layer, Image) and layer.metadata.get(ROLE_KEY) is None:
            return layer

        if isinstance(layer, Shapes):
            source = self.roi_to_image.get(id(layer))
            if source is None:
                source = self.scale_bar_to_image.get(id(layer))
            if source is None:
                source = self.vein_paths_to_image.get(id(layer))
            if source is not None and self.is_present(source):
                return source

        crop_layer = self.active_crop_layer()
        if crop_layer is None:
            return None
        for roi_id, candidate_crop in self.roi_to_crop.items():
            if candidate_crop is crop_layer:
                source = self.roi_to_image.get(roi_id)
                if source is not None and self.is_present(source):
                    return source
        return None

    def propagate_calibration(
        self,
        source_image: Image,
        metadata: dict[str, object],
    ) -> None:
        """Update calibration fields on existing layers derived from an image."""
        self._replace_calibration(source_image, metadata)

        scale_bar = self.image_to_scale_bar.get(id(source_image))
        if scale_bar is not None and self.is_present(scale_bar):
            self._replace_calibration(scale_bar, metadata)

        vein_paths = self.image_to_vein_paths.get(id(source_image))
        if vein_paths is not None and self.is_present(vein_paths):
            self._replace_calibration(vein_paths, metadata)

        for roi_id, candidate_source in self.roi_to_image.items():
            if candidate_source is not source_image:
                continue
            roi_layer = next(
                (
                    layer
                    for layer in self.viewer.layers
                    if id(layer) == roi_id
                ),
                None,
            )
            if roi_layer is not None:
                self._replace_calibration(roi_layer, metadata)

            crop_layer = self.roi_to_crop.get(roi_id)
            if crop_layer is None or not self.is_present(crop_layer):
                continue
            self._replace_calibration(crop_layer, metadata)

            smoothed_layer = self.crop_to_smoothed.get(id(crop_layer))
            if smoothed_layer is not None and self.is_present(smoothed_layer):
                self._replace_calibration(smoothed_layer, metadata)

            mask_layer = self.crop_to_mask.get(id(crop_layer))
            if mask_layer is None or not self.is_present(mask_layer):
                continue
            self._replace_calibration(mask_layer, metadata)

            order_layer = self.mask_to_order_labels.get(id(mask_layer))
            if order_layer is not None and self.is_present(order_layer):
                self._replace_calibration(order_layer, metadata)

    @staticmethod
    def inherited_calibration(layer: Any) -> dict[str, object]:
        """Return calibration metadata ready to copy to a derived layer."""
        return calibration_metadata_from(layer.metadata)

    @staticmethod
    def _replace_calibration(layer: Any, metadata: dict[str, object]) -> None:
        """Replace, rather than merge, one layer's calibration record."""
        for key in CALIBRATION_METADATA_KEYS:
            layer.metadata.pop(key, None)
        layer.metadata.update(metadata)

    def active_roi_layer(self) -> Shapes | None:
        """Return the selected ROI Shapes layer, if any."""
        layer = self.viewer.layers.selection.active
        if (
            isinstance(layer, Shapes)
            and layer.metadata.get(ROLE_KEY) == ROLE_ROI_SHAPES
        ):
            return layer
        return None

    def active_crop_layer(self) -> Image | None:
        """Resolve the selected workflow layer back to its ROI crop."""
        layer = self.viewer.layers.selection.active

        if isinstance(layer, Image) and layer.metadata.get(ROLE_KEY) == ROLE_ROI_CROP:
            return layer

        if (
            isinstance(layer, Image)
            and layer.metadata.get(ROLE_KEY) == ROLE_SMOOTHED_IMAGE
        ):
            crop_layer = self.smoothed_to_crop.get(id(layer))
            if crop_layer is not None and self.is_present(crop_layer):
                return crop_layer

        if isinstance(layer, Labels) and layer.metadata.get(ROLE_KEY) == ROLE_VEIN_MASK:
            crop_layer = self.mask_to_crop.get(id(layer))
            if crop_layer is not None and self.is_present(crop_layer):
                return crop_layer

        if (
            isinstance(layer, Labels)
            and layer.metadata.get(ROLE_KEY) == ROLE_ORDER_LABELS
        ):
            mask_layer = self.order_labels_to_mask.get(id(layer))
            if mask_layer is not None and self.is_present(mask_layer):
                crop_layer = self.mask_to_crop.get(id(mask_layer))
                if crop_layer is not None and self.is_present(crop_layer):
                    return crop_layer

        if (
            isinstance(layer, Shapes)
            and layer.metadata.get(ROLE_KEY) == ROLE_ROI_SHAPES
        ):
            crop_layer = self.roi_to_crop.get(id(layer))
            if crop_layer is not None and self.is_present(crop_layer):
                return crop_layer

        return None

    def active_intensity_layer(self) -> Image | None:
        """Return the smoothed image for the active crop, or the crop itself."""
        crop_layer = self.active_crop_layer()
        if crop_layer is None:
            return None

        smoothed_layer = self.crop_to_smoothed.get(id(crop_layer))
        if smoothed_layer is not None and self.is_present(smoothed_layer):
            return smoothed_layer
        return crop_layer

    def active_mask_layer(self) -> Labels | None:
        """Resolve the selected workflow layer to its editable vein mask."""
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Labels) and layer.metadata.get(ROLE_KEY) == ROLE_VEIN_MASK:
            return layer

        if (
            isinstance(layer, Labels)
            and layer.metadata.get(ROLE_KEY) == ROLE_ORDER_LABELS
        ):
            mask_layer = self.order_labels_to_mask.get(id(layer))
            if mask_layer is not None and self.is_present(mask_layer):
                return mask_layer

        crop_layer = self.active_crop_layer()
        if crop_layer is None:
            return None

        mask_layer = self.crop_to_mask.get(id(crop_layer))
        if mask_layer is not None and self.is_present(mask_layer):
            return mask_layer
        return None

    def activate_rectangle_mode(self, roi_layer: Shapes) -> None:
        """Select an ROI layer and put it in rectangle drawing mode."""
        self.activate(roi_layer)
        roi_layer.mode = "add_rectangle"

    def activate(self, layer: Any) -> None:
        """Make one layer the unambiguous active selection."""
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(layer)
        self.viewer.layers.selection.active = layer

    def move_to_top(self, layer: Any) -> None:
        """Move a derived layer above the other viewer layers."""
        current_index = self.viewer.layers.index(layer)
        top_index = len(self.viewer.layers) - 1
        if current_index != top_index:
            self.viewer.layers.move(current_index, top_index)

    def unique_name(self, base_name: str) -> str:
        """Return a napari layer name that is not currently in use."""
        existing = {layer.name for layer in self.viewer.layers}
        if base_name not in existing:
            return base_name

        index = 2
        while f"{base_name} {index}" in existing:
            index += 1
        return f"{base_name} {index}"

    def by_name(self, name: str):
        """Return a layer by name without exposing napari's KeyError."""
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def is_present(self, target_layer: Any) -> bool:
        """Return whether the exact layer object is still in the viewer."""
        return any(layer is target_layer for layer in self.viewer.layers)

    @staticmethod
    def single_shape_type(layer: Shapes) -> str | None:
        """Return the only shape type present in an ROI layer."""
        shape_type = layer.shape_type
        if isinstance(shape_type, str):
            return shape_type
        if len(shape_type) != 1:
            return None
        return str(shape_type[0])

    def clear_mask_products(self, crop_layer: Image) -> None:
        """Remove all derived layers below a crop after the ROI changes."""
        self.mask_updates_enabled.discard(id(crop_layer))

        smoothed_layer = self.crop_to_smoothed.pop(id(crop_layer), None)
        if smoothed_layer is not None:
            self.smoothed_to_crop.pop(id(smoothed_layer), None)
            if self.is_present(smoothed_layer):
                self.viewer.layers.remove(smoothed_layer)

        self.clear_mask_layer(crop_layer)

    def clear_mask_layer(self, crop_layer: Image) -> None:
        """Remove a crop's mask and its order-label layer."""
        mask_layer = self.crop_to_mask.pop(id(crop_layer), None)
        if mask_layer is None:
            return

        self.mask_to_crop.pop(id(mask_layer), None)
        self.clear_order_labels_layer(mask_layer)
        if self.is_present(mask_layer):
            self.viewer.layers.remove(mask_layer)

    def clear_order_labels_layer(self, mask_layer: Labels) -> None:
        """Remove order-label state owned by one mask."""
        order_layer = self.mask_to_order_labels.pop(id(mask_layer), None)
        if order_layer is not None:
            order_id = id(order_layer)
            self.order_labels_to_mask.pop(order_id, None)
            self.order_label_data_setitems.pop(order_id, None)
            self.order_label_clip_connected.discard(order_id)
            self.clipping_order_labels.discard(order_id)
            if self.is_present(order_layer):
                self.viewer.layers.remove(order_layer)
        self.mask_clip_connected.discard(id(mask_layer))
