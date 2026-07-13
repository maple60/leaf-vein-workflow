"""Editable vein-order labels constrained by the confirmed mask.

This module owns napari's paint interception and clip events.  It does not
create or modify the threshold mask itself.
"""

import numpy as np
from napari.layers import Labels
from napari.utils.colormaps import direct_colormap
from napari.utils.notifications import show_info

from .controls import WorkflowControls
from .layers import (
    ORDER_LABEL_COLORS,
    ORDER_LABELS_LAYER_NAME,
    ORDER_UNCERTAIN_LABEL,
    ROLE_KEY,
    ROLE_ORDER_LABELS,
    SOURCE_CROP_KEY,
    SOURCE_IMAGE_KEY,
    SOURCE_MASK_KEY,
    VEIN_MASK_LAYER_NAME,
    WorkflowLayers,
)


class OrderLabelsWorkflow:
    """Create and protect manual order-label layers."""

    def __init__(self, layers: WorkflowLayers, controls: WorkflowControls):
        self.layers = layers
        self.controls = controls

    def create_order_labels(self) -> None:
        """Create an editable order-label layer from the confirmed mask."""
        mask_layer = self.layers.active_mask_layer()
        if mask_layer is None:
            self.controls.warn(
                f"Create or select {VEIN_MASK_LAYER_NAME} before order labels."
            )
            return

        mask_data = np.asarray(mask_layer.data)
        if not np.any(mask_data == 0):
            self.controls.warn(f"{VEIN_MASK_LAYER_NAME} has no white-out pixels.")
            return

        if mask_data.ndim != 2:
            self.controls.warn("Order labels require a 2D vein mask.")
            return
        labels = np.where(mask_data == 0, ORDER_UNCERTAIN_LABEL, 0).astype(np.uint8)

        order_layer = self.create_or_select_order_labels_layer(mask_layer, labels)
        if order_layer is None:
            return

        crop_layer = self.layers.mask_to_crop.get(id(mask_layer))
        if crop_layer is not None:
            # 手描きの次数ラベルを閾値UIの遅延更新で壊さない。
            self.layers.mask_updates_enabled.discard(id(crop_layer))

        mask_layer.visible = False
        self.layers.move_to_top(order_layer)
        self.layers.activate(order_layer)
        try:
            order_layer.mode = "paint"
        except ValueError:
            pass

        self.controls.set_status(
            f"Created {ORDER_LABELS_LAYER_NAME}. Paint labels inside the confirmed "
            f"{VEIN_MASK_LAYER_NAME}."
        )
        show_info(f"Created {ORDER_LABELS_LAYER_NAME}.")

    def create_or_select_order_labels_layer(
        self,
        mask_layer: Labels,
        labels: np.ndarray,
    ) -> Labels | None:
        """Create order labels or reuse existing manual edits unchanged."""
        existing = self.layers.mask_to_order_labels.get(id(mask_layer))
        if existing is None or not self.layers.is_present(existing):
            named_layer = self.layers.by_name(ORDER_LABELS_LAYER_NAME)
            if named_layer is not None:
                if not isinstance(named_layer, Labels):
                    self.controls.warn(
                        f"{ORDER_LABELS_LAYER_NAME} exists but is not a Labels layer."
                    )
                    return None
                if named_layer.data.shape != labels.shape:
                    self.controls.warn(
                        f"{ORDER_LABELS_LAYER_NAME} exists but has a different shape."
                    )
                    return None
                existing = named_layer

        metadata = {
            ROLE_KEY: ROLE_ORDER_LABELS,
            SOURCE_IMAGE_KEY: mask_layer.metadata.get(SOURCE_IMAGE_KEY),
            SOURCE_CROP_KEY: mask_layer.metadata.get(SOURCE_CROP_KEY),
            SOURCE_MASK_KEY: mask_layer.name,
            **self.layers.inherited_calibration(mask_layer),
        }

        if existing is not None and self.layers.is_present(existing):
            existing.translate = tuple(mask_layer.translate)
            existing.opacity = 0.8
            existing.colormap = direct_colormap(ORDER_LABEL_COLORS)
            existing.metadata.update(metadata)
            order_layer = existing
        else:
            order_layer = self.layers.viewer.add_labels(
                labels,
                name=ORDER_LABELS_LAYER_NAME,
                translate=tuple(mask_layer.translate),
                opacity=0.8,
                colormap=direct_colormap(ORDER_LABEL_COLORS),
                metadata=metadata,
            )

        self.layers.mask_to_order_labels[id(mask_layer)] = order_layer
        self.layers.order_labels_to_mask[id(order_layer)] = mask_layer
        self.install_masked_paint(order_layer)
        self.connect_mask_clip(mask_layer, order_layer)
        self.clip_to_mask(order_layer)
        return order_layer

    def install_masked_paint(self, order_layer: Labels) -> None:
        """Intercept paint writes so nonzero labels stay inside the mask."""
        order_id = id(order_layer)
        if order_id in self.layers.order_label_data_setitems:
            return

        original_data_setitem = order_layer.data_setitem
        self.layers.order_label_data_setitems[order_id] = original_data_setitem

        def masked_data_setitem(indices, value, refresh=True):
            filtered = self.filter_indices(order_layer, indices, value)
            if filtered is None:
                return None
            filtered_indices, filtered_value = filtered
            return original_data_setitem(filtered_indices, filtered_value, refresh)

        order_layer.data_setitem = masked_data_setitem

    def filter_indices(self, order_layer: Labels, indices, value):
        """Filter a napari paint write against the associated mask."""
        if self.is_background_value(value):
            return indices, value

        mask_layer = self.layers.order_labels_to_mask.get(id(order_layer))
        if mask_layer is None or not self.layers.is_present(mask_layer):
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
            filtered_indices = tuple(
                np.asarray(axis_indices)[keep] for axis_indices in indices
            )
        except (TypeError, ValueError):
            return indices, value

        value_array = np.asarray(value)
        if value_array.shape == keep.shape:
            return filtered_indices, value_array[keep]
        return filtered_indices, value

    @staticmethod
    def is_background_value(value) -> bool:
        """Return whether a paint operation erases labels with value zero."""
        value_array = np.asarray(value)
        return value_array.shape == () and int(value_array) == 0

    def connect_mask_clip(
        self,
        mask_layer: Labels,
        order_layer: Labels,
    ) -> None:
        """Connect each order and mask event exactly once."""
        order_id = id(order_layer)
        if order_id not in self.layers.order_label_clip_connected:
            order_layer.events.data.connect(
                lambda event, order_layer=order_layer: self.clip_to_mask(order_layer)
            )
            order_layer.events.labels_update.connect(
                lambda event, order_layer=order_layer: self.clip_to_mask(order_layer)
            )
            self.layers.order_label_clip_connected.add(order_id)

        mask_id = id(mask_layer)
        if mask_id not in self.layers.mask_clip_connected:
            mask_layer.events.data.connect(
                lambda event, mask_layer=mask_layer: self.clip_for_mask(mask_layer)
            )
            self.layers.mask_clip_connected.add(mask_id)

    def clip_for_mask(self, mask_layer: Labels) -> None:
        """Clip the order layer after its mask changes."""
        order_layer = self.layers.mask_to_order_labels.get(id(mask_layer))
        if order_layer is not None and self.layers.is_present(order_layer):
            self.clip_to_mask(order_layer)

    def clip_to_mask(self, order_layer: Labels) -> None:
        """Clear nonzero order labels outside the associated mask."""
        order_id = id(order_layer)
        if order_id in self.layers.clipping_order_labels:
            return

        mask_layer = self.layers.order_labels_to_mask.get(order_id)
        if mask_layer is None or not self.layers.is_present(mask_layer):
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
        # data代入で同じeventが再発火するため、layer idで再入を抑止する。
        self.layers.clipping_order_labels.add(order_id)
        try:
            order_layer.data = clipped
        finally:
            self.layers.clipping_order_labels.discard(order_id)

