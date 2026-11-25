"""Explicit cleanup of an extracted vein mask, constrained to its confirmed leaf.

This stage edits the existing vein mask and records cleanup provenance. It does
not segment the leaf, extract veins, or change threshold controls.
"""

import numpy as np
from napari.utils.notifications import show_info

from ..provenance import array_digest
from ..threshold import postprocess_mask
from .controls import WorkflowControls
from .layers import VEIN_MASK_LAYER_NAME, WorkflowLayers


def clean_vein_mask(layers: WorkflowLayers, controls: WorkflowControls) -> bool | None:
    """Clean the current mask after explicit morphology settings."""
    mask_layer = layers.active_mask_layer()
    if mask_layer is None:
        controls.warn(
            f"Create or select {VEIN_MASK_LAYER_NAME} before cleaning."
        )
        return

    try:
        crop = layers.mask_to_crop.get(id(mask_layer))
        leaf = layers.confirmed_leaf_mask(crop)
        cleaned = postprocess_mask(
            np.asarray(mask_layer.data),
            small_object_size=controls.min_object_size_spinbox.value(),
            max_hole_size=controls.max_hole_size_spinbox.value(),
            closing_radius=controls.closing_radius_spinbox.value(),
            opening_radius=controls.opening_radius_spinbox.value(),
        )
        cleaned = (cleaned.astype(bool) & leaf).astype(np.uint8)
    except ValueError as exc:
        controls.warn(str(exc))
        return

    record = layers.processing(mask_layer)
    edited_before = (
        record.get("auto_mask_sha256") is not None
        and array_digest(np.asarray(mask_layer.data))
        != record.get("auto_mask_sha256")
    )
    record.setdefault("cleanup", []).append(
        {
            "small_object_px": int(controls.min_object_size_spinbox.value()),
            "max_hole_px": int(controls.max_hole_size_spinbox.value()),
            "closing_radius_px": int(controls.closing_radius_spinbox.value()),
            "opening_radius_px": int(controls.opening_radius_spinbox.value()),
            "mask_edited_by_hand_before": bool(edited_before),
        }
    )
    record["auto_mask_sha256"] = array_digest(cleaned)
    mask_layer.data = cleaned
    layers.record_processing(mask_layer, record)
    crop_layer = layers.mask_to_crop.get(id(mask_layer))
    if crop_layer is not None:
        # 手修正後のmaskを閾値UIの遅延更新で上書きしない。
        layers.mask_updates_enabled.discard(id(crop_layer))

    layers.move_to_top(mask_layer)
    layers.activate(mask_layer)
    try:
        mask_layer.mode = "paint"
    except ValueError:
        pass

    controls.set_status(
        f"Cleaned {VEIN_MASK_LAYER_NAME}. Paint on this Labels layer to make "
        "manual corrections."
    )
    show_info(f"Cleaned {VEIN_MASK_LAYER_NAME}.")
    return True
