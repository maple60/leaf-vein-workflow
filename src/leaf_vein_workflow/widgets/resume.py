"""Rebuild the editable workflow from validated saved arrays."""

from dataclasses import fields

import numpy as np
from napari.layers._source import layer_source
from napari.utils.colormaps import direct_colormap
from qtpy.QtCore import QTimer

from ..calibration import CALIBRATION_MM_PER_PIXEL_KEY
from ..auto_order import AutoOrderParams
from ..provenance import array_digest
from ..threshold import smooth_image
from .layers import (
    ROLE_KEY, ROLE_LEAF_MASK, ROLE_ORDER_LABELS, SOURCE_IMAGE_KEY,
    SOURCE_CROP_KEY, SOURCE_MASK_KEY, ORDER_LABEL_COLORS, ORDER_LABELS_OPACITY,
)


def restore_analysis(widget, saved):
    """Create a separate session, leaving any existing images and edits alone."""
    layers, controls = widget._layers, widget._controls
    record = saved.record
    processing = dict(record["order_labels"].get("processing") or {})
    processing["resumed_export"] = {
        "metadata": str(saved.metadata_path.resolve()),
        "mask_edited_at_export": processing.get("mask_edited_by_hand_after_last_step"),
    }
    processing["auto_mask_sha256"] = array_digest(saved.vein)
    source = None
    try:
        with layer_source(path=str(saved.source_path.resolve())):
            source = layers.viewer.add_image(
                saved.image, name=layers.unique_name(record.get("image_name") or saved.source_path.stem),
                rgb=saved.image.ndim == 3, metadata=saved.calibration.copy(),
            )
        roi = next(layer for layer in layers.viewer.layers if layers.roi_to_image.get(id(layer)) is source)
        r0, r1, c0, c1 = saved.roi
        roi.add_rectangles(np.array([[[r0, c0], [r0, c1], [r1, c1], [r1, c0]]], dtype=float))
        layers.activate(roi)
        widget._roi_workflow.add_roi()
        crop = layers.roi_to_crop[id(roi)]
        layers.record_processing(crop, processing)
        smoothed = widget._mask_workflow.create_or_update_smoothed_layer(
            crop, smooth_image(crop.data, saved.sigma)
        )
        layers.record_processing(smoothed, processing)
        common = {
            SOURCE_IMAGE_KEY: source.name, SOURCE_CROP_KEY: crop.name,
            **saved.calibration,
        }
        leaf = layers.viewer.add_labels(
            saved.leaf.copy(), name=layers.unique_name("leaf_mask"), translate=tuple(crop.translate),
            opacity=0.35, colormap=direct_colormap({0: "transparent", 1: "#00e5ff", None: "#00e5ff"}),
            metadata={**common, ROLE_KEY: ROLE_LEAF_MASK},
        )
        layers.crop_to_leaf[id(crop)] = leaf
        layers.leaf_to_crop[id(leaf)] = crop
        layers.leaf_masks[id(crop)] = saved.leaf.astype(bool)
        layers.record_processing(leaf, processing)
        mask = widget._mask_workflow.create_or_update_mask_layer(crop, saved.vein.copy())
        layers.record_processing(mask, processing)
        # Register this session's layer before the normal setup, so the shared
        # default name cannot accidentally select another leaf's annotations.
        order = layers.viewer.add_labels(
            saved.labels.copy(), name=layers.unique_name("order_labels"),
            translate=tuple(crop.translate), opacity=ORDER_LABELS_OPACITY,
            colormap=direct_colormap(ORDER_LABEL_COLORS),
            metadata={**common, ROLE_KEY: ROLE_ORDER_LABELS, SOURCE_MASK_KEY: mask.name},
        )
        layers.mask_to_order_labels[id(mask)] = order
        widget._order_labels_workflow.create_or_select_order_labels_layer(mask, saved.labels)
        auto = record["order_labels"].get("auto_order")
        if auto and auto.get("params"):
            names = {field.name for field in fields(AutoOrderParams)}
            controls.auto_order_options = {key: value for key, value in auto["params"].items() if key in names}
        layers.record_processing(order, {"auto_order": auto} if auto else {})
        layers.activate(order)
        if auto and "base_rc" in auto and "apex_rc" in auto:
            seeds = widget._auto_order_workflow.mark_base_and_apex()
            seeds.data = np.array([auto["base_rc"], auto["apex_rc"]], dtype=float)
            seeds.visible = False
        controls.gaussian_sigma_spinbox.setValue(saved.sigma)
        widget._mask_workflow.set_threshold_range_from_image(smoothed.data)
        threshold = processing.get("threshold") or {}
        if "value" in threshold:
            widget._mask_workflow.set_threshold_value(float(threshold["value"]), update_mask=False)
        checkbox = controls.invert_threshold_checkbox
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(threshold.get("invert", processing.get("otsu_vein_side") == "dark")))
        checkbox.blockSignals(False)
        controls.largest_leaf_checkbox.setChecked(bool((processing.get("leaf_mask") or {}).get("largest_component_only", False)))
        cleanup = processing.get("cleanup") or []
        if cleanup:
            for name, spinbox in (
                ("small_object_px", controls.min_object_size_spinbox),
                ("max_hole_px", controls.max_hole_size_spinbox),
                ("closing_radius_px", controls.closing_radius_spinbox),
                ("opening_radius_px", controls.opening_radius_spinbox),
            ):
                if name in cleanup[-1]:
                    spinbox.setValue(int(cleanup[-1][name]))
        area = record.get("leaf_area") or {}
        controls.leaf_area_unit_combo.setCurrentIndex(controls.leaf_area_unit_combo.findData("mm2"))
        controls.leaf_area_input.setText(str(area["mm2"]) if area.get("mm2") is not None else "")
        controls.leaf_area_source_combo.setCurrentIndex(
            controls.leaf_area_source_combo.findData("leaf_mask" if area.get("source") == "leaf_mask" else "input")
        )
        mm = saved.calibration.get(CALIBRATION_MM_PER_PIXEL_KEY)
        if mm is not None:
            controls.calibration_method_combo.setCurrentIndex(controls.calibration_method_combo.findData("mm_per_pixel"))
            controls.calibration_value_input.setText(str(mm))
        widget._order_labels_workflow.show_for_painting(order)
        widget._mask_workflow.refresh_leaf_status()
        controls.set_threshold_enabled(True)
        controls.set_cleanup_enabled(True)
        controls.open_section("orders")
        controls.set_status("Resumed saved labels. Continue painting, then export to a new folder.")
        # The source insertion also schedules ROI selection; restore painting last.
        QTimer.singleShot(0, lambda: layers.activate(order) if layers.is_present(order) else None)
        return order
    except Exception:
        if source is not None and layers.is_present(source):
            layers.reset_source(source)
        raise
