"""Auto-assign 1° and 2° veins inside the order-label layer.

The user marks the leaf base and the apex of the 1° on a Points layer.  The
algorithm itself lives in :mod:`leaf_vein_workflow.auto_order` and touches no
napari object, so it can run in a worker thread.  Its result only replaces
pixels that are still uncertain (4), so labels painted by hand are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from napari.layers import Labels, Points
from qtpy.QtCore import QTimer

from ..auto_order import AutoOrderParams, AutoOrderResult
from ..calibration import CALIBRATION_MM_PER_PIXEL_KEY
from .controls import WorkflowControls
from .layers import (
    ORDER_LABELS_LAYER_NAME,
    ORDER_UNCERTAIN_LABEL,
    ROLE_KEY,
    ROLE_ORDER_SEEDS,
    SOURCE_IMAGE_KEY,
    WorkflowLayers,
)

ORDER_SEEDS_LAYER_NAME = "base & apex"
SEED_NAMES = ("base", "apex")


@dataclass(slots=True)
class AutoOrderJob:
    """Copies of the inputs for :func:`auto_assign_orders`."""

    order_layer: Labels
    vein_mask: np.ndarray
    leaf_mask: np.ndarray | None
    base_rc: tuple[float, float]
    apex_rc: tuple[float, float]
    mm_per_pixel: float | None
    params: AutoOrderParams

    def kwargs(self) -> dict[str, Any]:
        return {
            "vein_mask": self.vein_mask,
            "base_rc": self.base_rc,
            "apex_rc": self.apex_rc,
            "leaf_mask": self.leaf_mask,
            "mm_per_pixel": self.mm_per_pixel,
            "params": self.params,
        }


class AutoOrderWorkflow:
    """Collect the two clicks, prepare the job, and apply its result."""

    def __init__(self, layers: WorkflowLayers, controls: WorkflowControls):
        self.layers = layers
        self.controls = controls

    def mark_base_and_apex(self) -> Points | None:
        """Create or select the Points layer for the base and apex clicks."""
        order_layer = self.layers.active_order_labels_layer()
        if order_layer is None:
            self.controls.warn("Press Create Order Labels before marking base and apex.")
            return None

        seeds = self.seeds_layer(order_layer)
        if seeds is None:
            seeds = self.layers.viewer.add_points(
                np.empty((0, 2)),
                name=ORDER_SEEDS_LAYER_NAME,
                translate=tuple(order_layer.translate),
                size=max(8, int(0.01 * max(order_layer.data.shape))),
                face_color="#f6aa00",
                border_color="black",
                metadata={
                    ROLE_KEY: ROLE_ORDER_SEEDS,
                    SOURCE_IMAGE_KEY: order_layer.metadata.get(SOURCE_IMAGE_KEY),
                },
            )
            seeds.events.data.connect(lambda event, layer=seeds: self.keep_two(layer))
        seeds.visible = True
        self.layers.move_to_top(seeds)
        self.layers.activate(seeds)
        seeds.mode = "add"
        self.controls.set_status(
            "Click the leaf base (where the 1° enters the blade), then the apex."
        )
        return seeds

    def seeds_layer(self, order_layer: Labels) -> Points | None:
        for layer in self.layers.viewer.layers:
            if (
                isinstance(layer, Points)
                and layer.metadata.get(ROLE_KEY) == ROLE_ORDER_SEEDS
                and layer.metadata.get(SOURCE_IMAGE_KEY)
                == order_layer.metadata.get(SOURCE_IMAGE_KEY)
            ):
                return layer
        return None

    def keep_two(self, seeds: Points) -> None:
        """Keep the two latest clicks: the first is the base, the second the apex."""
        if len(seeds.data) > 2:
            seeds.data = np.asarray(seeds.data)[-2:]
            return
        count = len(seeds.data)
        if count == 1:
            self.controls.set_status("Base marked. Now click the apex.")
        elif count == 2:
            self.controls.set_status("Base and apex marked. Press Auto-assign 1°/2°.")
            if self.controls.activate_labels_after_marking:
                # 点の追加イベントの途中でレイヤーを切り替えないよう、次の周回で行う。
                QTimer.singleShot(0, lambda: self.back_to_labels(seeds))

    def back_to_labels(self, seeds: Points) -> None:
        """Return to painting on the order labels once both points are set."""
        if not self.layers.is_present(seeds) or len(seeds.data) != 2:
            return
        for layer in self.layers.viewer.layers:
            if (
                isinstance(layer, Labels)
                and layer.metadata.get(ROLE_KEY) == "order_labels"
                and layer.metadata.get(SOURCE_IMAGE_KEY)
                == seeds.metadata.get(SOURCE_IMAGE_KEY)
            ):
                self.layers.activate(layer)
                try:
                    layer.mode = "paint"
                except ValueError:
                    pass
                return

    def prepare(self) -> AutoOrderJob | None:
        """Validate inputs and copy the arrays needed by the algorithm."""
        order_layer = self.layers.active_order_labels_layer()
        if order_layer is None:
            self.controls.warn("Press Create Order Labels first.")
            return None
        mask_layer = self.layers.order_labels_to_mask.get(id(order_layer))
        if mask_layer is None or not self.layers.is_present(mask_layer):
            self.controls.warn(f"{ORDER_LABELS_LAYER_NAME} has no vein mask to work from.")
            return None
        seeds = self.seeds_layer(order_layer)
        if seeds is None or len(seeds.data) != 2:
            self.controls.warn("Mark the base and the apex first (two clicks).")
            return None

        crop_layer = self.layers.mask_to_crop.get(id(mask_layer))
        leaf = self.layers.leaf_masks.get(id(crop_layer)) if crop_layer is not None else None
        # Points は order_labels と同じ translate で作るので、データ座標がそのまま画素位置。
        points = np.asarray(seeds.data, dtype=float)
        mm_per_pixel = order_layer.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        return AutoOrderJob(
            order_layer=order_layer,
            vein_mask=np.array(mask_layer.data, copy=True) > 0,
            leaf_mask=None if leaf is None else np.array(leaf, copy=True),
            base_rc=(float(points[0, 0]), float(points[0, 1])),
            apex_rc=(float(points[1, 0]), float(points[1, 1])),
            mm_per_pixel=float(mm_per_pixel) if mm_per_pixel else None,
            params=AutoOrderParams(**self.controls.auto_order_options),
        )

    def apply(self, job: AutoOrderJob, result: AutoOrderResult) -> int:
        """Write 1°/2° into pixels that are still uncertain; return their count."""
        order_layer = job.order_layer
        if not self.layers.is_present(order_layer):
            return 0
        current = np.asarray(order_layer.data)
        if current.shape != result.labels.shape:
            self.controls.warn("Order labels changed shape; auto result was not applied.")
            return 0
        update = (current == ORDER_UNCERTAIN_LABEL) & np.isin(result.labels, (1, 2, 3))
        labels = current.copy()
        labels[update] = result.labels[update]
        order_layer.data = labels

        record = self.layers.processing(order_layer)
        record["auto_order"] = {
            "base_rc": [round(v, 1) for v in job.base_rc],
            "apex_rc": [round(v, 1) for v in job.apex_rc],
            "leaf_outline_from_otsu": job.leaf_mask is not None,
            "pixels_changed": int(update.sum()),
            **result.summary(),
        }
        self.layers.record_processing(order_layer, record)
        seeds = self.seeds_layer(order_layer)
        if seeds is not None:
            seeds.visible = False
        tertiary = (
            "remaining veins of 3° gauge as 3°, thinner ones Uncertain. "
            if result.tertiary_gauge_range_px is not None
            else ""
        )
        self.controls.set_status(
            f"Auto-assigned the 1°, {result.n_secondary} secondaries, and "
            f"{tertiary}Check and fix with Paint as; your painting was kept."
        )
        return int(update.sum())
