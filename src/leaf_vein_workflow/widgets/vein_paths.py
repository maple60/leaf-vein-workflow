"""Hand-traced 1° and 2° vein paths on the whole-leaf source image.

Each path in the Shapes layer carries an ``order`` feature.  New paths take the
order selected in the dock; selecting paths and changing the order relabels
them.  Lengths are computed only at export.
"""

import numpy as np
import pandas as pd
from napari.layers import Shapes
from napari.utils.notifications import show_info

from ..calibration import CALIBRATION_MM_PER_PIXEL_KEY
from .controls import WorkflowControls
from .layers import (
    ORDER_LABEL_COLORS,
    ROLE_KEY,
    ROLE_VEIN_PATHS,
    SOURCE_IMAGE_KEY,
    VEIN_PATH_ORDER_KEY,
    WorkflowLayers,
)

PATH_SHAPE_TYPES = {"path", "line"}


class VeinPathsWorkflow:
    """Create the vein-path layer and keep path orders and colors in sync."""

    def __init__(self, layers: WorkflowLayers, controls: WorkflowControls):
        self.layers = layers
        self.controls = controls

    def current_order(self) -> int:
        """Return the order selected in the dock (1 or 2)."""
        return int(self.controls.path_order_combo.currentData())

    def draw_vein_paths(self) -> Shapes | None:
        """Create or activate the path layer of the selected source image."""
        source_image = self.layers.active_source_image()
        if source_image is None:
            self.controls.warn("Select a source image before tracing vein paths.")
            return None

        paths_layer = self.layers.image_to_vein_paths.get(id(source_image))
        if paths_layer is None or not self.layers.is_present(paths_layer):
            shape = np.shape(source_image.data)
            if getattr(source_image, "rgb", False):
                shape = shape[:-1]
            edge_width = max(2.0, 0.003 * float(min(shape[-2:])))
            order = self.current_order()
            paths_layer = self.layers.viewer.add_shapes(
                name=self.layers.unique_name(f"{source_image.name} vein paths"),
                ndim=2,
                translate=tuple(source_image.translate[-2:]),
                edge_width=edge_width,
                edge_color=ORDER_LABEL_COLORS[order],
                face_color="transparent",
                features=pd.DataFrame(
                    {VEIN_PATH_ORDER_KEY: pd.Series([], dtype=int)}
                ),
            )
            paths_layer.feature_defaults = {VEIN_PATH_ORDER_KEY: order}
            paths_layer.metadata.update(
                {
                    ROLE_KEY: ROLE_VEIN_PATHS,
                    SOURCE_IMAGE_KEY: source_image.name,
                    **self.layers.inherited_calibration(source_image),
                }
            )
            self.layers.image_to_vein_paths[id(source_image)] = paths_layer
            self.layers.vein_paths_to_image[id(paths_layer)] = source_image

        self.layers.move_to_top(paths_layer)
        self.layers.activate(paths_layer)
        paths_layer.mode = "add_path"
        self.controls.set_status(
            "Click along a vein to trace it; double-click to finish the path."
        )
        show_info("Trace veins: click vertices, double-click to finish each path.")
        return paths_layer

    def on_order_changed(self, *args) -> None:
        """Use the selected order for new paths and relabel selected paths."""
        paths_layer = self.active_paths_layer()
        if paths_layer is None:
            return
        order = self.current_order()
        paths_layer.feature_defaults = {VEIN_PATH_ORDER_KEY: order}
        paths_layer.current_edge_color = ORDER_LABEL_COLORS[order]

        selected = sorted(paths_layer.selected_data)
        if selected:
            features = paths_layer.features.copy()
            features.loc[selected, VEIN_PATH_ORDER_KEY] = order
            paths_layer.features = features
            self.recolor(paths_layer)

    def recolor(self, paths_layer: Shapes) -> None:
        """Color every path by its order feature."""
        if len(paths_layer.data) == 0:
            return
        orders = paths_layer.features[VEIN_PATH_ORDER_KEY].astype(int)
        paths_layer.edge_color = [ORDER_LABEL_COLORS.get(int(o), "#f6aa00") for o in orders]

    def active_paths_layer(self) -> Shapes | None:
        """Resolve the path layer for the active layer or source image."""
        layer = self.layers.viewer.layers.selection.active
        if isinstance(layer, Shapes) and layer.metadata.get(ROLE_KEY) == ROLE_VEIN_PATHS:
            return layer

        source_image = self.layers.active_source_image()
        if source_image is not None:
            paths_layer = self.layers.image_to_vein_paths.get(id(source_image))
            if paths_layer is not None and self.layers.is_present(paths_layer):
                return paths_layer

        candidates = [
            candidate
            for candidate in self.layers.viewer.layers
            if isinstance(candidate, Shapes)
            and candidate.metadata.get(ROLE_KEY) == ROLE_VEIN_PATHS
        ]
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def collect(paths_layer: Shapes) -> tuple[list[np.ndarray], list[int]]:
        """Return traced vertices and their orders, skipping non-path shapes."""
        orders = paths_layer.features[VEIN_PATH_ORDER_KEY].astype(int).tolist()
        paths: list[np.ndarray] = []
        path_orders: list[int] = []
        for data, shape_type, order in zip(
            paths_layer.data, paths_layer.shape_type, orders
        ):
            if shape_type in PATH_SHAPE_TYPES:
                paths.append(np.asarray(data, dtype=float))
                path_orders.append(int(order))
        return paths, path_orders

    @staticmethod
    def pixel_size_mm(paths_layer: Shapes) -> float | None:
        """Return the inherited calibration of the path layer."""
        value = paths_layer.metadata.get(CALIBRATION_MM_PER_PIXEL_KEY)
        return float(value) if value else None
