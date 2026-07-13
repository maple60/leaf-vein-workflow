"""Skeletonization helpers independent of napari."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skan import Skeleton
from skan.csr import summarize
from skimage.morphology import dilation, disk, medial_axis, skeletonize

from .calibration import millimetres_per_pixel


BRANCH_TYPE_LABELS = {
    0: "tip-tip",
    1: "tip-junction",
    2: "junction-junction",
    3: "cycle",
}

BRANCH_TABLE_COLUMNS = [
    "skeleton-id",
    "node-id-src",
    "node-id-dst",
    "branch-distance",
    "branch-type",
    "mean-pixel-value",
    "stdev-pixel-value",
    "image-coord-src-0",
    "image-coord-src-1",
    "image-coord-dst-0",
    "image-coord-dst-1",
    "coord-src-0",
    "coord-src-1",
    "coord-dst-0",
    "coord-dst-1",
    "euclidean-distance",
    "branch_id",
    "branch-type-label",
    "mean-diameter-px",
    "median-diameter-px",
    "max-diameter-px",
    "spur-candidate",
]

NODE_TABLE_COLUMNS = [
    "node-id",
    "coord-y-px",
    "coord-x-px",
    "degree",
    "node-type",
]

BRANCH_MM_COLUMN_MAP = {
    "branch-distance": "branch-distance-mm",
    "euclidean-distance": "euclidean-distance-mm",
    "mean-diameter-px": "mean-diameter-mm",
    "median-diameter-px": "median-diameter-mm",
    "max-diameter-px": "max-diameter-mm",
    "image-coord-src-0": "image-coord-src-0-mm",
    "image-coord-src-1": "image-coord-src-1-mm",
    "image-coord-dst-0": "image-coord-dst-0-mm",
    "image-coord-dst-1": "image-coord-dst-1-mm",
    "coord-src-0": "coord-src-0-mm",
    "coord-src-1": "coord-src-1-mm",
    "coord-dst-0": "coord-dst-0-mm",
    "coord-dst-1": "coord-dst-1-mm",
}

NODE_MM_COLUMN_MAP = {
    "coord-y-px": "coord-y-mm",
    "coord-x-px": "coord-x-mm",
}


@dataclass(slots=True)
class SkeletonHelperResult:
    """Computed helper images and tables for manual vein-order labeling."""

    mask: np.ndarray
    skeleton: np.ndarray
    skeleton_display: np.ndarray
    radius_px: np.ndarray
    diameter_px: np.ndarray
    diameter_on_skeleton: np.ndarray
    diameter_display: np.ndarray
    branch_table: pd.DataFrame
    node_table: pd.DataFrame
    endpoint_coordinates: np.ndarray
    branchpoint_coordinates: np.ndarray
    spur_paths: list[np.ndarray]


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 0/1 skeleton image from a 2D binary or label mask."""
    foreground = _as_2d_foreground(mask)
    return skeletonize(foreground).astype(np.uint8)


def create_skeleton_helpers(
    mask: np.ndarray,
    spur_length_px: float = 20.0,
    display_radius: int = 3,
    mm_per_pixel: float | None = None,
) -> SkeletonHelperResult:
    """Create skeleton, diameter, branch, node, and spur helper outputs."""
    foreground = _as_2d_foreground(mask)
    skeleton = skeletonize(foreground).astype(np.uint8)
    skeleton_bool = skeleton.astype(bool)

    _medial_skeleton, radius_px = medial_axis(foreground, return_distance=True)
    radius_px = radius_px.astype(float, copy=False)
    diameter_px = radius_px * 2.0
    diameter_on_skeleton = np.where(skeleton_bool, diameter_px, np.nan)

    display_radius = max(0, int(display_radius))
    if display_radius > 0:
        footprint = disk(display_radius)
        skeleton_display_bool = dilation(skeleton_bool, footprint=footprint)
        diameter_seed = np.where(skeleton_bool, diameter_px, 0.0)
        diameter_display_values = dilation(diameter_seed, footprint=footprint)
        diameter_display = np.where(
            skeleton_display_bool,
            diameter_display_values,
            np.nan,
        )
    else:
        skeleton_display_bool = skeleton_bool
        diameter_display = diameter_on_skeleton.copy()

    skeleton_graph = _make_skeleton_graph(skeleton_bool)
    branch_table = _make_branch_table_from_graph(
        skeleton_graph,
        diameter_px,
        spur_length_px=spur_length_px,
        mm_per_pixel=mm_per_pixel,
    )
    node_table = _make_node_table_from_graph(
        skeleton_graph,
        mm_per_pixel=mm_per_pixel,
    )
    spur_paths = _make_spur_paths(skeleton_graph, branch_table)

    endpoint_coordinates = _node_coordinates(node_table, "endpoint")
    branchpoint_coordinates = _node_coordinates(node_table, "branchpoint")

    return SkeletonHelperResult(
        mask=foreground,
        skeleton=skeleton,
        skeleton_display=skeleton_display_bool.astype(np.uint8),
        radius_px=radius_px,
        diameter_px=diameter_px,
        diameter_on_skeleton=diameter_on_skeleton,
        diameter_display=diameter_display,
        branch_table=branch_table,
        node_table=node_table,
        endpoint_coordinates=endpoint_coordinates,
        branchpoint_coordinates=branchpoint_coordinates,
        spur_paths=spur_paths,
    )


def make_branch_table(
    skeleton: np.ndarray,
    diameter_px: np.ndarray,
    spur_length_px: float = 20.0,
    mm_per_pixel: float | None = None,
) -> pd.DataFrame:
    """Return a skan branch table with diameter and spur-candidate columns."""
    skeleton_bool = _as_2d_skeleton(skeleton)
    diameter_array = np.asarray(diameter_px)
    if diameter_array.shape != skeleton_bool.shape:
        raise ValueError("diameter_px must have the same shape as skeleton.")

    skeleton_graph = _make_skeleton_graph(skeleton_bool)
    return _make_branch_table_from_graph(
        skeleton_graph,
        diameter_array.astype(float, copy=False),
        spur_length_px=spur_length_px,
        mm_per_pixel=mm_per_pixel,
    )


def make_node_table(
    skeleton: np.ndarray,
    mm_per_pixel: float | None = None,
) -> pd.DataFrame:
    """Return skeleton graph nodes with endpoint and branchpoint labels."""
    skeleton_bool = _as_2d_skeleton(skeleton)
    skeleton_graph = _make_skeleton_graph(skeleton_bool)
    return _make_node_table_from_graph(
        skeleton_graph,
        mm_per_pixel=mm_per_pixel,
    )


def _as_2d_foreground(mask: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(mask)

    if mask_array.ndim != 2:
        raise ValueError("Only 2D masks can be skeletonized.")

    if mask_array.size == 0:
        raise ValueError("Mask is empty.")

    return mask_array > 0


def _as_2d_skeleton(skeleton: np.ndarray) -> np.ndarray:
    skeleton_array = np.asarray(skeleton)

    if skeleton_array.ndim != 2:
        raise ValueError("Only 2D skeleton images are supported.")

    if skeleton_array.size == 0:
        raise ValueError("Skeleton is empty.")

    return skeleton_array > 0


def _make_skeleton_graph(skeleton: np.ndarray) -> Skeleton | None:
    if not np.any(skeleton):
        return None
    return Skeleton(skeleton.astype(bool), spacing=1)


def _make_branch_table_from_graph(
    skeleton_graph: Skeleton | None,
    diameter_px: np.ndarray,
    spur_length_px: float,
    mm_per_pixel: float | None,
) -> pd.DataFrame:
    pixel_size = (
        millimetres_per_pixel(mm_per_pixel)
        if mm_per_pixel is not None
        else None
    )
    if skeleton_graph is None:
        columns = list(BRANCH_TABLE_COLUMNS)
        if pixel_size is not None:
            columns.extend(BRANCH_MM_COLUMN_MAP.values())
        return pd.DataFrame(columns=columns)

    branch_table = summarize(skeleton_graph, separator="-").copy()
    branch_table["branch_id"] = np.arange(len(branch_table), dtype=int)
    branch_table["branch-type-label"] = branch_table["branch-type"].map(
        BRANCH_TYPE_LABELS
    )

    mean_diameter: list[float] = []
    median_diameter: list[float] = []
    max_diameter: list[float] = []

    for branch_id in range(skeleton_graph.n_paths):
        coords = skeleton_graph.path_coordinates(branch_id).astype(int)
        branch_diameter = diameter_px[coords[:, 0], coords[:, 1]]
        finite = branch_diameter[np.isfinite(branch_diameter)]
        if finite.size == 0:
            mean_diameter.append(float("nan"))
            median_diameter.append(float("nan"))
            max_diameter.append(float("nan"))
            continue

        mean_diameter.append(float(np.mean(finite)))
        median_diameter.append(float(np.median(finite)))
        max_diameter.append(float(np.max(finite)))

    branch_table["mean-diameter-px"] = mean_diameter
    branch_table["median-diameter-px"] = median_diameter
    branch_table["max-diameter-px"] = max_diameter
    branch_table["spur-candidate"] = (
        (branch_table["branch-type"] == 1)
        & (branch_table["branch-distance"] <= float(spur_length_px))
    )

    if pixel_size is not None:
        for pixel_column, mm_column in BRANCH_MM_COLUMN_MAP.items():
            branch_table[mm_column] = branch_table[pixel_column] * pixel_size

    return branch_table


def _make_node_table_from_graph(
    skeleton_graph: Skeleton | None,
    mm_per_pixel: float | None,
) -> pd.DataFrame:
    pixel_size = (
        millimetres_per_pixel(mm_per_pixel)
        if mm_per_pixel is not None
        else None
    )
    if skeleton_graph is None:
        columns = list(NODE_TABLE_COLUMNS)
        if pixel_size is not None:
            columns.extend(NODE_MM_COLUMN_MAP.values())
        return pd.DataFrame(columns=columns)

    coordinates = np.asarray(skeleton_graph.coordinates)
    degrees = np.asarray(skeleton_graph.degrees)
    node_table = pd.DataFrame(
        {
            "node-id": np.arange(len(degrees), dtype=int),
            "coord-y-px": coordinates[:, 0].astype(float),
            "coord-x-px": coordinates[:, 1].astype(float),
            "degree": degrees.astype(int),
        }
    )
    node_table["node-type"] = "path"
    node_table.loc[node_table["degree"] == 1, "node-type"] = "endpoint"
    node_table.loc[node_table["degree"] >= 3, "node-type"] = "branchpoint"
    if pixel_size is not None:
        for pixel_column, mm_column in NODE_MM_COLUMN_MAP.items():
            node_table[mm_column] = node_table[pixel_column] * pixel_size
    return node_table


def _node_coordinates(node_table: pd.DataFrame, node_type: str) -> np.ndarray:
    if node_table.empty:
        return np.empty((0, 2), dtype=float)

    selected = node_table[node_table["node-type"] == node_type]
    if selected.empty:
        return np.empty((0, 2), dtype=float)

    return selected[["coord-y-px", "coord-x-px"]].to_numpy(dtype=float)


def _make_spur_paths(
    skeleton_graph: Skeleton | None,
    branch_table: pd.DataFrame,
) -> list[np.ndarray]:
    if skeleton_graph is None or branch_table.empty:
        return []

    spur_paths: list[np.ndarray] = []
    spur_rows = branch_table[branch_table["spur-candidate"]]
    for branch_id in spur_rows["branch_id"]:
        coords = skeleton_graph.path_coordinates(int(branch_id)).astype(float)
        spur_paths.append(coords[:, :2])
    return spur_paths
