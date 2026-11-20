"""Assign 1° and 2° vein orders from a vein mask and two clicks.

The user marks the base and the apex of the primary vein.  From there the
rules follow the Manual of Leaf Architecture (Ellis et al., 2009):

1. **Primary**: the cheapest path on the vein skeleton from base to apex, where
   a step costs ``length / diameter**2`` so the path keeps to the widest vein.
2. **Secondary origins**: every vein that leaves the primary is measured in a
   ring just outside the primary's edge, where junction swelling no longer
   inflates the width.  A branch is a 2° candidate when its gauge is at least
   ``secondary_relative_gauge`` of the thickest branch nearby on the same side
   (secondaries thin toward the apex, so a fixed cutoff does not work).
3. **Secondary course**: each candidate is followed to the leaf margin on its
   own side with the same width-preferring cost, and the course is cut where
   its gauge drops abruptly to less than ``continue_gauge_ratio`` of the gauge
   just before (Ellis: a vein of nearly the same gauge is the same order).  A
   2° tapers gradually toward the tooth, while a detour into a thinner vein
   (for example a looping secondary in a brochidodromous leaf) drops at once.
4. **Tertiaries off the primary**: a candidate whose traced course mostly runs
   along a thicker secondary is a 3° that connects the 1° to a 2°, so it is
   dropped.  Everything that is not 1° or 2° stays uncertain (4) for manual
   painting.

The module does not depend on napari, so it can run in a worker thread.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skan import Skeleton
from skimage.morphology import skeletonize

from .skeleton import drop_isolated_pixels

PRIMARY, SECONDARY, UNCERTAIN = 1, 2, 4
# 試作は 3200 dpi の標本画像で調整した。校正値が無いときはこの解像度とみなす。
DEFAULT_MM_PER_PIXEL = 25.4 / 3200


@dataclass(slots=True)
class AutoOrderParams:
    """Tunable distances in millimetres and ratios (see module docstring)."""

    snap_radius_mm: float = 0.6
    ring_inner_mm: float = 0.05
    ring_outer_mm: float = 0.37
    secondary_window_mm: float = 1.75
    secondary_relative_gauge: float = 0.7
    secondary_min_gauge_mm: float = 0.048
    margin_band_mm: float = 0.2
    continue_gauge_ratio: float = 0.55
    max_shared_with_thicker: float = 0.5


@dataclass(slots=True)
class AutoOrderResult:
    """Order labels inside the vein mask plus a summary for metadata."""

    labels: np.ndarray
    primary_length_px: float
    n_secondary: int
    n_rejected: int
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "primary_length_px": round(self.primary_length_px, 1),
            "n_secondary": self.n_secondary,
            "n_rejected_merging_into_thicker": self.n_rejected,
            "params": self.params,
        }


def auto_assign_orders(
    vein_mask: np.ndarray,
    base_rc: tuple[float, float],
    apex_rc: tuple[float, float],
    *,
    leaf_mask: np.ndarray | None = None,
    mm_per_pixel: float | None = None,
    params: AutoOrderParams | None = None,
) -> AutoOrderResult:
    """Return labels with 1° = 1, 2° = 2, and other vein pixels = 4."""
    params = params or AutoOrderParams()
    vein = np.asarray(vein_mask) > 0
    if vein.ndim != 2 or not vein.any():
        raise ValueError("Auto order needs a 2D vein mask with vein pixels.")
    mmpp = float(mm_per_pixel) if mm_per_pixel else DEFAULT_MM_PER_PIXEL

    def px(mm: float) -> float:
        return mm / mmpp

    skeleton = drop_isolated_pixels(skeletonize(vein).astype(np.uint8)).astype(bool)
    diameter = 2.0 * ndimage.distance_transform_edt(vein)
    graph = _VeinGraph(skeleton, diameter)

    base = graph.snap(base_rc, px(params.snap_radius_mm))
    apex = graph.snap(apex_rc, px(params.snap_radius_mm))
    primary = graph.cheapest_path(base, apex)
    if primary is None:
        raise ValueError("The apex is not connected to the base through the vein mask.")

    geometry = _PrimaryGeometry(graph, primary, skeleton.shape)
    origins = _secondary_origins(graph, geometry, params, px)

    if leaf_mask is not None and np.shape(leaf_mask) == vein.shape:
        leaf = np.asarray(leaf_mask) > 0
    else:
        # 葉脈の網目を閉じて穴を埋め、葉の輪郭の代わりにする。
        leaf = ndimage.binary_fill_holes(ndimage.binary_closing(vein, iterations=8))
    margin_dist = ndimage.distance_transform_edt(leaf)[graph.rows, graph.cols]

    traced = _trace_secondaries(graph, geometry, origins, margin_dist, params, px)
    accepted, rejected = _drop_tertiaries(graph, traced, params)

    skeleton_labels = np.zeros(skeleton.shape, np.uint8)
    skeleton_labels[skeleton] = UNCERTAIN
    labels = _spread_to_mask(skeleton_labels, vein)
    # 太い脈の中にある小さな骨格の枝に引っ張られないよう、たどった経路から
    # その場所の半径までを、その次数の本体として塗る（1° を最後に塗って優先）。
    secondary_nodes = np.concatenate(accepted) if accepted else np.empty(0, np.int64)
    for value, nodes in ((SECONDARY, secondary_nodes), (PRIMARY, primary)):
        if nodes.size:
            labels[_vein_body(graph, nodes, vein)] = value
    return AutoOrderResult(
        labels=labels,
        primary_length_px=graph.length(primary),
        n_secondary=len(accepted),
        n_rejected=len(rejected),
        params={**asdict(params), "mm_per_pixel_used": mmpp},
    )


class _VeinGraph:
    """Skeleton pixels as graph nodes, with diameters for path costs."""

    def __init__(self, skeleton: np.ndarray, diameter: np.ndarray):
        sk = Skeleton(skeleton.astype(np.uint8), spacing=1)
        coords = np.asarray(sk.coordinates).astype(np.int64)
        self.rows, self.cols = coords[:, 0], coords[:, 1]
        self.n = len(coords)
        self.diameter = diameter[self.rows, self.cols].astype(float)
        self.index = np.full(skeleton.shape, -1, np.int64)
        self.index[self.rows, self.cols] = np.arange(self.n)
        g = sk.graph.tocoo()
        self.edge_row, self.edge_col, self.edge_len = g.row, g.col, g.data
        self.cost = self.cost_matrix(np.ones(self.n, bool))

    def cost_matrix(self, allowed: np.ndarray) -> csr_matrix:
        """Width-preferring step costs, only between allowed nodes."""
        keep = allowed[self.edge_row] & allowed[self.edge_col]
        r, c = self.edge_row[keep], self.edge_col[keep]
        mean_d = np.maximum(0.5 * (self.diameter[r] + self.diameter[c]), 1.0)
        return csr_matrix((self.edge_len[keep] / mean_d**2, (r, c)), shape=(self.n, self.n))

    def snap(self, point: tuple[float, float], radius: float) -> int:
        """Snap a click to the nearest pixel of the thick vein around it.

        "Thick" means at least half the widest diameter within ``radius``, so
        a click beside the midrib does not land on a thin vein, and a click at
        the apex does not jump back to a wider junction upstream.
        """
        r, c = point
        dist2 = (self.rows - r) ** 2 + (self.cols - c) ** 2
        near = np.flatnonzero(dist2 <= radius**2)
        if near.size == 0:
            return int(np.argmin(dist2))
        thick = near[self.diameter[near] >= 0.5 * self.diameter[near].max()]
        return int(thick[np.argmin(dist2[thick])])

    def cheapest_path(self, start: int, end: int, cost: csr_matrix | None = None):
        _, pred = dijkstra(
            self.cost if cost is None else cost, indices=start, return_predecessors=True
        )
        if pred[end] < 0 and end != start:
            return None
        seq = [end]
        while seq[-1] != start:
            seq.append(int(pred[seq[-1]]))
        return np.array(seq[::-1])

    def length(self, seq: np.ndarray) -> float:
        steps = np.diff(np.stack([self.rows[seq], self.cols[seq]], axis=1), axis=0)
        return float(np.linalg.norm(steps, axis=1).sum())


class _PrimaryGeometry:
    """Distance from the primary's edge and which side of it a pixel lies on."""

    def __init__(self, graph: _VeinGraph, primary: np.ndarray, shape: tuple[int, int]):
        self.primary = primary
        pr, pc = graph.rows[primary], graph.cols[primary]
        path_img = np.zeros(shape, bool)
        path_img[pr, pc] = True
        indices = np.zeros((2, *shape), dtype=np.int32)
        dist = ndimage.distance_transform_edt(~path_img, return_indices=True, indices=indices)

        # 主脈の局所半径（分岐点の太りを中央値でならす）と、進む向き。
        size = max(3, min(101, len(primary) // 3 * 2 + 1))
        self.primary_diameter = ndimage.median_filter(
            graph.diameter[primary], size=size, mode="nearest"
        )
        position = np.full(shape, -1, np.int32)
        position[pr, pc] = np.arange(len(primary))
        step = max(1, min(20, len(primary) // 4))
        ahead = np.clip(np.arange(len(primary)) + step, 0, len(primary) - 1)
        behind = np.clip(np.arange(len(primary)) - step, 0, len(primary) - 1)
        tangent = np.stack([pr[ahead] - pr[behind], pc[ahead] - pc[behind]], axis=1).astype(float)

        nr, nc = indices[0][graph.rows, graph.cols], indices[1][graph.rows, graph.cols]
        self.node_position = position[nr, nc]
        radius = self.primary_diameter[self.node_position] / 2
        self.node_offset = dist[graph.rows, graph.cols] - radius
        to_node = np.stack([graph.rows - nr, graph.cols - nc], axis=1).astype(float)
        t = tangent[self.node_position]
        self.node_side = np.sign(t[:, 0] * to_node[:, 1] - t[:, 1] * to_node[:, 0])


@dataclass(slots=True)
class _Origin:
    nodes: np.ndarray
    gauge: float
    position: float
    side: float
    centroid: np.ndarray


def _secondary_origins(graph, geometry, params, px) -> list[_Origin]:
    inner, outer = px(params.ring_inner_mm), px(params.ring_outer_mm)
    in_ring = (geometry.node_offset > inner) & (geometry.node_offset <= outer)
    ring_img = np.zeros(graph.index.shape, bool)
    ring_img[graph.rows[in_ring], graph.cols[in_ring]] = True
    comp, n = ndimage.label(ring_img, structure=np.ones((3, 3)))
    if n == 0:
        return []
    node_comp = comp[graph.rows, graph.cols]

    # 主脈の両端付近（基部の先の葉柄や先端）から出るものは 2° とみなさない。
    end_margin_base = 2 * geometry.primary_diameter[0]
    end_margin_apex = 2 * geometry.primary_diameter[-1]
    n_primary = len(geometry.primary)
    candidates: list[_Origin] = []
    for label in range(1, n + 1):
        nodes = np.flatnonzero(node_comp == label)
        # 輪を横切る枝だけ（輪の幅の6割以上の長さ）。
        if nodes.size < 0.6 * (outer - inner):
            continue
        where = float(np.median(geometry.node_position[nodes]))
        if where < end_margin_base or where > n_primary - 1 - end_margin_apex:
            continue
        candidates.append(_Origin(
            nodes=nodes,
            gauge=float(np.median(graph.diameter[nodes])),
            position=float(np.median(geometry.node_position[nodes])),
            side=float(np.sign(np.sum(geometry.node_side[nodes]))),
            centroid=np.array([graph.rows[nodes].mean(), graph.cols[nodes].mean()]),
        ))

    window = px(params.secondary_window_mm)
    floor = px(params.secondary_min_gauge_mm)
    chosen = []
    for origin in candidates:
        nearby = [
            other.gauge for other in candidates
            if other.side == origin.side and abs(other.position - origin.position) <= window
        ]
        if origin.gauge >= floor and origin.gauge >= params.secondary_relative_gauge * max(nearby):
            chosen.append(origin)

    # 同じ 2° が輪を2回横切っても、たどる経路が重なるので _drop_tertiaries で落ちる。
    return chosen


def _trace_secondaries(graph, geometry, origins, margin_dist, params, px):
    """Follow each 2° origin to the margin, cut where the gauge drops abruptly."""
    cost = graph.cost_matrix(geometry.node_offset > 2)  # 主脈の中は通らない
    in_band = margin_dist < px(params.margin_band_mm)
    near_margin = np.flatnonzero(in_band)
    # 帯の中は長さそのものを距離にする（延長は短い距離に限る）。
    keep = in_band[graph.edge_row] & in_band[graph.edge_col]
    band_len = csr_matrix(
        (graph.edge_len[keep], (graph.edge_row[keep], graph.edge_col[keep])),
        shape=(graph.n, graph.n),
    )
    max_extension = 2 * px(params.margin_band_mm)
    traced = []
    for origin in origins:
        start = int(origin.nodes[np.argmax(geometry.node_offset[origin.nodes])])
        targets = near_margin[geometry.node_side[near_margin] == origin.side]
        if targets.size == 0:
            continue
        dist, pred = dijkstra(cost, indices=start, return_predecessors=True)
        reachable = targets[np.isfinite(dist[targets])]
        if reachable.size == 0:
            continue
        seq = [int(reachable[np.argmin(dist[reachable])])]
        while seq[-1] != start:
            seq.append(int(pred[seq[-1]]))
        course = np.array(seq[::-1])
        cut = _gauge_drop(graph, course, origin.gauge, params)
        if cut == course.size:
            # 葉縁の帯に入った所で止めず、帯の中で最も外側の点まで延ばす。
            tail = _to_outermost(band_len, margin_dist, course[-1], max_extension)
            course = np.concatenate([course, tail])
        else:
            course = course[:cut]
        # 付け根側も主脈の縁まで含める（主脈の内側は 1° のまま）。
        inner = origin.nodes[geometry.node_offset[origin.nodes] > 0]
        inner = inner[np.argsort(geometry.node_offset[inner])]
        traced.append((origin.gauge, np.concatenate([inner, course])))
    return traced


def _to_outermost(band_len, margin_dist, start: int, max_length: float) -> np.ndarray:
    """Short path inside the margin band from ``start`` to its outermost node.

    The band is connected all along the leaf edge by the marginal vein, so the
    search is limited to ``max_length`` px to stay in the same tooth.
    """
    dist, pred = dijkstra(
        band_len, indices=start, return_predecessors=True, limit=max_length
    )
    reachable = np.flatnonzero(np.isfinite(dist))
    end = int(reachable[np.argmin(margin_dist[reachable])])
    seq = [end]
    while seq[-1] != start:
        seq.append(int(pred[seq[-1]]))
    return np.array(seq[::-1][1:], dtype=np.int64)


def _gauge_drop(graph, course: np.ndarray, gauge: float, params) -> int:
    """Index where the median gauge ahead falls abruptly below the one behind."""
    window = max(5, int(round(2 * gauge)))
    d = graph.diameter[course]
    if d.size < 2 * window + 1:
        return d.size
    # 区間の中央値を累積で求めるのは重いので、窓ごとの中央値を順に比べる。
    for i in range(window, d.size - window, max(1, window // 4)):
        behind = np.median(d[i - window:i])
        ahead = np.median(d[i:i + window])
        if ahead < params.continue_gauge_ratio * behind:
            return i
    return d.size


def _drop_tertiaries(graph, traced, params):
    """Drop courses that mostly run along a thicker secondary.

    These are 3° veins that connect the 1° to a 2°, or a second crossing of an
    already traced 2°.
    """
    accepted, rejected = [], []
    owned = np.zeros(graph.index.shape, bool)
    for _, seq in sorted(traced, key=lambda item: -item[0]):
        if owned.any():
            near = ndimage.binary_dilation(owned, iterations=3)
            shared = float(near[graph.rows[seq], graph.cols[seq]].mean())
        else:
            shared = 0.0
        if shared > params.max_shared_with_thicker:
            rejected.append(seq)
            continue
        accepted.append(seq)
        owned[graph.rows[seq], graph.cols[seq]] = True
    return accepted, rejected


def _vein_body(graph, nodes: np.ndarray, vein: np.ndarray) -> np.ndarray:
    """Vein pixels within the local radius (+1 px) of a traced path."""
    path_img = np.zeros(vein.shape, bool)
    path_img[graph.rows[nodes], graph.cols[nodes]] = True
    radius = np.zeros(vein.shape, np.float32)
    radius[graph.rows[nodes], graph.cols[nodes]] = graph.diameter[nodes] / 2
    indices = np.zeros((2, *vein.shape), dtype=np.int32)
    dist = ndimage.distance_transform_edt(~path_img, return_indices=True, indices=indices)
    return vein & (dist <= radius[indices[0], indices[1]] + 1)


def _spread_to_mask(skeleton_labels: np.ndarray, vein: np.ndarray) -> np.ndarray:
    """Give every vein pixel the label of its nearest skeleton pixel."""
    indices = np.zeros((2, *vein.shape), dtype=np.int32)
    ndimage.distance_transform_edt(skeleton_labels == 0, return_indices=True, indices=indices)
    labels = skeleton_labels[indices[0], indices[1]]
    labels[~vein] = 0
    labels[vein & (labels == 0)] = UNCERTAIN
    return labels.astype(np.uint8)
