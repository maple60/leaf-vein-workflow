"""Assign 1°, 2° (and a first guess of 3°) vein orders from two clicks.

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
   dropped.
5. **Tertiary guess** (optional): the remaining veins are split by gauge.
   Segments at least as thick as a cutoff, but thinner than the median 2°
   gauge (Ellis: 3° are the thickest veins of the mesh, thinner than 2°), get
   3; thinner ones (minor veins) stay uncertain (4).  The cutoff is chosen by
   Otsu on the length-weighted log gauge unless ``tertiary_min_gauge_mm`` is
   set.  Veins near both ends of the 1° (petiole, apex) stay uncertain.

The module does not depend on napari, so it can run in a worker thread.

コード中の単位と添字:
- パラメータの距離は mm、計算中の画像上の距離・直径は px。
- rc は画像の (row, column)、node_ids は骨格グラフのノード番号。
- primary_position_index は主脈をたどる配列上の位置であり、累積距離ではない。
  名前の整理で判定式やしきい値の意味を変えないよう、区別を明示する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skan import Skeleton
from skimage.morphology import skeletonize

from .skeleton import drop_isolated_pixels

PRIMARY, SECONDARY, TERTIARY, UNCERTAIN = 1, 2, 3, 4
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
    assign_tertiary: bool = True
    # None: 残りの区間の太さ（長さで重み付け）に Otsu をかけて決める。
    tertiary_min_gauge_mm: float | None = None


@dataclass(slots=True)
class AutoOrderResult:
    """Order labels inside the vein mask plus a summary for metadata."""

    labels: np.ndarray
    primary_length_px: float
    n_secondary: int
    n_rejected: int
    params: dict[str, Any] = field(default_factory=dict)
    tertiary_gauge_range_px: tuple[float, float] | None = None

    def summary(self) -> dict[str, Any]:
        summary = {
            "primary_length_px": round(self.primary_length_px, 1),
            "n_secondary": self.n_secondary,
            "n_rejected_merging_into_thicker": self.n_rejected,
            "params": self.params,
        }
        if self.tertiary_gauge_range_px is not None:
            low, high = self.tertiary_gauge_range_px
            summary["tertiary_gauge_range_px"] = [round(low, 2), round(high, 2)]
        return summary


def auto_assign_orders(
    vein_mask: np.ndarray,
    base_rc: tuple[float, float],
    apex_rc: tuple[float, float],
    *,
    leaf_mask: np.ndarray | None = None,
    mm_per_pixel: float | None = None,
    params: AutoOrderParams | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> AutoOrderResult:
    """Return labels with 1° = 1, 2° = 2, 3° guess = 3, and other veins = 4.

    ``progress(fraction, message)`` is called as the steps advance (0 to 1).
    """
    params = params or AutoOrderParams()

    def report(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    vein = np.asarray(vein_mask) > 0
    if vein.ndim != 2 or not vein.any():
        raise ValueError("Auto order needs a 2D vein mask with vein pixels.")
    mmpp = float(mm_per_pixel) if mm_per_pixel else DEFAULT_MM_PER_PIXEL

    def mm_to_px(mm: float) -> float:
        return mm / mmpp

    report(0.02, "Skeletonizing the vein mask")
    skeleton = drop_isolated_pixels(skeletonize(vein).astype(np.uint8)).astype(bool)
    diameter_px = 2.0 * ndimage.distance_transform_edt(vein)
    report(0.15, "Building the vein graph")
    graph = _VeinGraph(skeleton, diameter_px)

    base = graph.snap(base_rc, mm_to_px(params.snap_radius_mm))
    apex = graph.snap(apex_rc, mm_to_px(params.snap_radius_mm))
    report(0.25, "Tracing the 1°")
    primary = graph.cheapest_path(base, apex)
    if primary is None:
        raise ValueError("The apex is not connected to the base through the vein mask.")

    report(0.3, "Finding 2° origins")
    geometry = _PrimaryGeometry(graph, primary, skeleton.shape)
    origins = _secondary_origins(graph, geometry, params, mm_to_px)

    if leaf_mask is not None and np.shape(leaf_mask) == vein.shape:
        # 葉身の中の明るい点（穴）も葉縁とみなされないよう、穴を埋める。
        leaf = ndimage.binary_fill_holes(np.asarray(leaf_mask) > 0)
    else:
        # 葉脈の網目を閉じて穴を埋め、葉の輪郭の代わりにする。
        leaf = ndimage.binary_fill_holes(ndimage.binary_closing(vein, iterations=8))
    margin_dist = ndimage.distance_transform_edt(leaf)[graph.rows, graph.cols]

    def trace_progress(done: int, total: int) -> None:
        report(0.35 + 0.45 * done / max(total, 1), f"Tracing 2° candidates ({done}/{total})")

    traced = _trace_secondaries(
        graph, geometry, origins, margin_dist, params, mm_to_px, trace_progress
    )
    report(0.82, "Dropping candidates that merge into a thicker 2°")
    accepted, rejected = _drop_tertiaries(graph, traced, params)

    skeleton_labels = np.zeros(skeleton.shape, np.uint8)
    skeleton_labels[skeleton] = UNCERTAIN
    tertiary_range = None
    report(0.87, "Splitting the remaining veins by gauge")
    if params.assign_tertiary and accepted:
        tertiary_range = _assign_tertiary(
            graph, geometry, primary, accepted, skeleton_labels, params, mm_to_px
        )
    report(0.93, "Painting the labels")
    labels = _spread_to_mask(skeleton_labels, vein)
    # 太い脈の中にある小さな骨格の枝に引っ張られないよう、たどった経路から
    # その場所の半径までを、その次数の本体として塗る（1° を最後に塗って優先）。
    secondary_nodes = np.concatenate(accepted) if accepted else np.empty(0, np.int64)
    for value, node_ids in ((SECONDARY, secondary_nodes), (PRIMARY, primary)):
        if node_ids.size:
            labels[_vein_body(graph, node_ids, vein)] = value
    report(1.0, "Done")
    return AutoOrderResult(
        labels=labels,
        primary_length_px=graph.length(primary),
        n_secondary=len(accepted),
        n_rejected=len(rejected),
        params={**asdict(params), "mm_per_pixel_used": mmpp},
        tertiary_gauge_range_px=tertiary_range,
    )


def primary_path_skeleton(
    mask: np.ndarray,
    endpoints_rc: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Measure one width-preferring primary path, without terminal side branches.

    Marked base/apex are snapped to the nearest primary skeleton pixels. Without
    marks, two weighted-distance sweeps estimate the trunk endpoints, favouring
    long, thick branches. This fallback assumes a single primary vein.
    """
    vein = np.asarray(mask, dtype=bool)
    skeleton = drop_isolated_pixels(skeletonize(vein).astype(np.uint8))
    result = np.zeros(vein.shape, dtype=np.uint8)
    if not np.any(skeleton):
        return result, 0.0
    graph = _VeinGraph(skeleton, 2.0 * ndimage.distance_transform_edt(vein))
    if endpoints_rc is not None:
        node_ids = []
        for row, col in endpoints_rc:
            node_ids.append(int(np.argmin((graph.rows - row)**2 + (graph.cols - col)**2)))
        start, end = node_ids
        if start == end:
            raise ValueError("Primary base and apex snap to the same skeleton pixel.")
    else:
        # Select the component with the greatest length × diameter² support.
        # Two sweeps are exact on a tree; cyclic masks need QC of this estimate.
        mean_d = 0.5 * (graph.diameter_px[graph.edge_row] + graph.diameter_px[graph.edge_col])
        support = graph.edge_len * mean_d**2
        extent = csr_matrix(
            (support, (graph.edge_row, graph.edge_col)), shape=(graph.n, graph.n)
        )
        _, components = connected_components(extent)
        scores = np.bincount(components[graph.edge_row], weights=support)
        component = components == int(np.argmax(scores))
        seed = int(np.argmax(np.where(component, graph.diameter_px, -1.0)))
        tips = np.flatnonzero(component & (np.diff(extent.indptr) == 1))
        if tips.size < 2:
            raise ValueError("Primary mask has no two endpoints; check its labels or mark base/apex.")
        distances = dijkstra(extent, indices=seed)
        start = int(tips[np.argmax(distances[tips])])
        distances = dijkstra(extent, indices=start)
        end = int(tips[np.argmax(distances[tips])])
    path = graph.cheapest_path(start, end)
    if path is None:
        raise ValueError("Primary base and apex are disconnected; repair the primary mask.")
    result[graph.rows[path], graph.cols[path]] = 1
    return result, graph.length(path)


class _VeinGraph:
    """Skeleton pixels as graph nodes, with diameters for path costs."""

    def __init__(self, skeleton: np.ndarray, diameter_px: np.ndarray):
        sk = Skeleton(skeleton.astype(np.uint8), spacing=1)
        coords = np.asarray(sk.coordinates).astype(np.int64)
        self.rows, self.cols = coords[:, 0], coords[:, 1]
        self.n = len(coords)
        self.diameter_px = diameter_px[self.rows, self.cols].astype(float)
        self.index = np.full(skeleton.shape, -1, np.int64)
        self.index[self.rows, self.cols] = np.arange(self.n)
        # 分岐点・端点で区切った区間（3次脈の太さを区間ごとに測るのに使う）
        self.segments = [np.asarray(seg, dtype=np.int64) for seg in sk.paths_list()]
        g = sk.graph.tocoo()
        self.edge_row, self.edge_col, self.edge_len = g.row, g.col, g.data
        self.cost = self.cost_matrix(np.ones(self.n, bool))

    def cost_matrix(self, allowed: np.ndarray) -> csr_matrix:
        """Width-preferring step costs, only between allowed nodes."""
        keep = allowed[self.edge_row] & allowed[self.edge_col]
        r, c = self.edge_row[keep], self.edge_col[keep]
        mean_d = np.maximum(0.5 * (self.diameter_px[r] + self.diameter_px[c]), 1.0)
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
        thick = near[self.diameter_px[near] >= 0.5 * self.diameter_px[near].max()]
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
    """Locate graph nodes relative to the primary vein.

    node_primary_index は主脈の配列添字、node_offset_px は主脈の縁からの距離、
    node_side は主脈の向きに対する左右の符号。配列添字と画素距離は別物で、
    ここでは既存の判定を保つため、添字を累積距離へ置き換えない。
    """

    def __init__(self, graph: _VeinGraph, primary: np.ndarray, shape: tuple[int, int]):
        self.primary = primary
        pr, pc = graph.rows[primary], graph.cols[primary]
        path_img = np.zeros(shape, bool)
        path_img[pr, pc] = True
        indices = np.zeros((2, *shape), dtype=np.int32)
        dist = ndimage.distance_transform_edt(~path_img, return_indices=True, indices=indices)

        # 主脈の局所半径（分岐点の太りを中央値でならす）と、進む向き。
        size = max(3, min(101, len(primary) // 3 * 2 + 1))
        self.primary_diameter_px = ndimage.median_filter(
            graph.diameter_px[primary], size=size, mode="nearest"
        )
        primary_position_index = np.full(shape, -1, np.int32)
        primary_position_index[pr, pc] = np.arange(len(primary))
        step = max(1, min(20, len(primary) // 4))
        ahead = np.clip(np.arange(len(primary)) + step, 0, len(primary) - 1)
        behind = np.clip(np.arange(len(primary)) - step, 0, len(primary) - 1)
        tangent = np.stack([pr[ahead] - pr[behind], pc[ahead] - pc[behind]], axis=1).astype(float)

        nr, nc = indices[0][graph.rows, graph.cols], indices[1][graph.rows, graph.cols]
        self.node_primary_index = primary_position_index[nr, nc]
        radius = self.primary_diameter_px[self.node_primary_index] / 2
        self.node_offset_px = dist[graph.rows, graph.cols] - radius
        to_node = np.stack([graph.rows - nr, graph.cols - nc], axis=1).astype(float)
        t = tangent[self.node_primary_index]
        self.node_side = np.sign(t[:, 0] * to_node[:, 1] - t[:, 1] * to_node[:, 0])


@dataclass(slots=True)
class _Origin:
    """A secondary candidate at its intersection with the primary's outer ring.

    node_ids は画像の座標ではなくグラフのノード番号。diameter_px は直径、
    primary_position_index は主脈上の添字の中央値なので float になる。
    """
    node_ids: np.ndarray
    diameter_px: float
    primary_position_index: float
    side: float
    centroid: np.ndarray


def _secondary_origins(
    graph: _VeinGraph, geometry: _PrimaryGeometry, params: AutoOrderParams,
    mm_to_px: Callable[[float], float],
) -> list[_Origin]:
    """Select origins by local relative diameter, excluding junction swelling."""
    inner, outer = mm_to_px(params.ring_inner_mm), mm_to_px(params.ring_outer_mm)
    in_ring = (geometry.node_offset_px > inner) & (geometry.node_offset_px <= outer)
    ring_img = np.zeros(graph.index.shape, bool)
    ring_img[graph.rows[in_ring], graph.cols[in_ring]] = True
    comp, n = ndimage.label(ring_img, structure=np.ones((3, 3)))
    if n == 0:
        return []
    node_comp = comp[graph.rows, graph.cols]

    # 主脈の両端付近（基部の先の葉柄や先端）から出るものは 2° とみなさない。
    end_margin_base = 2 * geometry.primary_diameter_px[0]
    end_margin_apex = 2 * geometry.primary_diameter_px[-1]
    n_primary = len(geometry.primary)
    candidates: list[_Origin] = []
    for label in range(1, n + 1):
        node_ids = np.flatnonzero(node_comp == label)
        # 輪を横切る枝だけ（輪の幅の6割以上の長さ）。
        if node_ids.size < 0.6 * (outer - inner):
            continue
        where = float(np.median(geometry.node_primary_index[node_ids]))
        if where < end_margin_base or where > n_primary - 1 - end_margin_apex:
            continue
        candidates.append(_Origin(
            node_ids=node_ids,
            diameter_px=float(np.median(graph.diameter_px[node_ids])),
            primary_position_index=float(np.median(geometry.node_primary_index[node_ids])),
            side=float(np.sign(np.sum(geometry.node_side[node_ids]))),
            centroid=np.array([graph.rows[node_ids].mean(), graph.cols[node_ids].mean()]),
        ))

    window = mm_to_px(params.secondary_window_mm)
    floor = mm_to_px(params.secondary_min_gauge_mm)
    chosen = []
    for origin in candidates:
        nearby = [
            other.diameter_px for other in candidates
            if other.side == origin.side and abs(other.primary_position_index - origin.primary_position_index) <= window
        ]
        if origin.diameter_px >= floor and origin.diameter_px >= params.secondary_relative_gauge * max(nearby):
            chosen.append(origin)

    # 同じ 2° が輪を2回横切っても、たどる経路が重なるので _drop_tertiaries で落ちる。
    return chosen


def _trace_secondaries(
    graph: _VeinGraph, geometry: _PrimaryGeometry, origins: list[_Origin],
    margin_dist: np.ndarray, params: AutoOrderParams,
    mm_to_px: Callable[[float], float],
    on_step: Callable[[int, int], None] | None = None,
) -> list[tuple[float, np.ndarray]]:
    """Follow each 2° origin to the margin, cut where the gauge drops abruptly."""
    cost = graph.cost_matrix(geometry.node_offset_px > 2)  # 主脈の中は通らない
    in_band = margin_dist < mm_to_px(params.margin_band_mm)
    near_margin = np.flatnonzero(in_band)
    # 帯の中は長さそのものを距離にする（延長は短い距離に限る）。
    keep = in_band[graph.edge_row] & in_band[graph.edge_col]
    band_len = csr_matrix(
        (graph.edge_len[keep], (graph.edge_row[keep], graph.edge_col[keep])),
        shape=(graph.n, graph.n),
    )
    max_extension = 2 * mm_to_px(params.margin_band_mm)
    traced = []
    for i, origin in enumerate(origins):
        if on_step is not None:
            on_step(i, len(origins))
        start = int(origin.node_ids[np.argmax(geometry.node_offset_px[origin.node_ids])])
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
        cut = _diameter_drop_index(graph, course, origin.diameter_px, params)
        if cut == course.size:
            # 葉縁の帯に入った所で止めず、帯の中で最も外側の点まで延ばす。
            tail = _to_outermost(band_len, margin_dist, course[-1], max_extension)
            course = np.concatenate([course, tail])
        else:
            course = course[:cut]
        # 付け根側も主脈の縁まで含める（主脈の内側は 1° のまま）。
        inner = origin.node_ids[geometry.node_offset_px[origin.node_ids] > 0]
        inner = inner[np.argsort(geometry.node_offset_px[inner])]
        traced.append((origin.diameter_px, np.concatenate([inner, course])))
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


def _diameter_drop_index(
    graph: _VeinGraph, course: np.ndarray, diameter_px: float, params: AutoOrderParams,
) -> int:
    """Index where the median gauge ahead falls abruptly below the one behind."""
    window = max(5, int(round(2 * diameter_px)))
    diameters_px = graph.diameter_px[course]
    if diameters_px.size < 2 * window + 1:
        return diameters_px.size
    # 区間の中央値を累積で求めるのは重いので、窓ごとの中央値を順に比べる。
    for i in range(window, diameters_px.size - window, max(1, window // 4)):
        behind = np.median(diameters_px[i - window:i])
        ahead = np.median(diameters_px[i:i + window])
        if ahead < params.continue_gauge_ratio * behind:
            return i
    return diameters_px.size


def _assign_tertiary(
    graph: _VeinGraph, geometry: _PrimaryGeometry, primary: np.ndarray,
    accepted: list[np.ndarray], skeleton_labels: np.ndarray,
    params: AutoOrderParams, mm_to_px: Callable[[float], float],
) -> tuple[float, float] | None:
    """Label remaining segments of 3° gauge as 3 on the skeleton; return the range."""
    taken = np.zeros(graph.n, bool)
    taken[primary] = True
    taken[np.concatenate(accepted)] = True
    secondary_gauge = float(np.median(graph.diameter_px[np.concatenate(accepted)]))
    # 1° の両端（葉柄や先端）にいちばん近い所の脈は判定しない
    end_zone = (geometry.node_primary_index <= 0) | (geometry.node_primary_index >= len(primary) - 1)

    gauges, lengths, segments = [], [], []
    for seg in graph.segments:
        if taken[seg].mean() > 0.5 or end_zone[seg].mean() > 0.5:
            continue
        rr, cc = graph.rows[seg], graph.cols[seg]
        diameters_px = graph.diameter_px[seg]
        # 分岐点の太りを避け、両端から「端の半径 + 2 px」より内側で測る
        middle = (np.hypot(rr - rr[0], cc - cc[0]) > diameters_px[0] / 2 + 2) & (
            np.hypot(rr - rr[-1], cc - cc[-1]) > diameters_px[-1] / 2 + 2
        )
        gauges.append(float(np.median(diameters_px[middle] if middle.sum() >= 3 else diameters_px)))
        lengths.append(graph.length(seg))
        segments.append(seg)
    if not segments:
        return None
    gauges, lengths = np.asarray(gauges), np.asarray(lengths)

    if params.tertiary_min_gauge_mm is not None:
        low = mm_to_px(params.tertiary_min_gauge_mm)
    else:
        low = _weighted_log_otsu(gauges, lengths)
    high = secondary_gauge
    for seg, diameter_px in zip(segments, gauges):
        if low <= diameter_px < high:
            keep = ~taken[seg]
            skeleton_labels[graph.rows[seg[keep]], graph.cols[seg[keep]]] = TERTIARY
    return float(low), float(high)


def _weighted_log_otsu(values: np.ndarray, weights: np.ndarray, bins: int = 120) -> float:
    """Otsu threshold on log values, each weighted by its length."""
    logs = np.log(np.maximum(values, 1e-6))
    if np.ptp(logs) == 0:
        return float(values[0])
    counts, edges = np.histogram(logs, bins=bins, weights=weights)
    centers = 0.5 * (edges[1:] + edges[:-1])
    total = counts.sum()
    w0 = np.cumsum(counts)
    w1 = total - w0
    m0 = np.cumsum(counts * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    # 閾値は「この値以上が太い側」なので、選んだ区切りの上端を返す
    return float(np.exp(edges[int(np.argmax(between)) + 1]))


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


def _vein_body(graph, node_ids: np.ndarray, vein: np.ndarray) -> np.ndarray:
    """Vein pixels within the local radius (+1 px) of a traced path."""
    path_img = np.zeros(vein.shape, bool)
    path_img[graph.rows[node_ids], graph.cols[node_ids]] = True
    radius = np.zeros(vein.shape, np.float32)
    radius[graph.rows[node_ids], graph.cols[node_ids]] = graph.diameter_px[node_ids] / 2
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
