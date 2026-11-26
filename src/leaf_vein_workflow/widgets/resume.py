"""Restore a validated export as a separate editable session.

SavedAnalysis の配列検証は上位の resume.py が担当する。このモジュールは
GUI スレッド上で、画像 → ROI → マスク → 次数ラベル → 操作欄の順に復元する。
Otsu や自動次数判定は再実行しない。保存済みの手修正をそのまま戻すためである。
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image, Labels
from napari.layers._source import layer_source
from napari.utils.colormaps import direct_colormap
from qtpy.QtCore import QTimer

from ..auto_order import AutoOrderParams
from ..calibration import CALIBRATION_MM_PER_PIXEL_KEY
from ..provenance import array_digest
from ..resume import SavedAnalysis
from ..threshold import smooth_image
from .controls import WorkflowControls
from .layers import (
    LEAF_MASK_COLORS, LEAF_MASK_OPACITY, ORDER_LABEL_COLORS, ORDER_LABELS_OPACITY,
    ROLE_KEY, ROLE_LEAF_MASK, ROLE_ORDER_LABELS,
    SOURCE_IMAGE_KEY, SOURCE_CROP_KEY, SOURCE_MASK_KEY, WorkflowLayers,
)

if TYPE_CHECKING:
    from .auto_order import AutoOrderWorkflow
    from .mask import MaskWorkflow
    from .order_labels import OrderLabelsWorkflow
    from .roi import RoiWorkflow


class ResumeWorkflow:
    """Coordinate restoration through explicit, stage-specific dependencies.

    CropWidget 全体を受け取らず、必要な処理だけを引数に並べる。
    どの工程を呼ぶのか、コンストラクタから読み取れるようにする。
    """

    def __init__(
        self, *, layers: WorkflowLayers, controls: WorkflowControls,
        roi: RoiWorkflow, mask: MaskWorkflow, orders: OrderLabelsWorkflow,
        auto_order: AutoOrderWorkflow,
    ):
        self.layers = layers
        self.controls = controls
        self.roi = roi
        self.mask = mask
        self.orders = orders
        self.auto_order = auto_order

    def restore(self, saved: SavedAnalysis) -> Labels:
        """Build layers first, then restore controls and the normal painting view."""
        processing = dict(saved.processing)
        processing["resumed_export"] = {
            "metadata": str(saved.metadata_path.resolve()),
            "mask_edited_at_export": processing.get("mask_edited_by_hand_after_last_step"),
        }
        # 再開後の手修正は、読み込んだマスクを基準に追跡する。
        # 読み込み前の編集履歴は resumed_export に分けて残す。
        processing["auto_mask_sha256"] = array_digest(saved.vein)
        source = None
        try:
            source = self._create_source(saved)
            crop, smoothed = self._restore_roi(source, saved, processing)
            order = self._restore_masks(crop, source, saved, processing)
            self._restore_auto_order(order, saved.auto_order)
            self._restore_processing_controls(smoothed, saved.sigma, processing)
            self._restore_measurement_controls(saved)
            self.orders.show_for_painting(order)
            self.mask.refresh_leaf_status()
            self.controls.set_threshold_enabled(True)
            self.controls.set_cleanup_enabled(True)
            self.controls.open_section("orders")
            self.controls.set_status(
                "Resumed saved labels. Continue painting, then export to a new folder."
            )
            # 元画像の挿入処理も ROI の選択を予約する。その後に塗り分け層へ戻す。
            QTimer.singleShot(
                0, lambda: self.layers.activate(order) if self.layers.is_present(order) else None
            )
            return order
        except Exception:
            # 途中で失敗した場合、今回作ったレイヤーだけを除く。他の解析は触らない。
            if source is not None and self.layers.is_present(source):
                self.layers.reset_source(source)
            raise

    def _create_source(self, saved: SavedAnalysis) -> Image:
        # napari に読み込み元を渡しておくと、次のExportでも元画像のパスを記録できる。
        with layer_source(path=str(saved.source_path.resolve())):
            return self.layers.viewer.add_image(
                saved.image,
                name=self.layers.unique_name(saved.record.get("image_name") or saved.source_path.stem),
                rgb=saved.image.ndim == 3, metadata=saved.calibration.copy(),
            )

    def _restore_roi(self, source: Image, saved: SavedAnalysis, processing: dict) -> tuple[Image, Image]:
        roi = next(
            layer for layer in self.layers.viewer.layers
            if self.layers.roi_to_image.get(id(layer)) is source
        )
        row_start, row_stop, col_start, col_stop = saved.roi
        roi.add_rectangles(np.array([[
            [row_start, col_start], [row_start, col_stop],
            [row_stop, col_stop], [row_stop, col_start],
        ]], dtype=float))
        self.layers.activate(roi)
        self.roi.add_roi()
        crop = self.layers.roi_to_crop[id(roi)]
        self.layers.record_processing(crop, processing)
        # 表示用の平滑化画像だけを元のsigmaで作る。保存済みマスクは再計算しない。
        smoothed = self.mask.create_or_update_smoothed_layer(
            crop, smooth_image(crop.data, saved.sigma)
        )
        self.layers.record_processing(smoothed, processing)
        return crop, smoothed

    def _restore_masks(self, crop: Image, source: Image, saved: SavedAnalysis, processing: dict) -> Labels:
        common = {
            SOURCE_IMAGE_KEY: source.name, SOURCE_CROP_KEY: crop.name,
            **saved.calibration,
        }
        leaf = self.layers.viewer.add_labels(
            saved.leaf.copy(), name=self.layers.unique_name("leaf_mask"),
            translate=tuple(crop.translate), opacity=LEAF_MASK_OPACITY,
            colormap=direct_colormap(LEAF_MASK_COLORS),
            metadata={**common, ROLE_KEY: ROLE_LEAF_MASK},
        )
        self.layers.register_leaf(crop, leaf)
        self.layers.confirm_leaf_region(crop, saved.leaf)
        self.layers.record_processing(leaf, processing)
        mask = self.mask.create_or_update_mask_layer(crop, saved.vein.copy())
        self.layers.record_processing(mask, processing)
        order = self.layers.viewer.add_labels(
            saved.labels.copy(), name=self.layers.unique_name("order_labels"),
            translate=tuple(crop.translate), opacity=ORDER_LABELS_OPACITY,
            colormap=direct_colormap(ORDER_LABEL_COLORS),
            metadata={**common, ROLE_KEY: ROLE_ORDER_LABELS, SOURCE_MASK_KEY: mask.name},
        )
        # 先にこの解析との対応を登録し、他の葉の同名レイヤーを再利用させない。
        self.layers.register_order_labels(mask, order)
        self.orders.create_or_select_order_labels_layer(mask, saved.labels)
        return order

    def _restore_auto_order(self, order: Labels, auto: dict) -> None:
        if auto.get("params"):
            names = {field.name for field in fields(AutoOrderParams)}
            self.controls.auto_order_options = {
                key: value for key, value in auto["params"].items() if key in names
            }
        self.layers.record_processing(order, {"auto_order": auto} if auto else {})
        self.layers.activate(order)
        if "base_rc" in auto and "apex_rc" in auto:
            seeds = self.auto_order.mark_base_and_apex()
            seeds.data = np.array([auto["base_rc"], auto["apex_rc"]], dtype=float)
            seeds.visible = False

    def _restore_processing_controls(self, smoothed: Image, sigma: float, processing: dict) -> None:
        self.controls.gaussian_sigma_spinbox.setValue(sigma)
        self.mask.set_threshold_range_from_image(smoothed.data)
        threshold = processing.get("threshold") or {}
        if "value" in threshold:
            self.mask.set_threshold_value(float(threshold["value"]), update_mask=False)
        checkbox = self.controls.invert_threshold_checkbox
        # 復元中の値変更で、保存済みの手描きマスクを閾値から作り直さない。
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(threshold.get("invert", processing.get("otsu_vein_side") == "dark")))
        checkbox.blockSignals(False)
        self.controls.largest_leaf_checkbox.setChecked(
            bool((processing.get("leaf_mask") or {}).get("largest_component_only", False))
        )
        cleanup = processing.get("cleanup") or []
        if cleanup:
            for name, spinbox in (
                ("small_object_px", self.controls.min_object_size_spinbox),
                ("max_hole_px", self.controls.max_hole_size_spinbox),
                ("closing_radius_px", self.controls.closing_radius_spinbox),
                ("opening_radius_px", self.controls.opening_radius_spinbox),
            ):
                if name in cleanup[-1]:
                    spinbox.setValue(int(cleanup[-1][name]))

    def _restore_measurement_controls(self, saved: SavedAnalysis) -> None:
        area = saved.leaf_area
        # 外部面積の表示単位をmm²にそろえるが、面積の値と出どころは変えない。
        self.controls.leaf_area_unit_combo.setCurrentIndex(
            self.controls.leaf_area_unit_combo.findData("mm2")
        )
        self.controls.leaf_area_input.setText(str(area["mm2"]) if area.get("mm2") is not None else "")
        self.controls.leaf_area_source_combo.setCurrentIndex(
            self.controls.leaf_area_source_combo.findData(
                "leaf_mask" if area.get("source") == "leaf_mask" else "input"
            )
        )
        mm = saved.calibration.get(CALIBRATION_MM_PER_PIXEL_KEY)
        if mm is not None:
            self.controls.calibration_method_combo.setCurrentIndex(
                self.controls.calibration_method_combo.findData("mm_per_pixel")
            )
            self.controls.calibration_value_input.setText(str(mm))
