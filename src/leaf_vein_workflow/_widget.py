"""napari dock widget for the ROI crop MVP.

このファイルは napari の dock widget、つまり画面右側などに表示される
小さな操作パネルを定義する。画像の切り抜き計算そのものは `crop.py` に
分け、ここでは napari のレイヤー、ボタン、イベントだけを扱う。
"""

import numpy as np
from napari.layers import Image, Labels, Shapes
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .crop import crop_to_rectangle, rectangle_to_slices
from .skeleton import skeletonize_mask
from .threshold import otsu_threshold, threshold_image


# napari の各レイヤーには metadata という辞書を持たせられる。
# ここでは「このレイヤーはプラグインが作った ROI か crop 結果か」を
# metadata に記録し、あとで通常の画像レイヤーと区別する。
ROLE_KEY = "leaf_vein_workflow_role"
ROLE_ROI_SHAPES = "roi_shapes"
ROLE_ROI_CROP = "roi_crop"
ROLE_VEIN_MASK = "vein_mask"
ROLE_VEIN_SKELETON = "vein_skeleton"
ROLE_ORDER_LABELS = "order_labels"
SOURCE_IMAGE_KEY = "source_image"
SOURCE_CROP_KEY = "source_crop"
SOURCE_MASK_KEY = "source_mask"


class CropWidget(QWidget):
    """新しく読み込まれた画像に ROI レイヤーを自動で用意する widget。"""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # ROI Shapes レイヤーと元画像 Image レイヤーの対応表。
        # napari のレイヤー名はユーザーが変更できるため、辞書のキーには
        # レイヤーオブジェクト自身を識別しやすい `id(roi_layer)` を使う。
        self._roi_to_image: dict[int, Image] = {}

        # ROI Shapes レイヤーと、そこから作った crop Image レイヤーの対応表。
        # 同じ ROI で Add ROI をもう一度押したとき、新しいレイヤーを増やさず
        # 既存の crop レイヤーを更新するために使う。
        self._roi_to_crop: dict[int, Image] = {}

        # Crop Image レイヤーと、そこから作った二値 mask Labels レイヤーの対応表。
        # threshold や invert 設定を変えたとき、既存 mask を自動更新する。
        self._crop_to_mask: dict[int, Labels] = {}
        self._mask_to_crop: dict[int, Image] = {}

        # Mask Labels レイヤーと、そこから作った skeleton / order labels の対応表。
        # order labels はユーザーが手で編集するため、同じ shape なら上書きしない。
        self._mask_to_skeleton: dict[int, Labels] = {}
        self._mask_to_order_labels: dict[int, Labels] = {}

        # Qt の部品を作る。QLabel は状態表示、QPushButton は crop 実行ボタン。
        self.status_label = QLabel(
            "Open an image with napari. A rectangle ROI layer will be created "
            "automatically."
        )
        self.status_label.setWordWrap(True)
        self.add_roi_button = QPushButton("Add ROI")
        self.otsu_button = QPushButton("Use Otsu")
        self.create_skeleton_button = QPushButton("Create Skeleton")
        self.invert_threshold_checkbox = QCheckBox("Invert threshold")
        self.threshold_label = QLabel("Mask threshold")
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.threshold_spinbox.setDecimals(6)
        self.threshold_spinbox.setSingleStep(1.0)
        self.threshold_spinbox.setValue(0.5)

        self._mask_update_timer = QTimer(self)
        self._mask_update_timer.setSingleShot(True)
        self._mask_update_timer.setInterval(200)

        # 画像がまだない段階では ROI もないので、最初はボタンを押せない。
        self.add_roi_button.setEnabled(False)
        self._set_mask_controls_enabled(False)
        self.create_skeleton_button.setEnabled(False)

        # Qt では layout に部品を追加し、その layout を QWidget に設定する。
        # ここでは上から crop 操作、mask 操作、説明ラベルの順に縦に並べる。
        layout = QVBoxLayout()
        layout.addWidget(self.add_roi_button)
        layout.addWidget(self.otsu_button)
        layout.addWidget(self.invert_threshold_checkbox)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_spinbox)
        layout.addWidget(self.create_skeleton_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        # Qt の clicked signal と、napari の layer inserted event を接続する。
        # これにより、ボタンが押されたら add_roi()、新しいレイヤーが追加されたら
        # _on_layer_inserted() が自動的に呼ばれる。
        self.add_roi_button.clicked.connect(self.add_roi)
        self.otsu_button.clicked.connect(self.use_otsu_threshold)
        self.create_skeleton_button.clicked.connect(self.create_skeleton)
        self.invert_threshold_checkbox.toggled.connect(self._schedule_mask_update)
        self.threshold_spinbox.valueChanged.connect(self._schedule_mask_update)
        self._mask_update_timer.timeout.connect(self._update_mask_from_controls)
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)

        # widget を開く前から画像が viewer に存在している場合にも対応する。
        # すでにある各レイヤーを、新しく追加された場合と同じ手順で処理する。
        for layer in list(self.viewer.layers):
            self._prepare_image_layer(layer)

    def _on_layer_inserted(self, event) -> None:
        """新しいレイヤーが追加されたときに呼ばれる napari event handler。"""
        # event.value には、追加されたレイヤーオブジェクトが入っている。
        self._prepare_image_layer(event.value)

    def _prepare_image_layer(self, layer) -> None:
        """画像レイヤーに対応する ROI Shapes レイヤーを作る。"""
        # この widget が対象にするのは napari.layers.Image だけ。
        # Shapes や Points など、画像以外のレイヤーは何もしない。
        if not isinstance(layer, Image):
            return

        # 自分で作った crop や helper 結果も Image レイヤーなので、そのままだと
        # さらに ROI レイヤーを作ってしまう。metadata で除外する。
        if layer.metadata.get(ROLE_KEY) is not None:
            return

        # すでに ROI レイヤーを作った元画像には、二重に ROI を作らない。
        if any(source is layer for source in self._roi_to_image.values()):
            return

        # ROI を描くための Shapes レイヤーを追加する。
        # ndim=2 は、現在の MVP が 2D 画像だけを対象にしているため。
        roi_layer = self.viewer.add_shapes(
            name=self._unique_layer_name(f"{layer.name} ROI rectangle"),
            ndim=2,
            edge_color="yellow",
            face_color=[1, 1, 0, 0.15],
            edge_width=2,
        )

        # metadata に、この Shapes レイヤーが本プラグインの ROI であることと、
        # どの画像名から作られたかを記録する。主な判定には _roi_to_image を使うが、
        # metadata はレイヤー一覧を見たときの自己説明にもなる。
        roi_layer.metadata[ROLE_KEY] = ROLE_ROI_SHAPES
        roi_layer.metadata[SOURCE_IMAGE_KEY] = layer.name

        # ROI レイヤーから元画像レイヤーをたどれるようにする。
        # ここで保存しておくことで、Add ROI 時に「どの画像を切るか」が分かる。
        self._roi_to_image[id(roi_layer)] = layer

        # ROI の図形データが変わったときに状態表示を更新する。
        # lambda の既定引数に roi_layer を入れているのは、後から別の ROI が増えても
        # この signal が必ず自分自身の ROI レイヤーを参照するようにするため。
        roi_layer.events.data.connect(
            lambda event, roi_layer=roi_layer: self._on_roi_data_changed(
                roi_layer
            )
        )

        # ROI レイヤーを選択し、すぐ長方形を描けるモードにする。
        self._activate_rectangle_mode(roi_layer)

        # napari 側のレイヤー追加直後には、選択状態や mode が内部処理で
        # 上書きされることがある。そのため Qt のイベントループが一度戻った後に
        # もう一度だけ長方形描画モードを設定する。
        QTimer.singleShot(
            0,
            lambda roi_layer=roi_layer: self._activate_rectangle_mode(
                roi_layer
            )
            if self._layer_is_present(roi_layer)
            else None,
        )

        # 少なくとも 1 つ ROI レイヤーができたので、Add ROI を押せるようにする。
        self.add_roi_button.setEnabled(True)

        self._set_status(
            f"Created ROI rectangle layer for {layer.name}. Draw one rectangle."
        )
        show_info(f"Draw one rectangle ROI on {layer.name}.")

    def _on_roi_data_changed(self, roi_layer: Shapes) -> None:
        """ROI が編集されたあと、切り抜き可能な状態かを表示する。"""
        # この段階では crop は実行しない。ユーザーが長方形を微調整している途中で
        # 毎回 crop レイヤーを作ると煩雑なので、Add ROI ボタンを押すまで待つ。
        image_layer = self._roi_to_image.get(id(roi_layer))
        if image_layer is None:
            return

        # MVP では ROI は 1 つだけ。0 個や 2 個以上なら、どれを切るべきか
        # 曖昧になるため、ここでは状態表示だけ更新して終了する。
        if len(roi_layer.data) != 1:
            self._set_status("Draw exactly one rectangle, then click Add ROI.")
            return

        # Shapes レイヤーには rectangle 以外の図形も入りうる。
        # この widget は長方形 crop 専用なので、rectangle 以外は警告する。
        shape_type = self._single_shape_type(roi_layer)
        if shape_type != "rectangle":
            self._warn("The ROI must be one rectangle.")
            return

        self._set_status(f"Rectangle ROI ready for {image_layer.name}. Click Add ROI.")

    def add_roi(self) -> None:
        """選択中の ROI 長方形から crop Image レイヤーを作成または更新する。"""
        # Add ROI は、現在選択されている ROI レイヤーを対象にする。
        # 画像レイヤーや通常の Shapes レイヤーが選択されている場合は処理しない。
        roi_layer = self._active_roi_layer()
        if roi_layer is None:
            self._warn("Select an ROI rectangle layer before clicking Add ROI.")
            return

        # ROI レイヤーから元画像を取得する。対応がない場合は、プラグインが
        # 管理していない ROI なので安全に止める。
        image_layer = self._roi_to_image.get(id(roi_layer))
        if image_layer is None:
            self._warn("The selected ROI rectangle is not linked to an image.")
            return

        # 実行時にも ROI 数を確認する。_on_roi_data_changed() は状態表示用なので、
        # ボタン押下時の本処理でも同じ条件を再確認しておく。
        if len(roi_layer.data) != 1:
            self._warn("Draw exactly one rectangle before clicking Add ROI.")
            return

        # rectangle 以外の Shapes が入っていた場合は、crop 計算に渡す前に止める。
        shape_type = self._single_shape_type(roi_layer)
        if shape_type != "rectangle":
            self._warn("The ROI must be one rectangle.")
            return

        try:
            # napari の ROI 座標を、NumPy 配列の切り抜きに使える slice に変換する。
            # y_slice.start と x_slice.start は、後で crop レイヤーの表示位置にも使う。
            y_slice, x_slice = rectangle_to_slices(
                np.asarray(roi_layer.data[0]),
                np.asarray(image_layer.data).shape,
            )

            # 実際の画像切り抜きは napari 非依存の helper に任せる。
            cropped = crop_to_rectangle(
                np.asarray(image_layer.data),
                np.asarray(roi_layer.data[0]),
            )
        except ValueError as exc:
            # ROI が空、画像外、未対応 shape などの場合は helper が ValueError を出す。
            # widget 側ではそれを napari の警告として表示する。
            self._warn(str(exc))
            return

        # 同じ ROI から過去に crop レイヤーを作っていれば、そのレイヤーを更新する。
        # ユーザーが ROI を修正して Add ROI を押すたびにレイヤーが増えるのを避ける。
        existing_crop = self._roi_to_crop.get(id(roi_layer))
        if existing_crop is not None and self._layer_is_present(existing_crop):
            existing_crop.data = cropped

            # crop 画像は小さい配列だが、元画像上の ROI 位置に重ねて見えるように
            # 左上座標分だけ translate する。
            existing_crop.translate = (y_slice.start, x_slice.start)
            self._move_layer_to_top(existing_crop)
            image_layer.visible = False
            self._activate_layer(existing_crop)
            self._set_mask_controls_enabled(True)
            self._set_status(
                f"Updated ROI crop layer from {image_layer.name}. "
                "The vein mask will update automatically."
            )
            self._schedule_mask_update()
            return

        # 初回は新しい Image レイヤーとして crop 結果を追加する。
        crop_layer = self.viewer.add_image(
            cropped,
            name=self._unique_layer_name(f"{image_layer.name} ROI"),

            # RGB/RGBA 画像の場合は napari に色画像として扱わせる。
            rgb=cropped.ndim == 3 and cropped.shape[-1] in (3, 4),
            translate=(y_slice.start, x_slice.start),
            metadata={
                ROLE_KEY: ROLE_ROI_CROP,
                SOURCE_IMAGE_KEY: image_layer.name,
            },
        )

        # 次回以降はこの crop レイヤーを更新できるよう、対応表に保存する。
        self._roi_to_crop[id(roi_layer)] = crop_layer
        image_layer.visible = False
        self._activate_layer(crop_layer)
        self._set_mask_controls_enabled(True)
        self._set_status(
            f"Created ROI crop layer from {image_layer.name}. "
            "Use Otsu or adjust the threshold to update the vein mask."
        )
        self._schedule_mask_update()
        show_info(f"Created ROI crop layer from {image_layer.name}.")

    def use_otsu_threshold(self) -> None:
        """現在の crop 画像から Otsu 閾値を推定して入力欄へ反映する。"""
        crop_layer = self._active_crop_layer()
        if crop_layer is None:
            self._warn("Create or select an ROI crop layer before using Otsu.")
            return

        try:
            threshold = otsu_threshold(np.asarray(crop_layer.data))
        except ValueError as exc:
            self._warn(str(exc))
            return

        self.threshold_spinbox.setValue(threshold)
        self._schedule_mask_update()
        self._set_status(
            f"Set Otsu threshold for {crop_layer.name}: {threshold:.6g}."
        )

    def _schedule_mask_update(self, *args) -> None:
        """閾値 UI の連続変更を少し待ってから mask に反映する。"""
        if self._active_crop_layer() is None:
            return
        self._mask_update_timer.start()

    def _update_mask_from_controls(self) -> None:
        """現在の crop 画像から二値 mask Labels レイヤーを作成または更新する。"""
        crop_layer = self._active_crop_layer()
        if crop_layer is None:
            return

        try:
            mask = threshold_image(
                np.asarray(crop_layer.data),
                self.threshold_spinbox.value(),
                invert=self.invert_threshold_checkbox.isChecked(),
            ).astype(np.uint8)
        except ValueError as exc:
            self._warn(str(exc))
            return

        existing_mask = self._crop_to_mask.get(id(crop_layer))
        if existing_mask is not None and self._layer_is_present(existing_mask):
            existing_mask.data = mask
            existing_mask.translate = tuple(crop_layer.translate)
            self._move_layer_to_top(existing_mask)
            self.create_skeleton_button.setEnabled(True)
            self._set_status(f"Updated vein mask from {crop_layer.name}.")
            return

        mask_layer = self.viewer.add_labels(
            mask,
            name=self._unique_layer_name(f"{crop_layer.name} vein mask"),
            translate=tuple(crop_layer.translate),
            opacity=0.5,
            metadata={
                ROLE_KEY: ROLE_VEIN_MASK,
                SOURCE_IMAGE_KEY: crop_layer.metadata.get(SOURCE_IMAGE_KEY),
                SOURCE_CROP_KEY: crop_layer.name,
            },
        )

        self._crop_to_mask[id(crop_layer)] = mask_layer
        self._mask_to_crop[id(mask_layer)] = crop_layer
        self.create_skeleton_button.setEnabled(True)
        self._set_status(f"Created vein mask from {crop_layer.name}.")
        show_info(f"Created vein mask from {crop_layer.name}.")

    def create_skeleton(self) -> None:
        """現在の vein mask から skeleton と手動分類用 label レイヤーを作る。"""
        mask_layer = self._active_mask_layer()
        if mask_layer is None:
            self._warn("Create or select a vein mask layer before skeletonizing.")
            return

        try:
            skeleton = skeletonize_mask(np.asarray(mask_layer.data))
        except ValueError as exc:
            self._warn(str(exc))
            return

        skeleton_layer = self._create_or_update_skeleton_layer(
            mask_layer,
            skeleton,
        )
        order_layer = self._create_or_reuse_order_labels_layer(mask_layer)
        self._move_layer_to_top(order_layer)
        self._move_layer_to_top(skeleton_layer)
        self._activate_layer(order_layer)
        self._set_status(
            f"Created skeleton and manual order labels from {mask_layer.name}."
        )
        show_info(f"Created skeleton and manual order labels from {mask_layer.name}.")

    def _create_or_update_skeleton_layer(
        self,
        mask_layer: Labels,
        skeleton: np.ndarray,
    ) -> Labels:
        """Mask に対応する skeleton Labels レイヤーを作成または更新する。"""
        existing_skeleton = self._mask_to_skeleton.get(id(mask_layer))
        if (
            existing_skeleton is not None
            and self._layer_is_present(existing_skeleton)
        ):
            existing_skeleton.data = skeleton
            existing_skeleton.translate = tuple(mask_layer.translate)
            return existing_skeleton

        skeleton_layer = self.viewer.add_labels(
            skeleton,
            name=self._unique_layer_name(f"{mask_layer.name} skeleton"),
            translate=tuple(mask_layer.translate),
            opacity=0.75,
            metadata={
                ROLE_KEY: ROLE_VEIN_SKELETON,
                SOURCE_IMAGE_KEY: mask_layer.metadata.get(SOURCE_IMAGE_KEY),
                SOURCE_CROP_KEY: mask_layer.metadata.get(SOURCE_CROP_KEY),
                SOURCE_MASK_KEY: mask_layer.name,
            },
        )
        self._mask_to_skeleton[id(mask_layer)] = skeleton_layer
        return skeleton_layer

    def _create_or_reuse_order_labels_layer(self, mask_layer: Labels) -> Labels:
        """手動編集済み order labels を保ったまま、必要なら新規作成する。"""
        existing_order_labels = self._mask_to_order_labels.get(id(mask_layer))
        if (
            existing_order_labels is not None
            and self._layer_is_present(existing_order_labels)
            and existing_order_labels.data.shape == mask_layer.data.shape
        ):
            existing_order_labels.translate = tuple(mask_layer.translate)
            return existing_order_labels

        order_labels = np.zeros_like(np.asarray(mask_layer.data), dtype=np.uint8)
        order_layer = self.viewer.add_labels(
            order_labels,
            name=self._unique_layer_name(f"{mask_layer.name} order labels"),
            translate=tuple(mask_layer.translate),
            opacity=0.65,
            metadata={
                ROLE_KEY: ROLE_ORDER_LABELS,
                SOURCE_IMAGE_KEY: mask_layer.metadata.get(SOURCE_IMAGE_KEY),
                SOURCE_CROP_KEY: mask_layer.metadata.get(SOURCE_CROP_KEY),
                SOURCE_MASK_KEY: mask_layer.name,
                "label_0": "unclassified/background",
                "label_1": "primary",
                "label_2": "secondary",
                "label_3": "tertiary",
                "label_4": "higher order/minor",
            },
        )
        self._mask_to_order_labels[id(mask_layer)] = order_layer
        return order_layer

    def _active_roi_layer(self) -> Shapes | None:
        """現在選択されているレイヤーが本プラグインの ROI なら返す。"""
        layer = self.viewer.layers.selection.active
        if (
            isinstance(layer, Shapes)
            and layer.metadata.get(ROLE_KEY) == ROLE_ROI_SHAPES
        ):
            return layer
        return None

    def _active_crop_layer(self) -> Image | None:
        """現在の選択状態から、本プラグインの crop Image レイヤーを返す。"""
        layer = self.viewer.layers.selection.active

        if (
            isinstance(layer, Image)
            and layer.metadata.get(ROLE_KEY) == ROLE_ROI_CROP
        ):
            return layer

        if (
            isinstance(layer, Shapes)
            and layer.metadata.get(ROLE_KEY) == ROLE_ROI_SHAPES
        ):
            crop_layer = self._roi_to_crop.get(id(layer))
            if crop_layer is not None and self._layer_is_present(crop_layer):
                return crop_layer

        if (
            isinstance(layer, Labels)
            and layer.metadata.get(ROLE_KEY) == ROLE_VEIN_MASK
        ):
            crop_layer = self._mask_to_crop.get(id(layer))
            if crop_layer is not None and self._layer_is_present(crop_layer):
                return crop_layer

        return None

    def _active_mask_layer(self) -> Labels | None:
        """現在の選択状態から、本プラグインの vein mask レイヤーを返す。"""
        layer = self.viewer.layers.selection.active

        if (
            isinstance(layer, Labels)
            and layer.metadata.get(ROLE_KEY) == ROLE_VEIN_MASK
        ):
            return layer

        if (
            isinstance(layer, Image)
            and layer.metadata.get(ROLE_KEY) == ROLE_ROI_CROP
        ):
            mask_layer = self._crop_to_mask.get(id(layer))
            if mask_layer is not None and self._layer_is_present(mask_layer):
                return mask_layer

        if (
            isinstance(layer, Shapes)
            and layer.metadata.get(ROLE_KEY) == ROLE_ROI_SHAPES
        ):
            crop_layer = self._roi_to_crop.get(id(layer))
            if crop_layer is not None and self._layer_is_present(crop_layer):
                mask_layer = self._crop_to_mask.get(id(crop_layer))
                if mask_layer is not None and self._layer_is_present(mask_layer):
                    return mask_layer

        return None

    def _activate_rectangle_mode(self, roi_layer: Shapes) -> None:
        """ROI レイヤーを選択し、長方形を追加するモードに切り替える。"""
        # napari の選択状態をいったん空にしてから ROI レイヤーだけを選択する。
        # これにより、ユーザーがそのまま画像上で長方形を描ける状態になる。
        self._activate_layer(roi_layer)
        roi_layer.mode = "add_rectangle"

    def _activate_layer(self, layer) -> None:
        """指定したレイヤーだけを選択状態にする。"""
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(layer)
        self.viewer.layers.selection.active = layer

    def _unique_layer_name(self, base_name: str) -> str:
        """既存レイヤーと衝突しない名前を返す。"""
        existing = {layer.name for layer in self.viewer.layers}
        if base_name not in existing:
            return base_name

        # napari のレイヤー名は重複しないほうが扱いやすいので、
        # `name 2`, `name 3` のように番号を付ける。
        index = 2
        while f"{base_name} {index}" in existing:
            index += 1
        return f"{base_name} {index}"

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _set_mask_controls_enabled(self, enabled: bool) -> None:
        self.threshold_label.setEnabled(enabled)
        self.threshold_spinbox.setEnabled(enabled)
        self.invert_threshold_checkbox.setEnabled(enabled)
        self.otsu_button.setEnabled(enabled)

    def _warn(self, message: str) -> None:
        """ステータス表示と napari 通知の両方で警告を出す。"""
        self._set_status(message)
        show_warning(message)

    @staticmethod
    def _single_shape_type(layer: Shapes) -> str | None:
        """Shapes レイヤーに 1 種類だけ入っている shape_type を返す。"""
        shape_type = layer.shape_type

        # napari の状態によって、shape_type は文字列の場合と配列状の場合がある。
        # 文字列ならそのまま返す。
        if isinstance(shape_type, str):
            return shape_type

        # 複数図形がある場合は shape_type も複数になるため、MVP では扱わない。
        if len(shape_type) != 1:
            return None
        return str(shape_type[0])

    def _layer_is_present(self, target_layer) -> bool:
        """対象レイヤーがまだ viewer に存在するかを確認する。"""
        return any(layer is target_layer for layer in self.viewer.layers)

    def _move_layer_to_top(self, layer) -> None:
        """指定したレイヤーをレイヤー一覧の一番上へ移動する。"""
        current_index = self.viewer.layers.index(layer)
        top_index = len(self.viewer.layers) - 1
        if current_index != top_index:
            self.viewer.layers.move(current_index, top_index)
