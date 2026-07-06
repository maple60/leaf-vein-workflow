"""Image cropping helpers independent of napari.

このファイルは napari や Qt には依存しない、NumPy 配列だけを扱う helper。
GUI 側の処理は `_widget.py` に置き、ここでは「ROI 座標から画像を切り抜く」
ための計算だけを担当する。
"""

from collections.abc import Sequence

import numpy as np


def _image_plane_shape(image_shape: Sequence[int]) -> tuple[int, int]:
    """対応している画像配列から、y/x 平面の高さと幅を返す。"""
    # 2 次元配列は grayscale 画像として扱う。
    # shape の意味は (height, width)。
    if len(image_shape) == 2:
        return int(image_shape[0]), int(image_shape[1])

    # 3 次元配列でも、最後の軸が 3 または 4 なら RGB/RGBA 画像として扱う。
    # shape の意味は (height, width, channels)。
    if len(image_shape) == 3 and image_shape[-1] in (3, 4):
        return int(image_shape[0]), int(image_shape[1])

    # MVP では z stack や time series のような nD 画像はまだ扱わない。
    raise ValueError("Only 2D grayscale and 2D RGB/RGBA images are supported.")


def rectangle_to_slices(
    rectangle: np.ndarray,
    image_shape: Sequence[int],
) -> tuple[slice, slice]:
    """長方形 ROI の頂点座標を、画像切り抜き用の y/x slice に変換する。"""
    # まず画像側の高さと幅を取り出す。
    # RGB/RGBA の場合でも、切り抜き範囲の判定に必要なのは y/x 平面だけ。
    height, width = _image_plane_shape(image_shape)

    # napari などから渡ってくる座標を NumPy 配列にそろえる。
    # 小数座標を扱うため、dtype は float にする。
    coords = np.asarray(rectangle, dtype=float)

    # ROI が空なら、切り抜く範囲を決められない。
    if coords.size == 0:
        raise ValueError("ROI is empty.")

    # 長方形 ROI は、少なくとも複数の y/x 座標ペアを持っている必要がある。
    # ここでは厳密に「4 点の長方形」かどうかまでは見ず、座標範囲から
    # bounding box を作る方針にしている。
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
        raise ValueError("ROI must contain at least two y/x coordinate pairs.")

    # napari の Shapes data は、状況によって追加次元を含む可能性がある。
    # この MVP では先頭 2 列を y/x として使い、それ以外は無視する。
    coords = coords[:, :2]

    # NaN や inf が混ざると min/max が信用できないため、先に止める。
    if not np.all(np.isfinite(coords)):
        raise ValueError("ROI contains non-finite coordinates.")

    # ROI 座標は小数になりうる。最小値は floor、最大値は ceil にして、
    # ROI に少しでも含まれる画素を取りこぼしにくくする。
    y_min = int(np.floor(np.min(coords[:, 0])))
    y_max = int(np.ceil(np.max(coords[:, 0])))
    x_min = int(np.floor(np.min(coords[:, 1])))
    x_max = int(np.ceil(np.max(coords[:, 1])))

    # ROI が画像の外にはみ出していても、画像内に重なる部分だけ切れるように
    # 0 から height/width の範囲へ切り詰める。
    y_min = max(0, min(height, y_min))
    y_max = max(0, min(height, y_max))
    x_min = max(0, min(width, x_min))
    x_max = max(0, min(width, x_max))

    # 切り詰めた結果、縦または横の幅が 0 なら画像と重なっていない。
    if y_min >= y_max or x_min >= x_max:
        raise ValueError("ROI does not overlap the image.")

    # NumPy の slice は終点を含まない。たとえば slice(1, 4) は 1, 2, 3 行目。
    return slice(y_min, y_max), slice(x_min, x_max)


def crop_to_rectangle(image: np.ndarray, rectangle: np.ndarray) -> np.ndarray:
    """対応している画像配列を、長方形 ROI の範囲で切り抜く。"""
    # 入力が list などでも扱えるように NumPy 配列へそろえる。
    image_array = np.asarray(image)

    # ROI 座標から、NumPy の indexing に使える y/x slice を作る。
    y_slice, x_slice = rectangle_to_slices(rectangle, image_array.shape)

    # grayscale 画像は (y, x) の 2 軸だけを切る。
    if image_array.ndim == 2:
        return image_array[y_slice, x_slice].copy()

    # RGB/RGBA 画像は最後の色チャンネル軸をそのまま残す。
    # copy() して、元画像の view ではなく独立した crop 配列として返す。
    return image_array[y_slice, x_slice, :].copy()
