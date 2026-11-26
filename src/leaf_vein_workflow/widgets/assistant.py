"""A small draggable guide, independent of image layers and analysis state."""

from pathlib import Path

from qtpy.QtCore import QEvent, QPoint, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPixmap, QRegion
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


STAGE_TIPS = {
    "image": (
        "画像を開き、DPIやスケールバーで\n"
        "縮尺を設定してください。"
    ),
    "roi": (
        "解析範囲を四角で囲み、\n"
        "Add ROIを押してください。"
    ),
    "mask": (
        "Apply Gaussianでノイズを減らし、\n"
        "Apply Otsuで葉のマスクを作成してください。\n"
        "その後、葉身のみを残し、Confirm Leaf & Extract Veinsで\n"
        "葉脈の抽出に進んでください。"
    ),
    "vein_mask": (
        "抽出された葉脈マスクをご確認ください。\n"
        "必要に応じて閾値やClean Vein Maskで整え、\n"
        "塗る・消す操作で修正してください。"
    ),
    "orders": (
        "次数を選んで葉脈を塗ってください。\n"
        "ブラシの太さは左上で変更できます。\n\n"
        "【ショートカット：標準設定】\n"
        "・V：選択レイヤの表示／非表示\n"
        "・右クリック：多角形の作図中に頂点を1つ戻す\n"
        "・Ctrl+Z：確定済みの塗り操作を元に戻す\n\n"
        "投げ縄（多角形）で塗る際は、Vで\n"
        "表示を切り替えると元画像を確認しやすくなります。"
    ),
    "leaf_area": (
        "VLAに使う葉面積をご確認ください。\n"
        "外部の値を使う場合は、測定範囲を合わせてください。"
    ),
    "export": (
        "出力先をご確認のうえ、\n"
        "Export Measurementsで結果を保存してください。"
    ),
    "settings": (
        "設定ファイルを開いたり、\n"
        "既定値に戻したりできます。"
    ),
}


class SpeechBubble(QFrame):
    """Paint the message box and its tail as one continuous outline."""

    TAIL_HEIGHT = 20

    def outline(self):
        bottom = self.height() - self.TAIL_HEIGHT - 0.5
        body = QPainterPath()
        body.addRoundedRect(QRectF(0.5, 0.5, self.width() - 1, bottom - 0.5), 10, 10)
        tail = QPainterPath()
        tail.moveTo(self.width() - 88, bottom - 1)
        tail.lineTo(self.width() - 56, bottom - 1)
        tail.lineTo(self.width() - 92, self.height() - 0.5)
        tail.closeSubpath()
        return body.united(tail)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor("#a8a58b"))
        painter.setBrush(QColor("#fffde4"))
        painter.drawPath(self.outline())


class FolipukaAssistant(QWidget):
    """Overlay the current window; only the mascot handles drag events."""

    dismissed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("folipuka_assistant")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._host = parent
        self._drag_offset = None
        self._positioned = False

        self.bubble = SpeechBubble(self)
        self.bubble.setObjectName("folipuka_bubble")
        self.bubble.setStyleSheet(
            "QFrame#folipuka_bubble {background: transparent; border: none;}"
            "QFrame#folipuka_bubble QLabel {color: #252525; background: transparent;}"
            "QFrame#folipuka_bubble QToolButton {color: #555; background: transparent;"
            "border: none;}"
        )
        bubble_layout = QHBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(10, 8, 10, 8 + SpeechBubble.TAIL_HEIGHT)
        self.close_button = QToolButton()
        self.close_button.setText("×")
        self.close_button.setFixedSize(16, 16)
        self.close_button.setToolTip("ガイドを非表示 / Hide guide")
        self.close_button.clicked.connect(self.dismissed.emit)
        self.message = QLabel(STAGE_TIPS["image"])
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        bubble_layout.addWidget(self.message, 1)
        bubble_layout.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignTop)

        self.mascot = QLabel(self)
        self.mascot.setObjectName("folipuka_mascot")
        pixmap = QPixmap(str(Path(__file__).parents[1] / "assets" / "folipuka.png"))
        self.mascot.setPixmap(pixmap.scaled(
            128, 128,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ))
        self.mascot.setFixedSize(128, 128)
        self.mascot.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.mascot.setCursor(Qt.CursorShape.OpenHandCursor)
        self.mascot.setToolTip("ドラッグで移動 / Drag to move")
        self.mascot.installEventFilter(self)
        self.bubble.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.bubble)
        layout.addWidget(self.mascot, alignment=Qt.AlignmentFlag.AlignRight)
        self.setFixedWidth(260)
        self.hide()

    def show_on(self, host):
        """Follow the dock into its current window, including floating docks."""
        if host is not self._host:
            self._host.removeEventFilter(self)
            self.setParent(host)
            self._host = host
            self._positioned = False
        host.installEventFilter(self)
        self._fit_text()
        if not self._positioned:
            self.move(host.width() - self.width() - 20,
                      host.height() - self.height() - 30)
            self._positioned = True
        self._clamp_position()
        self.show()
        self.raise_()

    def set_stage(self, stage, expanded=True):
        if expanded:
            self.message.setText(STAGE_TIPS[stage])
            self._fit_text()
            self._clamp_position()

    def _fit_text(self):
        """Measure wrapped text at its actual width before sizing the bubble."""
        self.ensurePolished()
        layout = self.bubble.layout()
        margins = layout.contentsMargins()
        text_width = (
            self.width() - margins.left() - margins.right()
            - self.close_button.width() - layout.spacing()
        )
        self.message.setFixedWidth(text_width)
        self.message.setFixedHeight(self.message.heightForWidth(text_width))
        layout.invalidate()
        self.bubble.setFixedHeight(layout.sizeHint().height())
        self.setFixedHeight(self.bubble.height() + self.mascot.height())
        self.layout().activate()
        layout.activate()
        self._update_mask()

    def _clamp_position(self):
        host = self.parentWidget()
        self.move(
            max(0, min(self.x(), host.width() - self.width())),
            max(0, min(self.y(), host.height() - self.height())),
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_mask()

    def _update_mask(self):
        # Blank space around the dolphin must not intercept leaf editing.
        region = QRegion(self.bubble.outline().toFillPolygon().toPolygon())
        region.translate(self.bubble.pos())
        pixmap = self.mascot.pixmap()
        if pixmap is not None and not pixmap.isNull():
            region |= QRegion(pixmap.mask()).translated(self.mascot.pos())
        self.setMask(region)

    def eventFilter(self, watched, event):
        # Qt may deliver final events while Python attributes are being cleared.
        if (watched is getattr(self, "bubble", None)
                or watched is getattr(self, "mascot", None)):
            if event.type() in (QEvent.Type.Resize, QEvent.Type.Move):
                self._update_mask()
        if watched is getattr(self, "_host", None) and event.type() == QEvent.Type.Resize:
            self._clamp_position()
        if watched is getattr(self, "mascot", None):
            if (event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton):
                self._drag_offset = event.globalPos() - self.mapToGlobal(QPoint())
                self.mascot.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.raise_()
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_offset is not None:
                target = event.globalPos() - self._drag_offset
                self.move(self.parentWidget().mapFromGlobal(target))
                self._clamp_position()
                return True
            if (event.type() == QEvent.Type.MouseButtonRelease
                    and event.button() == Qt.MouseButton.LeftButton):
                self._drag_offset = None
                self.mascot.setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(watched, event)
