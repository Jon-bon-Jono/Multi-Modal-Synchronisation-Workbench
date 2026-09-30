"""Single-line status labels which cannot force a window wider or taller."""


def compact_label(text='', parent=None):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QPainter
    from PySide6.QtWidgets import QLabel, QSizePolicy

    class CompactLabel(QLabel):
        def __init__(self):
            super().__init__(parent)
            self.setTextFormat(Qt.PlainText)
            self.setWordWrap(False)
            self.setMinimumWidth(0)
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            self.setFixedHeight(self.fontMetrics().height() + 6)
            self.setText(text)

        def setText(self, value):
            super().setText(value)
            self.setToolTip(value)

        def changeEvent(self, event):
            super().changeEvent(event)
            if event.type() == QEvent.FontChange:
                self.setFixedHeight(self.fontMetrics().height() + 6)

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setPen(self.palette().windowText().color())
            text = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, self.contentsRect().width())
            painter.drawText(self.contentsRect(), Qt.AlignLeft | Qt.AlignVCenter, text)

    return CompactLabel()
