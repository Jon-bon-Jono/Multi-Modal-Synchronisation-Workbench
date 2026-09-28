"""Anchor table widget for the experimental GUI."""
from __future__ import annotations

import json


def _imports():
    from PySide6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem  # type: ignore
    return QAbstractItemView, QTableWidget, QTableWidgetItem


class AnchorTable:
    def __new__(cls):
        QAbstractItemView, QTableWidget, QTableWidgetItem = _imports()

        class _AnchorTable(QTableWidget):
            def __init__(self):
                super().__init__(0, 5)
                self.setHorizontalHeaderLabels(["anchor_id", "source_sample", "target_sample", "label", "viewed cloud"])
                self.setSelectionBehavior(QAbstractItemView.SelectRows)
                self.setSelectionMode(QAbstractItemView.SingleSelection)

            def set_anchors(self, anchors):
                self.setRowCount(0)
                if anchors is None or getattr(anchors, "empty", True):
                    return
                for row_idx, row in enumerate(anchors.to_dict("records")):
                    self.insertRow(row_idx)
                    try:
                        cloud = json.loads(row.get("notes") or "{}").get("provenance", {}).get("point_cloud", {})
                    except (ValueError, TypeError):
                        cloud = {}
                    version = cloud.get("point_cloud_version_id", "")
                    cloud_label = cloud.get("readable_label") or version or "not recorded"
                    values = [
                        row.get("anchor_id", ""),
                        row.get("sample_index_source", ""),
                        row.get("sample_index_target", ""),
                        row.get("label", ""),
                        cloud_label,
                    ]
                    for col, value in enumerate(values):
                        item = QTableWidgetItem(str(value))
                        if col == 4:
                            item.setToolTip(version or "Legacy anchor: displayed cloud version was not recorded")
                        self.setItem(row_idx, col, item)
                self.resizeColumnsToContents()

            def selected_anchor_id(self) -> str | None:
                rows = self.selectionModel().selectedRows() if self.selectionModel() is not None else []
                if not rows:
                    return None
                item = self.item(rows[0].row(), 0)
                return item.text() if item is not None else None

        return _AnchorTable()
