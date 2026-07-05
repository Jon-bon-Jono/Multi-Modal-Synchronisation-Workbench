"""Launch entry point for the experimental anchoring GUI."""
from __future__ import annotations

from pathlib import Path

from sync_workbench.experimental.anchoring_gui.controllers import AnchoringController
from sync_workbench.experimental.anchoring_gui.main_window import make_main_window_class


def run_anchoring_gui(
    *,
    sqlite_path: str | Path,
    artifact_root: str | Path,
    rgb_root: str | Path,
    subject_id: str,
    mapping_version_id: str,
    annotator_id: str = "",
) -> int:
    try:
        from PySide6.QtWidgets import QApplication  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The experimental anchoring GUI requires PySide6, pyqtgraph, and opencv-python.") from exc

    controller = AnchoringController(
        sqlite_path=sqlite_path,
        artifact_root=artifact_root,
        rgb_root=rgb_root,
        subject_id=subject_id,
        mapping_version_id=mapping_version_id,
        annotator_id=annotator_id,
    )
    app = QApplication.instance() or QApplication([])
    MainWindow = make_main_window_class()
    window = MainWindow(controller)
    window.show()
    return int(app.exec())
