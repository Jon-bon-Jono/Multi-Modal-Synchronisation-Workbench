"""Launch entry point for the experimental anchoring GUI."""
from __future__ import annotations

from pathlib import Path

from sync_workbench.experimental.anchoring_gui.controllers import AnchoringController
from sync_workbench.experimental.anchoring_gui.main_window import make_main_window_class
from sync_workbench.experimental.anchoring_gui.session_selection import make_session_dialog_class, resolve_session, session_choices


def run_anchoring_gui(
    *,
    sqlite_path: str | Path,
    artifact_root: str | Path,
    rgb_root: str | Path,
    subject_id: str,
    mapping_version_id: str | None = None,
    point_cloud_version_id: str | None = None,
    annotator_id: str = "",
    pose_predictions_path: str | Path | None = None,
    pose_prediction_array: str = "pred_globally_aligned",
) -> int:
    try:
        from PySide6.QtWidgets import QApplication  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The experimental anchoring GUI requires PySide6, pyqtgraph, and opencv-python.") from exc

    app = QApplication.instance() or QApplication([])
    if mapping_version_id and point_cloud_version_id:
        selection = resolve_session(sqlite_path, artifact_root, subject_id, mapping_version_id, point_cloud_version_id)
    else:
        choices = session_choices(sqlite_path, artifact_root, subject_id)
        if not choices:
            raise ValueError("No supported mapped clouds are available. Import point-cloud artifacts and create a compatible RGB-to-radar mapping first.")
        if mapping_version_id and not any(c.mapping_version_id == mapping_version_id for c in choices):
            raise ValueError("The requested mapping has no supported cloud available in this artifact store.")
        if point_cloud_version_id and not any(c.point_cloud_version_id == point_cloud_version_id for c in choices):
            raise ValueError("The requested cloud version has no compatible RGB-to-radar mapping.")
        dialog = make_session_dialog_class()(choices, preferred_mapping=mapping_version_id, preferred_version=point_cloud_version_id)
        if not dialog.exec():
            return 0
        selection = dialog.selection
    controller = AnchoringController(
        sqlite_path=sqlite_path,
        artifact_root=artifact_root,
        rgb_root=rgb_root,
        subject_id=subject_id,
        mapping_version_id=selection.mapping_version_id,
        point_cloud_version_id=selection.point_cloud_version_id,
        annotator_id=annotator_id,
        pose_predictions_path=pose_predictions_path,
        pose_prediction_array=pose_prediction_array,
    )
    try:
        MainWindow = make_main_window_class()
        window = MainWindow(controller)
        window.show()
        return int(app.exec())
    finally:
        controller.close()
