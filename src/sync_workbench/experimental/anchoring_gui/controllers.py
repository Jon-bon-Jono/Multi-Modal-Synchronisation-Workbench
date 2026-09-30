"""Backend-facing controller for the experimental anchoring GUI."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import json
from types import MappingProxyType
import uuid
from typing import Any

import pandas as pd
import numpy as np

from sync_workbench.core.time_utils import utc_now_str
from sync_workbench.experimental.anchoring_gui.session_selection import resolve_session
from sync_workbench.services.anchor_service import AnchorEndpoint, AnchorService
from sync_workbench.services.asset_service import AssetService
from sync_workbench.services.mapping_lookup_service import MappingLookupService
from sync_workbench.services.payload_service import PayloadService
from sync_workbench.services.video_frame_service import VideoFrameService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.experimental.anchoring_gui.prediction_overlay import (
    PosePredictionFrame,
    PosePredictionOverlay,
)


class AnchoringController:
    def __init__(
        self,
        *,
        sqlite_path: str | Path,
        artifact_root: str | Path,
        rgb_root: str | Path,
        subject_id: str,
        mapping_version_id: str,
        annotator_id: str = "",
        package_provenance: dict | None = None,
        point_cloud_version_id: str | None = None,
        pose_predictions_path: str | Path | None = None,
        pose_prediction_array: str = "pred_globally_aligned",
    ):
        self.sqlite_path = Path(sqlite_path)
        self.subject_id = subject_id
        self.mapping_version_id = mapping_version_id
        self.annotator_id = annotator_id
        self._asset_roots = (Path(artifact_root).resolve(),Path(rgb_root).resolve())
        self.last_write_result = {}
        self.recovery_root = self.sqlite_path.parent / ("recovery" if package_provenance else self.sqlite_path.stem + "_recovery")
        self._package_provenance = json.loads(json.dumps(package_provenance)) if package_provenance else None
        self.store = SQLiteCoreStore(sqlite_path)
        self.lookup = MappingLookupService(sqlite_path)
        self._selection = resolve_session(sqlite_path, artifact_root, subject_id, mapping_version_id, point_cloud_version_id)
        self.context = MappingProxyType(self.lookup.get_mapping_context(subject_id, mapping_version_id))
        self.session_id = str(uuid.uuid4())
        self.session_started_at = utc_now_str()
        if pose_predictions_path and self._selection.target_device_type == "radar_raw":
            raise ValueError("Existing pose-prediction files are indexed to online samples. Omit --pose-predictions for offline raw sessions; version-bound raw predictions are not yet supported.")
        self.payloads = PayloadService(sqlite_path, artifact_root)
        self.assets = AssetService(sqlite_path, {"rgb": rgb_root, "artifact_store": artifact_root})
        self.video = VideoFrameService(self.assets)
        self.anchors = AnchorService(sqlite_path)
        self.pose_predictions = (
            PosePredictionOverlay(
                pose_predictions_path,
                pose_array=pose_prediction_array,
            )
            if pose_predictions_path
            else None
        )
        self._run_sample_bounds: dict[tuple[str, str], int] = {}
        self._nominal_fps: dict[tuple[str, str], float] = {}
        self._payload_cache: dict[tuple[str, str, int, str, str | None], Any] = {}

    @property
    def selection(self):
        return self._selection

    @property
    def package_id(self):
        return self._package_provenance.get("package_id", "") if self._package_provenance else ""

    @property
    def point_cloud_version_id(self):
        return self._selection.point_cloud_version_id

    def session_metadata(self):
        return {"created_by": "experimental_anchoring_gui", "initial_mapping_version_id": self.mapping_version_id,
                "annotator_id": self.annotator_id, "session_id": self.session_id, "session_started_at": self.session_started_at,
                "point_cloud": self.selection.cloud_provenance(),
                **({"package": json.loads(json.dumps(self._package_provenance))} if self._package_provenance else {})}

    def initial_samples(self):
        with closing(sqlite3.connect(self.sqlite_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT source_sample_index,target_sample_index FROM SAMPLE_MAPPING WHERE subject_id=? AND mapping_version_id=? ORDER BY CAST(source_sample_index AS INTEGER), CAST(rank AS INTEGER) LIMIT 1", (self.subject_id, self.mapping_version_id)).fetchone()
        return tuple(map(int, row)) if row else (0, 0)

    def get_target_summary(self, sample_index):
        return self.payloads.get_sample_summary(self.subject_id, self.target_run_id, self.target_device_type,
                                               sample_index, point_cloud_version_id=self.point_cloud_version_id)

    @property
    def source_run_id(self) -> str:
        return str(self.context["source_run_id"])

    @property
    def source_device_type(self) -> str:
        return str(self.context["source_device_type"])

    @property
    def target_run_id(self) -> str:
        return str(self.context["target_run_id"])

    @property
    def target_device_type(self) -> str:
        return str(self.context["target_device_type"])

    def max_sample(self, run_id: str, device_type: str) -> int:
        key = (str(run_id), str(device_type))
        cached = self._run_sample_bounds.get(key)
        if cached is not None:
            return cached
        samples = self.store.read_table("RUN_SAMPLE")
        rows = samples[
            (samples["subject_id"].astype(str) == str(self.subject_id))
            & (samples["run_id"].astype(str) == str(run_id))
            & (samples["device_type"].astype(str) == str(device_type))
        ].copy()
        if rows.empty:
            value = 0
        else:
            value = int(pd.to_numeric(rows["sample_index"], errors="coerce").max())
        self._run_sample_bounds[key] = value
        return value

    def nominal_fps(self, run_id: str, device_type: str) -> float:
        key = (str(run_id), str(device_type))
        cached = self._nominal_fps.get(key)
        if cached is not None:
            return cached
        runs = self.store.read_table("DEVICE_RUN")
        rows = runs[
            (runs["subject_id"].astype(str) == str(self.subject_id))
            & (runs["run_id"].astype(str) == str(run_id))
            & (runs["device_type"].astype(str) == str(device_type))
        ].copy()
        if rows.empty:
            value = 15.0 if str(device_type) == "kinect_rgb" else 20.0
        else:
            raw_value = pd.to_numeric(rows.iloc[0].get("nominal_fps", 0), errors="coerce")
            value = float(raw_value) if pd.notna(raw_value) else 0.0
            if value <= 0:
                value = 15.0 if str(device_type) == "kinect_rgb" else 20.0
        self._nominal_fps[key] = value
        return value

    def get_rgb_frame(self, source_sample_index: int):
        return self.video.get_rgb_frame(self.subject_id, self.source_run_id, source_sample_index, device_type=self.source_device_type)

    def _cached_payload(self, run_id: str, device_type: str, sample_index: int, role: str):
        version = self.point_cloud_version_id if (run_id, device_type) == (self.target_run_id, self.target_device_type) else None
        key = (str(run_id), str(device_type), int(sample_index), str(role), version)
        if key in self._payload_cache:
            return self._payload_cache[key]
        value = self.payloads.get_payload(
            self.subject_id,
            run_id,
            device_type,
            int(sample_index),
            role,
            point_cloud_version_id=version,
        )
        # Keep the cache deliberately small enough for interactive browsing.
        if len(self._payload_cache) > 512:
            self._payload_cache.pop(next(iter(self._payload_cache)))
        self._payload_cache[key] = value
        return value

    def get_source_pose2d(self, source_sample_index: int):
        return self._cached_payload(self.source_run_id, self.source_device_type, source_sample_index, "pose2d")

    def get_source_pose3d(self, source_sample_index: int):
        return self._cached_payload(self.source_run_id, self.source_device_type, source_sample_index, "pose3d")

    def get_target_points(self, target_sample_index: int):
        return self._cached_payload(self.target_run_id, self.target_device_type, target_sample_index, "radar_points")

    @property
    def has_pose_predictions(self) -> bool:
        return self.pose_predictions is not None

    def get_target_pose_prediction(
        self,
        target_sample_index: int,
    ) -> PosePredictionFrame | None:
        if self.pose_predictions is None:
            return None
        return self.pose_predictions.get(target_sample_index)

    def pose_prediction_summary(self) -> dict[str, object] | None:
        if self.pose_predictions is None:
            return None
        return self.pose_predictions.summary()

    def get_target_points_window(self, target_sample_index: int, radius: int = 0):
        """Return radar points from target_sample_index +/- radius frames.

        radius=0 preserves the original single-frame behaviour. For radius > 0,
        use PayloadService's ragged-NPZ window reader so interactive browsing
        does not repeatedly retrieve and stack every neighbouring sample.
        """
        radius = max(0, int(radius or 0))

        if radius == 0:
            return self.get_target_points(target_sample_index)

        return self.payloads.get_ragged_payload_window(
            self.subject_id,
            self.target_run_id,
            self.target_device_type,
            int(target_sample_index),
            "radar_points",
            radius=radius,
            point_cloud_version_id=self.point_cloud_version_id,
            min_sample_index=0,
            max_sample_index=self.max_sample(self.target_run_id, self.target_device_type),
        )

    def mapping_for_source(self, source_sample_index: int) -> dict[str, Any]:
        return self.lookup.map_source_to_target(self.subject_id, self.mapping_version_id, source_sample_index)

    def sync_target_to_source(self, source_sample_index: int) -> int:
        row = self.mapping_for_source(source_sample_index)
        return int(row["target_sample_index"])

    def sync_source_to_target(self, target_sample_index: int) -> int:
        row = self.lookup.map_target_to_source(self.subject_id, self.mapping_version_id, target_sample_index)
        return int(row["source_sample_index"])

    def place_anchor(self, source_sample_index: int, target_sample_index: int, *, label: str = "", confidence: float | None = None, notes: str = "", point_window_radius: int = 0, filter_noise: bool = False) -> str:
        # Do not claim an annotator saw a cloud when the selected frame/window is unavailable.
        self.get_target_points_window(target_sample_index, radius=point_window_radius)
        # RGB is required even when the user has hidden its rendering layer.
        self.get_rgb_frame(source_sample_index)
        summary = self.get_target_summary(target_sample_index)
        provenance = self.session_metadata()
        provenance["display"] = {"target_sample_index": int(target_sample_index), "point_window_radius": max(0, int(point_window_radius)),
                                 "excluded_association_ids": [253, 254, 255] if filter_noise else [],
                                 "point_status": summary.get("point_status", "available")}
        anchor_id = self.anchors.create_pair_anchor(
            subject_id=self.subject_id,
            source=AnchorEndpoint(self.source_run_id, self.source_device_type, int(source_sample_index), "source", confidence),
            target=AnchorEndpoint(self.target_run_id, self.target_device_type, int(target_sample_index), "target", confidence),
            label=label,
            confidence=confidence,
            user_notes=notes,
            provenance=provenance,
        )
        self._record_saved("Saved anchor " + anchor_id)
        return anchor_id

    def delete_anchor(self, anchor_id: str) -> None:
        # Preserve the deleted anchor before committing the deletion.
        self._recovery_snapshot()
        self.anchors.delete_anchor(self.subject_id, anchor_id)
        self._record_saved("Deleted anchor " + anchor_id)

    def _record_saved(self, message):
        self.last_write_result = {"message":message,"database_saved":True,"recovery_path":None,"warning":""}
        try:
            self.last_write_result["recovery_path"] = str(self._recovery_snapshot())
        except Exception as exc:
            self.last_write_result["warning"] = f"Saved in database, but recovery copy failed: {exc}. Retry Export anchors."

    def _recovery_snapshot(self):
        path = self.recovery_root / ("anchors_" + utc_now_str().replace(":", "").replace(".", "") + "_" + uuid.uuid4().hex[:8] + ".json")
        self._export_anchors(path)
        return path

    def default_export_path(self):
        name = f"{self.package_id or self.subject_id}_{self.annotator_id or 'anchors'}.json"
        path = self.sqlite_path.parent / "exports" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def export_anchors(self, path: str | Path) -> dict[str, Any]:
        path = Path(path).resolve()
        protected = (*self._asset_roots, self.recovery_root.resolve())
        if any(path == p or path.is_relative_to(p) for p in protected):
            raise ValueError("Choose an export location outside source assets and recovery snapshots")
        if self.package_id:
            package_root = self.sqlite_path.parent.parent.resolve()
            export_root = (self.sqlite_path.parent / "exports").resolve()
            if path.is_relative_to(package_root) and not path.is_relative_to(export_root):
                raise ValueError("Within a student package, save return files under work/exports")
        payload = self._export_anchors(path)
        self.last_write_result = {"message":f"Exported {len(payload['ANCHOR'])} anchors to {path}",
                                  "database_saved":True,"recovery_path":str(path),"warning":""}
        return payload

    def _export_anchors(self, path):
        return self.anchors.export_pair_anchors_json(
            path,
            subject_id=self.subject_id,
            source_run_id=self.source_run_id,
            source_device_type=self.source_device_type,
            target_run_id=self.target_run_id,
            target_device_type=self.target_device_type,
            session_metadata=self.session_metadata(),
        )

    def list_anchors(self):
        return self.anchors.list_pair_anchors(
            subject_id=self.subject_id,
            source_run_id=self.source_run_id,
            source_device_type=self.source_device_type,
            target_run_id=self.target_run_id,
            target_device_type=self.target_device_type,
        )

    def close(self) -> None:
        self.video.close()
        self._payload_cache.clear()
        self.payloads.clear_cache()
