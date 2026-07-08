"""Backend-facing controller for the experimental anchoring GUI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from sync_workbench.services.anchor_service import AnchorEndpoint, AnchorService
from sync_workbench.services.asset_service import AssetService
from sync_workbench.services.mapping_lookup_service import MappingLookupService
from sync_workbench.services.payload_service import PayloadService
from sync_workbench.services.video_frame_service import VideoFrameService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore


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
    ):
        self.sqlite_path = Path(sqlite_path)
        self.subject_id = subject_id
        self.mapping_version_id = mapping_version_id
        self.annotator_id = annotator_id
        self.store = SQLiteCoreStore(sqlite_path)
        self.lookup = MappingLookupService(sqlite_path)
        self.context = self.lookup.get_mapping_context(subject_id, mapping_version_id)
        self.payloads = PayloadService(sqlite_path, artifact_root)
        self.assets = AssetService(sqlite_path, {"rgb": rgb_root, "artifact_store": artifact_root})
        self.video = VideoFrameService(self.assets)
        self.anchors = AnchorService(sqlite_path)
        self._run_sample_bounds: dict[tuple[str, str], int] = {}
        self._nominal_fps: dict[tuple[str, str], float] = {}
        self._payload_cache: dict[tuple[str, str, int, str], Any] = {}

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
        key = (str(run_id), str(device_type), int(sample_index), str(role))
        if key in self._payload_cache:
            return self._payload_cache[key]
        value = self.payloads.get_payload(
            self.subject_id,
            run_id,
            device_type,
            int(sample_index),
            role,
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

    def place_anchor(self, source_sample_index: int, target_sample_index: int, *, label: str = "", confidence: float | None = None, notes: str = "") -> str:
        return self.anchors.create_pair_anchor(
            subject_id=self.subject_id,
            source=AnchorEndpoint(self.source_run_id, self.source_device_type, int(source_sample_index), "source", confidence),
            target=AnchorEndpoint(self.target_run_id, self.target_device_type, int(target_sample_index), "target", confidence),
            label=label,
            confidence=confidence,
            user_notes=notes,
            provenance={
                "created_by": "experimental_anchoring_gui",
                "initial_mapping_version_id": self.mapping_version_id,
                "annotator_id": self.annotator_id,
            },
        )

    def delete_anchor(self, anchor_id: str) -> None:
        self.anchors.delete_anchor(self.subject_id, anchor_id)

    def export_anchors(self, path: str | Path) -> dict[str, Any]:
        return self.anchors.export_pair_anchors_json(
            path,
            subject_id=self.subject_id,
            source_run_id=self.source_run_id,
            source_device_type=self.source_device_type,
            target_run_id=self.target_run_id,
            target_device_type=self.target_device_type,
            session_metadata={
                "created_by": "experimental_anchoring_gui",
                "initial_mapping_version_id": self.mapping_version_id,
                "annotator_id": self.annotator_id,
            },
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
