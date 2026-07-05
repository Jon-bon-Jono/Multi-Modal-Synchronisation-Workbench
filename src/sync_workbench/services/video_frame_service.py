"""RGB video frame retrieval service for experimental frontends."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sync_workbench.services.asset_service import AssetService


class VideoFrameService:
    """Retrieve RGB frames by canonical sample_index.

    The current v0.2.2 assumption is that RUN_SAMPLE.sample_index is the
    zero-based frame position in the MP4. The temporary rgb_samples.frame_number
    may start at 1 in real data; that raw source frame number is not used for
    seeking here.
    """

    def __init__(self, asset_service: AssetService):
        self.asset_service = asset_service
        self._captures: dict[Path, object] = {}
        self._last_frames: dict[Path, tuple[int, np.ndarray]] = {}

    def get_rgb_video_path(self, subject_id: str, run_id: str, device_type: str = "kinect_rgb") -> Path:
        return self.asset_service.resolve_run_asset(
            subject_id=subject_id,
            run_id=run_id,
            device_type=device_type,
            asset_role="rgb_video",
        )

    def get_rgb_frame(
        self,
        subject_id: str,
        run_id: str,
        sample_index: int,
        *,
        device_type: str = "kinect_rgb",
    ) -> np.ndarray:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("VideoFrameService requires opencv-python to decode MP4 frames.") from exc

        path = self.get_rgb_video_path(subject_id, run_id, device_type=device_type)
        cap = self._captures.get(path)
        if cap is None:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open RGB video: {path}")
            self._captures[path] = cap

        idx = int(sample_index)
        if idx < 0:
            raise ValueError("sample_index must be non-negative.")

        cached = self._last_frames.get(path)
        if cached is not None and cached[0] == idx:
            return cached[1].copy()

        # Sequential playback is much faster if we do not seek for every next
        # frame. For arbitrary jumps, seek explicitly.
        if cached is None or cached[0] + 1 != idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise IndexError(f"Could not decode frame sample_index={idx} from {path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._last_frames[path] = (idx, frame_rgb)
        return frame_rgb.copy()

    def close(self) -> None:
        for cap in self._captures.values():
            try:
                cap.release()
            except Exception:
                pass
        self._captures.clear()
        self._last_frames.clear()
