"""Playback state containers for the experimental anchoring GUI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamPlaybackState:
    subject_id: str
    run_id: str
    device_type: str
    sample_index: int = 0
    nominal_fps: float = 15.0
    playing: bool = False

    def step_frames(self, delta_frames: int, *, min_sample: int = 0, max_sample: int | None = None) -> None:
        value = self.sample_index + int(delta_frames)
        if max_sample is not None:
            value = min(value, int(max_sample))
        self.sample_index = max(int(min_sample), value)

    def step_seconds(self, seconds: float, *, min_sample: int = 0, max_sample: int | None = None) -> None:
        self.step_frames(round(float(seconds) * float(self.nominal_fps)), min_sample=min_sample, max_sample=max_sample)
