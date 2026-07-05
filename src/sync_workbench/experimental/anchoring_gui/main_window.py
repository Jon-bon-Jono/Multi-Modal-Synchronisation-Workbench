"""Bare-bones experimental anchoring GUI main window.

This GUI is intentionally experimental, but it should still be usable for the
v0.2.2 feasibility task: browse source RGB and target radar streams, move each
stream independently or together, use the initial mapping as a navigation aid,
and place/delete/export canonical anchors.
"""
from __future__ import annotations

from pathlib import Path
import time

from sync_workbench.experimental.anchoring_gui.anchor_table import AnchorTable
from sync_workbench.experimental.anchoring_gui.controllers import AnchoringController
from sync_workbench.experimental.anchoring_gui.pointcloud_panel import PointCloudPanel
from sync_workbench.experimental.anchoring_gui.video_panel import VideoPanel


def _imports():
    from PySide6.QtCore import QTimer  # type: ignore
    from PySide6.QtWidgets import (  # type: ignore
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    return (
        QTimer,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )


def make_main_window_class():
    (
        QTimer,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    ) = _imports()

    class MainWindow(QMainWindow):
        FRAME_DELTAS = [-100, -10, -5, -1, 1, 5, 10, 100]
        SECOND_DELTAS = [-60, -10, -5, -1, 1, 5, 10, 60]

        def __init__(self, controller: AnchoringController):
            super().__init__()
            self.controller = controller
            self.source_sample = 0
            self.target_sample = 0
            self.source_playing = False
            self.target_playing = False
            self.both_playing = False
            self._both_play_start_monotonic = 0.0
            self._both_base_source_sample = 0
            self._both_base_target_sample = 0
            self._both_target_offset_from_mapping = 0

            self.source_max = self.controller.max_sample(self.controller.source_run_id, self.controller.source_device_type)
            self.target_max = self.controller.max_sample(self.controller.target_run_id, self.controller.target_device_type)
            self.source_fps = self.controller.nominal_fps(self.controller.source_run_id, self.controller.source_device_type)
            self.target_fps = self.controller.nominal_fps(self.controller.target_run_id, self.controller.target_device_type)

            self.setWindowTitle("SyncWB experimental anchoring GUI")
            self.video_panel = VideoPanel()
            self.point_panel = PointCloudPanel()
            self.anchor_table = AnchorTable()
            self.status = QLabel("")
            self.status.setWordWrap(True)

            self.source_spin = QSpinBox()
            self.source_spin.setRange(0, max(0, self.source_max))
            self.source_spin.setKeyboardTracking(False)
            self.source_spin.editingFinished.connect(self.go_source_from_spin)
            self.target_spin = QSpinBox()
            self.target_spin.setRange(0, max(0, self.target_max))
            self.target_spin.setKeyboardTracking(False)
            self.target_spin.editingFinished.connect(self.go_target_from_spin)

            self.label_edit = QLineEdit()
            self.label_edit.setPlaceholderText("anchor label")
            self.notes_edit = QLineEdit()
            self.notes_edit.setPlaceholderText("anchor notes")

            self.source_play_button = QPushButton("play source")
            self.target_play_button = QPushButton("play target")
            self.both_play_button = QPushButton("play both")

            self.source_timer = QTimer(self)
            self.source_timer.timeout.connect(self._advance_source_playback)
            self.source_timer.setInterval(self._timer_interval_ms(self.source_fps))
            self.target_timer = QTimer(self)
            self.target_timer.timeout.connect(self._advance_target_playback)
            self.target_timer.setInterval(self._timer_interval_ms(self.target_fps))
            self.both_timer = QTimer(self)
            self.both_timer.timeout.connect(self._advance_both_playback)
            self.both_timer.setInterval(min(self._timer_interval_ms(self.source_fps), self._timer_interval_ms(self.target_fps)))

            self._build_layout()
            self.refresh_all(refresh_anchors=True)

        @staticmethod
        def _timer_interval_ms(fps: float) -> int:
            fps = max(1.0, float(fps or 1.0))
            return max(1, int(round(1000.0 / fps)))

        def _build_layout(self):
            root = QWidget()
            outer = QVBoxLayout(root)

            panels = QHBoxLayout()
            panels.addWidget(self.video_panel, stretch=1)
            panels.addWidget(self.point_panel, stretch=1)
            outer.addLayout(panels, stretch=4)

            controls = QHBoxLayout()
            controls.addWidget(self._make_stream_group("Source", self.source_spin, self.source_play_button, "source"))
            controls.addWidget(self._make_stream_group("Target", self.target_spin, self.target_play_button, "target"))
            controls.addWidget(self._make_both_group())
            controls.addWidget(self._make_anchor_group())
            outer.addLayout(controls)

            outer.addWidget(self.status)
            outer.addWidget(self.anchor_table, stretch=1)
            self.setCentralWidget(root)

        def _make_stream_group(self, title: str, spin, play_button, stream: str):
            group = QGroupBox(title)
            layout = QVBoxLayout(group)
            top = QHBoxLayout()
            top.addWidget(QLabel("sample"))
            top.addWidget(spin)
            play_button.clicked.connect(lambda: self.toggle_play(stream))
            top.addWidget(play_button)
            layout.addLayout(top)

            frames = QGridLayout()
            frames.addWidget(QLabel("frames"), 0, 0)
            for idx, delta in enumerate(self.FRAME_DELTAS, start=1):
                button = QPushButton(f"{delta:+d}")
                if stream == "source":
                    button.clicked.connect(lambda _=False, d=delta: self.step_source(d))
                else:
                    button.clicked.connect(lambda _=False, d=delta: self.step_target(d))
                frames.addWidget(button, 0, idx)
            layout.addLayout(frames)

            seconds = QGridLayout()
            seconds.addWidget(QLabel("seconds"), 0, 0)
            for idx, delta in enumerate(self.SECOND_DELTAS, start=1):
                button = QPushButton(f"{delta:+d}s")
                if stream == "source":
                    button.clicked.connect(lambda _=False, d=delta: self.step_source_seconds(d))
                else:
                    button.clicked.connect(lambda _=False, d=delta: self.step_target_seconds(d))
                seconds.addWidget(button, 0, idx)
            layout.addLayout(seconds)
            return group

        def _make_both_group(self):
            group = QGroupBox("Both streams")
            layout = QVBoxLayout(group)
            self.both_play_button.clicked.connect(lambda: self.toggle_play("both"))
            layout.addWidget(self.both_play_button)

            frames = QGridLayout()
            frames.addWidget(QLabel("frames"), 0, 0)
            for idx, delta in enumerate(self.FRAME_DELTAS, start=1):
                button = QPushButton(f"{delta:+d}")
                button.clicked.connect(lambda _=False, d=delta: self.step_both(d))
                frames.addWidget(button, 0, idx)
            layout.addLayout(frames)

            seconds = QGridLayout()
            seconds.addWidget(QLabel("seconds"), 0, 0)
            for idx, delta in enumerate(self.SECOND_DELTAS, start=1):
                button = QPushButton(f"{delta:+d}s")
                button.clicked.connect(lambda _=False, d=delta: self.step_both_seconds(d))
                seconds.addWidget(button, 0, idx)
            layout.addLayout(seconds)

            sync_row = QHBoxLayout()
            sync_target = QPushButton("sync target to source")
            sync_target.clicked.connect(self.sync_target_to_source)
            sync_source = QPushButton("sync source to target")
            sync_source.clicked.connect(self.sync_source_to_target)
            sync_row.addWidget(sync_target)
            sync_row.addWidget(sync_source)
            layout.addLayout(sync_row)
            return group

        def _make_anchor_group(self):
            group = QGroupBox("Anchors")
            layout = QVBoxLayout(group)
            layout.addWidget(self.label_edit)
            layout.addWidget(self.notes_edit)
            row1 = QHBoxLayout()
            place = QPushButton("place anchor")
            place.clicked.connect(self.place_anchor)
            delete = QPushButton("delete selected")
            delete.clicked.connect(self.delete_selected_anchor)
            row1.addWidget(place)
            row1.addWidget(delete)
            layout.addLayout(row1)
            row2 = QHBoxLayout()
            export = QPushButton("export anchors")
            export.clicked.connect(self.export_anchors)
            finish = QPushButton("finish session")
            finish.clicked.connect(self.close)
            row2.addWidget(export)
            row2.addWidget(finish)
            layout.addLayout(row2)
            return group

        def _set_spin_value(self, spin, value: int) -> None:
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)

        def _clamp_source(self, value: int) -> int:
            return max(0, min(int(value), int(self.source_max)))

        def _clamp_target(self, value: int) -> int:
            return max(0, min(int(value), int(self.target_max)))

        def _stop_playback(self, *, update_labels: bool = True) -> None:
            self.source_playing = False
            self.target_playing = False
            self.both_playing = False
            self.source_timer.stop()
            self.target_timer.stop()
            self.both_timer.stop()
            if update_labels:
                self._update_play_button_labels()

        def go_source_from_spin(self) -> None:
            self._stop_playback()
            self.source_sample = self._clamp_source(self.source_spin.value())
            self._set_spin_value(self.source_spin, self.source_sample)
            self.refresh_source()
            self.refresh_status()

        def go_target_from_spin(self) -> None:
            self._stop_playback()
            self.target_sample = self._clamp_target(self.target_spin.value())
            self._set_spin_value(self.target_spin, self.target_sample)
            self.refresh_target()
            self.refresh_status()

        def _step_source_no_stop(self, delta: int) -> None:
            self.source_sample = self._clamp_source(self.source_sample + int(delta))
            self._set_spin_value(self.source_spin, self.source_sample)
            self.refresh_source()
            self.refresh_status()

        def _step_target_no_stop(self, delta: int) -> None:
            self.target_sample = self._clamp_target(self.target_sample + int(delta))
            self._set_spin_value(self.target_spin, self.target_sample)
            self.refresh_target()
            self.refresh_status()

        def step_source(self, delta: int) -> None:
            self._stop_playback()
            self._step_source_no_stop(delta)

        def step_target(self, delta: int) -> None:
            self._stop_playback()
            self._step_target_no_stop(delta)

        def step_both(self, delta: int) -> None:
            self._stop_playback()
            self.source_sample = self._clamp_source(self.source_sample + int(delta))
            self.target_sample = self._clamp_target(self.target_sample + int(delta))
            self._set_spin_value(self.source_spin, self.source_sample)
            self._set_spin_value(self.target_spin, self.target_sample)
            self.refresh_all(refresh_anchors=False)

        def step_source_seconds(self, seconds: float) -> None:
            self.step_source(round(float(seconds) * float(self.source_fps)))

        def step_target_seconds(self, seconds: float) -> None:
            self.step_target(round(float(seconds) * float(self.target_fps)))

        def step_both_seconds(self, seconds: float) -> None:
            self._stop_playback()
            self.source_sample = self._clamp_source(self.source_sample + round(float(seconds) * float(self.source_fps)))
            self.target_sample = self._clamp_target(self.target_sample + round(float(seconds) * float(self.target_fps)))
            self._set_spin_value(self.source_spin, self.source_sample)
            self._set_spin_value(self.target_spin, self.target_sample)
            self.refresh_all(refresh_anchors=False)

        def _advance_source_playback(self) -> None:
            if not self.source_playing or self.both_playing:
                return
            self._step_source_no_stop(1)
            if self.source_sample >= self.source_max:
                self.source_playing = False
                self.source_timer.stop()
                self._update_play_button_labels()

        def _advance_target_playback(self) -> None:
            if not self.target_playing or self.both_playing:
                return
            self._step_target_no_stop(1)
            if self.target_sample >= self.target_max:
                self.target_playing = False
                self.target_timer.stop()
                self._update_play_button_labels()

        def _mapped_target_or_fallback(self, source_sample: int, elapsed: float) -> int:
            """Target sample for linked two-stream playback.

            During play-both, source is the master stream. The target is derived
            from the active mapping for the exact displayed source sample, plus
            the target offset that existed when playback started. This avoids the
            drift caused by independently rounding source and target from wall
            clock elapsed time. If mapping lookup fails, fall back to elapsed-
            time target playback so the GUI remains usable.
            """
            try:
                mapped_target = int(self.controller.sync_target_to_source(int(source_sample)))
                return self._clamp_target(mapped_target + int(self._both_target_offset_from_mapping))
            except Exception:
                return self._clamp_target(self._both_base_target_sample + round(float(elapsed) * float(self.target_fps)))

        def _advance_both_playback(self) -> None:
            if not self.both_playing:
                return
            elapsed = max(0.0, time.monotonic() - self._both_play_start_monotonic)
            next_source = self._clamp_source(self._both_base_source_sample + round(elapsed * float(self.source_fps)))
            next_target = self._mapped_target_or_fallback(next_source, elapsed)

            if next_source == self.source_sample and next_target == self.target_sample:
                return

            self.source_sample = next_source
            self.target_sample = next_target
            self._set_spin_value(self.source_spin, self.source_sample)
            self._set_spin_value(self.target_spin, self.target_sample)
            self.refresh_all(refresh_anchors=False)

            if self.source_sample >= self.source_max and self.target_sample >= self.target_max:
                self._stop_playback()

        def toggle_play(self, stream: str) -> None:
            if stream == "source":
                if self.source_playing and not self.both_playing:
                    self.source_playing = False
                    self.source_timer.stop()
                else:
                    self.both_playing = False
                    self.both_timer.stop()
                    self.source_playing = True
                    self.source_timer.start()
            elif stream == "target":
                if self.target_playing and not self.both_playing:
                    self.target_playing = False
                    self.target_timer.stop()
                else:
                    self.both_playing = False
                    self.both_timer.stop()
                    self.target_playing = True
                    self.target_timer.start()
            elif stream == "both":
                if self.both_playing or (self.source_playing and self.target_playing):
                    # First bring the playhead to the current elapsed position,
                    # then stop. Without this, pressing pause between timer
                    # callbacks can leave the displayed samples behind the actual
                    # playback time.
                    self._advance_both_playback()
                    self._stop_playback(update_labels=False)
                else:
                    self.source_timer.stop()
                    self.target_timer.stop()
                    self.source_playing = True
                    self.target_playing = True
                    self.both_playing = True
                    self._both_base_source_sample = int(self.source_sample)
                    self._both_base_target_sample = int(self.target_sample)
                    try:
                        mapped_target = int(self.controller.sync_target_to_source(self._both_base_source_sample))
                        self._both_target_offset_from_mapping = int(self._both_base_target_sample - mapped_target)
                    except Exception:
                        self._both_target_offset_from_mapping = 0
                    self._both_play_start_monotonic = time.monotonic()
                    self.both_timer.start()
            self._update_play_button_labels()

        def _update_play_button_labels(self) -> None:
            self.source_play_button.setText("pause source" if self.source_playing and not self.both_playing else "play source")
            self.target_play_button.setText("pause target" if self.target_playing and not self.both_playing else "play target")
            self.both_play_button.setText("pause both" if self.both_playing or (self.source_playing and self.target_playing) else "play both")

        def sync_target_to_source(self) -> None:
            self.go_source_from_spin()
            try:
                self.target_sample = self._clamp_target(self.controller.sync_target_to_source(self.source_sample))
                self._set_spin_value(self.target_spin, self.target_sample)
                self.refresh_target()
                self.refresh_status()
            except Exception as exc:
                self._show_error("sync target to source failed", exc)

        def sync_source_to_target(self) -> None:
            self.go_target_from_spin()
            try:
                self.source_sample = self._clamp_source(self.controller.sync_source_to_target(self.target_sample))
                self._set_spin_value(self.source_spin, self.source_sample)
                self.refresh_source()
                self.refresh_status()
            except Exception as exc:
                self._show_error("sync source to target failed", exc)

        def place_anchor(self) -> None:
            self.go_source_from_spin()
            self.go_target_from_spin()
            try:
                self.controller.place_anchor(
                    self.source_sample,
                    self.target_sample,
                    label=self.label_edit.text(),
                    notes=self.notes_edit.text(),
                )
                self.refresh_anchors()
            except Exception as exc:
                self._show_error("place anchor failed", exc)

        def delete_selected_anchor(self) -> None:
            anchor_id = self.anchor_table.selected_anchor_id()
            if not anchor_id:
                return
            try:
                self.controller.delete_anchor(anchor_id)
                self.refresh_anchors()
            except Exception as exc:
                self._show_error("delete anchor failed", exc)

        def export_anchors(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export anchors", "anchors.json", "JSON files (*.json);;All files (*)")
            if not path:
                return
            try:
                self.controller.export_anchors(Path(path))
            except Exception as exc:
                self._show_error("export anchors failed", exc)

        def refresh_all(self, *, refresh_anchors: bool = False) -> None:
            self.refresh_source()
            self.refresh_target()
            self.refresh_status()
            if refresh_anchors:
                self.refresh_anchors()

        def refresh_source(self) -> None:
            try:
                self.video_panel.set_frame(self.controller.get_rgb_frame(self.source_sample))
            except Exception as exc:
                self.video_panel.setText(f"RGB unavailable: {exc}")

        def refresh_target(self) -> None:
            try:
                self.point_panel.set_points(self.controller.get_target_points(self.target_sample))
            except Exception as exc:
                self.status.setText(f"Radar points unavailable: {exc}")

        def refresh_status(self) -> None:
            mapping_text = "mapping unavailable"
            try:
                row = self.controller.mapping_for_source(self.source_sample)
                mapped_target = row.get("target_sample_index", "")
                delta = row.get("predicted_minus_estimated_ms", "")
                support = row.get("support_status", "")
                rank = row.get("rank", "")
                primary = row.get("is_primary", "")
                mapping_text = (
                    f"mapped target={mapped_target}; delta_ms={delta}; "
                    f"support={support}; rank={rank}; primary={primary}"
                )
            except Exception:
                pass
            self.status.setText(
                f"source {self.controller.source_device_type}/{self.controller.source_run_id} sample={self.source_sample} "
                f"of {self.source_max}; target {self.controller.target_device_type}/{self.controller.target_run_id} "
                f"sample={self.target_sample} of {self.target_max}; {mapping_text}"
            )

        def refresh_anchors(self) -> None:
            self.anchor_table.set_anchors(self.controller.list_anchors())

        def _show_error(self, title: str, exc: Exception) -> None:
            QMessageBox.warning(self, title, str(exc))

        def closeEvent(self, event):  # noqa: N802 - Qt API name
            self.source_timer.stop()
            self.target_timer.stop()
            self.both_timer.stop()
            try:
                self.controller.close()
            except Exception:
                pass
            super().closeEvent(event)

    return MainWindow
