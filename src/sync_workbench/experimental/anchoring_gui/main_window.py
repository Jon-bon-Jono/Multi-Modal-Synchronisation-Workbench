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
from sync_workbench.experimental.anchoring_gui.visualization_utils import filter_noise_points, project_pc_to_digital


def _imports():
    from PySide6.QtCore import QRectF, Qt, QTimer  # type: ignore
    from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap  # type: ignore
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
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
        QInputDialog
    )

    return (
        Qt,
        QRectF,
        QColor, 
        QLinearGradient, 
        QPainter, 
        QPixmap,
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
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
        QInputDialog
    )


def make_main_window_class():
    (
        Qt,
        QRectF,
        QColor, 
        QLinearGradient, 
        QPainter, 
        QPixmap,
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
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
        QInputDialog
    ) = _imports()

    class MainWindow(QMainWindow):
        FRAME_DELTAS = [-100, -10, -5, -1, 1, 5, 10, 100]
        FINE_SECOND_DELTAS = [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5]
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
            self.point_color_mode = "constant"
            self.filter_noise_points = False
            self.show_pose3d_in_pointcloud = False
            self.show_predicted_pose3d_in_pointcloud = bool(
                self.controller.has_pose_predictions
            )
            self.show_pose2d_overlay = False
            self.show_projected_pc_overlay = False
            self.show_video_frames = True
            self.point_view_pan_mode = False
            self._projected_pc_cache: dict[tuple[int, bool], object] = {}
            self.pc_window_radius = 0
            self._target_points_cache_key: tuple[int, int] | None = None
            self._target_points_cache_value: object | None = None
            self._projected_pc_cache_key: tuple[int, int, bool] | None = None
            self._projected_pc_cache_value: object | None = None

            self.source_max = self.controller.max_sample(self.controller.source_run_id, self.controller.source_device_type)
            self.target_max = self.controller.max_sample(self.controller.target_run_id, self.controller.target_device_type)
            self.source_fps = self.controller.nominal_fps(self.controller.source_run_id, self.controller.source_device_type)
            self.target_fps = self.controller.nominal_fps(self.controller.target_run_id, self.controller.target_device_type)

            self.setWindowTitle("SyncWB experimental anchoring GUI")
            self.video_panel = VideoPanel()
            self.point_panel = PointCloudPanel()

            self.point_colour_legend = QLabel("")
            self.point_colour_legend.setMinimumHeight(24)
            self.point_colour_legend.setVisible(False)

            self.point_colour_legend_ticks = QLabel("")
            self.point_colour_legend_ticks.setAlignment(Qt.AlignCenter)
            self.point_colour_legend_ticks.setVisible(False)

            self.anchor_table = AnchorTable()
            self.status = QLabel("")
            self.point_colour_legend.setMinimumHeight(22)
            self.point_colour_legend.setWordWrap(True)
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
            self.color_mode_button = QPushButton("colour: none")
            self.filter_noise_button = QPushButton("filter noise: off")
            self.pose3d_button = QPushButton("3D pose: off")
            self.predicted_pose3d_button = QPushButton(
                "predicted pose: on"
                if self.show_predicted_pose3d_in_pointcloud
                else "predicted pose: off"
            )
            self.pose2d_button = QPushButton("2D pose: off")
            self.pc2d_button = QPushButton("2D PC: off")
            self.video_toggle_button = QPushButton("video: on")
            self.pc_window_button = QPushButton("PC window: ±0")
            self.point_pan_button = QPushButton("3D pan: off")
            for button in [
                self.filter_noise_button,
                self.pose3d_button,
                self.predicted_pose3d_button,
                self.pose2d_button,
                self.pc2d_button,
                self.video_toggle_button,
                self.point_pan_button,
            ]:
                button.setCheckable(True)
            self.video_toggle_button.setChecked(True)
            self.predicted_pose3d_button.setChecked(
                self.show_predicted_pose3d_in_pointcloud
            )
            self.predicted_pose3d_button.setEnabled(
                self.controller.has_pose_predictions
            )
            if not self.controller.has_pose_predictions:
                self.predicted_pose3d_button.setToolTip(
                    "Launch the GUI with --pose-predictions to enable this overlay."
                )

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
            self._update_point_colour_legend()
            self.refresh_all(refresh_anchors=True)

        @staticmethod
        def _timer_interval_ms(fps: float) -> int:
            fps = max(1.0, float(fps or 1.0))
            return max(1, int(round(1000.0 / fps)))

        @staticmethod
        def _seconds_button_label(seconds: float) -> str:
            return f"{float(seconds):+g}s"
        
        def _build_layout(self):
            root = QWidget()
            outer = QVBoxLayout(root)

            panels = QHBoxLayout()
            panels.addWidget(self.video_panel, stretch=1)

            point_container = QWidget()
            point_layout = QVBoxLayout(point_container)
            point_layout.setContentsMargins(0, 0, 0, 0)
            point_layout.addWidget(self.point_panel, stretch=1)
            point_layout.addWidget(self.point_colour_legend)
            point_layout.addWidget(self.point_colour_legend_ticks)

            panels.addWidget(point_container, stretch=1)
            outer.addLayout(panels, stretch=4)

            controls_widget = QWidget()
            controls = QHBoxLayout(controls_widget)
            controls.setContentsMargins(0, 0, 0, 0)

            controls.addWidget(self._make_stream_group("Source", self.source_spin, self.source_play_button, "source"))
            controls.addWidget(self._make_stream_group("Target", self.target_spin, self.target_play_button, "target"))
            controls.addWidget(self._make_both_group())
            controls.addWidget(self._make_visual_group())
            controls.addWidget(self._make_anchor_group())
            controls.addStretch(1)

            controls_scroll = QScrollArea()
            controls_scroll.setWidget(controls_widget)
            controls_scroll.setWidgetResizable(False)
            controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            controls_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Maximum)
            controls_scroll.setMinimumWidth(0)
            controls_scroll.setMaximumHeight(260)

            outer.addWidget(controls_scroll)

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

            fine_seconds = QGridLayout()
            fine_seconds.addWidget(QLabel("fine sec"), 0, 0)
            for idx, delta in enumerate(self.FINE_SECOND_DELTAS, start=1):
                button = QPushButton(self._seconds_button_label(delta))
                if stream == "source":
                    button.clicked.connect(lambda _=False, d=delta: self.step_source_seconds(d))
                else:
                    button.clicked.connect(lambda _=False, d=delta: self.step_target_seconds(d))
                fine_seconds.addWidget(button, 0, idx)
            layout.addLayout(fine_seconds)

            seconds = QGridLayout()
            seconds.addWidget(QLabel("seconds"), 0, 0)
            for idx, delta in enumerate(self.SECOND_DELTAS, start=1):
                button = QPushButton(self._seconds_button_label(delta))
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

            fine_seconds = QGridLayout()
            fine_seconds.addWidget(QLabel("fine sec"), 0, 0)
            for idx, delta in enumerate(self.FINE_SECOND_DELTAS, start=1):
                button = QPushButton(self._seconds_button_label(delta))
                button.clicked.connect(lambda _=False, d=delta: self.step_both_seconds(d))
                fine_seconds.addWidget(button, 0, idx)
            layout.addLayout(fine_seconds)

            seconds = QGridLayout()
            seconds.addWidget(QLabel("seconds"), 0, 0)
            for idx, delta in enumerate(self.SECOND_DELTAS, start=1):
                button = QPushButton(self._seconds_button_label(delta))
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

        def _make_visual_group(self):
            group = QGroupBox("Visualisation")
            layout = QVBoxLayout(group)

            self.color_mode_button.clicked.connect(self.cycle_point_colour_mode)
            self.filter_noise_button.clicked.connect(self.toggle_filter_noise)
            self.pose3d_button.clicked.connect(self.toggle_pose3d)
            self.predicted_pose3d_button.clicked.connect(
                self.toggle_predicted_pose3d
            )
            self.pose2d_button.clicked.connect(self.toggle_pose2d)
            self.pc2d_button.clicked.connect(self.toggle_projected_pc)
            self.video_toggle_button.clicked.connect(self.toggle_video_frames)
            self.pc_window_button.clicked.connect(self.set_pc_window_radius)
            self.point_pan_button.clicked.connect(self.toggle_point_pan_mode)

            layout.addWidget(self.color_mode_button)
            layout.addWidget(self.filter_noise_button)
            layout.addWidget(self.pose3d_button)
            layout.addWidget(self.predicted_pose3d_button)
            layout.addWidget(self.pose2d_button)
            layout.addWidget(self.pc2d_button)
            layout.addWidget(self.video_toggle_button)
            layout.addWidget(self.pc_window_button)
            layout.addWidget(self.point_pan_button)
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
            if self.show_pose3d_in_pointcloud:
                self.refresh_target()
            self.refresh_status()

        def _step_target_no_stop(self, delta: int) -> None:
            self.target_sample = self._clamp_target(self.target_sample + int(delta))
            self._set_spin_value(self.target_spin, self.target_sample)
            self.refresh_target()
            if self.show_projected_pc_overlay:
                self.refresh_source()
            self.refresh_status()

        def step_source(self, delta: int) -> None:
            self._stop_playback()
            self._step_source_no_stop(delta)

        def step_target(self, delta: int) -> None:
            self._stop_playback()
            self._step_target_no_stop(delta)

        def _current_target_offset_from_mapping(self) -> int:
            """Return current target offset relative to the active source->target mapping.

            If the current pair is already synced, this is 0. If the user has manually
            nudged the target ahead/behind the mapped target, this preserves that offset
            when stepping both streams.
            """
            try:
                mapped_target = int(self.controller.sync_target_to_source(int(self.source_sample)))
                return int(self.target_sample) - mapped_target
            except Exception:
                return 0


        def _target_for_source_preserving_offset(
            self,
            source_sample: int,
            target_offset: int,
            *,
            fallback_target_delta: int = 0,
        ) -> int:
            """Map source_sample to target, preserving the current manual target offset."""
            try:
                mapped_target = int(self.controller.sync_target_to_source(int(source_sample)))
                return self._clamp_target(mapped_target + int(target_offset))
            except Exception:
                # If the mapping lookup fails, fall back to the old behaviour.
                return self._clamp_target(int(self.target_sample) + int(fallback_target_delta))
        
        def step_both(self, delta: int) -> None:
            self._stop_playback()

            target_offset = self._current_target_offset_from_mapping()

            new_source = self._clamp_source(self.source_sample + int(delta))
            new_target = self._target_for_source_preserving_offset(
                new_source,
                target_offset,
                fallback_target_delta=int(delta),
            )

            self.source_sample = new_source
            self.target_sample = new_target

            self._set_spin_value(self.source_spin, self.source_sample)
            self._set_spin_value(self.target_spin, self.target_sample)
            self.refresh_all(refresh_anchors=False)

        def step_source_seconds(self, seconds: float) -> None:
            self.step_source(round(float(seconds) * float(self.source_fps)))

        def step_target_seconds(self, seconds: float) -> None:
            self.step_target(round(float(seconds) * float(self.target_fps)))

        def step_both_seconds(self, seconds: float) -> None:
            self._stop_playback()

            seconds = float(seconds)
            source_delta = round(seconds * float(self.source_fps))
            target_delta_fallback = round(seconds * float(self.target_fps))

            target_offset = self._current_target_offset_from_mapping()

            new_source = self._clamp_source(self.source_sample + source_delta)
            new_target = self._target_for_source_preserving_offset(
                new_source,
                target_offset,
                fallback_target_delta=target_delta_fallback,
            )

            self.source_sample = new_source
            self.target_sample = new_target

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
                self.refresh_all(refresh_anchors=False)
            except Exception as exc:
                self._show_error("sync target to source failed", exc)

        def sync_source_to_target(self) -> None:
            self.go_target_from_spin()
            try:
                self.source_sample = self._clamp_source(self.controller.sync_source_to_target(self.target_sample))
                self._set_spin_value(self.source_spin, self.source_sample)
                self.refresh_all(refresh_anchors=False)
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

        def cycle_point_colour_mode(self) -> None:
            modes = ["constant", "snr", "doppler"]
            idx = modes.index(self.point_color_mode) if self.point_color_mode in modes else 0
            self.point_color_mode = modes[(idx + 1) % len(modes)]

            label = {
                "constant": "none",
                "snr": "SNR",
                "doppler": "Doppler",
            }[self.point_color_mode]

            self.color_mode_button.setText(f"colour: {label}")
            self._update_point_colour_legend()
            self.refresh_target()

            if self.show_projected_pc_overlay:
                self.refresh_source()

        def _make_doppler_legend_pixmap(self, width: int = 420, height: int = 18):
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            gradient = QLinearGradient(0, 0, width, 0)

            # Must match visualization_utils.point_rgba:
            # norm=0 -> blue, norm=0.5 -> green, norm=1 -> red.
            gradient.setColorAt(0.0, QColor(0, 80, 255))
            gradient.setColorAt(0.5, QColor(40, 220, 40))
            gradient.setColorAt(1.0, QColor(255, 60, 30))

            painter.fillRect(QRectF(0, 0, width, height), gradient)
            painter.setPen(QColor(220, 220, 220))
            painter.drawRect(0, 0, width - 1, height - 1)
            painter.end()

            return pixmap


        def _make_snr_legend_pixmap(self, width: int = 420, height: int = 18):
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            gradient = QLinearGradient(0, 0, width, 0)

            # Same blue -> green -> red ramp, but SNR is still auto-normalised.
            gradient.setColorAt(0.0, QColor(0, 80, 255))
            gradient.setColorAt(0.5, QColor(40, 220, 40))
            gradient.setColorAt(1.0, QColor(255, 60, 30))

            painter.fillRect(QRectF(0, 0, width, height), gradient)
            painter.setPen(QColor(220, 220, 220))
            painter.drawRect(0, 0, width - 1, height - 1)
            painter.end()

            return pixmap


        def _update_point_colour_legend(self) -> None:
            if self.point_color_mode == "doppler":
                self.point_colour_legend.setPixmap(self._make_doppler_legend_pixmap())
                self.point_colour_legend_ticks.setText("-3 m/s        0 m/s        +3 m/s")
                self.point_colour_legend.setVisible(True)
                self.point_colour_legend_ticks.setVisible(True)
            elif self.point_color_mode == "snr":
                self.point_colour_legend.setPixmap(self._make_snr_legend_pixmap())
                self.point_colour_legend_ticks.setText("low SNR        mid SNR        high SNR")
                self.point_colour_legend.setVisible(True)
                self.point_colour_legend_ticks.setVisible(True)
            else:
                self.point_colour_legend.clear()
                self.point_colour_legend_ticks.clear()
                self.point_colour_legend.setVisible(False)
                self.point_colour_legend_ticks.setVisible(False)

        def toggle_filter_noise(self) -> None:
            self.filter_noise_points = bool(self.filter_noise_button.isChecked())
            self.filter_noise_button.setText("filter noise: on" if self.filter_noise_points else "filter noise: off")
            self._clear_projected_point_cache()
            self.refresh_target()
            if self.show_projected_pc_overlay:
                self.refresh_source()

        def toggle_pose3d(self) -> None:
            self.show_pose3d_in_pointcloud = bool(self.pose3d_button.isChecked())
            self.pose3d_button.setText("3D pose: on" if self.show_pose3d_in_pointcloud else "3D pose: off")
            self.refresh_target()

        def toggle_predicted_pose3d(self) -> None:
            self.show_predicted_pose3d_in_pointcloud = bool(
                self.predicted_pose3d_button.isChecked()
                and self.controller.has_pose_predictions
            )
            self.predicted_pose3d_button.setText(
                "predicted pose: on"
                if self.show_predicted_pose3d_in_pointcloud
                else "predicted pose: off"
            )
            self.refresh_target()
            self.refresh_status()

        def toggle_pose2d(self) -> None:
            self.show_pose2d_overlay = bool(self.pose2d_button.isChecked())
            self.pose2d_button.setText("2D pose: on" if self.show_pose2d_overlay else "2D pose: off")
            self.refresh_source()

        def toggle_projected_pc(self) -> None:
            self.show_projected_pc_overlay = bool(self.pc2d_button.isChecked())
            self.pc2d_button.setText("2D PC: on" if self.show_projected_pc_overlay else "2D PC: off")
            self.refresh_source()

        def toggle_video_frames(self) -> None:
            self.show_video_frames = bool(self.video_toggle_button.isChecked())
            self.video_toggle_button.setText("video: on" if self.show_video_frames else "video: off")
            self.refresh_source()
        
        def toggle_point_pan_mode(self) -> None:
            self.point_view_pan_mode = bool(self.point_pan_button.isChecked())
            self.point_pan_button.setText("3D pan: on" if self.point_view_pan_mode else "3D pan: off")
            self.point_panel.set_pan_mode(self.point_view_pan_mode)

        def set_pc_window_radius(self) -> None:
            value, ok = QInputDialog.getInt(
                self,
                "Set radar point-cloud frame window",
                "Display target point clouds from current target sample ± n frames:",
                int(self.pc_window_radius),
                0,
                200,
                1,
            )

            if not ok:
                return

            self.pc_window_radius = int(value)
            self.pc_window_button.setText(f"PC window: ±{self.pc_window_radius}")

            self._clear_point_window_caches()

            self.refresh_target()

            if self.show_projected_pc_overlay:
                self.refresh_source()

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

        def _current_target_points(self):
            key = (int(self.target_sample), int(self.pc_window_radius))

            if key != self._target_points_cache_key:
                self._target_points_cache_value = self.controller.get_target_points_window(
                    self.target_sample,
                    self.pc_window_radius,
                )
                self._target_points_cache_key = key

                # Projected points depend on the current target points.
                self._clear_projected_point_cache()

            return self._target_points_cache_value

        def _clear_point_window_caches(self) -> None:
            self._target_points_cache_key = None
            self._target_points_cache_value = None
            self._clear_projected_point_cache()

        def _clear_projected_point_cache(self) -> None:
            self._projected_pc_cache_key = None
            self._projected_pc_cache_value = None

        def _current_target_points_for_display(self):
            points = self._current_target_points()
            return filter_noise_points(points) if self.filter_noise_points else points

        def _current_projected_points(self):
            key = (
                int(self.target_sample),
                int(self.pc_window_radius),
                bool(self.filter_noise_points),
            )

            if key != self._projected_pc_cache_key:
                self._projected_pc_cache_value = project_pc_to_digital(
                    self._current_target_points(),
                    filter_noise=self.filter_noise_points,
                )
                self._projected_pc_cache_key = key

            return self._projected_pc_cache_value

        def refresh_source(self) -> None:
            try:
                frame = self.controller.get_rgb_frame(self.source_sample) if self.show_video_frames else None
                pose2d = self.controller.get_source_pose2d(self.source_sample) if self.show_pose2d_overlay else None
                projected_points = self._current_projected_points() if self.show_projected_pc_overlay else None
                self.video_panel.set_options(
                    show_video=self.show_video_frames,
                    show_pose2d=self.show_pose2d_overlay,
                    show_projected_pc=self.show_projected_pc_overlay,
                    projected_pc_color_mode=self.point_color_mode,
                )
                self.video_panel.set_scene(frame_rgb=frame, pose2d=pose2d, projected_points=projected_points)
            except Exception as exc:
                self.video_panel.setText(f"RGB/overlay unavailable: {exc}")

        def refresh_target(self) -> None:
            try:
                pose3d = (
                    self.controller.get_source_pose3d(self.source_sample)
                    if self.show_pose3d_in_pointcloud
                    else None
                )
                prediction_frame = (
                    self.controller.get_target_pose_prediction(self.target_sample)
                    if self.show_predicted_pose3d_in_pointcloud
                    else None
                )
                predicted_pose3d = (
                    prediction_frame.pose_sensor_xyz_m
                    if prediction_frame is not None
                    else None
                )
                self.point_panel.set_options(
                    color_mode=self.point_color_mode,
                    filter_noise=self.filter_noise_points,
                    show_pose3d=self.show_pose3d_in_pointcloud,
                    show_predicted_pose3d=self.show_predicted_pose3d_in_pointcloud,
                )
                self.point_panel.set_scene(
                    self._current_target_points(),
                    pose3d=pose3d,
                    predicted_pose3d=predicted_pose3d,
                )
            except Exception as exc:
                self.status.setText(f"Radar points/pose overlay unavailable: {exc}")

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
            prediction_text = ""
            if self.controller.has_pose_predictions:
                prediction_frame = self.controller.get_target_pose_prediction(
                    self.target_sample
                )
                if prediction_frame is None:
                    prediction_text = "; predicted_pose=unavailable"
                else:
                    eligibility = prediction_frame.metric_eligible_single_person
                    if eligibility is None:
                        eligibility_text = "unknown"
                    else:
                        eligibility_text = "yes" if eligibility else "no"
                    prediction_text = (
                        "; predicted_pose=available"
                        f"; prediction_single_person_metric_eligible={eligibility_text}"
                    )

            self.status.setText(
                f"source {self.controller.source_device_type}/{self.controller.source_run_id} sample={self.source_sample} "
                f"of {self.source_max}; target {self.controller.target_device_type}/{self.controller.target_run_id} "
                f"sample={self.target_sample} of {self.target_max}; {mapping_text}{prediction_text}"
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
