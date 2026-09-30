import json
import numpy as np
import pytest

from test_versioned_raw_point_clouds import backend
from test_gui_cloud_sessions import sessions, controller, qt_app
from sync_workbench.experimental.anchoring_gui.main_window import make_main_window_class


@pytest.fixture
def window(sessions,qt_app,monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox,'warning',lambda *args:None)
    c=controller(sessions,sessions[2][0])
    w=make_main_window_class()(c)
    yield w,c
    w.close()


def test_rgb_failure_clears_display_and_blocks_button_and_slot(window,monkeypatch):
    w,c=window
    assert w.place_anchor_button.isEnabled()
    def fail(i):raise OSError('test decoder failure')
    monkeypatch.setattr(c,'get_rgb_frame',fail)
    w.refresh_all()
    assert w.video_panel.pixmap().isNull()
    assert 'test decoder failure' in w.status.text()
    assert not w.place_anchor_button.isEnabled()
    w.refresh_target()
    w.refresh_status()
    assert not w.place_anchor_button.isEnabled()
    w.place_anchor()  # Shortcut/programmatic invocation must also reject.
    assert c.list_anchors().empty
    assert 'required RGB/cloud' in w.action_status.text()
    w.show_video_frames=False
    w.refresh_all()
    assert not w.place_anchor_button.isEnabled()
    monkeypatch.setattr(c,'get_rgb_frame',lambda i:np.zeros((72,128,3),np.uint8))
    w.refresh_all()
    assert w.place_anchor_button.isEnabled()


def test_processed_empty_cloud_is_anchorable_and_corrupt_cloud_is_not(window,monkeypatch):
    w,c=window
    w.target_spin.setValue(1)
    w.go_target_from_spin()
    assert 'empty processed' in w.cloud_status.text()
    assert len(w.point_panel.scatter.pos)==0
    assert w.place_anchor_button.isEnabled()
    w.place_anchor()
    assert len(c.list_anchors())==1
    assert 'Saved anchor' in w.action_status.text()
    assert 'Recoverable copy' in w.action_status.text()
    previous=w.action_status.text()
    w.refresh_all()
    assert w.action_status.text()==previous
    def fail(*args):raise ValueError('corrupt bundle')
    monkeypatch.setattr(c,'get_target_points_window',fail)
    w._clear_point_window_caches()
    w.refresh_all()
    assert 'corrupt bundle' in w.cloud_status.text()
    assert len(w.point_panel.scatter.pos)==0
    assert not w.place_anchor_button.isEnabled()


def test_fallback_is_visible_and_persists_through_status_refresh(window,monkeypatch):
    w,c=window
    def missing(sample):raise KeyError('outside mapping coverage')
    monkeypatch.setattr(c,'sync_target_to_source',missing)
    w.toggle_play('both')
    assert 'Nominal-rate playback fallback' in w.navigation_status.text()
    w._mapped_target_or_fallback(1,0.5)
    w.refresh_status()
    assert 'outside mapping coverage' in w.navigation_status.text()
    w._stop_playback()
    w.step_both(1)
    assert 'fallback' in w.navigation_status.text()


def test_export_success_and_failure_stay_visible(window,tmp_path,monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    w,c=window
    w.place_anchor()
    out=tmp_path/'return.json'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args:(str(out),'JSON'))
    w.export_anchors()
    assert 'Exported 1 anchors' in w.action_status.text()
    assert c.default_export_path().parent.is_dir()
    assert len(json.loads(out.read_text())['ANCHOR'])==1
    def fail(*args):raise OSError('destination not writable')
    monkeypatch.setattr(c,'export_anchors',fail)
    w.export_anchors()
    w.refresh_all()
    assert 'destination not writable' in w.action_status.text()
    monkeypatch.setattr(c,'default_export_path',fail)
    w.export_anchors()
    assert 'destination not writable' in w.action_status.text()
