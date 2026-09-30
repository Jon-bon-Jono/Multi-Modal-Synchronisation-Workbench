"""Small-screen layout checks use offscreen Qt and a non-GL scene stand-in."""
from pathlib import Path
import os

import numpy as np
import pytest

from test_versioned_raw_point_clouds import backend
from test_gui_cloud_sessions import sessions, controller, qt_app


@pytest.mark.parametrize('width,height,font_size',[(1024,640,9),(1280,720,12),(1366,768,12)])
def test_compact_status_does_not_squeeze_controls(sessions,qt_app,monkeypatch,tmp_path,width,height,font_size):
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QLabel,QDialog,QPlainTextEdit
    from sync_workbench.experimental.anchoring_gui import main_window

    class SceneStandIn(QLabel):
        def set_options(self,**kwargs):pass
        def set_scene(self,*args,**kwargs):self.setText('Point-cloud scene (layout test)')

    monkeypatch.setattr(main_window,'PointCloudPanel',SceneStandIn)
    old_font=qt_app.font()
    # Windows offscreen Qt has no system font discovery; load a real font
    # so layout measurements match the student's desktop rather than tofu glyphs.
    family = old_font.family()
    system_font = Path(os.environ.get('WINDIR', '/nonexistent')) / 'Fonts/segoeui.ttf'
    if system_font.is_file():
        font_id = QFontDatabase.addApplicationFont(str(system_font))
        assert font_id >= 0
        family = QFontDatabase.applicationFontFamilies(font_id)[0]
    qt_app.setFont(QFont(family,font_size))
    c=controller(sessions,sessions[2][0])
    # Real video dimensions previously inflated QLabel's preferred layout size.
    c.get_rgb_frame=lambda i:np.zeros((720,1280,3),np.uint8)
    w=main_window.make_main_window_class()(c)
    try:
        w.resize(width,height)
        w.show()
        qt_app.processEvents()
        assert w.width()==width and w.height()==height
        assert w.controls_scroll.height()>=180
        assert w.controls_scroll.viewport().height()>=155
        assert w.summary_status.height()<=w.summary_status.fontMetrics().height()+6
        assert w.notice_status.isHidden()
        for label in (w.status,w.action_status,w.navigation_status):assert label.isHidden()
        assert not w.session_label.wordWrap()
        assert 'run_id' not in w.summary_status.text() and 'rank=' not in w.summary_status.text()
        long_path=str(tmp_path/'very long recovery directory'/'anchor_snapshot.json')*5
        c.last_write_result={'message':'Saved anchor long-anchor-identifier','recovery_path':long_path,'warning':''}
        w._show_write_result()
        qt_app.processEvents()
        assert w.notice_status.isVisible()
        assert w.notice_status.text()=='Anchor saved | Recovery copy saved'
        assert long_path in w.notice_status.toolTip()
        assert w.notice_status.height()<=w.notice_status.fontMetrics().height()+6
        assert w.controls_scroll.height()>=180 and w.width()==width
        # The complete diagnostic state is inspectable without consuming main-window height.
        captured=[]
        def inspect_dialog(dialog):
            captured.append(dialog.findChild(QPlainTextEdit).toPlainText())
            return 0
        monkeypatch.setattr(QDialog,'exec',inspect_dialog)
        w.show_session_details()
        assert long_path in captured[0] and c.point_cloud_version_id in captured[0]
        w._set_navigation_status('Nominal-rate playback fallback: '+('long lookup failure ' * 30))
        w._source_error='RGB unavailable: '+('long decode failure ' * 30)
        w.refresh_status()
        qt_app.processEvents()
        assert w.summary_status.text().startswith('FALLBACK')
        assert 'RGB unavailable' in w.notice_status.text()
        assert w.notice_status.height()<=w.notice_status.fontMetrics().height()+6
        assert w.controls_scroll.height()>=180
        assert w.height()==height
        if width==1280:
            # Save the real controls/status widgets as an automated visual-review artifact.
            w._source_error=''
            w._set_navigation_status('Both-stream playback: navigation mapping with manual offset.')
            w._action_summary=''
            w.source_max,w.target_max,w.target_sample=34285,61786,4749
            w._update_compact_status()
            qt_app.processEvents()
            destination=tmp_path/'compact_controls_preview.png'
            top=w.controls_scroll.geometry().top()
            assert w.grab(QRect(0,top,width,height-top)).save(str(destination))
    finally:
        w.close()
        qt_app.setFont(old_font)
