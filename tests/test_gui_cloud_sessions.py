"""Session selection, GUI wiring, and anchor provenance with two raw versions."""
from dataclasses import FrozenInstanceError
import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from test_versioned_raw_point_clouds import backend, write_package
from sync_workbench.experimental.anchoring_gui.controllers import AnchoringController
from sync_workbench.experimental.anchoring_gui.session_selection import session_choices, resolve_session, make_session_dialog_class
from sync_workbench.services.raw_point_cloud_import_service import RawPointCloudImportService
from sync_workbench.services.anchor_service import AnchorService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.storage.ragged_npz import RaggedNpzWriter


@pytest.fixture
def sessions(backend, tmp_path):
    db, root = backend
    ids = []
    for seed in ('a', 'b'):
        vid, _ = write_package(tmp_path/seed, seed, partial=(seed == 'b'))
        RawPointCloudImportService().import_package(tmp_path/seed, db, root)
        ids.append(vid)
    store = SQLiteCoreStore(db)
    store.write_table('DEVICE_RUN', pd.DataFrame([dict(subject_id='P',run_id='RGB',device_type='kinect_rgb',nominal_fps=1)]), if_exists='append')
    store.write_table('RUN_SAMPLE', pd.DataFrame([dict(subject_id='P',run_id='RGB',device_type='kinect_rgb',sample_index=i,sample_kind='frame') for i in range(3)]), if_exists='append')
    store.write_table('MAPPING_VERSION', pd.DataFrame([dict(subject_id='P',mapping_version_id=mid,source_run_id='RGB',source_device_type='kinect_rgb',target_run_id='R',target_device_type=dev) for mid,dev in [('raw_map','radar_raw'),('online_map','radar_pc')]]))
    store.write_table('SAMPLE_MAPPING', pd.DataFrame([dict(subject_id='P',mapping_version_id=mid,source_run_id='RGB',source_device_type='kinect_rgb',source_sample_index=i,target_run_id='R',target_device_type=dev,target_sample_index=i,rank=1,is_primary=True) for mid,dev in [('raw_map','radar_raw'),('online_map','radar_pc')] for i in range(3)]))
    points = np.array([[1,2,3,4,5,6]],dtype=np.float32)
    RaggedNpzWriter.write(root/'online.npz', {0:points},tail_shape=(6,),dtype='float32')
    store.write_table('SAMPLE_ARTIFACT', pd.DataFrame([dict(subject_id='P',run_id='R',device_type='radar_pc',sample_index=0,artifact_role='radar_points',artifact_ref='online.npz',artifact_format='ragged_npz',artifact_member_key='{"sample_index":0}')]),if_exists='append')
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE POINT_CLOUD_VERSION SET artifact_ref='online.npz' WHERE device_type='radar_pc'")
    return db, root, ids


def controller(sessions, version, mapping='raw_map', **kwargs):
    db, root, _ = sessions
    return AnchoringController(sqlite_path=db,artifact_root=root,rgb_root=root,subject_id='P',mapping_version_id=mapping,
                               point_cloud_version_id=version,annotator_id='tester',**kwargs)


def test_session_choices_and_strict_acquisition_binding(sessions):
    db, root, ids = sessions
    assert len(session_choices(db,root,'P')) == 3
    raw = resolve_session(db,root,'P','raw_map',ids[0])
    assert raw.target_device_type == 'radar_raw'
    with pytest.raises(FrozenInstanceError):
        raw.point_cloud_version_id = ids[1]
    for mapping,version in [('online_map',ids[0]),('raw_map','online_original'),('raw_map',None)]:
        with pytest.raises(ValueError):
            resolve_session(db,root,'P',mapping,version)
    assert resolve_session(db,root,'P','online_map').point_cloud_version_id == 'online_original'


def test_controller_reads_selected_version_and_preserves_anchor_provenance(sessions,tmp_path):
    db,root,ids=sessions
    first=controller(sessions,ids[0])
    assert first.initial_samples() == (0,0)
    assert first.get_target_points(0).shape == (6,6)
    assert first.get_target_points_window(0,radius=2).shape == (12,6)
    aid=first.place_anchor(0,0,label='v1',point_window_radius=2,filter_noise=True)
    first.close()
    second=controller(sessions,ids[1])
    assert second.get_target_points(0).shape == (7,6)
    assert aid in second.list_anchors().anchor_id.values
    second.place_anchor(2,2,label='v2')
    export=second.export_anchors(tmp_path/'anchors.json')
    assert export['session']['point_cloud']['point_cloud_version_id'] == ids[1]
    provenance=[json.loads(row['notes'])['provenance'] for row in export['ANCHOR']]
    assert {row['point_cloud']['point_cloud_version_id'] for row in provenance} == set(ids)
    original=next(p for p in provenance if p['point_cloud']['point_cloud_version_id']==ids[0])
    assert original['display']['point_window_radius']==2
    assert original['display']['excluded_association_ids']==[253,254,255]
    assert original['point_cloud']['payload_fingerprint']
    assert original['session_id'] != export['session']['session_id']
    # The existing JSON importer preserves notes/provenance without adding versions to endpoint keys.
    copy=tmp_path/'import.sqlite'
    SQLiteCoreStore(copy).initialise_empty()
    AnchorService(copy).import_anchors_json(tmp_path/'anchors.json')
    imported=SQLiteCoreStore(copy).read_table('ANCHOR')
    assert set(imported.notes)=={row['notes'] for row in export['ANCHOR']}
    assert 'point_cloud_version_id' not in export['ANCHOR_MEMBER'][0]
    second.close()
    assert not second.payloads._npz_cache


def test_missing_cloud_disallows_anchor_and_online_predictions_do_not_transfer(sessions):
    _,_,ids=sessions
    c=controller(sessions,ids[1])
    assert c.get_target_summary(1)['point_status']=='unprocessed'
    with pytest.raises(KeyError):
        c.place_anchor(1,1)
    with pytest.raises(KeyError,match='unprocessed'):
        c.place_anchor(0,0,point_window_radius=1)
    with pytest.raises(ValueError,match='online samples'):
        controller(sessions,ids[0],pose_predictions_path='online-predictions.npz')
    with pytest.raises(AttributeError):
        c.point_cloud_version_id=ids[0]
    assert c.list_anchors().empty
    c.close()


@pytest.fixture
def qt_app():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    yield app


def test_launch_dialog_changes_source_before_accepting(sessions,qt_app):
    db,root,ids=sessions
    dialog=make_session_dialog_class()(session_choices(db,root,'P'),preferred_mapping='raw_map',preferred_version=ids[1])
    assert dialog.selection.point_cloud_version_id==ids[1]
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData(json.dumps(['radar_pc','R'])))
    assert dialog.selection.point_cloud_version_id=='online_original'
    assert dialog.selection.mapping_version_id=='online_map'
    dialog.source_combo.setCurrentIndex(dialog.source_combo.findData(json.dumps(['radar_raw','R'])))
    dialog.version_combo.setCurrentIndex(dialog.version_combo.findData(ids[0]))
    assert dialog.selection.point_cloud_version_id==ids[0]
    dialog.accept()
    assert dialog.result()==1
    empty=make_session_dialog_class()([])
    assert empty.selection is None
    empty.reject()


def test_window_clears_stale_cloud_and_records_display_context(sessions,qt_app,monkeypatch):
    pytest.importorskip('pyqtgraph')
    from PySide6.QtWidgets import QComboBox
    from sync_workbench.experimental.anchoring_gui.main_window import make_main_window_class
    db,root,ids=sessions
    c=controller(sessions,ids[1])
    monkeypatch.setattr(c,'initial_samples',lambda: (0,2))
    monkeypatch.setattr(c,'get_rgb_frame',lambda sample: np.zeros((72,128,3),np.uint8))
    window=make_main_window_class()(c)
    assert window.target_spin.value()==window.target_sample==2
    assert len(window.point_panel.scatter.pos)==7
    assert window.place_anchor_button.isEnabled()
    assert not window.findChildren(QComboBox)  # Session source/version cannot change mid-session.
    window.target_sample=1
    window.show_projected_pc_overlay=True
    window.refresh_all()
    assert len(window.point_panel.scatter.pos)==0
    assert not window.video_panel.pixmap().isNull()
    assert 'not processed' in window.cloud_status.text()
    assert not window.place_anchor_button.isEnabled()
    window.target_sample=2
    window.refresh_all()
    assert len(window.point_panel.scatter.pos)==7
    assert window.place_anchor_button.isEnabled()
    window.close()


def test_online_session_provenance_and_legacy_online_reading(sessions,tmp_path):
    db,root,_=sessions
    c=controller(sessions,'online_original',mapping='online_map')
    assert c.get_target_points(0).shape==(1,6)
    c.place_anchor(0,0)
    notes=json.loads(c.list_anchors().iloc[0]['notes'])
    assert notes['provenance']['point_cloud']['point_cloud_version_id']=='online_original'
    c.close()
    # A legacy online store can still launch without a write/migration at startup.
    with sqlite3.connect(db) as conn:
        conn.execute('DROP TABLE POINT_CLOUD_VERSION')
    choices=session_choices(db,root,'P')
    assert len(choices)==1 and choices[0].point_cloud_version_id=='online_original'


def test_cli_forwards_explicit_cloud_selection(monkeypatch):
    from sync_workbench.cli.main import main
    from sync_workbench.experimental.anchoring_gui import app
    received={}
    def launch(**kwargs):
        received.update(kwargs)
        return 0
    monkeypatch.setattr(app,'run_anchoring_gui',launch)
    assert main(['anchoring-gui','--sqlite','db','--artifact-root','artifacts','--rgb-root','rgb',
                 '--subject','P','--mapping-version','raw_map','--point-cloud-version','raw_id'])==0
    assert received['point_cloud_version_id']=='raw_id'
    assert received['mapping_version_id']=='raw_map'


def test_cancel_launch_does_not_create_session_or_write_database(sessions,qt_app,monkeypatch):
    from sync_workbench.experimental.anchoring_gui import app
    from sync_workbench.ingestion.raw_point_cloud_package import file_hash
    db,root,_=sessions
    class CancelDialog:
        def __init__(self,*args,**kwargs): pass
        def exec(self): return 0
    monkeypatch.setattr(app,'make_session_dialog_class',lambda:CancelDialog)
    monkeypatch.setattr(app,'AnchoringController',lambda **kwargs: pytest.fail('Controller created after cancellation'))
    before=file_hash(db)
    assert app.run_anchoring_gui(sqlite_path=db,artifact_root=root,rgb_root=root,subject_id='P')==0
    assert file_hash(db)==before
