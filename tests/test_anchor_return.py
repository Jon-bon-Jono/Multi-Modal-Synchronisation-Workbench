"""Return/import, durability and recovery checks without manual GUI activity."""
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pytest

from test_v0_2_2_piecewise import _write_toy_sqlite
from test_student_package import inputs
from sync_workbench.services.anchor_service import AnchorService, AnchorEndpoint
from sync_workbench.services.anchor_transfer import atomic_json
from sync_workbench.deployment.package_layout import sha256, json_digest
from sync_workbench.deployment.student_package import export_student_package
from sync_workbench.deployment.student_runtime import prepare_student_launch
from sync_workbench.experimental.anchoring_gui.controllers import AnchoringController
from sync_workbench.services.piecewise_sync_service import PiecewiseSyncService
from sync_workbench.sync.mapping import TimelineSelection

PAIR = dict(subject_id='P001', source_run_id='RGB-A',source_device_type='kinect_rgb',target_run_id='PC-A',target_device_type='radar_pc')


@pytest.fixture
def exchange(tmp_path):
    source,dest = tmp_path/'source.sqlite',tmp_path/'master.sqlite'
    _write_toy_sqlite(source)
    shutil.copyfile(source,dest)
    service = AnchorService(source)
    for aid,i in [('first',0),('last',4)]:
        service.create_pair_anchor(subject_id='P001',anchor_id=aid,source=AnchorEndpoint('RGB-A','kinect_rgb',i,'source'),target=AnchorEndpoint('PC-A','radar_pc',i,'target'))
    path = tmp_path/'return.json'
    payload = service.export_pair_anchors_json(path,**PAIR)
    return source,dest,path,payload


def reseal(path,payload):
    payload['content_sha256'] = json_digest({k:v for k,v in payload.items() if k!='content_sha256'})
    atomic_json(path,payload)


def test_two_anchor_return_retry_and_fit(exchange):
    source,dest,path,payload=exchange
    before=sha256(dest)
    service=AnchorService(dest)
    assert service.import_anchors_json(path,dry_run=True)['anchors']==2
    assert sha256(dest)==before
    assert service.import_anchors_json(path)['anchors']==2
    first=sha256(dest)
    assert service.import_anchors_json(path)['unchanged']==2
    assert sha256(dest)==first
    result=PiecewiseSyncService(dest).fit_piecewise_and_generate_mapping(
        TimelineSelection('P001','RGB-A','kinect_rgb','rgb_t'),TimelineSelection('P001','PC-A','radar_pc','pc_t'),
        sync_model_id='returned_fit',mapping_version_id='returned_map',top_k=1)
    assert len(result.model_anchor)==2
    assert result.sample_mapping.source_sample_index.nunique()==5
    assert service.import_anchors_json(path)['unchanged']==2


def test_conflict_rejects_entire_batch_and_never_overwrites(exchange):
    source,dest,path,payload=exchange
    service=AnchorService(dest)
    service.import_anchors_json(path)
    payload['ANCHOR'][0]['label']='different'
    extra=dict(payload['ANCHOR'][1],anchor_id='new')
    payload['ANCHOR'].append(extra)
    payload['ANCHOR_MEMBER'] += [dict(r,anchor_id='new') for r in payload['ANCHOR_MEMBER'] if r['anchor_id']=='last']
    reseal(path,payload)
    before=sha256(dest)
    with pytest.raises(ValueError,match='conflicts; nothing imported'):
        service.import_anchors_json(path)
    with pytest.raises(ValueError,match='never overwrite'):
        service.import_anchors_json(path,overwrite=True)
    assert sha256(dest)==before


@pytest.mark.parametrize('change,match',[
    ('sample','unavailable sample'),('orphan','Orphan'),('missing_member','exactly one'),
    ('duplicate','Duplicate'),('wrong_pair','[Aa]cquisition'),('fractional','integers'),('confidence','Confidence'),
])
def test_malformed_batch_is_rejected_without_any_write(exchange,change,match):
    _,dest,path,payload=exchange
    if change=='sample':payload['ANCHOR_MEMBER'][0]['sample_index']=999999
    elif change=='orphan':payload['ANCHOR_MEMBER'][0]['anchor_id']='unknown'
    elif change=='missing_member':payload['ANCHOR_MEMBER'].pop()
    elif change=='duplicate':payload['ANCHOR'].append(payload['ANCHOR'][0])
    elif change=='wrong_pair':payload['pair']['target_run_id']='another'
    elif change=='fractional':payload['ANCHOR_MEMBER'][0]['sample_index']=1.5
    elif change=='confidence':payload['ANCHOR'][0]['confidence']=2
    reseal(path,payload)
    before=sha256(dest)
    with pytest.raises(ValueError,match=match):AnchorService(dest).import_anchors_json(path)
    assert sha256(dest)==before


def test_checksum_and_acquisition_mismatch(exchange):
    _,dest,path,payload=exchange
    payload['ANCHOR'][0]['label']='unsealed change'
    atomic_json(path,payload)
    with pytest.raises(ValueError,match='checksum'):AnchorService(dest).import_anchors_json(path)
    reseal(path,payload)
    with sqlite3.connect(dest) as conn:conn.execute("UPDATE DEVICE_RUN SET nominal_fps=999")
    before=sha256(dest)
    with pytest.raises(ValueError,match='acquisition'):AnchorService(dest).import_anchors_json(path)
    assert sha256(dest)==before


def test_cli_reports_rejected_return(exchange,capsys):
    from sync_workbench.cli.main import main
    _,dest,path,payload=exchange
    payload['ANCHOR'][0]['label']='changed without checksum'
    atomic_json(path,payload)
    assert main(['import-anchors','--sqlite',str(dest),'--input',str(path)])==2
    assert json.loads(capsys.readouterr().out)['status']=='rejected'


def test_sql_failure_rolls_back_whole_import(exchange):
    _,dest,path,_=exchange
    with sqlite3.connect(dest) as conn:
        conn.execute("CREATE TRIGGER reject_member BEFORE INSERT ON ANCHOR_MEMBER BEGIN SELECT RAISE(ABORT,'disk simulation'); END")
    before=sha256(dest)
    with pytest.raises(sqlite3.IntegrityError):AnchorService(dest).import_anchors_json(path)
    assert sha256(dest)==before


def test_create_delete_are_atomic_and_keep_indexes(exchange):
    source,_,_,_=exchange
    with sqlite3.connect(source) as conn:
        conn.execute('CREATE UNIQUE INDEX anchor_key ON ANCHOR(subject_id,anchor_id)')
        conn.execute("CREATE TRIGGER reject_member BEFORE INSERT ON ANCHOR_MEMBER BEGIN SELECT RAISE(ABORT,'disk simulation'); END")
    before=sha256(source)
    service=AnchorService(source)
    with pytest.raises(sqlite3.IntegrityError):
        service.create_pair_anchor(subject_id='P001',anchor_id='broken',source=AnchorEndpoint('RGB-A','kinect_rgb',1,'source'),target=AnchorEndpoint('PC-A','radar_pc',1,'target'))
    assert sha256(source)==before
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON ANCHOR BEGIN SELECT RAISE(ABORT,'disk simulation'); END")
    before=sha256(source)
    with pytest.raises(sqlite3.IntegrityError):service.delete_anchor('P001','first')
    assert sha256(source)==before
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='anchor_key'").fetchone()


def test_atomic_export_preserves_previous_file_on_failure(exchange,monkeypatch):
    source,_,path,_=exchange
    before=path.read_bytes()
    def fail(*args):raise OSError('replace failed')
    monkeypatch.setattr('sync_workbench.services.anchor_transfer.os.replace',fail)
    with pytest.raises(OSError):AnchorService(source).export_pair_anchors_json(path,**PAIR)
    assert path.read_bytes()==before
    assert not list(path.parent.glob('.return.json.*.tmp'))


@pytest.fixture
def student(inputs):
    export_student_package(**inputs)
    root=inputs['output']
    options=prepare_student_launch(root,'student01')
    c=AnchoringController(**options)
    c.get_rgb_frame=lambda i:np.zeros((72,128,3),np.uint8)
    yield inputs,root,c
    c.close()


def test_student_return_requires_original_manifest_and_preserves_identity(student,tmp_path):
    inputs,root,c=student
    aid=c.place_anchor(0,0)
    assert Path(c.last_write_result['recovery_path']).is_file()
    path=tmp_path/'return.json'
    c.export_anchors(path)
    service=AnchorService(inputs['sqlite_path'])
    before=sha256(inputs['sqlite_path'])
    with pytest.raises(ValueError,match='original --package-manifest'):service.import_anchors_json(path)
    assert sha256(inputs['sqlite_path'])==before
    result=service.import_anchors_json(path,expected_manifest=root/'manifest.json')
    assert result['anchors']==1
    notes=service.store.read_table('ANCHOR').set_index('anchor_id').loc[aid,'notes']
    assert json.loads(notes)['provenance']['package']['package_id']==c.package_id
    assert service.import_anchors_json(path,expected_manifest=root/'manifest.json')['unchanged']==1


@pytest.mark.parametrize('change',['package','annotator','cloud','manifest'])
def test_student_wrong_identity_is_rejected(student,tmp_path,change):
    inputs,root,c=student
    c.place_anchor(0,0)
    path=tmp_path/'return.json'
    payload=c.export_anchors(path)
    expected=root/'manifest.json'
    if change=='package':payload['session']['package']['package_id']='wrong'
    elif change=='annotator':payload['session']['annotator_id']='someone-else'
    elif change=='cloud':
        notes=json.loads(payload['ANCHOR'][0]['notes'])
        notes['provenance']['point_cloud']['point_cloud_version_id']='different'
        payload['ANCHOR'][0]['notes']=json.dumps(notes)
    else:
        manifest=json.loads(expected.read_text())
        manifest['package_id']='different'
        manifest['manifest_sha256']=json_digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})
        expected=tmp_path/'other-manifest.json'
        atomic_json(expected,manifest)
    reseal(path,payload)
    before=sha256(inputs['sqlite_path'])
    with pytest.raises(ValueError):AnchorService(inputs['sqlite_path']).import_anchors_json(path,expected_manifest=expected)
    assert sha256(inputs['sqlite_path'])==before


def test_return_and_reexport_do_not_require_displayed_cloud_installed(student,tmp_path):
    inputs,root,c=student
    c.place_anchor(0,0)
    path=tmp_path/'student.json'
    c.export_anchors(path)
    with sqlite3.connect(inputs['sqlite_path']) as conn:
        conn.execute('DELETE FROM POINT_CLOUD_VERSION')
    service=AnchorService(inputs['sqlite_path'])
    assert service.import_anchors_json(path,expected_manifest=root/'manifest.json')['anchors']==1
    # A coordinator export may combine imported student and original master anchors.
    master_export=tmp_path/'master.json'
    service.export_pair_anchors_json(master_export,subject_id='P001',source_run_id='Session-A',source_device_type='kinect_rgb',target_run_id='Session-A',target_device_type='radar_pc')
    assert service.import_anchors_json(master_export)['unchanged']==2


def test_recovery_preserves_deleted_anchor_and_failure_reports_saved(student,monkeypatch):
    _,root,c=student
    aid=c.place_anchor(0,0)
    snapshot=Path(c.last_write_result['recovery_path'])
    c.delete_anchor(aid)
    assert c.list_anchors().empty
    assert json.loads(snapshot.read_text())['ANCHOR'][0]['anchor_id']==aid
    assert not json.loads(Path(c.last_write_result['recovery_path']).read_text())['ANCHOR']
    restored=c.anchors.import_anchors_json(snapshot,expected_manifest=root/'manifest.json')
    assert restored['anchors']==1 and aid in c.list_anchors().anchor_id.values
    def fail():raise OSError('no space for recovery')
    monkeypatch.setattr(c,'_recovery_snapshot',fail)
    aid=c.place_anchor(1,1)
    assert aid in c.list_anchors().anchor_id.values
    assert c.last_write_result['database_saved']
    assert 'recovery copy failed' in c.last_write_result['warning']
    with pytest.raises(OSError):c.delete_anchor(aid)
    assert aid in c.list_anchors().anchor_id.values


def test_rgb_failure_rejects_anchor_and_export_protects_package(student):
    _,root,c=student
    def fail(i):raise OSError('missing RGB')
    c.get_rgb_frame=fail
    with pytest.raises(OSError,match='missing RGB'):c.place_anchor(0,0)
    assert c.list_anchors().empty
    for path in (root/'config.json',root/'work/annotator.json',root/'assets/rgb/video.json'):
        with pytest.raises(ValueError):c.export_anchors(path)


def test_linked_model_anchor_cannot_be_deleted(exchange):
    source,_,_,_=exchange
    with sqlite3.connect(source) as conn:conn.execute("INSERT INTO MODEL_ANCHOR VALUES ('P001','existing_fit','first')")
    before=sha256(source)
    with pytest.raises(ValueError,match='synchronization model'):AnchorService(source).delete_anchor('P001','first')
    assert sha256(source)==before
