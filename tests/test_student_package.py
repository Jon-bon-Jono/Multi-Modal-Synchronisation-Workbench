"""WP1 checks use synthetic assets and never open the student GUI."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from test_v0_2_1_artifacts import _write_temp_package
from sync_workbench.deployment.package_layout import local_path, read_manifest, sha256, verify_package
from sync_workbench.deployment.student_package import export_student_package
from sync_workbench.deployment.student_runtime import prepare_student_launch, saved_annotator
from sync_workbench.services.artifact_build_service import ArtifactBuildService
from sync_workbench.services.ingestion_service import IngestionService
from sync_workbench.services.mapping_service import MappingService
from sync_workbench.services.anchor_service import AnchorEndpoint, AnchorService
from sync_workbench.services.payload_service import PayloadService
from sync_workbench.storage.sqlite_store import SQLiteCoreStore
from sync_workbench.sync.mapping import TimelineSelection


@pytest.fixture
def inputs(tmp_path):
    temp = tmp_path/'temp'
    _write_temp_package(temp)
    db = tmp_path/'master.sqlite'
    artifacts, rgb = tmp_path/'master-assets', tmp_path/'recordings'
    IngestionService().ingest_temp_package(temp, db)
    ArtifactBuildService().build_from_temp_package(temp,db,artifacts)
    video = rgb/'P001/Session-A/kinect_camera_recording_rgb_lq.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'synthetic-video-for-path-tests')
    MappingService(db).generate_nearest_mapping(TimelineSelection('P001','Session-A','kinect_rgb','rgb_wallclock_from_pts'),
        TimelineSelection('P001','Session-A','radar_pc','radar_pc_linear_from_index'), mapping_version_id='student_initial',top_k=1)
    AnchorService(db).create_pair_anchor(subject_id='P001',anchor_id='master_only',source=AnchorEndpoint('Session-A','kinect_rgb',0,'source'),target=AnchorEndpoint('Session-A','radar_pc',0,'target'))
    # Unrelated subject/run rows must not leak into the assignment.
    store=SQLiteCoreStore(db)
    store.write_table('SUBJECT',pd.DataFrame([dict(subject_id='OTHER')]),if_exists='append')
    store.write_table('DEVICE_RUN',pd.DataFrame([dict(subject_id='OTHER',run_id='Other',device_type='radar_pc')]),if_exists='append')
    return dict(sqlite_path=db,artifact_root=artifacts,rgb_root=rgb,output=tmp_path/'student package',
        subject_id='P001',mapping_version_id='student_initial',point_cloud_version_id='online_original',
        application_root=Path(__file__).resolve().parents[1])


def test_export_clean_subset_complete_assets_and_read_only_sources(inputs):
    original=sha256(inputs['sqlite_path'])
    video=next(inputs['rgb_root'].rglob('*.mp4'))
    before=sha256(video)
    result=export_student_package(**inputs)
    assert sha256(inputs['sqlite_path'])==original and sha256(video)==before
    root=inputs['output']
    manifest=read_manifest(root)
    assert result['package_id']==manifest['package_id']
    assert verify_package(root)['full_checksums']
    store=SQLiteCoreStore(root/'database/template.sqlite')
    assert store.read_table('SUBJECT').subject_id.tolist()==['P001']
    assert len(store.read_table('DEVICE_RUN'))==2
    for table in ['ANCHOR','ANCHOR_MEMBER','MODEL_ANCHOR']:
        assert store.read_table(table).empty
    assert len(store.read_table('MAPPING_VERSION'))==1
    assert len(store.read_table('POINT_CLOUD_VERSION'))==1
    assert {'pose2d','pose3d'}.issubset(manifest['pose_roles'])
    payloads=PayloadService(root/'database/template.sqlite',root/'assets/artifacts')
    np.testing.assert_array_equal(payloads.get_payload('P001','Session-A','radar_pc',0,'radar_points'),[[1,2,3,4,5,6]])
    assert payloads.get_payload('P001','Session-A','kinect_rgb',0,'pose3d').shape==(1,32,4)
    for row in store.read_table('RUN_ASSET').itertuples(index=False):
        base=root/('assets/rgb' if row.storage_key=='rgb' else 'assets/artifacts')
        assert local_path(base,row.asset_ref).is_file()
    assert not (root/'work').exists()
    assert str(inputs['rgb_root']) not in (root/'config.json').read_text()


def test_relocation_identity_and_annotation_persistence(inputs,tmp_path):
    export_student_package(**inputs)
    moved=tmp_path/'renamed with spaces'
    inputs['output'].rename(moved)
    options=prepare_student_launch(moved,'student01')
    assert options['rgb_root']==moved/'assets/rgb'
    assert options['package_provenance']['package_id']==read_manifest(moved)['package_id']
    assert saved_annotator(moved)=='student01'
    # Source media can disappear after packaging; all returned paths are self-contained.
    inputs['rgb_root'].rename(tmp_path/'disconnected-recordings')
    AnchorService(options['sqlite_path']).create_pair_anchor(subject_id='P001',anchor_id='student_anchor',
        source=AnchorEndpoint('Session-A','kinect_rgb',0,'source'),target=AnchorEndpoint('Session-A','radar_pc',0,'target'),
        provenance={'annotator_id':options['annotator_id'],'package':options['package_provenance']})
    resumed=prepare_student_launch(moved,'student01')
    assert len(SQLiteCoreStore(resumed['sqlite_path']).read_table('ANCHOR'))==1
    verify_package(moved)  # Mutable work does not invalidate immutable package checksums.
    with pytest.raises(ValueError,match='another annotator'):
        prepare_student_launch(moved,'student02')


def test_missing_asset_aborts_without_publishing(inputs):
    next(inputs['rgb_root'].rglob('*.mp4')).unlink()
    with pytest.raises(FileNotFoundError):
        export_student_package(**inputs)
    assert not inputs['output'].exists()


def test_corruption_detected_before_working_database_created(inputs):
    export_student_package(**inputs)
    asset=next((inputs['output']/'assets/rgb').iterdir())
    data=asset.read_bytes()
    asset.write_bytes(b'x'+data[1:])
    with pytest.raises(ValueError,match='checksum mismatch'):
        prepare_student_launch(inputs['output'],'student01',full_checksums=True)
    assert not (inputs['output']/'work').exists()


def test_output_protection_and_no_overwrite(inputs):
    readonly=dict(inputs,output=inputs['rgb_root']/'student-output')
    with pytest.raises(ValueError,match='read-only'):
        export_student_package(**readonly)
    export_student_package(**inputs)
    with pytest.raises(FileExistsError):
        export_student_package(**inputs)


@pytest.mark.parametrize('ref',['../escape','C:/absolute','/absolute','sub/../../escape','sub\\file'])
def test_portable_path_rejects_escape(tmp_path,ref):
    with pytest.raises(ValueError):local_path(tmp_path,ref)


def test_wrong_working_database_is_retained_and_rejected(inputs,tmp_path):
    export_student_package(**inputs)
    opts=prepare_student_launch(inputs['output'],'student01')
    with sqlite3.connect(opts['sqlite_path']) as conn:
        conn.execute("UPDATE STUDENT_PACKAGE SET package_id='wrong'")
    before=sha256(opts['sqlite_path'])
    with pytest.raises(ValueError,match='different package'):
        prepare_student_launch(inputs['output'],'student01')
    assert sha256(opts['sqlite_path'])==before


def test_assignment_cannot_be_changed_in_local_work(inputs):
    export_student_package(**inputs)
    opts=prepare_student_launch(inputs['output'],'student01')
    with sqlite3.connect(opts['sqlite_path']) as conn:
        conn.execute("UPDATE MAPPING_VERSION SET target_run_id='wrong'")
    with pytest.raises(ValueError,match='assignment differs'):
        prepare_student_launch(inputs['output'],'student01')


def test_package_identity_reaches_gui_anchor_and_export(inputs,tmp_path):
    from sync_workbench.experimental.anchoring_gui.controllers import AnchoringController
    export_student_package(**inputs)
    options=prepare_student_launch(inputs['output'],'student01')
    c=AnchoringController(**options)
    c.get_rgb_frame=lambda sample: np.zeros((72,128,3),np.uint8)
    c.place_anchor(0,0)
    result=c.export_anchors(tmp_path/'return.json')
    provenance=json.loads(result['ANCHOR'][0]['notes'])['provenance']
    assert provenance['package']==result['session']['package']==options['package_provenance']
    assert provenance['annotator_id']=='student01'
    c.close()


def test_navigation_only_and_launchers_are_portable(inputs):
    with sqlite3.connect(inputs['sqlite_path']) as conn:
        conn.execute("UPDATE SYNC_MODEL SET model_type='piecewise_affine'")
    with pytest.raises(ValueError,match='initial nearest-time'):
        export_student_package(**inputs)
    assert not inputs['output'].exists()


def test_launch_scripts_do_not_embed_master_paths(inputs):
    export_student_package(**inputs)
    for name in ['setup_windows.cmd','launch_windows.cmd','setup_macos.command','launch_macos.command']:
        content=(inputs['output']/name).read_text()
        assert str(inputs['sqlite_path']) not in content and str(inputs['rgb_root']) not in content
        assert 'find_environment_' in content
        assert 'PYTHONPATH' in content
    assert b'\r' not in (inputs['output']/'launch_macos.command').read_bytes()


def test_normal_startup_skips_bulk_checks_and_accepts_legacy_marker(inputs,monkeypatch):
    from sync_workbench.deployment import package_layout, student_runtime
    export_student_package(**inputs)
    root=inputs['output']
    real_hash=package_layout.sha256
    def only_small_files(path):
        path=Path(path)
        assert not path.is_relative_to(root/'assets') and path.suffix!='.sqlite', 'Unexpected bulk hashing'
        return real_hash(path)
    monkeypatch.setattr(package_layout,'sha256',only_small_files)
    real_connect=sqlite3.connect
    queries=[]
    def traced_connect(*args,**kwargs):
        conn=real_connect(*args,**kwargs)
        conn.set_trace_callback(queries.append)
        return conn
    monkeypatch.setattr(student_runtime.sqlite3,'connect',traced_connect)
    opts=prepare_student_launch(root,'student01')
    with sqlite3.connect(opts['sqlite_path']) as conn:
        # Legacy marker columns are accepted; per-sample notes are no longer fingerprinted.
        conn.execute('ALTER TABLE STUDENT_PACKAGE ADD COLUMN core_sha256 TEXT')
        conn.execute("UPDATE STUDENT_PACKAGE SET core_sha256='old-unused-digest'")
        conn.execute("UPDATE RUN_SAMPLE SET notes='changed' WHERE sample_index=0")
    queries.clear()
    prepare_student_launch(root,'student01')
    assert not any('pragma' in q.lower() for q in queries)
    assert not any('run_sample' in q.lower() or 'sample_mapping' in q.lower() for q in queries)
    assert all('limit 2' in q.lower() for q in queries if q.lstrip().lower().startswith('select'))


def test_checks_are_opt_in_and_verification_never_creates_work(inputs,capsys):
    from sync_workbench.deployment.student_runtime import main
    export_student_package(**inputs)
    root=inputs['output']
    asset=next((root/'assets/rgb').iterdir())
    asset.write_bytes(b'x'+asset.read_bytes()[1:])  # Same size: lightweight checks intentionally pass.
    args=['--package',str(root),'--verify-only']
    assert main(args)==0
    assert '"full_checksums": false' in capsys.readouterr().out
    assert main(args+['--sqlite-integrity-check'])==0
    output=capsys.readouterr().out
    assert 'database/template.sqlite' in output and '"full_checksums": false' in output
    assert main(args+['--full-checksums'])==2
    assert 'checksum mismatch' in capsys.readouterr().out
    assert not (root/'work').exists()


def test_sqlite_integrity_flag_detects_error_in_mutable_database(inputs,capsys):
    from sync_workbench.deployment.student_runtime import main
    export_student_package(**inputs)
    root=inputs['output']
    opts=prepare_student_launch(root,'student01')
    with closing(sqlite3.connect(opts['sqlite_path'])) as conn, conn:
        conn.execute('CREATE TABLE damaged_test_page (value INTEGER)')
        conn.execute('INSERT INTO damaged_test_page VALUES (1)')
        page=conn.execute("SELECT rootpage FROM sqlite_master WHERE name='damaged_test_page'").fetchone()[0]
        page_size=conn.execute('PRAGMA page_size').fetchone()[0]
    # Corrupt only the synthetic extra table's B-tree page, leaving identity rows readable.
    with opts['sqlite_path'].open('r+b') as stream:
        stream.seek((page-1)*page_size)
        stream.write(b'\xff')
    before=sha256(opts['sqlite_path'])
    args=['--package',str(root),'--verify-only']
    assert main(args)==0
    assert main(args+['--full-checksums'])==0  # Immutable package scan does not scan student work.
    assert main(args+['--sqlite-integrity-check'])==2
    assert 'Cannot open student package:' in capsys.readouterr().out
    assert sha256(opts['sqlite_path'])==before


def test_light_checks_reject_changed_cloud_identity(inputs):
    export_student_package(**inputs)
    opts=prepare_student_launch(inputs['output'],'student01')
    with sqlite3.connect(opts['sqlite_path']) as conn:
        conn.execute("UPDATE POINT_CLOUD_VERSION SET payload_fingerprint='wrong'")
    with pytest.raises(ValueError,match='point-cloud identity'):
        prepare_student_launch(inputs['output'],'student01')


def test_check_flags_reach_gui_launch_preparation(monkeypatch):
    from sync_workbench.deployment import student_runtime
    from sync_workbench.experimental.anchoring_gui import app
    seen=[]
    monkeypatch.setattr(student_runtime,'prepare_student_launch',lambda *args,**kw: seen.append(kw) or {})
    monkeypatch.setattr(app,'run_anchoring_gui',lambda **kw:0)
    assert student_runtime.main(['--package','unused','--annotator-id','student01',
                                 '--full-checksums','--sqlite-integrity-check'])==0
    assert seen==[dict(full_checksums=True,sqlite_integrity_check=True)]
