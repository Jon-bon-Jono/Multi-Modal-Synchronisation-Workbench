"""Exercise Windows preset argument forwarding without starting Conda or the GUI."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows batch launchers')
@pytest.mark.parametrize('name,calibrated', [
    ('anchoring_gui_versioned_19_MM.bat', True),
    ('anchoring_gui_versioned_09_SY.bat', True),
    ('anchoring_gui_19_MM.bat', False),
    ('anchoring_gui_19_MM_with_finetuned_predictions.bat', False),
    ('anchoring_gui_19_MM_with_mmfi_predictions.bat', False),
    ('export_training_data_19_MM.bat', True),
    ('export_training_data_living_lab.bat', True),
])
def test_presets_embed_calibration_without_caller_arguments(tmp_path, name, calibrated):
    root = Path(__file__).resolve().parents[1]
    fake_conda = tmp_path/'mock conda.cmd'
    fake_conda.write_text('@echo off\necho CAPTURE %*\nexit /b 0\n')
    rgb = tmp_path/'rgb root'
    video = rgb / ('09_SY/Session-2023-December-04 15-44-03-011950/kinect_camera_recording_rgb_lq.mp4'
                   if name == 'anchoring_gui_versioned_09_SY.bat' else
                   '19_MM/Session-2024-January-15 09-47-41-274452/kinect_camera_recording_rgb_lq.mp4')
    video.parent.mkdir(parents=True)
    video.write_bytes(b'placeholder')
    env = dict(os.environ, SYNCWB_CONDA_EXE=str(fake_conda), SYNCWB_RGB_ROOT=str(rgb),
               SYNCWB_NO_PAUSE='1', SYNCWB_SPATIAL_CALIBRATION='invalid inherited value')
    result = subprocess.run(['cmd.exe', '/d', '/c', str(root/'scripts/syncwb'/name)],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    captured = next(line for line in result.stdout.splitlines() if line.startswith('CAPTURE '))
    assert ('--spatial-calibration' in captured) == calibrated
    if calibrated:
        assert 'calibration\\kinect_radar\\2026-10-06-desk\\desk_all.json' in captured
    assert 'invalid inherited value' not in captured
    if name == 'export_training_data_living_lab.bat':
        assert 'living_lab_09_SY_19_MM_selection.json' in captured
        assert '--kinect-root' in captured
        assert '--output' in captured
    if name == 'anchoring_gui_versioned_09_SY.bat':
        assert '--subject 09_SY' in captured
        assert '--mapping-version initial_rgb_to_raw_v001' in captured
        assert '--point-cloud-version raw_fbf432d2908ce574ec3daa437ab78762c7ed850bac3aeb6e7cb8b1aef9aeae7c' in captured
