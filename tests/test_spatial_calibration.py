import json

import numpy as np
import pytest

from sync_workbench.core.geometry import (
    load_spatial_calibration, pose3d_to_pc, pose3d_to_world, geometry_metadata,
)


def write_calibration(tmp_path):
    axis = np.eye(4)
    axis[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    inverse = np.array([[0, -1, 0, .1], [1, 0, 0, .2], [0, 0, 1, .3], [0, 0, 0, 1]])
    forward = np.linalg.inv(inverse)
    document = dict(schema="radar-kinect-desk-calibration-v2", units="m", name="synthetic",
                    kinect_native_to_radar_axes_3x3=axis[:3, :3].tolist(),
                    radar_to_kinect_4x4=forward.tolist(), kinect_to_radar_4x4=inverse.tolist(),
                    native_kinect_to_radar_4x4=(inverse @ axis).tolist(),
                    radar_to_native_kinect_4x4=(axis.T @ forward).tolist())
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))
    return path


def test_native_mm_direction_no_double_axis_conversion_and_world_composition(tmp_path):
    path = write_calibration(tmp_path)
    calibration = load_spatial_calibration(path)
    pose = np.array([[[1000., 2000., 3000., .75]]])
    original = pose.copy()
    sensor = pose3d_to_pc(pose, calibration=calibration)
    np.testing.assert_allclose(sensor[0, 0], [-2.9, 1.2, -1.7])
    np.testing.assert_allclose(calibration.radar_m_to_kinect_mm(sensor), pose[..., :3])
    expected_world = [-2.9, np.sqrt(3)/2*1.2-.5*1.7, -.5*1.2-np.sqrt(3)/2*1.7+1.76]
    np.testing.assert_allclose(pose3d_to_world(pose, calibration=calibration)[0, 0], expected_world)
    np.testing.assert_array_equal(pose, original)
    metadata = geometry_metadata(calibration)
    np.testing.assert_allclose(np.asarray(metadata["kinect_mm_to_world_linear"]) @ pose[0, 0, :3]
                              + metadata["kinect_world_translation_m"], expected_world)
    before = calibration.provenance()
    path.write_text("{}")
    assert calibration.provenance() == before  # Open sessions retain the loaded snapshot.
    assert pose3d_to_world(np.empty((0,32,4)), calibration=calibration).size == 0


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema="audit"),
    lambda d: d.update(units="mm"),
    lambda d: d.update(kinect_native_to_radar_axes_3x3=np.eye(3).tolist()),
    lambda d: d.pop("native_kinect_to_radar_4x4"),
    lambda d: d["kinect_to_radar_4x4"][0].__setitem__(3, 100),
    lambda d: d["native_kinect_to_radar_4x4"][0].__setitem__(0, 2),
    lambda d: d["radar_to_native_kinect_4x4"][1].__setitem__(3, float("nan")),
])
def test_reject_invalid_or_inconsistent_calibrations(tmp_path, change):
    path = write_calibration(tmp_path)
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Invalid spatial calibration"):
        load_spatial_calibration(path)


def test_projection_uses_inverse_native_transform(tmp_path, monkeypatch):
    from sync_workbench.experimental.anchoring_gui import visualization_utils as visual
    calibration = load_spatial_calibration(write_calibration(tmp_path))
    seen = []
    def project(xyz):
        seen.append(xyz)
        return np.array([100., 200.])
    monkeypatch.setattr(visual, "_project_kinect_xyz_to_digital", project)
    radar = np.array([[-2.9, 1.2, -1.7, 4, 5, 6]])
    result = visual.project_pc_to_digital(radar, calibration=calibration)
    np.testing.assert_allclose(seen[0], [1000, 2000, 3000])
    np.testing.assert_array_equal(result[:, 2:], radar[:, 2:])


def test_legacy_geometry_is_preserved():
    assert load_spatial_calibration(None) is None
    pose = np.array([[[1000,2000,3000,1]]])
    np.testing.assert_array_equal(pose3d_to_pc(pose), [[[1,3,-2]]])
    assert geometry_metadata()["profile_id"] == "syncwb.gui_world.v1"
