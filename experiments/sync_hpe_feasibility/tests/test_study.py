import numpy as np
import pytest

from hpe_sync.common import (block_bootstrap_difference, inverse_map, load_config,
                             nearest, safe_output, split_blocks, standardize_fit)
from hpe_sync.prepare import make_queries, timeline
from hpe_sync.targets import pose_targets, radar_features
from hpe_sync.train import make_plan, job_id


def poses(n=1):
    p = np.zeros((n, 26, 3))
    p[..., :2] = np.random.default_rng(23).normal(300, 20, (n, 26, 2))
    p[..., 2] = .9
    p[:, [5, 6], 1] = 100
    p[:, [11, 12], 1] = 200
    return p


def test_no_identity_required_for_targets():
    cfg = load_config()
    p = poses(2)
    a, _ = pose_targets(p, 2, [.9, .8], cfg)
    b, _ = pose_targets(p[::-1], 2, [.8, .9], cfg)
    for t in a:
        np.testing.assert_allclose(a[t][0], b[t][0])
        np.testing.assert_array_equal(a[t][1], b[t][1])


@pytest.mark.parametrize("expected", [0, 2, np.nan, 1.5])
def test_count_mismatch_rejects_every_target(expected):
    out, _ = pose_targets(poses(), expected, [.9], load_config())
    assert not any(mask.any() for _, mask in out.values())


def test_missing_3d_is_not_an_input_requirement():
    out, _ = pose_targets(poses(), 1, [.9], load_config())
    assert out["image"][1].any() and out["shape"][1].any()


def test_bad_person_joint_invalidates_that_scene_joint_only():
    p = poses(2)
    p[1, 4, 2] = .1
    out, _ = pose_targets(p, 2, [.9, .9], load_config())
    mask = out["image"][1].reshape(26, 4)
    assert not mask[4].any()
    assert mask[3].all()


def test_single_person_zero_spread_not_scored():
    out, _ = pose_targets(poses(), 1, [.9], load_config())
    assert not out["image"][1].reshape(26, 4)[:, 2:].any()


def test_shape_translation_and_scale_invariant_but_not_image():
    p = poses()
    q = p.copy()
    q[..., :2] = 2 * p[..., :2] + [40, 10]
    a, _ = pose_targets(p, 1, [.9], load_config())
    b, _ = pose_targets(q, 1, [.9], load_config())
    np.testing.assert_allclose(a["shape"][0], b["shape"][0], atol=1e-6)
    assert not np.allclose(a["image"][0], b["image"][0])


def test_missing_root_rejects_shape_but_keeps_image():
    p = poses()
    p[0, 11, 2] = .01
    out, _ = pose_targets(p, 1, [.9], load_config())
    assert out["image"][1].any()
    assert not out["shape"][1].any()


def test_nearest_refuses_large_gap_and_extrapolation():
    idx, valid, residual = nearest([0., .067, .6], [-.01, .033, .3, .7], .05)
    assert valid.tolist() == [False, True, False, False]
    assert idx[1] == 0


def test_inverse_offset_drift_and_unsupported():
    a = [dict(source_time=0., target_time=2.),
         dict(source_time=10., target_time=12.1)]
    np.testing.assert_allclose(inverse_map([2., 7.05, 12.1], a), [0., 5., 10.])
    assert np.isnan(inverse_map([1.9, 12.2], a)).all()


def test_datetime_unit_is_seconds_across_pandas_versions():
    import sqlite3
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE SAMPLE_TIME_ESTIMATE (subject_id,run_id,device_type,timeline_model_id,"
              "sample_index,time_value_datetime,time_value_sec)")
    c.executemany("INSERT INTO SAMPLE_TIME_ESTIMATE VALUES (?,?,?,?,?,?,?)",
                  [("p", "r", "rgb", "t", 0, "2024-01-15T09:48:05.445703", None),
                   ("p", "r", "rgb", "t", 1, "2024-01-15T09:48:05.512370", None)])
    t = timeline(c, ("p", "r", "rgb"), "t", [0, 1])
    assert abs(t[0] - 1705312085.445703) < 1e-6
    assert abs(t[1] - t[0] - .066667) < 1e-6


def test_shift_sign_and_negative_control_stays_in_block():
    cfg = load_config()
    a = [dict(source_time=0., target_time=0.), dict(source_time=1000., target_time=1000.)]
    ref = np.array([10., 115., 130.])
    q = make_queries(ref, ref, a, cfg)
    np.testing.assert_allclose(q["shift_p200"] - ref, .2)
    assert np.all(np.floor(q["unrelated"] / 120) == np.floor(ref / 120))
    assert np.all(np.abs(q["unrelated"] - ref) >= 10 - 1e-6)


def test_block_buffers_and_training_only_scaling():
    cfg = load_config()
    block, split, valid = split_blocks(np.array([0., 5., 119., 125., 245., 605.]), cfg)
    assert valid.tolist() == [False, True, False, True, True, True]
    assert split[-2:].tolist() == [1, 2]
    train = np.arange(12.).reshape(-1, 1)
    mean, scale, active = standardize_fit(train)
    assert mean[0] == 5.5 and active[0]
    # A remote test-set value must not change fitted mean/scale.
    assert ((1000 - mean[0]) / scale[0]) > 100


def test_empty_and_unassociated_stationary_points_are_preserved():
    f, p = radar_features(np.empty((0, 6)))
    assert f.shape == (27,) and len(p) == 0 and np.isfinite(f).all()
    f, p = radar_features(np.array([[1, 2, 3, 0, 5, 255], [1, 2, 3, 0, 5, np.nan]]))
    assert len(p) == 2 and f[-1] == 0


def test_block_bootstrap_not_frame_pseudoreplication():
    a = np.ones(60)
    b = np.zeros(60)
    r = block_bootstrap_difference(a, b, np.repeat([1, 2, 3], 20), repeats=50)
    assert r["blocks"] == 3 and r["ci95"] == [1., 1.]
    assert block_bootstrap_difference(a, b, np.zeros(60))["ci95"] is None


def test_outputs_cannot_escape_and_core_is_unique():
    with pytest.raises(ValueError):
        safe_output("../../outside")
    plan = make_plan(load_config(), "core")
    assert len({job_id(j) for j in plan}) == len(plan)
    assert all(not j["smoke"] for j in plan)
