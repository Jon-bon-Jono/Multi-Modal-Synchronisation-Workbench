"""Instantaneous 2D set descriptors: invariant to permutation of people."""
import numpy as np


def pose_targets(pose, expected_people, bbox_scores, config):
    dims = 26 * 4
    out = {t: (np.zeros(dims, np.float32), np.zeros(dims, bool)) for t in ("image", "shape")}
    pose = np.asarray(pose)
    if (not np.isfinite(expected_people) or expected_people < 1
            or expected_people != int(expected_people)
            or pose.shape != (int(expected_people), 26, 3)):
        return out, "pose_count_mismatch_or_no_person"
    bbox_scores = np.asarray(bbox_scores)
    if (bbox_scores.shape != (len(pose),) or not np.isfinite(bbox_scores).all()
            or np.any(bbox_scores < config["bbox_confidence"])):
        return out, "bbox_confidence"
    xy = pose[..., :2]
    good = np.isfinite(pose).all(axis=-1) & (pose[..., 2] >= config["joint_confidence"])
    # A joint statistic is accepted only when that joint is valid for every expected person.
    joint_good = good.all(axis=0)

    def moments(coords, joint_mask):
        coords = np.where(np.isfinite(coords), coords, 0)
        values = np.concatenate((coords.mean(axis=0), coords.std(axis=0)), axis=1)
        mask = np.repeat(joint_mask[:, None], 4, axis=1)
        if len(coords) == 1:
            mask[:, 2:] = False  # Constant zero spread must not inflate scores.
        if joint_mask.sum() < config["minimum_valid_joints"]:
            mask[:] = False
        return np.where(mask, values, 0).reshape(-1).astype("float32"), mask.reshape(-1)

    out["image"] = moments(xy, joint_good)
    # Halpe26: shoulders 5,6; hips 11,12. No cross-frame or cross-modality identity.
    if good[:, [5, 6, 11, 12]].all():
        root = xy[:, [11, 12]].mean(axis=1)
        shoulder = xy[:, [5, 6]].mean(axis=1)
        scale = np.linalg.norm(shoulder - root, axis=1)
        if np.all(scale >= config["minimum_torso_pixels"]):
            out["shape"] = moments((xy - root[:, None, :]) / scale[:, None, None], joint_good)
    return out, "eligible" if out["image"][1].any() else "joint_confidence"


def radar_features(points):
    """Radar-only features; retain stationary and unassociated returns."""
    p = np.asarray(points, dtype=np.float32)
    p = p[np.isfinite(p[:, :5]).all(axis=1), :5]
    if not len(p):
        return np.zeros(27, np.float32), p
    stats = np.concatenate((p.mean(axis=0), p.std(axis=0),
                            np.quantile(p, [.1, .5, .9], axis=0).ravel(),
                            [np.log1p(len(p)), np.mean(np.abs(p[:, 3]) > .1)]))
    return stats.astype("float32"), p
