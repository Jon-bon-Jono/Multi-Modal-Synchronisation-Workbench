"""Controlled vocabularies used by the v0.1 backend.

The values are intentionally compact. They should be treated as recommended
controlled values rather than an attempt to close every future extension point.
"""

DEVICE_TYPES = (
    "kinect_rgb",
    "kinect_depth",
    "radar_pc",
    "radar_raw",
    "imu_accel",
    "imu_gyro",
    "other",
)

SAMPLE_KINDS = ("frame", "pose_frame", "packet", "window", "event", "other")

ASSET_ROLES = (
    "rgb_video",
    "depth_video",
    "pc_file",
    "pc_folder",
    "raw_capture_folder",
    "preview_bundle",
    "temporary_ingestion_package",
    "other",
)

ARTIFACT_ROLES = (
    "preview_image",
    "pose2d_json",
    "pose3d_json",
    "pc_frame_file",
    "raw_frame_file",
    "activity_json",
    "diagnostic_plot",
    "other",
)

TIMELINE_MODEL_TYPES = (
    "identity_observed",
    "linear_from_index",
    "start_end_uniform",
    "regressed_observed_vs_index",
    "regressed_from_pts",
    "piecewise",
    "spline",
    "manual",
    "other",
)

TIME_KINDS = (
    "observed_wallclock",
    "device_elapsed",
    "pts_based",
    "estimated_wallclock",
    "latent_acquisition_time",
    "sync_target_time",
    "other",
)

SYNC_MODEL_TYPES = (
    "identity_time",
    "piecewise_affine",
    "global_affine",
    "piecewise_linear",
    "spline",
    "manual",
    "other",
)

EXTRAPOLATION_POLICIES = (
    "disallow",
    "allow_nearest_only",
    "allow_linear",
    "allow_with_penalty",
    "manual",
)

MAPPING_METHODS = (
    "nearest_predicted_time",
    "nearest_observed_wallclock",
    "windowed_nearest",
    "top_k_candidates",
    "manual",
    "other",
)

MAPPING_REGION_TYPES = (
    "interpolation",
    "left_extrapolation",
    "right_extrapolation",
    "unsupported",
    "manual",
)

SUPPORT_STATUSES = (
    "supported",
    "weak_support",
    "extrapolated",
    "outside_run",
    "missing_source_time",
    "missing_target",
    "manual_override",
)
