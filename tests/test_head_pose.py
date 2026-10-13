"""Unit tests for head pose estimation."""
from __future__ import annotations

import math
import numpy as np
import pytest

from trustsync.synth import make_pts, _POSE_MODEL
from trustsync.vision.head_pose import estimate_pose, POSE_MODEL


def test_pose_model_matches_synth() -> None:
    """Verify POSE_MODEL in head_pose.py matches _POSE_MODEL in synth.py."""
    assert _POSE_MODEL == POSE_MODEL


def test_estimate_pose_grid() -> None:
    """Verify estimated pose angles across a grid of known yaw, pitch, roll values."""
    w, h = 640, 480
    for y in (-40, -20, 0, 20, 40):
        for p in (-20, 0, 20):
            for r in (-15, 0, 15):
                # pts shape: (478, 3)
                pts = make_pts(yaw=float(y), pitch=float(p), roll=float(r), w=w, h=h)
                est_yaw, est_pitch, est_roll = estimate_pose(pts, w, h)
                assert abs(est_yaw - y) < 3.0, f"Yaw error too large: {est_yaw} vs {y}"
                assert abs(est_pitch - p) < 3.0, f"Pitch error too large: {est_pitch} vs {p}"
                assert abs(est_roll - r) < 3.0, f"Roll error too large: {est_roll} vs {r}"


def test_estimate_pose_noise() -> None:
    """Verify pose estimation robustness to 0.5 px Gaussian noise on landmarks."""
    rng = np.random.default_rng(1234)
    w, h = 640, 480
    yaw, pitch, roll = 10.0, -10.0, 5.0
    pts = make_pts(yaw=yaw, pitch=pitch, roll=roll, w=w, h=h)

    # Add 0.5 px noise to normalized coordinates
    # noise_px shape: (478, 2)
    noise_px = rng.standard_normal((478, 2)) * 0.5
    noise_pts = pts.copy()
    noise_pts[:, 0] += noise_px[:, 0] / w
    noise_pts[:, 1] += noise_px[:, 1] / h

    est_yaw, est_pitch, est_roll = estimate_pose(noise_pts, w, h)
    assert abs(est_yaw - yaw) < 5.0
    assert abs(est_pitch - pitch) < 5.0
    assert abs(est_roll - roll) < 5.0


def test_estimate_pose_scale_and_center() -> None:
    """Verify pose estimation invariant to scale and center variation."""
    w, h = 640, 480
    yaw, pitch, roll = 15.0, 10.0, -5.0
    # pts shape: (478, 3)
    pts = make_pts(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        w=w,
        h=h,
        scale=0.5,
        center=(0.3, 0.6),
    )
    est_yaw, est_pitch, est_roll = estimate_pose(pts, w, h)
    assert abs(est_yaw - yaw) < 3.0
    assert abs(est_pitch - pitch) < 3.0
    assert abs(est_roll - roll) < 3.0


def test_estimate_pose_zero_and_nan() -> None:
    """Verify all-zero pts and NaN pts return (NaN, NaN, NaN)."""
    w, h = 640, 480
    # pts_zero shape: (478, 3)
    pts_zero = np.zeros((478, 3), dtype=np.float64)
    y, p, r = estimate_pose(pts_zero, w, h)
    assert math.isnan(y) and math.isnan(p) and math.isnan(r)

    pts_nan = make_pts(w=w, h=h).copy()
    pts_nan[1, 0] = np.nan
    y, p, r = estimate_pose(pts_nan, w, h)
    assert math.isnan(y) and math.isnan(p) and math.isnan(r)
