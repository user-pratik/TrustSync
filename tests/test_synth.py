"""Tests for synthetic signal generation and mock tracking helpers."""
import numpy as np

from trustsync.synth import (
    FakeFaceTracker,
    make_env,
    make_lip,
    make_pts,
    make_static_lip,
)
from trustsync.types import FaceObs


def test_make_env() -> None:
    """Verify make_env exact length and burst-to-baseline contrast."""
    rng = np.random.default_rng(1234)
    duration_s = 3.0
    env_hz = 100.0
    bursts = [(1.0, 2.0)]
    env = make_env(duration_s, env_hz, bursts, rng)

    assert len(env) == round(duration_s * env_hz)
    assert env.dtype == np.float32

    t = np.arange(len(env)) / env_hz
    inside = env[(t >= 1.0) & (t <= 2.0)]
    outside = env[(t < 0.9) | (t > 2.1)]

    # Mean inside a burst is more than 10 times the mean outside
    assert np.mean(inside) > 10.0 * np.mean(outside)


def test_make_lip_correlation_and_lag() -> None:
    """Verify make_lip correlation with envelope and lag accuracy."""
    rng = np.random.default_rng(1234)
    duration_s = 4.0
    env_hz = 100.0
    fps = 30.0
    bursts = [(0.5, 3.5)]
    env = make_env(duration_s, env_hz, bursts, rng)

    # 1. Zero lag and zero noise -> Pearson correlation > 0.9
    lip_zero = make_lip(env, env_hz, fps, lag_s=0.0, gain=0.05, noise=0.0, rng=rng)
    t_fps = np.arange(round(duration_s * fps)) / fps
    t_env = np.arange(len(env)) / env_hz
    env_resampled = np.interp(t_fps, t_env, env)

    r_zero = np.corrcoef(lip_zero, env_resampled)[0, 1]
    assert r_zero > 0.9

    # 2. lag_s = 0.1 at fps 30 -> best lag 3 frames +/- 1
    lag_s = 0.1
    lip_lag = make_lip(env, env_hz, fps, lag_s=lag_s, gain=0.05, noise=0.0, rng=rng)

    n = len(lip_lag)
    best_lag = 0
    best_corr = -1.0
    for k in range(-10, 11):
        if k >= 0:
            a = lip_lag[: n - k]
            b = env_resampled[k:]
        else:
            a = lip_lag[-k:]
            b = env_resampled[: n + k]
        corr = np.corrcoef(a, b)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_lag = k

    # Expected lag is round(lag_s * fps) = 3 frames (+/- 1)
    assert abs(best_lag - 3) <= 1


def test_make_static_lip() -> None:
    """Verify make_static_lip std is bounded by noise amplitude."""
    rng = np.random.default_rng(1234)
    lip = make_static_lip(duration_s=3.0, fps=30.0, noise=0.001, rng=rng)
    assert len(lip) == round(3.0 * 30.0)
    assert np.std(lip) < 0.002


def test_make_pts_geometry() -> None:
    """Verify make_pts shape, range, facial symmetry, and mouth gap ratio."""
    pts = make_pts(
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        w=640,
        h=480,
        mouth_gap_ratio=0.05,
        iris_ratio=0.5,
        center=(0.5, 0.5),
        scale=1.0,
    )

    assert pts.shape == (478, 3)

    used_indices = [1, 152, 33, 263, 61, 291, 13, 14, 133, 362, 468, 473]
    for idx in used_indices:
        assert 0.0 <= pts[idx, 0] <= 1.0
        assert 0.0 <= pts[idx, 1] <= 1.0

    # At yaw 0, points 33 and 263 are symmetric around center (0.5) within 0.02
    mid_x = (pts[33, 0] + pts[263, 0]) / 2.0
    assert abs(mid_x - 0.5) < 0.02

    # Ratio dist(13,14) / dist(33,263) in pixels equals mouth_gap_ratio within 1e-6
    p33_px = pts[33, :2] * np.array([640.0, 480.0])
    p263_px = pts[263, :2] * np.array([640.0, 480.0])
    p13_px = pts[13, :2] * np.array([640.0, 480.0])
    p14_px = pts[14, :2] * np.array([640.0, 480.0])

    iod_px = float(np.linalg.norm(p263_px - p33_px))
    mouth_px = float(np.linalg.norm(p14_px - p13_px))
    assert abs((mouth_px / iod_px) - 0.05) < 1e-6


def test_fake_face_tracker() -> None:
    """Verify FakeFaceTracker returns scripted observations in order then []."""
    f1 = FaceObs(track_id=0, bbox=(0.1, 0.1, 0.5, 0.5), pts=np.zeros((478, 3)), mouth_open=0.05, quality=0.9)
    f2 = FaceObs(track_id=1, bbox=(0.2, 0.2, 0.6, 0.6), pts=np.zeros((478, 3)), mouth_open=0.08, quality=0.85)

    script = [[f1], [f1, f2], []]
    tracker = FakeFaceTracker(script)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    assert tracker.process(dummy) == [f1]
    assert tracker.process(dummy) == [f1, f2]
    assert tracker.process(dummy) == []
    # Post-script exhaustion returns []
    assert tracker.process(dummy) == []
    tracker.close()
