"""TrustSync synthetic signal generation and mock tracking helpers."""
from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage

from trustsync.types import FaceObs

_POSE_MODEL: dict[int, tuple[float, float, float]] = {
    1: (0.0, 0.0, 0.0),
    152: (0.0, 330.0, 65.0),
    33: (-225.0, -170.0, 135.0),
    263: (225.0, -170.0, 135.0),
    61: (-150.0, 150.0, 125.0),
    291: (150.0, 150.0, 125.0),
}


def make_env(
    duration_s: float,
    env_hz: float,
    bursts: list[tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic audio envelope with speech bursts.

    Args:
        duration_s: Signal duration in seconds.
        env_hz: Envelope sampling rate in Hz.
        bursts: List of (start_s, end_s) active speech burst windows.
        rng: Seeded numpy RandomGenerator.

    Returns:
        np.ndarray shape (round(duration_s * env_hz),) float32 envelope values.
    """
    n = round(duration_s * env_hz)
    t = np.arange(n, dtype=np.float64) / env_hz
    env = 0.002 + 0.001 * np.abs(rng.standard_normal(n))

    for start_s, end_s in bursts:
        burst_mask = (t >= start_s) & (t <= end_s)
        if not np.any(burst_mask):
            continue
        env[burst_mask] += 0.03
        centre = start_s + 0.1
        while centre <= end_s:
            amp = 0.1 * float(rng.uniform(0.5, 1.0))
            u = (t - centre) / 0.09
            bump_mask = np.abs(u) <= 1.0
            if np.any(bump_mask):
                bump = 0.5 * (1.0 + np.cos(np.pi * u[bump_mask])) * amp
                env[bump_mask] += bump
            centre += float(rng.uniform(0.12, 0.30))
        n_burst = np.count_nonzero(burst_mask)
        env[burst_mask] += 0.005 * np.abs(rng.standard_normal(n_burst))

    return env.astype(np.float32)


def make_lip(
    env: np.ndarray,
    env_hz: float,
    fps: float,
    lag_s: float,
    gain: float = 0.05,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic lip opening signal aligned to an envelope.

    Positive lag_s means the audio is later than the lip motion.

    Args:
        env: Audio envelope signal array.
        env_hz: Envelope sample rate in Hz.
        fps: Video frame rate.
        lag_s: Lag offset in seconds (positive = audio later than lip).
        gain: Amplitude scaling factor for normalised envelope (default 0.05).
        noise: Standard deviation of additive Gaussian noise.
        rng: Seeded numpy RandomGenerator.

    Returns:
        np.ndarray shape (round(len(env) / env_hz * fps),) lip opening signal.
    """
    n_frames = round(len(env) / env_hz * fps)
    t = np.arange(n_frames, dtype=np.float64) / fps
    env_times = np.arange(len(env), dtype=np.float64) / env_hz
    sampled = np.interp(
        t + lag_s,
        env_times,
        env,
        left=float(env[0]),
        right=float(env[-1]),
    )
    smoothed = scipy.ndimage.uniform_filter1d(sampled, size=3, mode="nearest")
    max_env = max(float(np.max(env)), 1e-6)
    normalised = smoothed / max_env
    lip = 0.02 + gain * normalised
    if noise > 0.0:
        if rng is None:
            rng = np.random.default_rng(1234)
        lip = lip + noise * rng.standard_normal(n_frames)
    return lip


def make_static_lip(
    duration_s: float,
    fps: float,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate static lip opening signal for a non-speaking face.

    Args:
        duration_s: Signal duration in seconds.
        fps: Video frame rate.
        noise: Standard deviation of baseline jitter.
        rng: Seeded numpy RandomGenerator.

    Returns:
        np.ndarray shape (round(duration_s * fps),) static lip signal.
    """
    n_frames = round(duration_s * fps)
    return 0.02 + noise * rng.standard_normal(n_frames)


def make_pts(
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    w: int = 640,
    h: int = 480,
    mouth_gap_ratio: float = 0.05,
    iris_ratio: float = 0.5,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 1.0,
) -> np.ndarray:
    """Generate synthetic 478 MediaPipe face landmarks with specified pose and geometry.

    Args:
        yaw: Head yaw in degrees.
        pitch: Head pitch in degrees.
        roll: Head roll in degrees.
        w: Frame width in pixels.
        h: Frame height in pixels.
        mouth_gap_ratio: Ratio of mouth opening gap to inter-ocular distance.
        iris_ratio: Iris position ratio (0.5 = centred).
        center: Normalized face center coordinates (cx, cy).
        scale: Face scale factor.

    Returns:
        np.ndarray shape (478, 3) float64 normalized landmark coordinates in [0, 1].
    """
    rad_pitch = np.radians(pitch)
    rad_yaw = np.radians(yaw)
    rad_roll = np.radians(roll)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rad_pitch), -np.sin(rad_pitch)],
            [0.0, np.sin(rad_pitch), np.cos(rad_pitch)],
        ],
        dtype=np.float64,
    )

    ry = np.array(
        [
            [np.cos(rad_yaw), 0.0, np.sin(rad_yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(rad_yaw), 0.0, np.cos(rad_yaw)],
        ],
        dtype=np.float64,
    )

    rz = np.array(
        [
            [np.cos(rad_roll), -np.sin(rad_roll), 0.0],
            [np.sin(rad_roll), np.cos(rad_roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    r_mat = rz @ ry @ rx
    rvec, _ = cv2.Rodrigues(r_mat)

    f = float(w)
    tz = 3000.0 / scale
    cx, cy = center
    tvec = np.array([(cx - 0.5) * tz, (cy - 0.5) * h * tz / w, tz], dtype=np.float64)

    camera_matrix = np.array(
        [
            [f, 0.0, float(w) / 2.0],
            [0.0, f, float(h) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    model_indices = [1, 152, 33, 263, 61, 291]
    model_pts = np.array([_POSE_MODEL[idx] for idx in model_indices], dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    projected, _ = cv2.projectPoints(model_pts, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)

    pts = np.zeros((478, 3), dtype=np.float64)
    for i, lm_idx in enumerate(model_indices):
        pts[lm_idx, 0] = projected[i, 0] / w
        pts[lm_idx, 1] = projected[i, 1] / h
        pts[lm_idx, 2] = 0.0

    # (d) Mouth opening
    p33_px = np.array([pts[33, 0] * w, pts[33, 1] * h])
    p263_px = np.array([pts[263, 0] * w, pts[263, 1] * h])
    iod = float(np.linalg.norm(p263_px - p33_px))
    gap = mouth_gap_ratio * iod

    p61_px = np.array([pts[61, 0] * w, pts[61, 1] * h])
    p291_px = np.array([pts[291, 0] * w, pts[291, 1] * h])
    m_px = 0.5 * (p61_px + p291_px)

    p13_px = m_px + np.array([0.0, -gap / 2.0])
    p14_px = m_px + np.array([0.0, +gap / 2.0])

    pts[13, 0] = p13_px[0] / w
    pts[13, 1] = p13_px[1] / h
    pts[14, 0] = p14_px[0] / w
    pts[14, 1] = p14_px[1] / h

    # (e) Eyes and irises
    p33 = pts[33, :2]
    p263 = pts[263, :2]
    p133 = p33 + 0.3 * (p263 - p33)
    p362 = p263 - 0.3 * (p263 - p33)
    p468 = p33 + iris_ratio * (p133 - p33)
    p473 = p362 + iris_ratio * (p263 - p362)

    pts[133, :2] = p133
    pts[362, :2] = p362
    pts[468, :2] = p468
    pts[473, :2] = p473

    return pts


class FakeFaceTracker:
    """Deterministic face tracker test helper replaying scripted FaceObs detections."""

    def __init__(self, script: list[list[FaceObs]]) -> None:
        self.script = script
        self._frame_idx = 0

    def process(self, frame_bgr: np.ndarray) -> list[FaceObs]:
        """Return scripted detections for current frame, or [] past script end."""
        if self._frame_idx < len(self.script):
            res = self.script[self._frame_idx]
            self._frame_idx += 1
            return res
        return []

    def close(self) -> None:
        """Close tracker resources."""
        pass
