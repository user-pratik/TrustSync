"""Head pose estimation using SolvePnP on facial landmarks."""
from __future__ import annotations

import math
import cv2
import numpy as np

POSE_MODEL: dict[int, tuple[float, float, float]] = {
    1: (0.0, 0.0, 0.0),
    152: (0.0, 330.0, 65.0),
    33: (-225.0, -170.0, 135.0),
    263: (225.0, -170.0, 135.0),
    61: (-150.0, 150.0, 125.0),
    291: (150.0, 150.0, 125.0),
}


def estimate_pose(
    pts: np.ndarray, w: int, h: int
) -> tuple[float, float, float]:
    """Estimate head pose (yaw, pitch, roll in degrees) from facial landmarks.

    Args:
        pts: np.ndarray shape (478, 3) normalized landmarks.
        w: Frame width in pixels.
        h: Frame height in pixels.

    Returns:
        tuple[float, float, float]: (yaw, pitch, roll) in degrees, or (NaN, NaN, NaN) on failure.
    """
    model_indices = [1, 152, 33, 263, 61, 291]

    # Check for NaN in landmarks
    for idx in model_indices:
        if np.isnan(pts[idx]).any():
            return (float("nan"), float("nan"), float("nan"))

    # Convert landmarks to pixel coordinates
    p33_px = pts[33, :2] * np.array([w, h], dtype=np.float64)
    p263_px = pts[263, :2] * np.array([w, h], dtype=np.float64)
    iod_px = float(np.linalg.norm(p263_px - p33_px))

    # Reject if inter-ocular distance is below 5 pixels (e.g. all-zero points or extreme occlusion)
    if iod_px < 5.0:
        return (float("nan"), float("nan"), float("nan"))

    model_points = np.array([POSE_MODEL[idx] for idx in model_indices], dtype=np.float64)
    image_points = np.array(
        [[pts[idx, 0] * w, pts[idx, 1] * h] for idx in model_indices],
        dtype=np.float64,
    )

    f = float(w)
    cx = float(w) / 2.0
    cy = float(h) / 2.0
    camera_matrix = np.array(
        [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist_coeffs = np.zeros(4, dtype=np.float64)

    try:
        success, rvec, _tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return (float("nan"), float("nan"), float("nan"))

        r_mat, _ = cv2.Rodrigues(rvec)

        pitch = math.atan2(r_mat[2, 1], r_mat[2, 2])
        yaw = math.atan2(-r_mat[2, 0], math.sqrt(r_mat[2, 1] ** 2 + r_mat[2, 2] ** 2))
        roll = math.atan2(r_mat[1, 0], r_mat[0, 0])

        return (
            float(math.degrees(yaw)),
            float(math.degrees(pitch)),
            float(math.degrees(roll)),
        )
    except Exception:
        return (float("nan"), float("nan"), float("nan"))
