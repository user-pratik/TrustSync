"""Gaze offset computation from eye iris and corner landmarks."""
from __future__ import annotations

import numpy as np


def gaze_off(pts: np.ndarray, w: int, h: int) -> float:
    """Compute normalized gaze deviation from center.

    Args:
        pts: np.ndarray shape (478, 3) normalized landmarks.
        w: Frame width in pixels.
        h: Frame height in pixels.

    Returns:
        float: Normalized gaze deviation in [0.0, 1.0], where 0.0 is looking directly at camera.
    """
    den1 = pts[133, 0] - pts[33, 0]
    den2 = pts[263, 0] - pts[362, 0]

    if abs(den1) < 1e-6 or abs(den2) < 1e-6:
        return 0.0

    r1 = (pts[468, 0] - pts[33, 0]) / den1
    r2 = (pts[473, 0] - pts[362, 0]) / den2

    mean_diff = (abs(r1 - 0.5) + abs(r2 - 0.5)) / 2.0
    return float(np.clip(mean_diff * 2.0, 0.0, 1.0))
