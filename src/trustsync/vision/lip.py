"""Lip aperture computation from facial landmarks."""
from __future__ import annotations

import numpy as np


def mouth_open(pts: np.ndarray, w: int, h: int) -> float:
    """Compute normalized mouth opening ratio.

    Args:
        pts: np.ndarray shape (478, 3) normalized landmarks.
        w: Frame width in pixels.
        h: Frame height in pixels.

    Returns:
        float: Ratio of inner lip gap to inter-ocular distance, or 0.0 if iod < 1e-6.
    """
    p13 = pts[13, :2] * np.array([w, h], dtype=np.float64)
    p14 = pts[14, :2] * np.array([w, h], dtype=np.float64)
    p33 = pts[33, :2] * np.array([w, h], dtype=np.float64)
    p263 = pts[263, :2] * np.array([w, h], dtype=np.float64)

    gap = float(np.linalg.norm(p14 - p13))
    iod = float(np.linalg.norm(p263 - p33))

    if iod < 1e-6:
        return 0.0

    return float(gap / iod)
