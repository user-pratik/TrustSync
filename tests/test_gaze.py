"""Unit tests for gaze offset computation."""
from __future__ import annotations

import numpy as np
import pytest

from trustsync.synth import make_pts
from trustsync.vision.gaze import gaze_off


def test_gaze_off_values() -> None:
    """Verify gaze_off values match expected formulas for various iris ratios."""
    w, h = 640, 480

    # iris_ratio 0.5 gives 0.0
    pts_05 = make_pts(iris_ratio=0.5, w=w, h=h)
    assert abs(gaze_off(pts_05, w, h) - 0.0) < 1e-6

    # iris_ratio 0.3 gives 0.4
    pts_03 = make_pts(iris_ratio=0.3, w=w, h=h)
    assert abs(gaze_off(pts_03, w, h) - 0.4) < 1e-6

    # iris_ratio 0.9 gives 0.8
    pts_09 = make_pts(iris_ratio=0.9, w=w, h=h)
    assert abs(gaze_off(pts_09, w, h) - 0.8) < 1e-6

    # iris_ratio 0.0 gives 1.0
    pts_00 = make_pts(iris_ratio=0.0, w=w, h=h)
    assert abs(gaze_off(pts_00, w, h) - 1.0) < 1e-6


def test_gaze_off_zero_pts() -> None:
    """Verify all-zero pts landmarks return 0.0 without division by zero error."""
    w, h = 640, 480
    # pts_zero shape: (478, 3)
    pts_zero = np.zeros((478, 3), dtype=np.float64)
    assert gaze_off(pts_zero, w, h) == 0.0
