"""Unit tests for lip aperture computation."""
import numpy as np

from trustsync.synth import make_pts
from trustsync.vision.lip import mouth_open


def test_mouth_open_matches_ratios() -> None:
    """Verify mouth_open matches the mouth_gap_ratio used in make_pts within 1e-6."""
    ratios = [0.0, 0.02, 0.05, 0.2]
    w, h = 640, 480

    for r in ratios:
        pts = make_pts(mouth_gap_ratio=r, w=w, h=h)
        computed = mouth_open(pts, w, h)
        assert abs(computed - r) < 1e-6


def test_mouth_open_invariant_to_scale_and_resolution() -> None:
    """Verify mouth_open is invariant to face scale and frame resolution."""
    r = 0.05

    # Scale 0.5
    pts_scaled = make_pts(mouth_gap_ratio=r, scale=0.5, w=640, h=480)
    computed_scaled = mouth_open(pts_scaled, 640, 480)
    assert abs(computed_scaled - r) < 1e-6

    # Higher resolution (1280x720)
    pts_hd = make_pts(mouth_gap_ratio=r, w=1280, h=720)
    computed_hd = mouth_open(pts_hd, 1280, 720)
    assert abs(computed_hd - r) < 1e-6


def test_mouth_open_coincident_eye_landmarks() -> None:
    """Verify mouth_open returns 0.0 when eye landmarks 33 and 263 coincide."""
    pts = make_pts()
    pts[263] = pts[33].copy()
    computed = mouth_open(pts, 640, 480)
    assert computed == 0.0
