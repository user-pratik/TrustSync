"""Unit tests for face image quality assessment."""
import numpy as np

from trustsync.config import Config
from trustsync.vision.quality import face_quality


def test_bright_large_face_quality(spec_cfg: Config) -> None:
    """Verify bright 480x640 frame with a 200 px face height gives quality > 0.9."""
    h, w = 480, 640
    frame = np.full((h, w, 3), 180, dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.2 + 200.0 / h)
    pts = np.full((478, 3), 0.5, dtype=np.float64)

    q = face_quality(frame, bbox, pts, spec_cfg.quality)
    assert q > 0.9


def test_dim_frame_low_quality(spec_cfg: Config) -> None:
    """Verify frame with mean grey 10 gives quality < 0.3."""
    h, w = 480, 640
    frame = np.full((h, w, 3), 10, dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.2 + 200.0 / h)
    pts = np.full((478, 3), 0.5, dtype=np.float64)

    q = face_quality(frame, bbox, pts, spec_cfg.quality)
    assert q < 0.3


def test_small_face_height_quality(spec_cfg: Config) -> None:
    """Verify face of 40 px height gives quality <= 0.5."""
    h, w = 480, 640
    frame = np.full((h, w, 3), 180, dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.2 + 40.0 / h)
    pts = np.full((478, 3), 0.5, dtype=np.float64)

    q = face_quality(frame, bbox, pts, spec_cfg.quality)
    assert q <= 0.5


def test_landmarks_outside_edge_zero_quality(spec_cfg: Config) -> None:
    """Verify > 5% landmarks outside [0.02, 0.98] gives exactly 0.0 quality."""
    h, w = 480, 640
    frame = np.full((h, w, 3), 180, dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.2 + 200.0 / h)
    pts = np.full((478, 3), 0.5, dtype=np.float64)

    # Place 30 landmarks (30 / 478 = 6.27% > 5%) outside [0.02, 0.98]
    pts[:30, 0] = 0.01

    q = face_quality(frame, bbox, pts, spec_cfg.quality)
    assert q == 0.0
