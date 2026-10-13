"""Unit tests for FaceTracker and face fixture verification."""
from pathlib import Path
import cv2
import numpy as np
import pytest

from trustsync.config import Config
from trustsync.vision.face_tracker import FaceTracker


def test_black_frame_and_unmodified_input(spec_cfg: Config) -> None:
    """Verify black frame returns [] and input frame is not mutated."""
    tracker = FaceTracker(spec_cfg)
    try:
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        copy_black = black.copy()

        faces = tracker.process(black)
        assert faces == []
        np.testing.assert_array_equal(black, copy_black)
    finally:
        tracker.close()


def test_face_frontal_fixture(spec_cfg: Config) -> None:
    """Verify face_frontal.jpg detection consistency, coordinates, and quality."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    frontal_path = fixtures_dir / "face_frontal.jpg"
    if not frontal_path.is_file():
        pytest.skip("run tools/capture_sample.py face_frontal")

    img = cv2.imread(str(frontal_path))
    if img is None:
        pytest.skip("run tools/capture_sample.py face_frontal")

    tracker = FaceTracker(spec_cfg)
    try:
        results = [tracker.process(img) for _ in range(3)]
        for faces in results:
            assert len(faces) >= 1
            f0 = faces[0]
            assert f0.pts.shape == (478, 3)
            x1, y1, x2, y2 = f0.bbox
            assert x1 < x2
            assert y1 < y2
            assert 0.0 <= f0.mouth_open <= 0.3
            assert f0.quality >= 0.5

        # Verify same track_id across all 3 calls
        track_ids = [faces[0].track_id for faces in results]
        assert track_ids[0] == track_ids[1] == track_ids[2]
    finally:
        tracker.close()


def test_face_mouth_open_fixture(spec_cfg: Config) -> None:
    """Verify face_mouth_open.jpg mouth_open exceeds face_frontal.jpg by >= 0.10."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    frontal_path = fixtures_dir / "face_frontal.jpg"
    open_path = fixtures_dir / "face_mouth_open.jpg"

    if not frontal_path.is_file():
        pytest.skip("run tools/capture_sample.py face_frontal")
    if not open_path.is_file():
        pytest.skip("run tools/capture_sample.py face_mouth_open")

    img_frontal = cv2.imread(str(frontal_path))
    img_open = cv2.imread(str(open_path))

    tracker = FaceTracker(spec_cfg)
    try:
        faces_frontal = tracker.process(img_frontal)
        faces_open = tracker.process(img_open)

        assert len(faces_frontal) >= 1
        assert len(faces_open) >= 1

        mouth_frontal = faces_frontal[0].mouth_open
        mouth_open_val = faces_open[0].mouth_open

        assert (mouth_open_val - mouth_frontal) >= 0.10
    finally:
        tracker.close()
