"""Unit tests and fuzz testing for Hungarian multi-face IdTracker."""
import numpy as np

from trustsync.vision.id_tracker import IdTracker


def test_one_face_jitter_keeps_id() -> None:
    """Verify one face with 0.005 jitter over 100 calls consistently keeps id 0."""
    rng = np.random.default_rng(1234)
    tracker = IdTracker(max_dist=0.15, max_missed=15)
    base = np.array([0.5, 0.5])

    for _ in range(100):
        pos = base + rng.uniform(-0.005, 0.005, size=2)
        ids = tracker.assign(pos[None, :])
        assert ids == [0]


def test_two_faces_moving_apart_keep_ids() -> None:
    """Verify two diverging faces maintain their respective IDs."""
    tracker = IdTracker(max_dist=0.15, max_missed=15)

    for i in range(20):
        c1 = np.array([0.4 - i * 0.01, 0.5])
        c2 = np.array([0.6 + i * 0.01, 0.5])
        ids = tracker.assign(np.array([c1, c2]))
        assert ids == [0, 1]


def test_face_missing_less_vs_more_than_max_missed() -> None:
    """Verify face keeps ID if missing <= max_missed, but gets new ID if > max_missed."""
    tracker = IdTracker(max_dist=0.15, max_missed=15)
    pos = np.array([[0.5, 0.5]])
    assert tracker.assign(pos) == [0]

    # Missing for 10 calls (< max_missed 15)
    for _ in range(10):
        tracker.assign(np.zeros((0, 2)))
    assert tracker.assign(pos) == [0]

    # Missing for 20 calls (> max_missed 15)
    for _ in range(20):
        tracker.assign(np.zeros((0, 2)))
    new_ids = tracker.assign(pos)
    assert new_ids == [1]


def test_two_faces_crossing_paths() -> None:
    """Verify two faces crossing paths with a step of 0.02 per call keep their IDs."""
    tracker = IdTracker(max_dist=0.15, max_missed=15)

    for k in range(25):
        # Face 0 moves left-to-right at y=0.46
        c0 = np.array([0.25 + k * 0.02, 0.46])
        # Face 1 moves right-to-left at y=0.54
        c1 = np.array([0.75 - k * 0.02, 0.54])
        ids = tracker.assign(np.array([c0, c1]))
        assert ids == [0, 1]


def test_ids_never_reused() -> None:
    """Verify deleted track IDs are never reallocated."""
    tracker = IdTracker(max_dist=0.15, max_missed=2)
    tracker.assign(np.array([[0.1, 0.1]]))

    for _ in range(5):
        tracker.assign(np.zeros((0, 2)))

    ids = tracker.assign(np.array([[0.1, 0.1]]))
    assert ids == [1]


def test_far_detection_gets_new_id() -> None:
    """Verify detection farther than max_dist from existing track gets a new ID."""
    tracker = IdTracker(max_dist=0.15, max_missed=15)
    tracker.assign(np.array([[0.1, 0.1]]))

    ids = tracker.assign(np.array([[0.8, 0.8]]))
    assert ids == [1]


def test_fuzz_random_centroids() -> None:
    """Fuzz: 1000 calls with 0 to 4 random centroids never raise, return correct len and unique IDs."""
    rng = np.random.default_rng(1234)
    tracker = IdTracker(max_dist=0.15, max_missed=15)

    for _ in range(1000):
        num_faces = rng.integers(0, 5)
        centroids = rng.uniform(0.0, 1.0, size=(num_faces, 2))
        ids = tracker.assign(centroids)
        assert len(ids) == num_faces
        assert len(set(ids)) == num_faces
