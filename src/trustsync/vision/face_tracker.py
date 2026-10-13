"""Real-time multi-face mesh detector and tracker."""
from __future__ import annotations

import cv2
import numpy as np

from trustsync.config import Config
from trustsync.types import FaceObs
from trustsync.vision.id_tracker import IdTracker
from trustsync.vision.lip import mouth_open
from trustsync.vision.quality import face_quality


class FaceTracker:
    """MediaPipe FaceMesh detector coupled with centroid IdTracker."""

    def __init__(self, cfg: Config) -> None:
        import mediapipe as mp

        self.cfg = cfg
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=cfg.video.max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._id_tracker = IdTracker(
            max_dist=cfg.tracker.max_dist,
            max_missed=cfg.tracker.max_missed,
        )
        self._prev_frame: np.ndarray | None = None

    def process(self, frame_bgr: np.ndarray) -> list[FaceObs]:
        """Detect faces in BGR frame and return persistent FaceObs list.

        Args:
            frame_bgr: Input BGR video frame.

        Returns:
            list[FaceObs]: Detected faces, or [] if no faces found.
        """
        h, w = frame_bgr.shape[:2]
        if self._prev_frame is not None and self._prev_frame.shape == frame_bgr.shape:
            diff = float(
                np.mean(
                    np.abs(
                        frame_bgr.astype(np.float32)
                        - self._prev_frame.astype(np.float32)
                    )
                )
            )
            if diff > 15.0:
                self._mesh.reset()
        self._prev_frame = frame_bgr.copy()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self._id_tracker.assign(np.zeros((0, 2), dtype=np.float64))
            return []

        raw_faces: list[tuple[np.ndarray, tuple[float, float, float, float]]] = []
        centroids: list[list[float]] = []

        for face_landmarks in results.multi_face_landmarks:
            pts = np.zeros((478, 3), dtype=np.float64)
            for i, lm in enumerate(face_landmarks.landmark):
                if i >= 478:
                    break
                pts[i, 0] = lm.x
                pts[i, 1] = lm.y
                pts[i, 2] = lm.z

            x1 = float(np.clip(np.min(pts[:, 0]), 0.0, 1.0))
            y1 = float(np.clip(np.min(pts[:, 1]), 0.0, 1.0))
            x2 = float(np.clip(np.max(pts[:, 0]), 0.0, 1.0))
            y2 = float(np.clip(np.max(pts[:, 1]), 0.0, 1.0))
            bbox = (x1, y1, x2, y2)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centroids.append([cx, cy])
            raw_faces.append((pts, bbox))

        track_ids = self._id_tracker.assign(np.array(centroids, dtype=np.float64))

        obs_list: list[FaceObs] = []
        for i, (pts, bbox) in enumerate(raw_faces):
            tid = track_ids[i]
            m_open = mouth_open(pts, w, h)
            q = face_quality(frame_bgr, bbox, pts, self.cfg.quality)
            obs_list.append(
                FaceObs(
                    track_id=tid,
                    bbox=bbox,
                    pts=pts,
                    mouth_open=m_open,
                    quality=q,
                )
            )

        return obs_list

    def close(self) -> None:
        """Release FaceMesh resources."""
        if hasattr(self, "_mesh") and self._mesh is not None:
            self._mesh.close()
