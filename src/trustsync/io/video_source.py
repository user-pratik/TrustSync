"""Video source implementations for live camera and session files."""
from __future__ import annotations

import csv
import time
from pathlib import Path
import cv2
import numpy as np

from trustsync.config import Config


class VideoSource:
    """Base interface for video sources."""

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Read next frame from source.

        Returns:
            tuple[bool, np.ndarray | None, float]: (ok, frame_bgr, timestamp).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release underlying video capture resources."""
        pass


class WebcamSource(VideoSource):
    """Live video capture source using OpenCV VideoCapture."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._cap = cv2.VideoCapture(cfg.video.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.video.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.video.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.video.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Read live webcam frame."""
        ok, frame = self._cap.read()
        t = time.monotonic()
        if not ok:
            return (False, None, t)
        return (True, frame, t)

    def close(self) -> None:
        """Release webcam resources."""
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()


class FileVideoSource(VideoSource):
    """File video playback synchronized with frame_times CSV."""

    def __init__(
        self, video_path: Path | str, frame_times_csv: Path | str
    ) -> None:
        self.video_path = Path(video_path)
        self.frame_times_csv = Path(frame_times_csv)

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self.video_path}")

        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timestamps: list[float] = []
        with open(self.frame_times_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row["t"]))

        if len(timestamps) != frame_count:
            self._cap.release()
            raise ValueError(
                f"csv row count ({len(timestamps)}) differs from "
                f"video frame count ({frame_count})"
            )

        self._timestamps = timestamps
        self._idx = 0
        self._last_t = timestamps[-1] if timestamps else 0.0

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Read next frame from video file."""
        if self._idx < len(self._timestamps):
            ok, frame = self._cap.read()
            t = self._timestamps[self._idx]
            self._last_t = t
            self._idx += 1
            if not ok:
                return (False, None, self._last_t)
            return (True, frame, t)
        return (False, None, self._last_t)

    def close(self) -> None:
        """Release video file capture."""
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()
