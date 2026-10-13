"""Unit tests for video sources."""
from __future__ import annotations

import csv
from pathlib import Path
import cv2
import numpy as np
import pytest

from trustsync.config import Config
from trustsync.io.video_source import FileVideoSource, WebcamSource


def test_file_video_source(tmp_path: Path) -> None:
    """Verify FileVideoSource reads all frames and csv times, and raises ValueError on count mismatch."""
    video_path = tmp_path / "test_video.mp4"
    csv_path = tmp_path / "test_times.csv"

    # Write a 30-frame 64x48 mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 48))
    for _ in range(30):
        frame = np.full((48, 64, 3), 120, dtype=np.uint8)
        out.write(frame)
    out.release()

    # Write matching 30-row csv
    times = [10.0 + k / 30.0 for k in range(30)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "t"])
        for idx, t in enumerate(times):
            writer.writerow([idx, f"{t:.4f}"])

    src = FileVideoSource(video_path, csv_path)
    read_frames = 0
    read_times = []

    for _ in range(35):
        ok, frame, t = src.read()
        if ok:
            read_frames += 1
            read_times.append(t)
            assert frame is not None
            assert frame.shape == (48, 64, 3)
        else:
            assert frame is None
            assert abs(t - times[-1]) < 1e-4

    assert read_frames == 30
    assert len(read_times) == 30
    src.close()

    # Write mismatched 29-row csv -> raises ValueError
    csv_bad = tmp_path / "bad_times.csv"
    with open(csv_bad, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "t"])
        for idx, t in enumerate(times[:29]):
            writer.writerow([idx, f"{t:.4f}"])

    with pytest.raises(ValueError, match="differs from video frame count"):
        FileVideoSource(video_path, csv_bad)


@pytest.mark.hardware
def test_webcam_source_hardware(spec_cfg: Config) -> None:
    """Read 10 frames from live WebcamSource with strictly increasing timestamps."""
    src = WebcamSource(spec_cfg)
    timestamps: list[float] = []
    try:
        for _ in range(10):
            ok, frame, t = src.read()
            if ok:
                assert frame is not None
                timestamps.append(t)

        assert len(timestamps) >= 5
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]
    finally:
        src.close()
