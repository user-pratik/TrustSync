"""Tests for SessionRecorder using synthetic audio and video sources."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from trustsync.config import Config
from trustsync.io.recorder import SessionRecorder


class FakeVideo:
    """Mock video source supplying 60 frames of black 480x640."""

    def __init__(self, n_frames: int = 60, fps: float = 30.0, t0: float = 100.0) -> None:
        self.n_frames = n_frames
        self.fps = fps
        self.t0 = t0
        self.idx = 0

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self.idx >= self.n_frames:
            return False, None, self.t0 + (self.n_frames - 1) / self.fps
        t = self.t0 + self.idx / self.fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.idx += 1
        return True, frame, t

    def close(self) -> None:
        pass


class FakeAudio:
    """Mock audio source writing 2 s 16 kHz noise WAV on start()."""

    def __init__(self, record_to: Path | str, t0: float = 100.0) -> None:
        self.record_to = Path(record_to)
        self.t0 = t0

    def start(self) -> None:
        rng = np.random.default_rng(1234)
        noise = (rng.standard_normal(32000) * 0.1).astype(np.float32)
        sf.write(str(self.record_to), noise, 16000)

    def close(self) -> None:
        pass


def test_session_recorder(tmp_path: Path, spec_cfg: Config) -> None:
    session_dir = tmp_path / "rec_test_001"
    meta = {
        "id": "rec_test_001",
        "scenario": "live_single",
        "subjects": ["tester"],
        "speaker_side": "center",
        "addressed": True,
        "expected": "ACCEPT",
        "lighting": "normal",
        "occlusion": False,
        "spoof_device": None,
        "notes": "recorder test",
    }

    video = FakeVideo(n_frames=60, fps=30.0, t0=100.0)
    audio = FakeAudio(record_to=session_dir / "audio.wav", t0=100.0)

    recorder = SessionRecorder(
        cfg=spec_cfg,
        out_dir=session_dir,
        meta=meta,
        video=video,
        audio=audio,
    )
    result_path = recorder.run(duration_s=2.0, preview=False)
    assert result_path == session_dir

    # All four files written
    assert (session_dir / "video.mp4").is_file()
    assert (session_dir / "audio.wav").is_file()
    assert (session_dir / "frame_times.csv").is_file()
    assert (session_dir / "meta.json").is_file()

    # frame_times has 60 rows
    with open(session_dir / "frame_times.csv", "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 60

    # meta.json contains audio_t0 100.0 and video_fps within 1 of 30
    with open(session_dir / "meta.json", "r", encoding="utf-8") as f:
        saved_meta = json.load(f)
    assert abs(saved_meta["audio_t0"] - 100.0) < 1e-4
    assert abs(saved_meta["video_fps"] - 30.0) <= 1.0

    # A second run into the same folder raises FileExistsError
    video2 = FakeVideo(n_frames=60, fps=30.0, t0=100.0)
    audio2 = FakeAudio(record_to=session_dir / "audio.wav", t0=100.0)
    recorder2 = SessionRecorder(
        cfg=spec_cfg,
        out_dir=session_dir,
        meta=meta,
        video=video2,
        audio=audio2,
    )
    with pytest.raises(FileExistsError):
        recorder2.run(duration_s=2.0, preview=False)
