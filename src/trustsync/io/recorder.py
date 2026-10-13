"""Session recording utility capturing synchronized video, audio, and metadata."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np

from trustsync.audio.sources import AudioSource, LiveAudioSource
from trustsync.config import Config
from trustsync.io.video_source import VideoSource, WebcamSource


class SessionRecorder:
    """Session recorder writing video.mp4, audio.wav, frame_times.csv, and meta.json."""

    def __init__(
        self,
        cfg: Config,
        out_dir: Path | str,
        meta: dict,
        video: VideoSource | None = None,
        audio: AudioSource | None = None,
    ) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.meta = dict(meta)

        self.audio = (
            audio
            if audio is not None
            else LiveAudioSource(cfg.audio, record_to=self.out_dir / "audio.wav")
        )
        self.video = video if video is not None else WebcamSource(cfg)

    def run(self, duration_s: float, preview: bool = True) -> Path:
        """Execute session recording for duration_s.

        Args:
            duration_s: Target recording length in seconds.
            preview: If True, render real-time preview window with REC indicator.

        Returns:
            Path: Output directory containing recorded session assets.

        Raises:
            FileExistsError: If video.mp4 already exists in out_dir.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        video_path = self.out_dir / "video.mp4"
        if video_path.exists():
            raise FileExistsError(f"video.mp4 already exists at {video_path}")

        # Start audio first
        if hasattr(self.audio, "start"):
            self.audio.start()

        writer: cv2.VideoWriter | None = None
        frame_records: list[tuple[int, float]] = []
        t_start = time.monotonic()
        frame_idx = 0

        try:
            while True:
                ok, frame, t = self.video.read()
                if not ok or frame is None:
                    break

                now = time.monotonic()
                elapsed = now - t_start
                remaining = max(0.0, duration_s - elapsed)

                h, w = frame.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(video_path), fourcc, float(self.cfg.video.fps), (w, h)
                    )

                writer.write(frame)
                frame_records.append((frame_idx, t))
                frame_idx += 1

                if preview:
                    disp = frame.copy()
                    # Red REC indicator dot
                    cv2.circle(disp, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(
                        disp,
                        f"REC {elapsed:.1f}s / {remaining:.1f}s",
                        (50, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    prompt_text = self.meta.get("prompt_text")
                    if prompt_text:
                        cv2.putText(
                            disp,
                            str(prompt_text),
                            (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    cv2.imshow("TrustSync Recording", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if elapsed >= duration_s:
                    break
        finally:
            if writer is not None:
                writer.release()
            if preview:
                cv2.destroyAllWindows()
            if hasattr(self.audio, "close"):
                self.audio.close()
            if hasattr(self.video, "close"):
                self.video.close()

        # Write frame_times.csv
        csv_path = self.out_dir / "frame_times.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["frame_idx", "t"])
            for f_idx, f_t in frame_records:
                csv_writer.writerow([f_idx, f"{f_t:.4f}"])

        # Calculate video_fps
        n = len(frame_records)
        if n >= 2 and (frame_records[-1][1] - frame_records[0][1]) > 0.0:
            video_fps = (n - 1) / (frame_records[-1][1] - frame_records[0][1])
        else:
            video_fps = float(self.cfg.video.fps)

        audio_t0 = getattr(self.audio, "t0", None)
        if audio_t0 is None:
            audio_t0 = frame_records[0][1] if frame_records else 0.0

        # Write meta.json
        meta_out = dict(self.meta)
        meta_out["audio_t0"] = float(audio_t0)
        meta_out["video_fps"] = float(video_fps)

        meta_path = self.out_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

        return self.out_dir
