"""Runtime manager executing TrustSync pipeline in a background thread."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from trustsync.config import Config, load_config, validate
from trustsync.fusion.gate import make_gate
from trustsync.pipeline import TrustSyncPipeline
from trustsync.types import FrameResult, frame_result_to_dict

logger = logging.getLogger(__name__)


@dataclass
class Sources:
    """Sources container for video, audio, tracker, and playback mode."""

    video: Any
    audio: Any
    tracker: Any
    realtime: bool


def default_build_sources(
    kind: str, session_id: str | None, cfg: Config, root: Path
) -> Sources:
    """Default factory instantiating live hardware or recorded session sources."""
    if kind == "webcam":
        from trustsync.audio.sources import LiveAudioSource
        from trustsync.io.video_source import WebcamSource

        try:
            video = WebcamSource(cfg)
        except Exception as e:
            raise RuntimeError(f"cannot open camera: {e}") from e

        audio = LiveAudioSource(cfg.audio)
        return Sources(video=video, audio=audio, tracker=None, realtime=False)

    if kind == "session":
        if not session_id:
            raise ValueError("session_id required for session source")
        session_path = root / "data" / "sessions" / session_id
        if not session_path.is_dir():
            raise FileNotFoundError(f"Session not found: {session_id}")
        meta_file = session_path / "meta.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"Session meta.json not found: {session_id}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        from trustsync.audio.sources import FileAudioSource
        from trustsync.io.video_source import FileVideoSource

        video = FileVideoSource(session_path / "video.mp4", session_path / "frame_times.csv")
        audio = FileAudioSource(
            session_path / "audio.wav", t0=float(meta["audio_t0"]), cfg=cfg.audio
        )
        return Sources(video=video, audio=audio, tracker=None, realtime=True)

    raise ValueError(f"Unknown source kind: {kind}")


class Runtime:
    """Background pipeline runtime orchestrating video capture, audio, and websocket publishing."""

    def __init__(
        self,
        cfg: Config,
        root: Path | None = None,
        factory: Callable[[str, str | None, Config, Path], Sources] | None = None,
    ) -> None:
        self.cfg = cfg
        self.root = root if root is not None else Path(__file__).resolve().parents[3]
        self.factory = factory or default_build_sources

        self._running = False
        self._source_name: str | None = None
        self._error: str | None = None

        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

        self.sources: Sources | None = None
        self.pipeline: TrustSyncPipeline | None = None

        self._subscribers: set[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = set()
        self._sub_lock = threading.Lock()

        # Pre-render 640x480 dark placeholder image with 'No source'
        placeholder = np.full((480, 640, 3), (20, 14, 10), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "No source",
            (220, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (230, 237, 243),
            2,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(
            ".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.server.jpeg_quality]
        )
        self._placeholder_jpeg = buf.tobytes() if ok else b""
        self._latest_jpeg = self._placeholder_jpeg

    def subscribe(self) -> asyncio.Queue:
        """Subscribe an asyncio queue to receive realtime server messages."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        with self._sub_lock:
            self._subscribers.add((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Unsubscribe a queue from receiving messages."""
        with self._sub_lock:
            self._subscribers = {item for item in self._subscribers if item[0] is not q}

    def _publish(self, msg: dict[str, Any]) -> None:
        """Publish message thread-safely to all subscriber queues, dropping oldest if full."""
        with self._sub_lock:
            subs = list(self._subscribers)

        def _put_message(queue: asyncio.Queue, message: dict[str, Any]) -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

        for q, loop in subs:
            try:
                if not loop.is_closed():
                    if loop.is_running():
                        loop.call_soon_threadsafe(_put_message, q, msg)
                    else:
                        _put_message(q, msg)
            except Exception:
                pass

    def start(self, kind: str, session_id: str | None = None) -> None:
        """Start streaming pipeline with specified source kind."""
        self.stop()

        self.sources = self.factory(kind, session_id, self.cfg, self.root)
        if hasattr(self.sources.audio, "start"):
            self.sources.audio.start()
        tracker = self.sources.tracker
        self.pipeline = TrustSyncPipeline(
            self.cfg, audio=self.sources.audio, gate=make_gate(self.cfg), tracker=tracker
        )

        self._source_name = f"session:{session_id}" if kind == "session" else kind
        self._error = None
        self._stop_requested = False
        self._running = True

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background worker thread and release sources."""
        self._stop_requested = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if self.sources is not None:
            if hasattr(self.sources.video, "close"):
                self.sources.video.close()
            if hasattr(self.sources.audio, "close"):
                self.sources.audio.close()
            self.sources = None

        self._running = False
        self._latest_jpeg = self._placeholder_jpeg

    def status(self) -> dict[str, Any]:
        """Return runtime execution status."""
        return {
            "running": self._running,
            "source": self._source_name if self._running else None,
            "error": self._error,
        }

    def get_config_dict(self) -> dict[str, Any]:
        """Return current configuration as JSON-safe dictionary."""
        return self.cfg.to_dict()

    def set_config(self, section: str, key: str, value: Any) -> dict[str, Any]:
        """Update configurable parameters in gate, addressee, or sync."""
        if section not in ("gate", "addressee", "sync"):
            raise ValueError(f"Section '{section}' is not modifiable")

        if section == "sync" and key != "min_lip_std":
            raise ValueError(f"Only sync.min_lip_std is modifiable in sync, got {key}")

        current_sec = getattr(self.cfg, section)
        if not hasattr(current_sec, key):
            raise ValueError(f"Unknown key '{key}' in section '{section}'")

        if section == "addressee" and key == "use_gaze":
            if not isinstance(value, bool):
                raise ValueError("addressee.use_gaze must be a boolean")
            val = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{section}.{key} must be numeric")
            val = float(value)

        # Value constraints
        if section == "gate":
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Gate threshold {key} must be in [0, 1]")
        elif section == "addressee":
            if key not in ("yaw_offset_deg", "pitch_offset_deg") and isinstance(val, (int, float)):
                if val <= 0.0:
                    raise ValueError(f"Addressee parameter {key} must be positive")
        elif section == "sync" and key == "min_lip_std":
            if val <= 0.0:
                raise ValueError("min_lip_std must be positive")

        # Build new config and validate
        updated_sec = dataclasses.replace(current_sec, **{key: val})
        new_cfg = dataclasses.replace(self.cfg, **{section: updated_sec})
        validate(new_cfg)
        self.cfg = new_cfg

        # If running, rebuild pipeline under lock to update thresholds
        with self._lock:
            if self._running and self.sources is not None:
                self.pipeline = TrustSyncPipeline(
                    self.cfg,
                    audio=self.sources.audio,
                    gate=make_gate(self.cfg),
                    tracker=self.sources.tracker,
                )

        return self.cfg.to_dict()

    def latest_jpeg(self) -> bytes:
        """Return most recent camera frame JPEG or placeholder."""
        return self._latest_jpeg

    def _worker_loop(self) -> None:
        """Worker loop processing frames, encoding JPEG, and publishing over WebSocket."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.server.jpeg_quality]
        ws_interval = 1.0 / self.cfg.server.ws_hz
        last_ws_pub = 0.0

        t_first_frame: float | None = None
        wall_start: float | None = None

        self._publish({"type": "status", "data": self.status()})

        consecutive_failures = 0

        try:
            while not self._stop_requested:
                try:
                    with self._lock:
                        if self.sources is None or self.pipeline is None:
                            break
                        sources = self.sources
                        pipeline = self.pipeline

                    ok, frame, t = sources.video.read()
                    if not ok or frame is None:
                        break

                    if sources.realtime:
                        if t_first_frame is None:
                            t_first_frame = t
                            wall_start = time.monotonic()
                        else:
                            target_wall = wall_start + (t - t_first_frame)
                            sleep_s = target_wall - time.monotonic()
                            if sleep_s > 0:
                                time.sleep(sleep_s)

                    with self._lock:
                        result = pipeline.process(frame, t)

                    # Encode raw frame as JPEG
                    ok_enc, buf = cv2.imencode(".jpg", frame, encode_param)
                    if ok_enc:
                        self._latest_jpeg = buf.tobytes()

                    now = time.monotonic()
                    if (now - last_ws_pub) >= ws_interval:
                        last_ws_pub = now
                        self._publish({"type": "frame", "data": frame_result_to_dict(result)})

                    if result.utterance_final:
                        self._publish({"type": "utterance", "data": frame_result_to_dict(result)})

                    consecutive_failures = 0

                except Exception as frame_exc:
                    consecutive_failures += 1
                    logger.warning(
                        "Frame processing exception (%d/30): %s",
                        consecutive_failures,
                        frame_exc,
                    )
                    if consecutive_failures >= 30:
                        self._error = str(frame_exc)
                        break

        except Exception as e:
            logger.exception("Runtime worker loop encountered an error")
            self._error = str(e)
        finally:
            self._running = False
            self._publish({"type": "status", "data": self.status()})
