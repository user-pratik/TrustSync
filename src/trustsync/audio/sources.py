"""Audio stream source implementations for offline arrays, files and live hardware."""
from __future__ import annotations

import collections
import math
from pathlib import Path
import threading
import time
from typing import Any
import numpy as np

from trustsync.audio.envelope import compute_envelope
from trustsync.config import AudioCfg


class AudioSource:
    """Base interface for all audio sources."""

    def envelope_at(self, times: np.ndarray) -> np.ndarray:
        """Sample audio envelope at specified query timestamps via interpolation.

        Args:
            times: 1-D numpy array of query timestamps in seconds.

        Returns:
            np.ndarray float32 envelope values (0.0 outside covered timeline).
        """
        raise NotImplementedError

    def latest(self) -> tuple[float, float]:
        """Return the latest timestamp and envelope sample."""
        raise NotImplementedError

    def close(self) -> None:
        """Release underlying resources."""
        pass


class NumpyAudioSource(AudioSource):
    """Audio source backed by a pre-computed envelope array."""

    def __init__(self, env: np.ndarray, env_hz: float, t0: float) -> None:
        self._env = np.asarray(env, dtype=np.float32)
        self._env_hz = float(env_hz)
        self._t0 = float(t0)
        n = len(self._env)
        self._timeline = self._t0 + np.arange(n, dtype=np.float64) / self._env_hz

    def envelope_at(self, times: np.ndarray) -> np.ndarray:
        if len(self._env) == 0:
            return np.zeros_like(times, dtype=np.float32)
        return np.interp(times, self._timeline, self._env, left=0.0, right=0.0).astype(
            np.float32
        )

    def latest(self) -> tuple[float, float]:
        if len(self._env) == 0:
            return (self._t0, 0.0)
        return (float(self._timeline[-1]), float(self._env[-1]))

    def close(self) -> None:
        pass


class FileAudioSource(AudioSource):
    """Audio source loaded from a WAV file on disk."""

    def __init__(self, wav_path: Path | str, t0: float, cfg: AudioCfg) -> None:
        import soundfile as sf

        p = Path(wav_path)
        data, sr = sf.read(str(p), dtype="float32")
        if sr != cfg.sample_rate:
            raise ValueError(
                f"WAV sample rate {sr} does not match audio.sample_rate {cfg.sample_rate}"
            )

        if data.ndim == 2:
            mono = np.mean(data, axis=1, dtype=np.float32)
        elif data.ndim == 1:
            mono = data
        else:
            raise ValueError(f"WAV data has unsupported shape {data.shape}")

        env = compute_envelope(mono, sr, cfg.env_hz)
        self._delegate = NumpyAudioSource(env, float(cfg.env_hz), t0)

    def envelope_at(self, times: np.ndarray) -> np.ndarray:
        return self._delegate.envelope_at(times)

    def latest(self) -> tuple[float, float]:
        return self._delegate.latest()

    def close(self) -> None:
        self._delegate.close()


class LiveAudioSource(AudioSource):
    """Thread-safe real-time microphone input stream."""

    def __init__(self, cfg: AudioCfg, record_to: Path | str | None = None) -> None:
        # Import sounddevice and soundfile lazily inside __init__
        import sounddevice as sd
        import soundfile as sf

        self.cfg = cfg
        self._lock = threading.Lock()
        max_samples = max(10, int(math.ceil(cfg.ring_s * cfg.env_hz)) + 10)
        self._buffer: collections.deque[tuple[float, float]] = collections.deque(
            maxlen=max_samples
        )
        self.t0: float | None = None
        self._wav_writer: Any = None

        if record_to is not None:
            rec_path = Path(record_to)
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            self._wav_writer = sf.SoundFile(
                str(rec_path),
                mode="w",
                samplerate=cfg.sample_rate,
                channels=1,
                subtype="FLOAT",
            )

        blocksize = cfg.sample_rate // cfg.env_hz
        block_s = blocksize / cfg.sample_rate

        def _callback(
            indata: np.ndarray, frames: int, time_info: Any, status: Any
        ) -> None:
            now = time.monotonic()
            mono = indata[:, 0]
            rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
            timestamp = now - self.cfg.latency_s - 0.5 * block_s
            with self._lock:
                if self._wav_writer is not None:
                    if self.t0 is None:
                        self.t0 = timestamp
                    self._wav_writer.write(mono)
                self._buffer.append((timestamp, rms))

        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=1,
            blocksize=blocksize,
            dtype="float32",
            device=cfg.device,
            callback=_callback,
        )

    def start(self) -> None:
        if self._stream is not None:
            self._stream.start()

    def envelope_at(self, times: np.ndarray) -> np.ndarray:
        with self._lock:
            if not self._buffer:
                return np.zeros_like(times, dtype=np.float32)
            t_grid = np.array([pt[0] for pt in self._buffer], dtype=np.float64)
            e_grid = np.array([pt[1] for pt in self._buffer], dtype=np.float32)

        return np.interp(times, t_grid, e_grid, left=0.0, right=0.0).astype(np.float32)

    def latest(self) -> tuple[float, float]:
        with self._lock:
            if not self._buffer:
                return (0.0, 0.0)
            return self._buffer[-1]

    def close(self) -> None:
        if hasattr(self, "_stream") and self._stream is not None:
            self._stream.stop()
            self._stream.close()
        with self._lock:
            if self._wav_writer is not None:
                self._wav_writer.close()
                self._wav_writer = None
