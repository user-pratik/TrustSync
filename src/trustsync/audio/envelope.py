"""Audio envelope extraction, Voice Activity Detection, and segmentation."""
from __future__ import annotations

import math
import numpy as np

from trustsync.config import VadCfg


def compute_envelope(x: np.ndarray, sr: int, env_hz: int) -> np.ndarray:
    """Compute float32 RMS envelope of an audio signal.

    Args:
        x: 1-D numpy array of audio samples.
        sr: Audio sample rate in Hz.
        env_hz: Target envelope sample rate in Hz.

    Returns:
        np.ndarray shape (ceil(len(x) / (sr // env_hz)),) float32 envelope values.

    Raises:
        ValueError: If x is not a 1-D array.
    """
    if x.ndim != 1:
        raise ValueError(f"Input audio array x must be 1-D, got shape {x.shape}")

    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    hop = sr // env_hz
    window_len = 2 * hop
    m = math.ceil(n / hop)

    # Pad x so that hop i has 2*hop window centred on the middle of that hop.
    # Hop i spans [i * hop, (i + 1) * hop) with middle at (i + 0.5) * hop.
    # A 2*hop window centred on (i + 0.5) * hop spans [i * hop - hop // 2, i * hop + 3 * hop // 2).
    # In padded coordinates with pad_left = hop // 2, window i starts at i * hop and has length 2 * hop.
    pad_left = hop // 2
    total_needed = (m - 1) * hop + window_len
    pad_right = max(0, total_needed - (n + pad_left))

    x_padded = np.pad(
        x.astype(np.float64),
        (pad_left, pad_right),
        mode="constant",
        constant_values=0.0,
    )

    sq_padded = x_padded**2
    windows = np.lib.stride_tricks.sliding_window_view(sq_padded, window_shape=window_len)[::hop][:m]

    mean_sq = np.mean(windows, axis=-1)
    return np.sqrt(mean_sq).astype(np.float32)


class VAD:
    """Energy-based Voice Activity Detector with hangover and noise adaptation."""

    def __init__(self, cfg: VadCfg, dt: float) -> None:
        self.cfg = cfg
        self.dt = dt
        self._calib_env: list[float] = []
        self._noise_floor: float = 0.0
        self._hangover_rem: float = 0.0
        self.last_on: bool = False

    @property
    def calibrated(self) -> bool:
        """Whether initial calibration period has elapsed."""
        return len(self._calib_env) * self.dt >= self.cfg.calib_s

    @property
    def noise_floor(self) -> float:
        """Current estimated background noise floor."""
        return self._noise_floor

    def update(self, env: float) -> bool:
        """Process one envelope sample and return speech activity decision."""
        if not self.calibrated:
            self._calib_env.append(env)
            self._noise_floor = float(np.mean(self._calib_env))
            self.last_on = False
            return False

        threshold = max(self._noise_floor * self.cfg.on_factor, self.cfg.min_abs)
        is_on = env > threshold
        self.last_on = is_on

        if is_on:
            self._hangover_rem = self.cfg.hangover_s
            output = True
        else:
            if self._hangover_rem > 0.0:
                output = True
                self._hangover_rem -= self.dt
            else:
                output = False

        if not output:
            self._noise_floor = (1.0 - self.cfg.adapt) * self._noise_floor + self.cfg.adapt * env
            if self._noise_floor < 0.0:
                self._noise_floor = 0.0

        return output


def segment_speech(
    env: np.ndarray, env_hz: float, cfg: VadCfg
) -> list[tuple[float, float]]:
    """Segment an envelope timeline into active speech intervals.

    Trailing hangover duration is trimmed so each segment boundaries are
    (time of first on sample, time of last on sample).

    Args:
        env: Audio envelope array.
        env_hz: Envelope sampling rate in Hz.
        cfg: VAD configuration.

    Returns:
        List of (t_start, t_end) speech segment timestamps in seconds.
    """
    dt = 1.0 / env_hz
    vad = VAD(cfg, dt)
    segments: list[tuple[float, float]] = []

    in_segment = False
    seg_start = 0.0
    last_on_time = 0.0

    for i in range(len(env)):
        t = i * dt
        e = float(env[i])
        active = vad.update(e)

        if active:
            if not in_segment:
                in_segment = True
                seg_start = t
                last_on_time = t
            if vad.last_on:
                last_on_time = t
        else:
            if in_segment:
                in_segment = False
                segments.append((seg_start, last_on_time))

    if in_segment:
        segments.append((seg_start, last_on_time))

    return segments
