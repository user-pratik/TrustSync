"""Unit tests for audio envelope computation."""
import math
import numpy as np
import pytest

from trustsync.audio.envelope import compute_envelope


def test_sine_rms_and_length() -> None:
    """Verify 1 kHz sine envelope accuracy and output length."""
    sr = 16000
    env_hz = 100
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr
    sine = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)

    env = compute_envelope(sine, sr, env_hz)
    hop = sr // env_hz
    expected_len = math.ceil(len(sine) / hop)
    assert len(env) == expected_len

    # Interior values (away from edges) should be within 5% of theoretical RMS 0.3536
    interior = env[10:-10]
    expected_rms = 0.3536
    assert np.all(np.abs(interior - expected_rms) / expected_rms < 0.05)


def test_zero_and_empty_inputs() -> None:
    """Verify all-zero input and empty input handling."""
    sr = 16000
    env_hz = 100

    # All-zero input produces values below 1e-9
    zeros = np.zeros(1600, dtype=np.float32)
    env_zeros = compute_envelope(zeros, sr, env_hz)
    assert np.all(env_zeros < 1e-9)

    # Empty input gives empty output
    empty = np.zeros(0, dtype=np.float32)
    env_empty = compute_envelope(empty, sr, env_hz)
    assert len(env_empty) == 0
    assert env_empty.dtype == np.float32


def test_invalid_dimension_raises() -> None:
    """Verify 2-D array input raises ValueError."""
    array_2d = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        compute_envelope(array_2d, 16000, 100)
